// =============================================================================
// PtlasWriteUpdateRepro - minimal repro for NVIDIA driver investigation
// =============================================================================
//
// Symptom:
//   GPU TDR (DXGI_ERROR_DEVICE_HUNG) on a PTLAS build whose
//   ExecuteIndirectRTASOperations call contains BOTH a WRITE_INSTANCE op
//   AND an UPDATE_INSTANCE op, when the WRITE's InstanceIndex is NON-ZERO
//   AND the UPDATE's InstanceIndex is also NON-ZERO.
//
// If EITHER op targets InstanceIndex=0, no TDR.
// If only one op type is present, no TDR.
// WARP runs the same code with no TDR.
//
// Driver: NVIDIA RTX 4090, preview driver (2026-05).
//
// Default scene (the minimum that reproduces): 3 instances, 1 partition,
// 1 BLAS.  Per frame: a single ExecuteIndirectRTASOperations call with
//   - WRITE_INSTANCE  (InstanceIndex = 1)
//   - UPDATE_INSTANCE (InstanceIndex = 2)
// No DispatchRays, no TRANSLATE_PARTITION, no ENABLE_EXPLICIT_AABB, no
// shaders, no pipeline state.  TDR on frame 1.
//
// (W,U) bisect grid -- "TDR iff W>0 AND U>0":
//
//     U=0 U=1 U=2 U=3
//   W=0  -   OK   OK   OK
//   W=1  OK   -   TDR  TDR
//   W=2  OK   TDR   -   TDR
//   W=3  OK   TDR  TDR   -
//
// CLI:
//   --frames N         : run N frames (default 30)
//   --instances N      : PTLAS instance count (default 3; minimum 3 to trigger)
//   --write IDX        : InstanceIndex for WRITE_INSTANCE arg (default 1; -1 = frame % N)
//   --update IDX       : InstanceIndex for UPDATE_INSTANCE arg (default 2; -1 = (frame+2) % N)
//   --no-write         : skip WRITE_INSTANCE  -> baseline (OK)
//   --no-update        : skip UPDATE_INSTANCE -> baseline (OK)
//   --use-warp         : force WARP adapter   -> baseline (OK)
//
// Exit: 0 = no TDR, 1 = TDR detected, 2 = setup error.
// =============================================================================

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wrl/client.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <atomic>
#include <string>

using Microsoft::WRL::ComPtr;

extern "C" __declspec(dllexport) extern const UINT  D3D12SDKVersion = 721;
extern "C" __declspec(dllexport) extern const char* D3D12SDKPath    = ".\\D3D12\\";

struct Config {
    UINT frames    = 30;
    bool noWrite   = false;
    bool noUpdate  = false;
    bool useWarp   = false;
    int  writeInst = 1;     // diag: hardcoded W's InstanceIndex (negative = use frame % instCount)
    int  updateInst = 2;    // diag: hardcoded U's InstanceIndex (negative = use (frame+2) % instCount)
    UINT instCount = 3;     // diag: total instance count in PTLAS (minimum 3 to trigger this bug)
} g_cfg;

static std::atomic<bool> g_deviceRemoved{false};
static std::string       g_deviceRemovedReason;

static void Die(const char* msg, HRESULT hr = S_OK) {
    fprintf(stderr, "FATAL: %s (hr=0x%08lx)\n", msg, (unsigned long)hr);
    exit(2);
}
static void CheckHR(HRESULT hr, const char* msg) { if (FAILED(hr)) Die(msg, hr); }

static void __stdcall InfoQueueCallback(D3D12_MESSAGE_CATEGORY,
                                        D3D12_MESSAGE_SEVERITY sev,
                                        D3D12_MESSAGE_ID id,
                                        LPCSTR description, void*) {
    fprintf(stderr, "[d3d12 sev=%d id=%u] %s\n", (int)sev, (unsigned)id, description ? description : "");
    if (id == 232 || sev == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
        g_deviceRemoved = true;
        if (description) g_deviceRemovedReason = description;
    }
}

static ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device* d, UINT64 size, D3D12_HEAP_TYPE heap,
                                           D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES hp = {};                          hp.Type = heap;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;  rd.Height = 1;  rd.DepthOrArraySize = 1;  rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;  rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;  rd.Flags = flags;
    ComPtr<ID3D12Resource> r;
    CheckHR(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&r)),
            "CreateCommittedResource");
    return r;
}

static void ParseCli(int argc, wchar_t** argv) {
    for (int i = 1; i < argc; ++i) {
        if      (!wcscmp(argv[i], L"--frames")  && i + 1 < argc) g_cfg.frames      = (UINT)_wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--no-write"))                 g_cfg.noWrite     = true;
        else if (!wcscmp(argv[i], L"--no-update"))                g_cfg.noUpdate    = true;
        else if (!wcscmp(argv[i], L"--use-warp"))                 g_cfg.useWarp     = true;
        else if (!wcscmp(argv[i], L"--write")   && i + 1 < argc)  g_cfg.writeInst   = _wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--update")  && i + 1 < argc)  g_cfg.updateInst  = _wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--instances") && i + 1 < argc) g_cfg.instCount  = (UINT)_wtoi(argv[++i]);
        else fwprintf(stderr, L"unknown arg: %s\n", argv[i]);
    }
}

int wmain(int argc, wchar_t** argv) {
    ParseCli(argc, argv);
    printf("PtlasWriteUpdateRepro: frames=%u noWrite=%d noUpdate=%d warp=%d\n",
           g_cfg.frames, g_cfg.noWrite, g_cfg.noUpdate, g_cfg.useWarp);

    UUID experimentalFeatures[] = { D3D12ExperimentalShaderModels, D3D12RaytracingExperiment };
    if (FAILED(D3D12EnableExperimentalFeatures(_countof(experimentalFeatures),
                                                experimentalFeatures, nullptr, nullptr))) {
        fprintf(stderr, "D3D12EnableExperimentalFeatures failed.\n"); return 2;
    }

    ComPtr<IDXGIFactory6> factory;
    CheckHR(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

    ComPtr<IDXGIAdapter1> adapter;
    if (g_cfg.useWarp) {
        CheckHR(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "EnumWarpAdapter");
    } else {
        for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                              IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc;  adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter.Reset(); continue; }
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_2, _uuidof(ID3D12Device), nullptr))) break;
            adapter.Reset();
        }
    }
    if (!adapter) Die("no D3D12 adapter found");
    DXGI_ADAPTER_DESC1 desc; adapter->GetDesc1(&desc);
    wprintf(L"  Adapter: %s\n", desc.Description);

    ComPtr<ID3D12Device5> device;
    CheckHR(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_2, IID_PPV_ARGS(&device)),
            "D3D12CreateDevice");
    {
        ComPtr<ID3D12InfoQueue1> iq;
        if (SUCCEEDED(device.As(&iq))) {
            DWORD cookie = 0;
            iq->RegisterMessageCallback(InfoQueueCallback, D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                                        nullptr, &cookie);
        }
    }
    ComPtr<ID3D12DeviceRaytracing2> deviceRT2;
    CheckHR(device.As(&deviceRT2), "QI(ID3D12DeviceRaytracing2)");

    D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL exp = {};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL, &exp, sizeof(exp)))
        || !exp.ClustersAndPTLASSupported) {
        fprintf(stderr, "ClustersAndPTLASSupported = NO.\n");  return 2;
    }

    // Queue / list / fence.
    D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    CheckHR(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
    ComPtr<ID3D12CommandAllocator> alloc;
    CheckHR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)),
            "CreateCommandAllocator");
    ComPtr<ID3D12GraphicsCommandList4> cl;
    {
        ComPtr<ID3D12GraphicsCommandList> cl0;
        CheckHR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr,
                                          IID_PPV_ARGS(&cl0)), "CreateCommandList");
        CheckHR(cl0.As(&cl), "QI(CL4)");
    }
    ComPtr<ID3D12CommandListRaytracing2> cl2;
    CheckHR(cl.As(&cl2), "QI(CL_RT2)");

    ComPtr<ID3D12Fence> fence;
    CheckHR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    HANDLE fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fenceValue = 0;
    auto flushAndWait = [&](const char* tag) {
        CheckHR(cl->Close(), "Close");
        ID3D12CommandList* cls[] = { cl.Get() };
        queue->ExecuteCommandLists(1, cls);
        CheckHR(queue->Signal(fence.Get(), ++fenceValue), "Signal");
        if (fence->GetCompletedValue() < fenceValue) {
            fence->SetEventOnCompletion(fenceValue, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        if (g_deviceRemoved.load()) {
            printf("!!! DEVICE REMOVED after %s: %s\n", tag, g_deviceRemovedReason.c_str());
            fprintf(stderr, "    GetDeviceRemovedReason = 0x%08lx\n",
                    (unsigned long)device->GetDeviceRemovedReason());
            exit(1);
        }
        CheckHR(alloc->Reset(), "alloc->Reset");
        CheckHR(cl->Reset(alloc.Get(), nullptr), "cl->Reset");
    };

    // -- Two triangle BLASes -------------------------------------------------
    // Two so we can alternate between them in WRITE/UPDATE args -- the
    // original sample's TDR happened across LOD swaps where the BLAS GPUVA
    // changed per frame.  We test below whether the alternation matters.
    struct V { float p[3]; };
    V verts[] = { {-0.5f,-0.5f,0}, {0.5f,-0.5f,0}, {0,0.5f,0} };
    uint32_t idx[] = { 0, 1, 2 };

    auto vb = CreateBuffer(device.Get(), sizeof(verts), D3D12_HEAP_TYPE_UPLOAD,
                           D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto ib = CreateBuffer(device.Get(), sizeof(idx), D3D12_HEAP_TYPE_UPLOAD,
                           D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    { void* p; D3D12_RANGE no = {0,0}; vb->Map(0, &no, &p); memcpy(p, verts, sizeof(verts)); vb->Unmap(0, nullptr); }
    { void* p; D3D12_RANGE no = {0,0}; ib->Map(0, &no, &p); memcpy(p, idx,   sizeof(idx));   ib->Unmap(0, nullptr); }

    auto buildOneBlas = [&]() -> ComPtr<ID3D12Resource> {
        D3D12_RAYTRACING_GEOMETRY_DESC geo = {};
        geo.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geo.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geo.Triangles.VertexBuffer.StartAddress  = vb->GetGPUVirtualAddress();
        geo.Triangles.VertexBuffer.StrideInBytes = sizeof(V);
        geo.Triangles.VertexCount  = 3;
        geo.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        geo.Triangles.IndexBuffer  = ib->GetGPUVirtualAddress();
        geo.Triangles.IndexCount   = 3;
        geo.Triangles.IndexFormat  = DXGI_FORMAT_R32_UINT;
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bi = {};
        bi.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        bi.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        bi.NumDescs = 1;
        bi.pGeometryDescs = &geo;
        bi.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bpre = {};
        device->GetRaytracingAccelerationStructurePrebuildInfo(&bi, &bpre);
        auto b = CreateBuffer(device.Get(), bpre.ResultDataMaxSizeInBytes,
                              D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                              D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
        auto s = CreateBuffer(device.Get(), bpre.ScratchDataSizeInBytes,
                              D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_COMMON);
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd = {};
        bd.Inputs = bi;
        bd.DestAccelerationStructureData    = b->GetGPUVirtualAddress();
        bd.ScratchAccelerationStructureData = s->GetGPUVirtualAddress();
        cl->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);
        D3D12_RESOURCE_BARRIER uav = {};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = b.Get();
        cl->ResourceBarrier(1, &uav);
        // keep scratch alive past the build
        static std::vector<ComPtr<ID3D12Resource>> s_keepAlive;
        s_keepAlive.push_back(s);
        return b;
    };
    auto blasA = buildOneBlas();
    auto blasB = buildOneBlas();
    flushAndWait("BLAS build");
    printf("  blasA = 0x%llx  blasB = 0x%llx\n",
           (unsigned long long)blasA->GetGPUVirtualAddress(),
           (unsigned long long)blasB->GetGPUVirtualAddress());

    // -- PTLAS: g_cfg.instCount instances, 1 partition, no flags ------------
    D3D12_RTAS_PARTITIONED_TLAS_INPUTS_DESC ptlasIn = {};
    ptlasIn.InstanceCount  = g_cfg.instCount;
    ptlasIn.PartitionCount = 1;
    ptlasIn.MaxInstancePerPartitionCount = g_cfg.instCount;
    ptlasIn.MaxInstanceInGlobalPartitionCount = 0;
    ptlasIn.Flags = D3D12_RTAS_PARTITIONED_TLAS_FLAG_NONE;

    D3D12_RTAS_OPERATION_INPUTS opIn = {};
    opIn.Type = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
    opIn.pPartitionedTLASInputsDesc = &ptlasIn;
    D3D12_RTAS_OPERATION_PREBUILD_INFO ppre = {};
    deviceRT2->GetRTASOperationPrebuildInfo(&opIn, &ppre);
    auto ptlas = CreateBuffer(device.Get(), ppre.ResultDataMaxSizeInBytes,
                              D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                              D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    auto ptlasScratch = CreateBuffer(device.Get(), ppre.ScratchDataSizeInBytes,
                                     D3D12_HEAP_TYPE_DEFAULT,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                     D3D12_RESOURCE_STATE_COMMON);

    // -- One upload buffer.  We wait fence every frame, so we can reuse it. -
    constexpr UINT64 kArenaBytes = 64 * 1024;
    auto arena = CreateBuffer(device.Get(), kArenaBytes, D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* arenaPtr = nullptr;
    { D3D12_RANGE no = {0,0}; arena->Map(0, &no, &arenaPtr); }

    // Builds a PTLAS op array of {write, update} (each empty or 1 entry) into
    // the arena and submits ExecuteIndirectRTASOperations.
    // Both BLAS pointer AND instance index vary per frame (W targets
    // (frame%4); U targets ((frame+2)%4)).
    auto buildPtlas = [&](UINT frame, bool firstBuild, bool emitWrite, bool emitUpdate) {
        D3D12_GPU_VIRTUAL_ADDRESS blasW = (frame & 1) ? blasB->GetGPUVirtualAddress()
                                                       : blasA->GetGPUVirtualAddress();
        D3D12_GPU_VIRTUAL_ADDRESS blasU = (frame & 1) ? blasA->GetGPUVirtualAddress()
                                                       : blasB->GetGPUVirtualAddress();
        const UINT writeInst  = (UINT)((g_cfg.writeInst  < 0) ? (int)(frame % g_cfg.instCount) : g_cfg.writeInst);
        const UINT updateInst = (UINT)((g_cfg.updateInst < 0) ? (int)((frame + 2) % g_cfg.instCount) : g_cfg.updateInst);
        uint8_t* head = (uint8_t*)arenaPtr;
        D3D12_GPU_VIRTUAL_ADDRESS baseGva = arena->GetGPUVirtualAddress();

        D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS write = {};
        write.Transform[0][0] = 1; write.Transform[1][1] = 1; write.Transform[2][2] = 1;
        write.Transform[2][3] = 5.0f;
        write.InstanceID = writeInst;
        write.InstanceMask = 0xFF;
        write.AccelerationStructure = blasW;
        write.InstanceIndex = writeInst;
        write.PartitionIndex = 0;

        D3D12_RTAS_PARTITIONED_TLAS_OPERATION_UPDATE_INSTANCE_ARGS update = {};
        update.InstanceIndex = updateInst;
        update.AccelerationStructure = blasU;

        // Lay out: write_args, update_args, op_headers, op_count.
        D3D12_GPU_VIRTUAL_ADDRESS writeGva = 0, updateGva = 0;
        UINT64 head64 = 0;
        if (emitWrite) {
            memcpy(head + head64, &write, sizeof(write));
            writeGva = baseGva + head64;
            head64 += sizeof(write);
        }
        if (emitUpdate) {
            head64 = (head64 + 7) & ~UINT64(7);
            memcpy(head + head64, &update, sizeof(update));
            updateGva = baseGva + head64;
            head64 += sizeof(update);
        }

        D3D12_RTAS_PARTITIONED_TLAS_OPERATION ops[2] = {};
        UINT numOps = 0;
        if (emitWrite) {
            ops[numOps].Type = D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE_WRITE_INSTANCE;
            ops[numOps].ArgCount = 1;
            ops[numOps].ArgData.StartAddress  = writeGva;
            ops[numOps].ArgData.StrideInBytes = sizeof(write);
            numOps++;
        }
        if (emitUpdate) {
            ops[numOps].Type = D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE_UPDATE_INSTANCE;
            ops[numOps].ArgCount = 1;
            ops[numOps].ArgData.StartAddress  = updateGva;
            ops[numOps].ArgData.StrideInBytes = sizeof(update);
            numOps++;
        }
        if (numOps == 0) return;

        head64 = (head64 + 7) & ~UINT64(7);
        memcpy(head + head64, ops, numOps * sizeof(ops[0]));
        D3D12_GPU_VIRTUAL_ADDRESS opsGva = baseGva + head64;
        head64 += numOps * sizeof(ops[0]);

        head64 = (head64 + 3) & ~UINT64(3);
        UINT numOps32 = numOps;
        memcpy(head + head64, &numOps32, sizeof(numOps32));
        D3D12_GPU_VIRTUAL_ADDRESS cntGva = baseGva + head64;
        head64 += sizeof(numOps32);

        D3D12_RTAS_PARTITIONED_TLAS_OPERATION_DATA opData = {};
        opData.SourceAccelerationStructureData  = firstBuild ? 0 : ptlas->GetGPUVirtualAddress();
        opData.DestAccelerationStructureData    = ptlas->GetGPUVirtualAddress();
        opData.ScratchAccelerationStructureData = ptlasScratch->GetGPUVirtualAddress();
        opData.IndirectPartitionedTlasOpCount   = cntGva;
        opData.IndirectPartitionedTlasOps       = opsGva;

        D3D12_RTAS_OPERATION_DESC opDesc = {};
        opDesc.Inputs.Type                       = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
        opDesc.Inputs.pPartitionedTLASInputsDesc = &ptlasIn;
        opDesc.pPartitionedTlasOperationData     = &opData;

        cl2->ExecuteIndirectRTASOperations(1, &opDesc,
            D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    };

    // -- Initial WRITE pass: populate all g_cfg.instCount instances.
    //    UPDATE_INSTANCE needs a prior non-zero-AS WRITE to target.
    {
        std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS> w(g_cfg.instCount);
        for (UINT i = 0; i < g_cfg.instCount; ++i) {
            w[i] = {};
            w[i].Transform[0][0] = 1; w[i].Transform[1][1] = 1; w[i].Transform[2][2] = 1;
            w[i].Transform[0][3] = (float)i;
            w[i].Transform[2][3] = 5.0f;
            w[i].InstanceID = i;
            w[i].InstanceMask = 0xFF;
            w[i].AccelerationStructure = blasA->GetGPUVirtualAddress();
            w[i].InstanceIndex = i;
            w[i].PartitionIndex = 0;
        }
        uint8_t* head = (uint8_t*)arenaPtr;
        memcpy(head, w.data(), w.size() * sizeof(w[0]));
        D3D12_GPU_VIRTUAL_ADDRESS writesGva = arena->GetGPUVirtualAddress();
        UINT64 head64 = w.size() * sizeof(w[0]);
        head64 = (head64 + 7) & ~UINT64(7);
        D3D12_RTAS_PARTITIONED_TLAS_OPERATION op = {};
        op.Type = D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE_WRITE_INSTANCE;
        op.ArgCount = (UINT)w.size();
        op.ArgData.StartAddress  = writesGva;
        op.ArgData.StrideInBytes = sizeof(w[0]);
        memcpy(head + head64, &op, sizeof(op));
        D3D12_GPU_VIRTUAL_ADDRESS opsGva = arena->GetGPUVirtualAddress() + head64;
        head64 += sizeof(op);
        head64 = (head64 + 3) & ~UINT64(3);
        UINT numOps32 = 1;
        memcpy(head + head64, &numOps32, sizeof(numOps32));
        D3D12_GPU_VIRTUAL_ADDRESS cntGva = arena->GetGPUVirtualAddress() + head64;
        D3D12_RTAS_PARTITIONED_TLAS_OPERATION_DATA opData = {};
        opData.DestAccelerationStructureData    = ptlas->GetGPUVirtualAddress();
        opData.ScratchAccelerationStructureData = ptlasScratch->GetGPUVirtualAddress();
        opData.IndirectPartitionedTlasOpCount   = cntGva;
        opData.IndirectPartitionedTlasOps       = opsGva;
        D3D12_RTAS_OPERATION_DESC opDesc = {};
        opDesc.Inputs.Type                       = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
        opDesc.Inputs.pPartitionedTLASInputsDesc = &ptlasIn;
        opDesc.pPartitionedTlasOperationData     = &opData;
        cl2->ExecuteIndirectRTASOperations(1, &opDesc,
            D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    }
    flushAndWait("initial WRITE pass");
    printf("  PTLAS initialized (%u instances).  Entering main loop.\n", g_cfg.instCount);

    // -- Main loop: per frame, one mixed WRITE+UPDATE build ------------------
    for (UINT frame = 0; frame < g_cfg.frames; ++frame) {
        buildPtlas(frame, /*firstBuild*/false, !g_cfg.noWrite, !g_cfg.noUpdate);
        char tag[64];
        sprintf_s(tag, "frame %u (w=%d u=%d)", frame, !g_cfg.noWrite, !g_cfg.noUpdate);
        flushAndWait(tag);
        if ((frame % 5) == 0 || frame + 1 == g_cfg.frames)
            printf("  frame %u OK\n", frame);
    }
    printf("\nCompleted %u frames -- no TDR observed.\n", g_cfg.frames);
    return 0;
}
