// =============================================================================
// PtlasWriteUpdateRepro - minimal DXR2 driver-bug repro
// =============================================================================
//
// Repros a GPU TDR (DXGI_ERROR_DEVICE_HUNG) on a PTLAS build whose
// ExecuteIndirectRTASOperations call contains BOTH a WRITE_INSTANCE op AND
// an UPDATE_INSTANCE op, WHEN both args target a non-zero InstanceIndex.
//
//   TDR iff WRITE.InstanceIndex != 0 AND UPDATE.InstanceIndex != 0.
//
//   If either op references InstanceIndex = 0, no TDR.
//   If only one op type is present, no TDR.
//   On WARP, no TDR for any (W, U).
//
// Driver: NVIDIA RTX 4090, preview (2026-05).
// Spec-compliant per Raytracing2.md ("Multiple operations of different
// types can be used in the same call.").
//
// Scene: 1 BLAS (one triangle).  1 PTLAS: 3 instances, 1 partition,
// Flags = NONE.  Initial WRITE pass populates all 3 instances.  Then per
// frame: one ExecuteIndirectRTASOperations call with a WRITE_INSTANCE
// arg (InstanceIndex = 1) and an UPDATE_INSTANCE arg (InstanceIndex = 2),
// both referencing the same BLAS.  No shaders, no pipeline state, no
// DispatchRays.  TDR on frame 1.
//
// CLI:
//   --frames N      : run N frames (default 5)
//   --write IDX     : InstanceIndex for the WRITE_INSTANCE arg (default 1)
//   --update IDX    : InstanceIndex for the UPDATE_INSTANCE arg (default 2)
//   --no-write      : skip WRITE_INSTANCE  -> baseline (OK)
//   --no-update     : skip UPDATE_INSTANCE -> baseline (OK)
//   --use-warp      : force WARP adapter   -> baseline (OK)
//
// Exit: 0 = no TDR, 1 = TDR, 2 = setup error.
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
#include <string>

using Microsoft::WRL::ComPtr;

extern "C" __declspec(dllexport) extern const UINT  D3D12SDKVersion = 721;
extern "C" __declspec(dllexport) extern const char* D3D12SDKPath    = ".\\D3D12\\";

static UINT  g_frames     = 5;
static int   g_writeInst  = 1;
static int   g_updateInst = 2;
static bool  g_noWrite    = false;
static bool  g_noUpdate   = false;
static bool  g_useWarp    = false;

static bool        g_deviceRemoved = false;
static std::string g_deviceRemovedReason;

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

static constexpr UINT kInstanceCount = 3;

int wmain(int argc, wchar_t** argv) {
    for (int i = 1; i < argc; ++i) {
        if      (!wcscmp(argv[i], L"--frames") && i + 1 < argc) g_frames     = (UINT)_wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--write")  && i + 1 < argc) g_writeInst  = _wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--update") && i + 1 < argc) g_updateInst = _wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--no-write"))                g_noWrite   = true;
        else if (!wcscmp(argv[i], L"--no-update"))               g_noUpdate  = true;
        else if (!wcscmp(argv[i], L"--use-warp"))                g_useWarp   = true;
        else fwprintf(stderr, L"unknown arg: %s\n", argv[i]);
    }
    printf("frames=%u write=%d update=%d noWrite=%d noUpdate=%d warp=%d\n",
           g_frames, g_writeInst, g_updateInst, g_noWrite, g_noUpdate, g_useWarp);

    // Experimental D3D12: shader models + DXR2.
    UUID feats[] = { D3D12ExperimentalShaderModels, D3D12RaytracingExperiment };
    if (FAILED(D3D12EnableExperimentalFeatures(_countof(feats), feats, nullptr, nullptr))) {
        fprintf(stderr, "D3D12EnableExperimentalFeatures failed.\n"); return 2;
    }

    ComPtr<IDXGIFactory6> factory;
    CheckHR(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    ComPtr<IDXGIAdapter1> adapter;
    if (g_useWarp) {
        CheckHR(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "EnumWarpAdapter");
    } else {
        for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                              IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc; adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter.Reset(); continue; }
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_2, _uuidof(ID3D12Device), nullptr))) break;
            adapter.Reset();
        }
    }
    if (!adapter) Die("no D3D12 adapter");
    DXGI_ADAPTER_DESC1 desc; adapter->GetDesc1(&desc);
    wprintf(L"  Adapter: %s\n", desc.Description);

    ComPtr<ID3D12Device5> device;
    CheckHR(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_2, IID_PPV_ARGS(&device)), "D3D12CreateDevice");
    {
        ComPtr<ID3D12InfoQueue1> iq;
        if (SUCCEEDED(device.As(&iq))) {
            DWORD cookie = 0;
            iq->RegisterMessageCallback(InfoQueueCallback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie);
        }
    }
    ComPtr<ID3D12DeviceRaytracing2> deviceRT2;
    CheckHR(device.As(&deviceRT2), "QI(DeviceRaytracing2)");
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
        if (g_deviceRemoved) {
            printf("!!! DEVICE REMOVED after %s: %s\n", tag, g_deviceRemovedReason.c_str());
            fprintf(stderr, "    GetDeviceRemovedReason = 0x%08lx\n",
                    (unsigned long)device->GetDeviceRemovedReason());
            exit(1);
        }
        CheckHR(alloc->Reset(), "alloc->Reset");
        CheckHR(cl->Reset(alloc.Get(), nullptr), "cl->Reset");
    };

    // -- 1-triangle BLAS -----------------------------------------------------
    float    verts[3][3] = { {-0.5f,-0.5f,0}, {0.5f,-0.5f,0}, {0,0.5f,0} };
    uint32_t idx[3]      = { 0, 1, 2 };
    auto vb = CreateBuffer(device.Get(), sizeof(verts), D3D12_HEAP_TYPE_UPLOAD,
                           D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto ib = CreateBuffer(device.Get(), sizeof(idx),   D3D12_HEAP_TYPE_UPLOAD,
                           D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    { void* p; D3D12_RANGE no = {0,0}; vb->Map(0,&no,&p); memcpy(p,verts,sizeof(verts)); vb->Unmap(0,nullptr); }
    { void* p; D3D12_RANGE no = {0,0}; ib->Map(0,&no,&p); memcpy(p,idx,  sizeof(idx));   ib->Unmap(0,nullptr); }
    D3D12_RAYTRACING_GEOMETRY_DESC geo = {};
    geo.Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geo.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geo.Triangles.VertexBuffer.StartAddress  = vb->GetGPUVirtualAddress();
    geo.Triangles.VertexBuffer.StrideInBytes = sizeof(verts[0]);
    geo.Triangles.VertexCount  = 3;
    geo.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geo.Triangles.IndexBuffer  = ib->GetGPUVirtualAddress();
    geo.Triangles.IndexCount   = 3;
    geo.Triangles.IndexFormat  = DXGI_FORMAT_R32_UINT;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bi = {};
    bi.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bi.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    bi.NumDescs = 1;  bi.pGeometryDescs = &geo;
    bi.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bpre = {};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&bi, &bpre);
    auto blas = CreateBuffer(device.Get(), bpre.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                             D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
                             D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    auto blasScratch = CreateBuffer(device.Get(), bpre.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd = {};
    bd.Inputs = bi;
    bd.DestAccelerationStructureData    = blas->GetGPUVirtualAddress();
    bd.ScratchAccelerationStructureData = blasScratch->GetGPUVirtualAddress();
    cl->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);
    D3D12_RESOURCE_BARRIER uav = {};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;  uav.UAV.pResource = blas.Get();
    cl->ResourceBarrier(1, &uav);
    flushAndWait("BLAS build");

    // -- PTLAS: kInstanceCount instances, 1 partition, no flags --------------
    D3D12_RTAS_PARTITIONED_TLAS_INPUTS_DESC ptlasIn = {};
    ptlasIn.InstanceCount  = kInstanceCount;
    ptlasIn.PartitionCount = 1;
    ptlasIn.MaxInstancePerPartitionCount = kInstanceCount;
    ptlasIn.MaxInstanceInGlobalPartitionCount = 0;
    ptlasIn.Flags = D3D12_RTAS_PARTITIONED_TLAS_FLAG_NONE;
    D3D12_RTAS_OPERATION_INPUTS opIn = {};
    opIn.Type = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
    opIn.pPartitionedTLASInputsDesc = &ptlasIn;
    D3D12_RTAS_OPERATION_PREBUILD_INFO ppre = {};
    deviceRT2->GetRTASOperationPrebuildInfo(&opIn, &ppre);
    auto ptlas = CreateBuffer(device.Get(), ppre.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                              D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    auto ptlasScratch = CreateBuffer(device.Get(), ppre.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

    // Single upload buffer.  We wait fence after every build, so reuse it.
    auto arena = CreateBuffer(device.Get(), 64 * 1024, D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* arenaPtr = nullptr;
    { D3D12_RANGE no = {0,0}; arena->Map(0, &no, &arenaPtr); }
    D3D12_GPU_VIRTUAL_ADDRESS arenaBase = arena->GetGPUVirtualAddress();

    // Stage a fresh op-array + ops + count and issue ExecuteIndirectRTASOperations.
    auto buildPtlas = [&](bool firstBuild,
                          UINT writeCount, const D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS* writeArgs,
                          UINT updateCount, const D3D12_RTAS_PARTITIONED_TLAS_OPERATION_UPDATE_INSTANCE_ARGS* updateArgs) {
        uint8_t* head = (uint8_t*)arenaPtr;
        UINT64 off = 0;
        auto alignTo = [&](UINT64 a) { off = (off + (a - 1)) & ~(a - 1); };

        D3D12_RTAS_PARTITIONED_TLAS_OPERATION ops[2] = {};
        UINT numOps = 0;
        if (writeCount) {
            alignTo(8);
            memcpy(head + off, writeArgs, writeCount * sizeof(writeArgs[0]));
            ops[numOps].Type = D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE_WRITE_INSTANCE;
            ops[numOps].ArgCount = writeCount;
            ops[numOps].ArgData.StartAddress  = arenaBase + off;
            ops[numOps].ArgData.StrideInBytes = sizeof(writeArgs[0]);
            off += writeCount * sizeof(writeArgs[0]);
            numOps++;
        }
        if (updateCount) {
            alignTo(8);
            memcpy(head + off, updateArgs, updateCount * sizeof(updateArgs[0]));
            ops[numOps].Type = D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE_UPDATE_INSTANCE;
            ops[numOps].ArgCount = updateCount;
            ops[numOps].ArgData.StartAddress  = arenaBase + off;
            ops[numOps].ArgData.StrideInBytes = sizeof(updateArgs[0]);
            off += updateCount * sizeof(updateArgs[0]);
            numOps++;
        }
        if (numOps == 0) return;

        alignTo(8);
        memcpy(head + off, ops, numOps * sizeof(ops[0]));
        D3D12_GPU_VIRTUAL_ADDRESS opsGva = arenaBase + off;
        off += numOps * sizeof(ops[0]);

        alignTo(4);
        UINT numOps32 = numOps;
        memcpy(head + off, &numOps32, sizeof(numOps32));
        D3D12_GPU_VIRTUAL_ADDRESS cntGva = arenaBase + off;

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
        cl2->ExecuteIndirectRTASOperations(1, &opDesc, D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    };

    // -- Init: WRITE all instances so each has a non-zero AS (UPDATE needs that)
    {
        D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS initWrites[kInstanceCount] = {};
        for (UINT i = 0; i < kInstanceCount; ++i) {
            initWrites[i].Transform[0][0] = 1;
            initWrites[i].Transform[1][1] = 1;
            initWrites[i].Transform[2][2] = 1;
            initWrites[i].Transform[0][3] = (float)i;
            initWrites[i].InstanceID    = i;
            initWrites[i].InstanceMask  = 0xFF;
            initWrites[i].AccelerationStructure = blas->GetGPUVirtualAddress();
            initWrites[i].InstanceIndex  = i;
            initWrites[i].PartitionIndex = 0;
        }
        buildPtlas(/*firstBuild*/true, kInstanceCount, initWrites, 0, nullptr);
        flushAndWait("initial WRITE pass");
    }

    // -- Per-frame mixed WRITE+UPDATE build ---------------------------------
    for (UINT frame = 0; frame < g_frames; ++frame) {
        D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS w = {};
        w.Transform[0][0] = 1;  w.Transform[1][1] = 1;  w.Transform[2][2] = 1;
        w.InstanceID = (UINT)g_writeInst;
        w.InstanceMask = 0xFF;
        w.AccelerationStructure = blas->GetGPUVirtualAddress();
        w.InstanceIndex  = (UINT)g_writeInst;
        w.PartitionIndex = 0;

        D3D12_RTAS_PARTITIONED_TLAS_OPERATION_UPDATE_INSTANCE_ARGS u = {};
        u.InstanceIndex          = (UINT)g_updateInst;
        u.AccelerationStructure  = blas->GetGPUVirtualAddress();

        buildPtlas(/*firstBuild*/false,
                   g_noWrite  ? 0 : 1, &w,
                   g_noUpdate ? 0 : 1, &u);
        char tag[64];
        sprintf_s(tag, "frame %u", frame);
        flushAndWait(tag);
        printf("  frame %u OK\n", frame);
    }
    printf("\nCompleted %u frames -- no TDR.\n", g_frames);
    return 0;
}
