// =============================================================================
// PtlasWriteUpdateRepro - minimal repro for NVIDIA driver investigation
// =============================================================================
//
// Symptom:
//   GPU TDR (DXGI_ERROR_DEVICE_HUNG) on the FIRST per-frame PTLAS build
//   that contains BOTH a WRITE_INSTANCE op AND an UPDATE_INSTANCE op
//   targeting DISJOINT instance indices in the same call.
//
// Driver: NVIDIA RTX 4090, preview driver (2026-05).
//
// This file is the MOST STRIPPED-DOWN version: no DispatchRays, no shaders,
// no pipeline state, no descriptor heap, no TRANSLATE_PARTITION, no
// ENABLE_EXPLICIT_AABB.  Just:
//   1) Build a 1-triangle BLAS (x2 -- alternated as the "swap pointer")
//   2) Create a PTLAS sized for N instances in M partitions
//   3) Initial WRITE_INSTANCE pass to populate
//   4) Per frame: WRITE_INSTANCE one instance + UPDATE_INSTANCE one
//      DIFFERENT instance, in the SAME ExecuteIndirectRTASOperations call.
//      Wait for fence.
//   -> TDR on frame 1.
//
// Verified spec-compliant: IndirectBuild.cpp conformance test exercises
// this exact mixed WRITE+UPDATE pattern on disjoint instances by default.
// Verified clean on WARP (Microsoft Basic Render Driver) with the same EXE
// via --use-warp.
//
// CLI:
//   --frames N         : run N frames (default 30)
//   --no-write         : skip WRITE_INSTANCE  -> sanity baseline (OK)
//   --no-update        : skip UPDATE_INSTANCE -> sanity baseline (OK)
//   --partitions N     : partition count (default 1)
//   --instances N      : instance count   (default 4)
//   --use-warp         : force WARP adapter  -> sanity baseline (OK)
//
// Exit: 0 = ran to completion (no TDR), 1 = TDR detected, 2 = setup error.
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

// =============================================================================
// Agility SDK exports -- system d3d12.dll uses these to find D3D12Core.dll
// =============================================================================
extern "C" __declspec(dllexport) extern const UINT D3D12SDKVersion = 721;
extern "C" __declspec(dllexport) extern const char* D3D12SDKPath  = ".\\D3D12\\";

// =============================================================================
// Knobs
// =============================================================================
struct Config {
    UINT frames         = 30;
    bool noWrite        = false;
    bool noUpdate       = false;
    UINT partitionCount = 1;
    UINT instanceCount  = 4;
    bool useWarp        = false;
} g_cfg;

static std::atomic<bool> g_deviceRemoved{false};
static std::string       g_deviceRemovedReason;

// =============================================================================
// Helpers
// =============================================================================
static void Die(const char* msg, HRESULT hr = S_OK) {
    fprintf(stderr, "FATAL: %s (hr=0x%08lx)\n", msg, (unsigned long)hr);
    exit(2);
}
static void CheckHR(HRESULT hr, const char* msg) {
    if (FAILED(hr)) Die(msg, hr);
}

static void __stdcall InfoQueueCallback(D3D12_MESSAGE_CATEGORY,
                                        D3D12_MESSAGE_SEVERITY sev,
                                        D3D12_MESSAGE_ID id,
                                        LPCSTR description,
                                        void*) {
    fprintf(stderr, "[d3d12 sev=%d id=%u] %s\n", (int)sev, (unsigned)id, description ? description : "");
    if (id == 232 /* DEVICE_REMOVAL */ || sev == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
        g_deviceRemoved = true;
        if (description) g_deviceRemovedReason = description;
    }
}

static UINT64 Align(UINT64 x, UINT64 a) { return (x + (a - 1)) & ~(a - 1); }

static ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device* device,
                                           UINT64 size,
                                           D3D12_HEAP_TYPE heap,
                                           D3D12_RESOURCE_FLAGS flags,
                                           D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width    = size;
    rd.Height   = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout    = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags     = flags;
    ComPtr<ID3D12Resource> r;
    CheckHR(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE,
                                            &rd, state, nullptr,
                                            IID_PPV_ARGS(&r)),
            "CreateCommittedResource");
    return r;
}

template<typename T>
static void UploadInline(ID3D12Resource* upload, const T* src, size_t bytes) {
    void* p = nullptr;
    D3D12_RANGE no = {0, 0};
    CheckHR(upload->Map(0, &no, &p), "Map upload");
    memcpy(p, src, bytes);
    upload->Unmap(0, nullptr);
}

// =============================================================================
// CLI parse
// =============================================================================
static void ParseCli(int argc, wchar_t** argv) {
    for (int i = 1; i < argc; ++i) {
        if      (!wcscmp(argv[i], L"--frames")     && i + 1 < argc) g_cfg.frames         = (UINT)_wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--no-write"))                    g_cfg.noWrite        = true;
        else if (!wcscmp(argv[i], L"--no-update"))                   g_cfg.noUpdate       = true;
        else if (!wcscmp(argv[i], L"--partitions") && i + 1 < argc)  g_cfg.partitionCount = (UINT)_wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--instances")  && i + 1 < argc)  g_cfg.instanceCount  = (UINT)_wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--use-warp"))                    g_cfg.useWarp        = true;
        else fwprintf(stderr, L"unknown arg: %s\n", argv[i]);
    }
}

// =============================================================================
// main()
// =============================================================================
int wmain(int argc, wchar_t** argv) {
    ParseCli(argc, argv);
    printf("PtlasWriteUpdateRepro: frames=%u inst=%u parts=%u  "
           "noWrite=%d noUpdate=%d  warp=%d\n",
           g_cfg.frames, g_cfg.instanceCount, g_cfg.partitionCount,
           g_cfg.noWrite, g_cfg.noUpdate, g_cfg.useWarp);

    // Experimental features: D3D12ExperimentalShaderModels + D3D12RaytracingExperiment
    // (both defined in the preview d3d12.h).
    UUID experimentalFeatures[] = {
        D3D12ExperimentalShaderModels,
        D3D12RaytracingExperiment,
    };
    HRESULT hr = D3D12EnableExperimentalFeatures(_countof(experimentalFeatures),
                                                  experimentalFeatures, nullptr, nullptr);
    if (FAILED(hr)) {
        fprintf(stderr, "D3D12EnableExperimentalFeatures failed (0x%08lx).\n", (unsigned long)hr);
        return 2;
    }

    // DXGI adapter + D3D12 device
    ComPtr<IDXGIFactory6> factory;
    UINT factoryFlags = 0;
#ifdef _DEBUG
    factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    { ComPtr<ID3D12Debug> dbg; if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer(); }
#endif
    CheckHR(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

    ComPtr<IDXGIAdapter1> adapter;
    if (g_cfg.useWarp) {
        CheckHR(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "EnumWarpAdapter");
    } else {
        for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc; adapter->GetDesc1(&desc);
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

    // Register callback so we see device-removed messages promptly.
    {
        ComPtr<ID3D12InfoQueue1> iq;
        if (SUCCEEDED(device.As(&iq))) {
            DWORD cookie = 0;
            iq->RegisterMessageCallback(InfoQueueCallback,
                                        D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                                        nullptr, &cookie);
        }
    }

    ComPtr<ID3D12DeviceRaytracing2> deviceRT2;
    CheckHR(device.As(&deviceRT2), "QI(ID3D12DeviceRaytracing2)");

    // Confirm PTLAS support.
    D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL exp = {};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL, &exp, sizeof(exp)))
        || !exp.ClustersAndPTLASSupported) {
        fprintf(stderr, "Adapter does not report ClustersAndPTLASSupported = YES.\n");
        return 2;
    }
    printf("  ClustersAndPTLASSupported = YES\n");

    // Command queue / list / fence.
    D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    CheckHR(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");

    ComPtr<ID3D12CommandAllocator> alloc;
    CheckHR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)),
            "CreateCommandAllocator");

    ComPtr<ID3D12GraphicsCommandList4> cl;
    {
        ComPtr<ID3D12GraphicsCommandList> cl0;
        CheckHR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          alloc.Get(), nullptr, IID_PPV_ARGS(&cl0)),
                "CreateCommandList");
        CheckHR(cl0.As(&cl), "QI(ID3D12GraphicsCommandList4)");
    }
    ComPtr<ID3D12CommandListRaytracing2> cl2;
    CheckHR(cl.As(&cl2), "QI(ID3D12CommandListRaytracing2)");

    ComPtr<ID3D12Fence> fence;
    CheckHR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    HANDLE fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fenceValue = 0;
    auto flushAndWait = [&](const char* tag) {
        CheckHR(cl->Close(), "cl->Close");
        ID3D12CommandList* cls[] = { cl.Get() };
        queue->ExecuteCommandLists(1, cls);
        ++fenceValue;
        CheckHR(queue->Signal(fence.Get(), fenceValue), "queue->Signal");
        if (fence->GetCompletedValue() < fenceValue) {
            fence->SetEventOnCompletion(fenceValue, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        if (g_deviceRemoved.load()) {
            printf("!!! DEVICE REMOVED after %s: %s\n", tag, g_deviceRemovedReason.c_str());
            HRESULT removedReason = device->GetDeviceRemovedReason();
            fprintf(stderr, "    GetDeviceRemovedReason = 0x%08lx\n", (unsigned long)removedReason);
            exit(1);
        }
        CheckHR(alloc->Reset(), "alloc->Reset");
        CheckHR(cl->Reset(alloc.Get(), nullptr), "cl->Reset");
    };

    // =========================================================================
    // BLAS x 2: one triangle each.  Two of them so we have two distinct BLAS
    // GPUVAs to alternate between in WRITE / UPDATE args -- mirrors the
    // LOD-swap pattern that triggered the original bug.
    // =========================================================================
    struct V { float p[3]; };
    V verts[] = {
        { -0.5f, -0.5f,  0.0f },
        {  0.5f, -0.5f,  0.0f },
        {  0.0f,  0.5f,  0.0f },
    };
    uint32_t idx[] = { 0, 1, 2 };
    auto buildBlas = [&](float scale, ID3D12GraphicsCommandList4* clb) -> ComPtr<ID3D12Resource> {
        std::vector<V> sv(_countof(verts));
        for (int i = 0; i < (int)_countof(verts); ++i) {
            sv[i].p[0] = verts[i].p[0] * scale;
            sv[i].p[1] = verts[i].p[1] * scale;
            sv[i].p[2] = verts[i].p[2] * scale;
        }
        auto vb = CreateBuffer(device.Get(), sizeof(verts), D3D12_HEAP_TYPE_UPLOAD,
                               D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto ib = CreateBuffer(device.Get(), sizeof(idx), D3D12_HEAP_TYPE_UPLOAD,
                               D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        UploadInline(vb.Get(), sv.data(), sizeof(verts));
        UploadInline(ib.Get(), idx, sizeof(idx));

        D3D12_RAYTRACING_GEOMETRY_DESC geo = {};
        geo.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geo.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geo.Triangles.VertexBuffer.StartAddress  = vb->GetGPUVirtualAddress();
        geo.Triangles.VertexBuffer.StrideInBytes = sizeof(V);
        geo.Triangles.VertexCount  = _countof(verts);
        geo.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        geo.Triangles.IndexBuffer  = ib->GetGPUVirtualAddress();
        geo.Triangles.IndexCount   = _countof(idx);
        geo.Triangles.IndexFormat  = DXGI_FORMAT_R32_UINT;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
        inputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        inputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
        inputs.NumDescs       = 1;
        inputs.pGeometryDescs = &geo;
        inputs.Flags          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pre = {};
        device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &pre);

        auto blas = CreateBuffer(device.Get(), pre.ResultDataMaxSizeInBytes,
                                 D3D12_HEAP_TYPE_DEFAULT,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                                 D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
                                 D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
        auto scratch = CreateBuffer(device.Get(), pre.ScratchDataSizeInBytes,
                                    D3D12_HEAP_TYPE_DEFAULT,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                    D3D12_RESOURCE_STATE_COMMON);

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
        buildDesc.Inputs                          = inputs;
        buildDesc.DestAccelerationStructureData    = blas->GetGPUVirtualAddress();
        buildDesc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
        clb->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
        D3D12_RESOURCE_BARRIER uav = {};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = blas.Get();
        clb->ResourceBarrier(1, &uav);
        // Keep VB/IB/scratch alive past the BLAS build.
        static std::vector<ComPtr<ID3D12Resource>> s_keepAlive;
        s_keepAlive.push_back(vb);
        s_keepAlive.push_back(ib);
        s_keepAlive.push_back(scratch);
        return blas;
    };
    auto blasA = buildBlas(1.0f, cl.Get());
    auto blasB = buildBlas(0.7f, cl.Get());
    flushAndWait("BLAS build");
    printf("  Built 2 triangle BLASes  (blasA=0x%llx  blasB=0x%llx)\n",
           (unsigned long long)blasA->GetGPUVirtualAddress(),
           (unsigned long long)blasB->GetGPUVirtualAddress());

    // =========================================================================
    // PTLAS: bare minimum -- N instances in M partitions.  NO
    // ENABLE_PARTITION_TRANSLATION flag (we don't TRANSLATE_PARTITION).
    // NO global partition.  NO ENABLE_EXPLICIT_AABB on writes (none of these
    // affect the TDR -- confirmed by bisect).
    // =========================================================================
    D3D12_RTAS_PARTITIONED_TLAS_INPUTS_DESC ptlasInputs = {};
    ptlasInputs.InstanceCount  = g_cfg.instanceCount;
    ptlasInputs.PartitionCount = g_cfg.partitionCount;
    ptlasInputs.MaxInstancePerPartitionCount =
        (g_cfg.instanceCount + g_cfg.partitionCount - 1) / g_cfg.partitionCount + 4;
    ptlasInputs.MaxInstanceInGlobalPartitionCount = 0;
    ptlasInputs.Flags = D3D12_RTAS_PARTITIONED_TLAS_FLAG_NONE;

    D3D12_RTAS_OPERATION_INPUTS opIn = {};
    opIn.Type = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
    opIn.pPartitionedTLASInputsDesc = &ptlasInputs;
    D3D12_RTAS_OPERATION_PREBUILD_INFO ppre = {};
    deviceRT2->GetRTASOperationPrebuildInfo(&opIn, &ppre);
    printf("  PTLAS prebuild: result=%llu scratch=%llu\n",
           (unsigned long long)ppre.ResultDataMaxSizeInBytes,
           (unsigned long long)ppre.ScratchDataSizeInBytes);

    auto ptlas = CreateBuffer(device.Get(), ppre.ResultDataMaxSizeInBytes,
                              D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                              D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    auto ptlasScratch = CreateBuffer(device.Get(), ppre.ScratchDataSizeInBytes,
                                     D3D12_HEAP_TYPE_DEFAULT,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                     D3D12_RESOURCE_STATE_COMMON);

    // Per-frame upload arena (cycled by frame; 4 slots so we don't reuse a
    // slot that might still be referenced by an in-flight build).
    constexpr UINT64 kArenaBytes = 1 << 20;
    constexpr UINT   kSlots = 4;
    ComPtr<ID3D12Resource> arenas[kSlots];
    void* arenaPtrs[kSlots] = {};
    UINT64 arenaHead[kSlots] = {};
    for (UINT i = 0; i < kSlots; ++i) {
        arenas[i] = CreateBuffer(device.Get(), kArenaBytes, D3D12_HEAP_TYPE_UPLOAD,
                                 D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        D3D12_RANGE no = {0, 0};
        arenas[i]->Map(0, &no, &arenaPtrs[i]);
    }
    auto AllocArena = [&](UINT slot, UINT64 bytes, UINT align)
        -> std::pair<void*, D3D12_GPU_VIRTUAL_ADDRESS> {
        arenaHead[slot] = Align(arenaHead[slot], align);
        if (arenaHead[slot] + bytes > kArenaBytes) Die("arena overflow");
        void* cpu = (uint8_t*)arenaPtrs[slot] + arenaHead[slot];
        D3D12_GPU_VIRTUAL_ADDRESS gpu = arenas[slot]->GetGPUVirtualAddress() + arenaHead[slot];
        arenaHead[slot] += bytes;
        return { cpu, gpu };
    };

    // =========================================================================
    // PTLAS build: stage the op-args into the per-frame arena, point the
    // operation desc at them, call ExecuteIndirectRTASOperations.
    // =========================================================================
    auto buildPtlas = [&](UINT slot, bool firstBuild,
                          const std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS>& writes,
                          const std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_UPDATE_INSTANCE_ARGS>& updates)
    {
        std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION> ops;
        if (!writes.empty()) {
            auto [cpu, gpu] = AllocArena(slot, writes.size() * sizeof(writes[0]), 8);
            memcpy(cpu, writes.data(), writes.size() * sizeof(writes[0]));
            D3D12_RTAS_PARTITIONED_TLAS_OPERATION op = {};
            op.Type = D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE_WRITE_INSTANCE;
            op.ArgCount = (UINT)writes.size();
            op.ArgData.StartAddress = gpu;
            op.ArgData.StrideInBytes = sizeof(writes[0]);
            ops.push_back(op);
        }
        if (!updates.empty()) {
            auto [cpu, gpu] = AllocArena(slot, updates.size() * sizeof(updates[0]), 8);
            memcpy(cpu, updates.data(), updates.size() * sizeof(updates[0]));
            D3D12_RTAS_PARTITIONED_TLAS_OPERATION op = {};
            op.Type = D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE_UPDATE_INSTANCE;
            op.ArgCount = (UINT)updates.size();
            op.ArgData.StartAddress = gpu;
            op.ArgData.StrideInBytes = sizeof(updates[0]);
            ops.push_back(op);
        }
        if (ops.empty()) return;

        auto [opsCpu, opsGpu] = AllocArena(slot, ops.size() * sizeof(ops[0]), 8);
        memcpy(opsCpu, ops.data(), ops.size() * sizeof(ops[0]));

        UINT32 numOps32 = (UINT32)ops.size();
        auto [cntCpu, cntGpu] = AllocArena(slot, sizeof(UINT32), 4);
        memcpy(cntCpu, &numOps32, sizeof(numOps32));

        D3D12_RTAS_OPERATION_INPUTS opInputs   = {};
        opInputs.Type                           = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
        opInputs.pPartitionedTLASInputsDesc     = &ptlasInputs;

        D3D12_RTAS_PARTITIONED_TLAS_OPERATION_DATA opData = {};
        opData.SourceAccelerationStructureData  = firstBuild ? 0 : ptlas->GetGPUVirtualAddress();
        opData.DestAccelerationStructureData    = ptlas->GetGPUVirtualAddress();
        opData.ScratchAccelerationStructureData = ptlasScratch->GetGPUVirtualAddress();
        opData.IndirectPartitionedTlasOpCount   = cntGpu;
        opData.IndirectPartitionedTlasOps       = opsGpu;

        D3D12_RTAS_OPERATION_DESC opDesc = {};
        opDesc.Inputs                        = opInputs;
        opDesc.pPartitionedTlasOperationData = &opData;

        cl2->ExecuteIndirectRTASOperations(1, &opDesc,
            D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    };

    // -------- Initial WRITE_INSTANCE pass: populate all instances ----------
    {
        std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS> writes;
        for (UINT i = 0; i < g_cfg.instanceCount; ++i) {
            D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS w = {};
            w.Transform[0][0] = 1; w.Transform[1][1] = 1; w.Transform[2][2] = 1;
            w.Transform[0][3] = (float)i;
            w.Transform[2][3] = 5.0f;
            w.InstanceID = i;
            w.InstanceMask = 0xFF;
            w.InstanceContributionToHitGroupIndex = 0;
            w.InstanceFlags = D3D12_RTAS_PARTITIONED_TLAS_INSTANCE_FLAG_NONE;
            w.AccelerationStructure = blasA->GetGPUVirtualAddress();
            w.InstanceIndex = i;
            w.PartitionIndex = i % g_cfg.partitionCount;
            writes.push_back(w);
        }
        buildPtlas(0, /*firstBuild*/true, writes, {});
        flushAndWait("initial PTLAS build");
        for (UINT i = 0; i < kSlots; ++i) arenaHead[i] = 0;
    }
    printf("  PTLAS initial WRITE pass done (%u instances).  Entering main loop.\n",
           g_cfg.instanceCount);

    // =========================================================================
    // Main loop: per frame, WRITE one instance + UPDATE a DIFFERENT instance.
    // =========================================================================
    for (UINT frame = 0; frame < g_cfg.frames; ++frame) {
        UINT slot = frame % kSlots;
        arenaHead[slot] = 0;

        UINT writeIdx  = frame % g_cfg.instanceCount;
        UINT updateIdx = (frame + g_cfg.instanceCount / 2) % g_cfg.instanceCount;
        if (updateIdx == writeIdx) updateIdx = (writeIdx + 1) % g_cfg.instanceCount;

        std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS> writes;
        if (!g_cfg.noWrite) {
            D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS w = {};
            w.Transform[0][0] = 1; w.Transform[1][1] = 1; w.Transform[2][2] = 1;
            w.Transform[0][3] = (float)writeIdx + 0.01f * frame;
            w.Transform[2][3] = 5.0f;
            w.InstanceID = writeIdx;
            w.InstanceMask = 0xFF;
            w.InstanceContributionToHitGroupIndex = 0;
            w.InstanceFlags = D3D12_RTAS_PARTITIONED_TLAS_INSTANCE_FLAG_NONE;
            w.AccelerationStructure = (frame & 1) ? blasB->GetGPUVirtualAddress()
                                                   : blasA->GetGPUVirtualAddress();
            w.InstanceIndex = writeIdx;
            w.PartitionIndex = writeIdx % g_cfg.partitionCount;
            writes.push_back(w);
        }

        std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_UPDATE_INSTANCE_ARGS> updates;
        if (!g_cfg.noUpdate) {
            D3D12_RTAS_PARTITIONED_TLAS_OPERATION_UPDATE_INSTANCE_ARGS u = {};
            u.InstanceIndex = updateIdx;
            u.InstanceContributionToHitGroupIndex = 0;
            u.AccelerationStructure = (frame & 1) ? blasA->GetGPUVirtualAddress()
                                                   : blasB->GetGPUVirtualAddress();
            updates.push_back(u);
        }

        buildPtlas(slot, /*firstBuild*/false, writes, updates);

        char tag[64];
        sprintf_s(tag, "frame %u (w=%zu u=%zu)", frame, writes.size(), updates.size());
        flushAndWait(tag);

        if ((frame % 5) == 0 || frame + 1 == g_cfg.frames) {
            printf("  frame %u OK (writes=%zu updates=%zu)\n",
                   frame, writes.size(), updates.size());
        }
    }

    printf("\nCompleted %u frames -- no TDR observed.\n", g_cfg.frames);
    return 0;
}
