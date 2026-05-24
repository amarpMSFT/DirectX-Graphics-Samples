// =============================================================================
// PtlasWriteUpdateRepro - minimal repro for NVIDIA driver investigation
// =============================================================================
//
// Symptom:
//   GPU TDR (DXGI_ERROR_DEVICE_HUNG) after ~8-10 successful frames when a
//   PTLAS build call contains BOTH a WRITE_INSTANCE op AND an UPDATE_INSTANCE
//   op targeting DISJOINT instance indices in the same call.
//
// Driver:
//   NVIDIA RTX 4090, preview driver (2026-05).
//
// Verified spec-compliant:
//   - The DXR2 conformance test (IndirectBuild.cpp) exercises exactly this
//     mixed-op pattern by default, and passes on the same driver.
//   - WARP (Microsoft Basic Render Driver) runs identical code with no TDR.
//
// Scope of this repro:
//   Single source file (this), inline HLSL compiled at runtime via DXC.
//   No DeviceResources / sample framework / overlay.  Headless.  Tiny scene:
//   16 triangle-BLAS instances split across 4 partitions, 64x64 dispatch.
//   Each frame:
//     - WRITE_INSTANCE one instance (transform + LOD-style BLAS swap)
//     - UPDATE_INSTANCE one DIFFERENT instance (BLAS-pointer swap only)
//     - TRANSLATE_PARTITION all 4 partitions
//     - DispatchRays at 64x64
//     - Wait for fence
//
// Default settings reproduce the TDR; CLI knobs let you bisect:
//   --frames N           : run N frames (default 30)
//   --no-write           : skip the WRITE_INSTANCE arg (sanity baseline)
//   --no-update          : skip the UPDATE_INSTANCE arg (sanity baseline)
//   --no-translate       : skip TRANSLATE_PARTITION
//   --no-dispatch        : skip DispatchRays (does build alone TDR?)
//   --no-aabb            : don't set ENABLE_EXPLICIT_AABB on writes
//   --partitions N       : partition count (default 4)
//   --instances N        : total instance count (default 16)
//   --use-warp           : force WARP adapter
//
// Exit code 0 = ran to completion (no TDR seen).
// Exit code 1 = TDR detected.
// Exit code 2 = setup error (no DXR2/PTLAS support, build failure, etc.).
//
// Build:
//   msbuild PtlasWriteUpdateRepro.vcxproj /p:Configuration=Debug /p:Platform=x64
//
// Run:
//   bin\x64\Debug\PtlasWriteUpdateRepro.exe                 (default: should TDR)
//   bin\x64\Debug\PtlasWriteUpdateRepro.exe --no-update      (baseline; no TDR expected)
//   bin\x64\Debug\PtlasWriteUpdateRepro.exe --no-dispatch    (does build alone TDR?)
//   bin\x64\Debug\PtlasWriteUpdateRepro.exe --use-warp       (WARP; no TDR)
//
// =============================================================================

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wrl/client.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <atomic>
#include <string>

using Microsoft::WRL::ComPtr;

// =============================================================================
// Agility SDK exports (required for the preview D3D12Core.dll to be picked up)
// =============================================================================
extern "C" __declspec(dllexport) extern const UINT D3D12SDKVersion = 721;
extern "C" __declspec(dllexport) extern const char* D3D12SDKPath  = ".\\D3D12\\";

// =============================================================================
// Inline HLSL.  Compiled at runtime via DXC (loaded from dxcompiler.dll which
// the props file copies next to the EXE).  This is the only shader source
// in the whole repro -- raygen + miss + hit, ~30 lines.
// =============================================================================
static const wchar_t kShaderSource[] = LR"_(
RaytracingAccelerationStructure g_tlas : register(t0);
RWTexture2D<float4>             g_out  : register(u0);

struct [raypayload] Payload {
    float3 color : read(caller) : write(caller, closesthit, miss);
};

[shader("raygeneration")]
void Raygen() {
    uint2 px  = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;
    // Just shoot a +Z ray from the pixel's pixel-space position.  Geometry
    // is at z~5; trace a ray to see if any instance is hit.  Not testing
    // shading -- the raytrace is here as a downstream cost so the PTLAS
    // contents are actually consumed by a trace each frame (we want any
    // PTLAS state corruption from the build to be exposed via TraceRay).
    float2 xy = (float2(px) + 0.5) / float2(dim) * 4.0 - 2.0;   // [-2, +2]
    RayDesc ray;
    ray.Origin    = float3(xy.x, xy.y, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin      = 0.001;
    ray.TMax      = 100.0;
    Payload p; p.color = float3(0,0,0);
    TraceRay(g_tlas, RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
             0xFF, 0, 0, 0, ray, p);
    g_out[px] = float4(p.color, 1);
}

[shader("miss")]
void Miss(inout Payload p) { p.color = float3(0.1, 0.1, 0.2); }

[shader("closesthit")]
void ClosestHit(inout Payload p, in BuiltInTriangleIntersectionAttributes a) {
    p.color = float3(0.8, 0.5, 0.2);
}
)_";

// =============================================================================
// Knobs / globals
// =============================================================================
struct Config {
    UINT  frames        = 30;
    bool  noWrite       = false;
    bool  noUpdate      = false;
    bool  noTranslate   = false;
    bool  noDispatch    = false;
    bool  noAabb        = false;
    UINT  partitionCount = 4;
    UINT  instanceCount  = 16;
    bool  useWarp        = false;
} g_cfg;

static std::atomic<bool> g_deviceRemoved{false};
static std::wstring      g_deviceRemovedReason;

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

// D3D12_MESSAGE_CALLBACK so we capture device-removed reason as soon as it
// fires (instead of only finding out after the next API call returns).
static void __stdcall InfoQueueCallback(D3D12_MESSAGE_CATEGORY,
                                        D3D12_MESSAGE_SEVERITY sev,
                                        D3D12_MESSAGE_ID id,
                                        LPCSTR description,
                                        void*) {
    fprintf(stderr, "[d3d12 sev=%d id=%u] %s\n", (int)sev, (unsigned)id, description ? description : "");
    if (id == 232 /* DEVICE_REMOVAL */ || sev == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
        g_deviceRemoved = true;
        if (description) {
            int wn = MultiByteToWideChar(CP_UTF8, 0, description, -1, nullptr, 0);
            g_deviceRemovedReason.resize(wn ? wn - 1 : 0);
            if (wn > 0) MultiByteToWideChar(CP_UTF8, 0, description, -1, g_deviceRemovedReason.data(), wn);
        }
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
    HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE,
                                                  &rd, state, nullptr,
                                                  IID_PPV_ARGS(&r));
    CheckHR(hr, "CreateCommittedResource");
    return r;
}

static ComPtr<ID3D12Resource> CreateTex2D(ID3D12Device* device,
                                          UINT w, UINT h,
                                          DXGI_FORMAT fmt,
                                          D3D12_RESOURCE_FLAGS flags,
                                          D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width    = w;
    rd.Height   = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Format    = fmt;
    rd.Flags     = flags;
    ComPtr<ID3D12Resource> r;
    HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE,
                                                  &rd, state, nullptr,
                                                  IID_PPV_ARGS(&r));
    CheckHR(hr, "CreateCommittedResource (tex)");
    return r;
}

template<typename T>
static void UploadInline(ID3D12Resource* upload, const T* src, size_t bytes) {
    void* p = nullptr;
    D3D12_RANGE no = {0, 0};
    HRESULT hr = upload->Map(0, &no, &p);
    CheckHR(hr, "Map upload");
    memcpy(p, src, bytes);
    upload->Unmap(0, nullptr);
}

// =============================================================================
// DXC compile (runtime, via dxcompiler.dll loaded from EXE dir)
// =============================================================================
static std::vector<uint8_t> CompileShader(const wchar_t* src, const wchar_t* entry,
                                          const wchar_t* target) {
    // Manual LoadLibrary so we don't need dxcompiler.lib (the preview DXC
    // drop ships dll+dxil.dll but no import library).  The dxcapi.h
    // interfaces come from the Windows SDK.
    HMODULE dll = LoadLibraryW(L"dxcompiler.dll");
    if (!dll) Die("LoadLibrary(dxcompiler.dll) failed -- ensure the preview DXC dll is next to the EXE");
    typedef HRESULT(__cdecl *DxcCreateInstanceFn)(REFCLSID, REFIID, void**);
    auto pCreate = (DxcCreateInstanceFn)GetProcAddress(dll, "DxcCreateInstance");
    if (!pCreate) Die("GetProcAddress(DxcCreateInstance) failed");

    ComPtr<IDxcCompiler3> comp;
    HRESULT hr = pCreate(CLSID_DxcCompiler, IID_PPV_ARGS(&comp));
    CheckHR(hr, "DxcCreateInstance(Compiler3)");

    UINT32 srcLen = (UINT32)wcslen(src);
    UINT32 srcBytes = srcLen * sizeof(wchar_t);
    DxcBuffer buf = {};
    buf.Ptr = src;
    buf.Size = srcBytes;
    buf.Encoding = DXC_CP_WIDE;

    const wchar_t* args[] = {
        L"-T", target,
        L"-Zi", L"-O0",
        L"-HV", L"2021",
    };
    ComPtr<IDxcResult> result;
    hr = comp->Compile(&buf, args, _countof(args), nullptr, IID_PPV_ARGS(&result));
    if (FAILED(hr)) {
        fprintf(stderr, "Compile() hr=0x%08lx\n", (unsigned long)hr);
        ComPtr<IDxcBlobUtf8> errs;
        if (result) result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errs), nullptr);
        if (errs && errs->GetStringLength()) {
            fprintf(stderr, "DXC errors:\n%s\n", errs->GetStringPointer());
        }
        Die("DXC compile failed");
    }
    HRESULT compStatus = S_OK;
    result->GetStatus(&compStatus);
    if (FAILED(compStatus)) {
        ComPtr<IDxcBlobUtf8> errs;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errs), nullptr);
        if (errs && errs->GetStringLength()) {
            fprintf(stderr, "DXC errors:\n%s\n", errs->GetStringPointer());
        }
        Die("DXC compile failed (compStatus)");
    }
    ComPtr<IDxcBlob> bcblob;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&bcblob), nullptr);
    if (!bcblob) Die("DXC: no object blob");
    std::vector<uint8_t> out((uint8_t*)bcblob->GetBufferPointer(),
                             (uint8_t*)bcblob->GetBufferPointer() + bcblob->GetBufferSize());
    return out;
}

// =============================================================================
// CLI parse
// =============================================================================
static void ParseCli(int argc, wchar_t** argv) {
    for (int i = 1; i < argc; ++i) {
        if      (!wcscmp(argv[i], L"--frames")        && i + 1 < argc) g_cfg.frames         = (UINT)_wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--no-write"))                       g_cfg.noWrite        = true;
        else if (!wcscmp(argv[i], L"--no-update"))                      g_cfg.noUpdate       = true;
        else if (!wcscmp(argv[i], L"--no-translate"))                   g_cfg.noTranslate    = true;
        else if (!wcscmp(argv[i], L"--no-dispatch"))                    g_cfg.noDispatch     = true;
        else if (!wcscmp(argv[i], L"--no-aabb"))                        g_cfg.noAabb         = true;
        else if (!wcscmp(argv[i], L"--partitions")    && i + 1 < argc)  g_cfg.partitionCount = (UINT)_wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--instances")     && i + 1 < argc)  g_cfg.instanceCount  = (UINT)_wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--use-warp"))                       g_cfg.useWarp        = true;
        else {
            fwprintf(stderr, L"unknown arg: %s\n", argv[i]);
        }
    }
}

// =============================================================================
// main()
// =============================================================================
int wmain(int argc, wchar_t** argv) {
    ParseCli(argc, argv);
    printf("PtlasWriteUpdateRepro: frames=%u inst=%u parts=%u  "
           "noWrite=%d noUpdate=%d noTranslate=%d noDispatch=%d noAabb=%d  warp=%d\n",
           g_cfg.frames, g_cfg.instanceCount, g_cfg.partitionCount,
           g_cfg.noWrite, g_cfg.noUpdate, g_cfg.noTranslate, g_cfg.noDispatch,
           g_cfg.noAabb, g_cfg.useWarp);

    // -- enable D3D12 experimental features (cluster + PTLAS) ---------------
    UUID experimentalFeatures[] = {
        D3D12ExperimentalShaderModels,
        D3D12RaytracingExperiment,
    };
    HRESULT hr = D3D12EnableExperimentalFeatures(_countof(experimentalFeatures),
                                                  experimentalFeatures, nullptr, nullptr);
    if (FAILED(hr)) {
        fprintf(stderr, "D3D12EnableExperimentalFeatures failed (0x%08lx).  "
                "Ensure the preview D3D12Core.dll is next to the EXE.\n", (unsigned long)hr);
        return 2;
    }

    // -- DXGI adapter + D3D12 device ----------------------------------------
    ComPtr<IDXGIFactory6> factory;
    UINT factoryFlags = 0;
#ifdef _DEBUG
    factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    { ComPtr<ID3D12Debug> dbg; if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer(); }
#endif
    hr = CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&factory));
    CheckHR(hr, "CreateDXGIFactory2");

    ComPtr<IDXGIAdapter1> adapter;
    if (g_cfg.useWarp) {
        hr = factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        CheckHR(hr, "EnumWarpAdapter");
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
    hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_2, IID_PPV_ARGS(&device));
    CheckHR(hr, "D3D12CreateDevice");

    // InfoQueue1 callback (captures TDR + corruption messages with no polling).
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
    hr = device.As(&deviceRT2);
    if (FAILED(hr)) Die("Device does not support ID3D12DeviceRaytracing2", hr);

    // Check PTLAS support.
    D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL exp = {};
    hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL, &exp, sizeof(exp));
    if (FAILED(hr) || !exp.ClustersAndPTLASSupported) {
        fprintf(stderr, "Adapter does not report ClustersAndPTLASSupported=YES.\n");
        return 2;
    }
    printf("  ClustersAndPTLASSupported = YES\n");

    // -- command queue / list / fence ---------------------------------------
    D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    hr = device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    CheckHR(hr, "CreateCommandQueue");

    ComPtr<ID3D12CommandAllocator> alloc;
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    CheckHR(hr, "CreateCommandAllocator");

    ComPtr<ID3D12GraphicsCommandList4> cl;
    {
        ComPtr<ID3D12GraphicsCommandList> cl0;
        hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       alloc.Get(), nullptr, IID_PPV_ARGS(&cl0));
        CheckHR(hr, "CreateCommandList");
        hr = cl0.As(&cl);
        CheckHR(hr, "QI(ID3D12GraphicsCommandList4)");
    }
    ComPtr<ID3D12CommandListRaytracing2> cl2;
    hr = cl.As(&cl2);
    if (FAILED(hr)) Die("CL does not support ID3D12CommandListRaytracing2", hr);

    ComPtr<ID3D12Fence> fence;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    CheckHR(hr, "CreateFence");
    HANDLE fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fenceValue = 0;
    auto flushAndWait = [&](const char* tag) {
        hr = cl->Close();
        CheckHR(hr, "cl->Close");
        ID3D12CommandList* cls[] = { cl.Get() };
        queue->ExecuteCommandLists(1, cls);
        ++fenceValue;
        hr = queue->Signal(fence.Get(), fenceValue);
        CheckHR(hr, "queue->Signal");
        if (fence->GetCompletedValue() < fenceValue) {
            fence->SetEventOnCompletion(fenceValue, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        if (g_deviceRemoved.load()) {
            wprintf(L"!!! DEVICE REMOVED after %hs: %s\n", tag, g_deviceRemovedReason.c_str());
            HRESULT removedReason = device->GetDeviceRemovedReason();
            fprintf(stderr, "    GetDeviceRemovedReason = 0x%08lx\n", (unsigned long)removedReason);
            exit(1);
        }
        hr = alloc->Reset();
        CheckHR(hr, "alloc->Reset");
        hr = cl->Reset(alloc.Get(), nullptr);
        CheckHR(hr, "cl->Reset");
    };

    // =========================================================================
    // BLAS: one triangle.  Two copies so we have two "BLAS GPUVA" values to
    // alternate between -- mirrors the LOD-swap pattern in the source sample
    // that triggers this TDR.
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
        // keep VB/IB/scratch alive until BLAS build is done.  Stash them
        // here so they survive flushAndWait -- pass back via static container.
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
    // PTLAS: m_cfg.instanceCount instances split across m_cfg.partitionCount
    // partitions.  ENABLE_PARTITION_TRANSLATION so we can also exercise
    // TRANSLATE_PARTITION (one of the three op types in the indirect
    // operation list).
    // =========================================================================
    D3D12_RTAS_PARTITIONED_TLAS_INPUTS_DESC ptlasInputs = {};
    ptlasInputs.InstanceCount  = g_cfg.instanceCount;
    ptlasInputs.PartitionCount = g_cfg.partitionCount;
    ptlasInputs.MaxInstancePerPartitionCount      = (g_cfg.instanceCount + g_cfg.partitionCount - 1) / g_cfg.partitionCount + 4;
    ptlasInputs.MaxInstanceInGlobalPartitionCount = 0;
    ptlasInputs.Flags = D3D12_RTAS_PARTITIONED_TLAS_FLAG_ENABLE_PARTITION_TRANSLATION;
    {
        D3D12_RTAS_OPERATION_INPUTS opIn = {};
        opIn.Type = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
        opIn.pPartitionedTLASInputsDesc = &ptlasInputs;
        D3D12_RTAS_OPERATION_PREBUILD_INFO ppre = {};
        deviceRT2->GetRTASOperationPrebuildInfo(&opIn, &ppre);
        printf("  PTLAS prebuild: result=%llu scratch=%llu\n",
               (unsigned long long)ppre.ResultDataMaxSizeInBytes,
               (unsigned long long)ppre.ScratchDataSizeInBytes);

        // Stash sizes for the build below; create the resources.
        // (Skipped — done below.)
    }
    // Re-query for the create-resource step (above call was just for the log).
    D3D12_RTAS_OPERATION_INPUTS opIn = {};
    opIn.Type = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
    opIn.pPartitionedTLASInputsDesc = &ptlasInputs;
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

    // Per-frame upload buffer arenas (just one upload buffer, cycled by frame).
    constexpr UINT64 kArenaBytes = 1 << 20;   // 1 MB per slot
    constexpr UINT kSlots = 4;
    ComPtr<ID3D12Resource> arenas[kSlots];
    void* arenaPtrs[kSlots] = {};
    for (UINT i = 0; i < kSlots; ++i) {
        arenas[i] = CreateBuffer(device.Get(), kArenaBytes, D3D12_HEAP_TYPE_UPLOAD,
                                 D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        D3D12_RANGE no = {0, 0};
        arenas[i]->Map(0, &no, &arenaPtrs[i]);
    }

    // =========================================================================
    // Raytracing pipeline state object + global root sig
    //   t0 = TLAS, u0 = output texture
    // =========================================================================
    ComPtr<ID3D12RootSignature> rootSig;
    {
        D3D12_DESCRIPTOR_RANGE uavRange = {};
        uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        uavRange.NumDescriptors = 1;
        uavRange.BaseShaderRegister = 0;

        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable.NumDescriptorRanges = 1;
        params[0].DescriptorTable.pDescriptorRanges = &uavRange;

        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1].Descriptor.ShaderRegister = 0;

        D3D12_ROOT_SIGNATURE_DESC rs = {};
        rs.NumParameters = 2;
        rs.pParameters = params;

        ComPtr<ID3DBlob> blob, err;
        hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
        if (FAILED(hr) && err) fprintf(stderr, "rs err: %.*s\n", (int)err->GetBufferSize(), (const char*)err->GetBufferPointer());
        CheckHR(hr, "SerializeRootSignature");
        hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                          IID_PPV_ARGS(&rootSig));
        CheckHR(hr, "CreateRootSignature");
    }

    auto bc = CompileShader(kShaderSource, L"", L"lib_6_10");
    printf("  Compiled shader: %zu bytes\n", bc.size());

    ComPtr<ID3D12StateObject> so;
    {
        D3D12_DXIL_LIBRARY_DESC lib = {};
        lib.DXILLibrary.pShaderBytecode = bc.data();
        lib.DXILLibrary.BytecodeLength = bc.size();
        D3D12_EXPORT_DESC exports[] = {
            { L"Raygen",     nullptr, D3D12_EXPORT_FLAG_NONE },
            { L"Miss",       nullptr, D3D12_EXPORT_FLAG_NONE },
            { L"ClosestHit", nullptr, D3D12_EXPORT_FLAG_NONE },
        };
        lib.NumExports = _countof(exports);
        lib.pExports = exports;

        D3D12_HIT_GROUP_DESC hg = {};
        hg.HitGroupExport = L"HitGroup";
        hg.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
        hg.ClosestHitShaderImport = L"ClosestHit";

        D3D12_RAYTRACING_SHADER_CONFIG sc = {};
        sc.MaxPayloadSizeInBytes   = 16;
        sc.MaxAttributeSizeInBytes = 8;

        D3D12_RAYTRACING_PIPELINE_CONFIG pc = {};
        pc.MaxTraceRecursionDepth = 1;

        D3D12_GLOBAL_ROOT_SIGNATURE grs = {};
        grs.pGlobalRootSignature = rootSig.Get();

        D3D12_STATE_SUBOBJECT subs[5] = {};
        subs[0].Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;          subs[0].pDesc = &lib;
        subs[1].Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;             subs[1].pDesc = &hg;
        subs[2].Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;  subs[2].pDesc = &sc;
        subs[3].Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG; subs[3].pDesc = &pc;
        subs[4].Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE; subs[4].pDesc = &grs;

        D3D12_STATE_OBJECT_DESC sod = {};
        sod.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
        sod.NumSubobjects = _countof(subs);
        sod.pSubobjects = subs;

        hr = device->CreateStateObject(&sod, IID_PPV_ARGS(&so));
        CheckHR(hr, "CreateStateObject");
    }

    // Shader table (1 raygen + 1 miss + 1 hit).
    ComPtr<ID3D12StateObjectProperties> soProps;
    so.As(&soProps);
    void* idRaygen = soProps->GetShaderIdentifier(L"Raygen");
    void* idMiss   = soProps->GetShaderIdentifier(L"Miss");
    void* idHit    = soProps->GetShaderIdentifier(L"HitGroup");
    const UINT idBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    const UINT recSize = (UINT)Align(idBytes, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
    const UINT tabAlign = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    UINT rgOff = 0;
    UINT msOff = (UINT)Align(rgOff + recSize, tabAlign);
    UINT hgOff = (UINT)Align(msOff + recSize, tabAlign);
    UINT total = hgOff + recSize;
    auto stable = CreateBuffer(device.Get(), total, D3D12_HEAP_TYPE_UPLOAD,
                               D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    {
        void* p = nullptr; D3D12_RANGE no = {0, 0};
        stable->Map(0, &no, &p);
        memset(p, 0, total);
        memcpy((uint8_t*)p + rgOff, idRaygen, idBytes);
        memcpy((uint8_t*)p + msOff, idMiss,   idBytes);
        memcpy((uint8_t*)p + hgOff, idHit,    idBytes);
        stable->Unmap(0, nullptr);
    }

    // Output texture + UAV descriptor heap.
    auto outTex = CreateTex2D(device.Get(), 64, 64, DXGI_FORMAT_R8G8B8A8_UNORM,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ComPtr<ID3D12DescriptorHeap> uavHeap;
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 1;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&uavHeap));
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(outTex.Get(), nullptr, &uavDesc,
                                          uavHeap->GetCPUDescriptorHandleForHeapStart());
    }

    // =========================================================================
    // Initial PTLAS build: WRITE_INSTANCE all instances.
    // =========================================================================
    auto WriteToArena = [&](UINT slot, const void* src, UINT64 bytes, UINT align) -> D3D12_GPU_VIRTUAL_ADDRESS {
        // SIMPLE bump allocator: each call writes immediately after the
        // previous one in the current slot, no head tracking across calls.
        // We track head in a static per-call counter below.
        static UINT64 heads[kSlots] = {};
        UINT64 head = Align(heads[slot], align);
        if (head + bytes > kArenaBytes) Die("arena overflow");
        memcpy((uint8_t*)arenaPtrs[slot] + head, src, (size_t)bytes);
        D3D12_GPU_VIRTUAL_ADDRESS gva = arenas[slot]->GetGPUVirtualAddress() + head;
        heads[slot] = head + bytes;
        return gva;
    };
    auto ResetArena = [&](UINT slot) {
        static UINT64 heads[kSlots] = {};  // Note: SHADOWS the lambda above's static.  // We track them differently below.
    };
    // We'll just allocate a new "logical head" for each slot at the start of
    // each frame.  Doing it via a closure-state vector:
    UINT64 arenaHead[kSlots] = {};
    auto AllocArena = [&](UINT slot, UINT64 bytes, UINT align) -> std::pair<void*, D3D12_GPU_VIRTUAL_ADDRESS> {
        arenaHead[slot] = Align(arenaHead[slot], align);
        if (arenaHead[slot] + bytes > kArenaBytes) Die("arena overflow");
        void* cpu = (uint8_t*)arenaPtrs[slot] + arenaHead[slot];
        D3D12_GPU_VIRTUAL_ADDRESS gpu = arenas[slot]->GetGPUVirtualAddress() + arenaHead[slot];
        arenaHead[slot] += bytes;
        return { cpu, gpu };
    };

    auto buildPtlas = [&](UINT slot, bool firstBuild,
                          const std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS>& writes,
                          const std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_UPDATE_INSTANCE_ARGS>& updates,
                          const std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TRANSLATE_PARTITION_ARGS>& translates)
    {
        // Build operation header array.  Order: WRITE first (matches the
        // source sample's order); the spec says order doesn't matter.
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
        if (!translates.empty()) {
            auto [cpu, gpu] = AllocArena(slot, translates.size() * sizeof(translates[0]), 8);
            memcpy(cpu, translates.data(), translates.size() * sizeof(translates[0]));
            D3D12_RTAS_PARTITIONED_TLAS_OPERATION op = {};
            op.Type = D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE_TRANSLATE_PARTITION;
            op.ArgCount = (UINT)translates.size();
            op.ArgData.StartAddress = gpu;
            op.ArgData.StrideInBytes = sizeof(translates[0]);
            ops.push_back(op);
        }
        if (ops.empty()) return;

        // Op header array + op count buffers.
        auto [opsCpu, opsGpu] = AllocArena(slot, ops.size() * sizeof(ops[0]), 8);
        memcpy(opsCpu, ops.data(), ops.size() * sizeof(ops[0]));

        UINT32 numOps32 = (UINT32)ops.size();
        auto [cntCpu, cntGpu] = AllocArena(slot, sizeof(UINT32), 4);
        memcpy(cntCpu, &numOps32, sizeof(numOps32));

        D3D12_RTAS_OPERATION_INPUTS opInputs   = {};
        opInputs.Type                          = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
        opInputs.pPartitionedTLASInputsDesc    = &ptlasInputs;

        D3D12_RTAS_PARTITIONED_TLAS_OPERATION_DATA opData = {};
        opData.SourceAccelerationStructureData     = firstBuild ? 0 : ptlas->GetGPUVirtualAddress();
        opData.DestAccelerationStructureData       = ptlas->GetGPUVirtualAddress();
        opData.ScratchAccelerationStructureData    = ptlasScratch->GetGPUVirtualAddress();
        opData.IndirectPartitionedTlasOpCount      = cntGpu;
        opData.IndirectPartitionedTlasOps          = opsGpu;

        D3D12_RTAS_OPERATION_DESC opDesc = {};
        opDesc.Inputs                       = opInputs;
        opDesc.pPartitionedTlasOperationData = &opData;

        cl2->ExecuteIndirectRTASOperations(1, &opDesc,
            D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

        D3D12_RESOURCE_BARRIER uav = {};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = ptlas.Get();
        cl->ResourceBarrier(1, &uav);
    };

    // -------------------- Initial WRITE pass --------------------
    {
        std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS> writes;
        for (UINT i = 0; i < g_cfg.instanceCount; ++i) {
            D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS w = {};
            w.Transform[0][0] = 1; w.Transform[1][1] = 1; w.Transform[2][2] = 1;
            // Spread instances in a row at z=5.
            w.Transform[0][3] = (float)(i % 8) * 0.4f - 1.6f;
            w.Transform[1][3] = (float)(i / 8) * 0.4f - 1.6f;
            w.Transform[2][3] = 5.0f;
            w.InstanceID = i;
            w.InstanceMask = 0xFF;
            w.InstanceContributionToHitGroupIndex = 0;
            w.InstanceFlags = g_cfg.noAabb ? D3D12_RTAS_PARTITIONED_TLAS_INSTANCE_FLAG_NONE
                                            : D3D12_RTAS_PARTITIONED_TLAS_INSTANCE_FLAG_ENABLE_EXPLICIT_AABB;
            w.AccelerationStructure = blasA->GetGPUVirtualAddress();
            w.InstanceIndex = i;
            w.PartitionIndex = i % g_cfg.partitionCount;
            // ExplicitAABB: 0.6 unit around the translation.
            if (!g_cfg.noAabb) {
                w.ExplicitAABB.MinX = w.Transform[0][3] - 0.6f;
                w.ExplicitAABB.MinY = w.Transform[1][3] - 0.6f;
                w.ExplicitAABB.MinZ = w.Transform[2][3] - 0.6f;
                w.ExplicitAABB.MaxX = w.Transform[0][3] + 0.6f;
                w.ExplicitAABB.MaxY = w.Transform[1][3] + 0.6f;
                w.ExplicitAABB.MaxZ = w.Transform[2][3] + 0.6f;
            }
            writes.push_back(w);
        }
        std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_UPDATE_INSTANCE_ARGS> nUpdates;
        std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TRANSLATE_PARTITION_ARGS> nTranslates;
        buildPtlas(0, /*firstBuild*/true, writes, nUpdates, nTranslates);
        flushAndWait("initial PTLAS build");
        for (UINT i = 0; i < kSlots; ++i) arenaHead[i] = 0;
    }

    printf("  PTLAS initial WRITE pass done (%u instances).  Entering main loop.\n",
           g_cfg.instanceCount);

    // =========================================================================
    // Main loop: per frame, build with WRITE 1 + UPDATE 1 + TRANSLATE N.
    // =========================================================================
    for (UINT frame = 0; frame < g_cfg.frames; ++frame) {
        UINT slot = frame % kSlots;
        arenaHead[slot] = 0;

        // Pick which instance to WRITE this frame (move it slightly and
        // alternate its BLAS) and which to UPDATE (different one, swap BLAS).
        UINT writeIdx  = frame % g_cfg.instanceCount;
        UINT updateIdx = (frame + g_cfg.instanceCount / 2) % g_cfg.instanceCount;
        if (updateIdx == writeIdx) updateIdx = (writeIdx + 1) % g_cfg.instanceCount;

        std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS> writes;
        if (!g_cfg.noWrite) {
            D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS w = {};
            w.Transform[0][0] = 1; w.Transform[1][1] = 1; w.Transform[2][2] = 1;
            w.Transform[0][3] = (float)(writeIdx % 8) * 0.4f - 1.6f
                              + 0.05f * sinf(frame * 0.1f);   // small per-frame jitter
            w.Transform[1][3] = (float)(writeIdx / 8) * 0.4f - 1.6f;
            w.Transform[2][3] = 5.0f;
            w.InstanceID = writeIdx;
            w.InstanceMask = 0xFF;
            w.InstanceContributionToHitGroupIndex = 0;
            w.InstanceFlags = g_cfg.noAabb ? D3D12_RTAS_PARTITIONED_TLAS_INSTANCE_FLAG_NONE
                                            : D3D12_RTAS_PARTITIONED_TLAS_INSTANCE_FLAG_ENABLE_EXPLICIT_AABB;
            w.AccelerationStructure = (frame & 1) ? blasB->GetGPUVirtualAddress()
                                                  : blasA->GetGPUVirtualAddress();
            w.InstanceIndex = writeIdx;
            w.PartitionIndex = writeIdx % g_cfg.partitionCount;
            if (!g_cfg.noAabb) {
                w.ExplicitAABB.MinX = w.Transform[0][3] - 0.7f;
                w.ExplicitAABB.MinY = w.Transform[1][3] - 0.7f;
                w.ExplicitAABB.MinZ = w.Transform[2][3] - 0.7f;
                w.ExplicitAABB.MaxX = w.Transform[0][3] + 0.7f;
                w.ExplicitAABB.MaxY = w.Transform[1][3] + 0.7f;
                w.ExplicitAABB.MaxZ = w.Transform[2][3] + 0.7f;
            }
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

        std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TRANSLATE_PARTITION_ARGS> translates;
        if (!g_cfg.noTranslate) {
            for (UINT p = 0; p < g_cfg.partitionCount; ++p) {
                D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TRANSLATE_PARTITION_ARGS t = {};
                t.PartitionIndex = p;
                t.PartitionTranslation[0] = 0.01f * frame;
                t.PartitionTranslation[1] = 0;
                t.PartitionTranslation[2] = 0;
                translates.push_back(t);
            }
        }

        buildPtlas(slot, /*firstBuild*/false, writes, updates, translates);

        if (!g_cfg.noDispatch) {
            ID3D12DescriptorHeap* heaps[] = { uavHeap.Get() };
            cl->SetDescriptorHeaps(1, heaps);
            cl->SetComputeRootSignature(rootSig.Get());
            cl->SetComputeRootDescriptorTable(0, uavHeap->GetGPUDescriptorHandleForHeapStart());
            cl->SetComputeRootShaderResourceView(1, ptlas->GetGPUVirtualAddress());
            cl->SetPipelineState1(so.Get());

            D3D12_DISPATCH_RAYS_DESC drd = {};
            auto stGva = stable->GetGPUVirtualAddress();
            drd.RayGenerationShaderRecord.StartAddress = stGva + rgOff;
            drd.RayGenerationShaderRecord.SizeInBytes  = recSize;
            drd.MissShaderTable.StartAddress  = stGva + msOff;
            drd.MissShaderTable.SizeInBytes   = recSize;
            drd.MissShaderTable.StrideInBytes = recSize;
            drd.HitGroupTable.StartAddress    = stGva + hgOff;
            drd.HitGroupTable.SizeInBytes     = recSize;
            drd.HitGroupTable.StrideInBytes   = recSize;
            drd.Width = 64; drd.Height = 64; drd.Depth = 1;
            cl->DispatchRays(&drd);
        }

        char tag[64];
        sprintf_s(tag, "frame %u (w=%zu u=%zu t=%zu)",
                  frame, writes.size(), updates.size(), translates.size());
        flushAndWait(tag);

        if ((frame % 5) == 0 || frame + 1 == g_cfg.frames) {
            printf("  frame %u OK (writes=%zu updates=%zu translates=%zu)\n",
                   frame, writes.size(), updates.size(), translates.size());
        }
    }

    printf("\nCompleted %u frames -- no TDR observed.\n", g_cfg.frames);
    return 0;
}
