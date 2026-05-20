//*********************************************************
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//*********************************************************
//
// Main.cpp - D3D12 DXR2 COMPRESSED1 cluster-geometry min repro.
//
// Builds a single 6-cluster cube (one CLAS per face) via DXR2's
// BUILD_CLAS_FROM_TRIANGLES + BUILD_BLAS_FROM_CLAS path, dispatches one
// ray per pixel.
//
//   Default                 (D3D12_VERTEX_FORMAT_COMPRESSED1) : BUG - only
//                                                                1 of 6
//                                                                faces hits.
//   --vertex-format float   (D3D12_VERTEX_FORMAT_FLOAT32_3   ) : OK   - full
//                                                                cube renders.
//
// Requires an experimental D3D12 build with DXR2 + lib_6_10 support; see
// ExperimentalD3D12.props for the runtime/DXC hookup.
//
// Single-file repro - no DXSample / DeviceResources / DirectXTK / PCH /
// procedural-geometry scaffolding. Goal: minimum fluff.
//*********************************************************

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <wrl/client.h>

#include <dxgi1_6.h>
#include <d3d12.h>
#include <d3dx12.h>           // CD3DX12_* helpers
#include <dxcapi.h>           // runtime HLSL compile via dxcompiler.dll
#include <DirectXMath.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <cfloat>
#include <algorithm>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

// ============================================================================
//  Log + error helpers
// ============================================================================

static void Log(const wchar_t* fmt, ...)
{
    wchar_t buf[1024];
    va_list a; va_start(a, fmt);
    _vsnwprintf_s(buf, _TRUNCATE, fmt, a);
    va_end(a);
    OutputDebugStringW(buf);
}

static void TF(HRESULT hr, const wchar_t* what = L"HRESULT failed")
{
    if (FAILED(hr))
    {
        wchar_t msg[256];
        _snwprintf_s(msg, _TRUNCATE, L"%ls (hr=0x%08X)\n", what, (unsigned)hr);
        Log(L"%s", msg);
        MessageBoxW(nullptr, msg, L"min repro", MB_OK | MB_ICONERROR);
        throw std::runtime_error("D3D12 call failed");
    }
}

// ============================================================================
//  COMPRESSED1 encoder
//
//  Per-cluster vertex-buffer layout (matches the public spec / d3d12conf
//  reference encoder):
//    12-byte D3D12_VERTEX_FORMAT_COMPRESSED1_HEADER
//      (8-bit biased exponent, three 24-bit signed anchors,
//       three 4-bit (bit_count - 1) widths per axis, range [1..16])
//    immediately followed by per-vertex (xBits + yBits + zBits) packed
//    deltas, LSB-first, no padding between components or vertices.
//
//  Decode:  pos[c] = (anchor[c] + delta[c]) * 2^(exponent - 127)
// ============================================================================
namespace Compressed1
{
    struct float3 { float x, y, z; };

    struct EncodedCluster
    {
        D3D12_VERTEX_FORMAT_COMPRESSED1_HEADER header = {};
        std::vector<uint8_t> bitstream;   // packed deltas
        unsigned vertexCount = 0;
        size_t TotalBytes() const { return sizeof(header) + bitstream.size(); }
    };

    static uint32_t BitsForValue(uint32_t v)
    {
        uint32_t b = 1;
        while ((v >> b) != 0u && b < 32u) ++b;
        return b;
    }

    static void WriteBitsLSB(std::vector<uint8_t>& dst, uint64_t& bitOff,
                             uint32_t value, uint32_t nBits)
    {
        uint32_t written = 0;
        while (written < nBits)
        {
            const uint64_t bytePos   = bitOff / 8;
            const uint32_t bitInByte = (uint32_t)(bitOff & 7);
            const uint32_t take      = std::min<uint32_t>(8 - bitInByte, nBits - written);
            const uint32_t mask      = (1u << take) - 1u;
            const uint32_t chunk     = (value >> written) & mask;
            dst[(size_t)bytePos] &= (uint8_t)~(mask << bitInByte);
            dst[(size_t)bytePos] |= (uint8_t)(chunk << bitInByte);
            written += take;
            bitOff  += take;
        }
    }

    // Encode one cluster.
    //   maxPrecisionBits caps per-axis bit count (1..16 per spec).
    //   forcedBiasedExponent in [1..232]: search starts here and never
    //     decreases (lets a scene-wide pick keep shared cluster-edge
    //     vertices bit-identical). Pass -1 for an unconstrained per-cluster
    //     pick.
    static EncodedCluster Encode(const std::vector<float3>& verts,
                                 int maxPrecisionBits = 12,
                                 int forcedBiasedExponent = -1)
    {
        EncodedCluster out;
        out.vertexCount = (unsigned)verts.size();
        if (verts.empty()) return out;

        maxPrecisionBits = std::clamp(maxPrecisionBits, 1, 16);

        float mn[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
        for (auto& v : verts)
        {
            const float p[3] = { v.x, v.y, v.z };
            for (int c = 0; c < 3; ++c) if (p[c] < mn[c]) mn[c] = p[c];
        }

        const uint64_t maxDeltaValue =
            (maxPrecisionBits >= 32) ? ~0ull : ((1ull << maxPrecisionBits) - 1ull);
        int exponent = (forcedBiasedExponent > 0) ? forcedBiasedExponent : 1;
        int32_t anchor[3] = { 0,0,0 };
        std::vector<uint32_t> deltas; deltas.reserve(verts.size() * 3);
        uint32_t maxDelta[3] = { 0,0,0 };

        for (;;)
        {
            const double scale    = std::ldexp(1.0, exponent - 127);
            const double invScale = 1.0 / scale;

            // anchor = floor(min/scale + 0.5) (monotonic with per-vertex
            // rounding, so anchor IS the min of the quantized set =>
            // all deltas >= 0 with no separate clamp).
            for (int i = 0; i < 3; ++i)
            {
                double a = std::floor((double)mn[i] * invScale + 0.5);
                a = std::clamp(a, -8388608.0, 8388607.0);   // 24-bit signed
                anchor[i] = (int32_t)a;
            }

            deltas.clear();
            maxDelta[0] = maxDelta[1] = maxDelta[2] = 0;
            bool overflow = false;
            for (const auto& v : verts)
            {
                const float p[3] = { v.x, v.y, v.z };
                for (int i = 0; i < 3; ++i)
                {
                    const double qd = std::floor((double)p[i] * invScale + 0.5);
                    const uint64_t d = (uint64_t)(int64_t)qd - (uint64_t)anchor[i];
                    if (d > 0xFFFFFFFFull || d > maxDeltaValue) overflow = true;
                    deltas.push_back((uint32_t)d);
                    if ((uint32_t)d > maxDelta[i]) maxDelta[i] = (uint32_t)d;
                }
            }

            bool anchorOk = !overflow;
            for (int i = 0; anchorOk && i < 3; ++i)
                if ((int64_t)anchor[i] + (int64_t)maxDelta[i] > 8388607ll) anchorOk = false;
            if (anchorOk) break;
            if (++exponent > 232) break;   // saturate
        }

        const uint32_t bitsNeeded[3] = {
            std::clamp<uint32_t>(BitsForValue(maxDelta[0]), 1u, (uint32_t)maxPrecisionBits),
            std::clamp<uint32_t>(BitsForValue(maxDelta[1]), 1u, (uint32_t)maxPrecisionBits),
            std::clamp<uint32_t>(BitsForValue(maxDelta[2]), 1u, (uint32_t)maxPrecisionBits),
        };

        ENCODE_D3D12_COMPRESSED1(out.header,
                                 (uint32_t)exponent,
                                 anchor[0], anchor[1], anchor[2],
                                 bitsNeeded[0] - 1, bitsNeeded[1] - 1, bitsNeeded[2] - 1);

        const uint64_t totalBits =
            (uint64_t)(bitsNeeded[0] + bitsNeeded[1] + bitsNeeded[2]) * verts.size();
        out.bitstream.assign((size_t)((totalBits + 7) / 8), 0);
        uint64_t bitOff = 0;
        for (size_t v = 0; v < verts.size(); ++v)
        {
            const uint32_t masks[3] = {
                (bitsNeeded[0] >= 32) ? ~0u : (1u << bitsNeeded[0]) - 1u,
                (bitsNeeded[1] >= 32) ? ~0u : (1u << bitsNeeded[1]) - 1u,
                (bitsNeeded[2] >= 32) ? ~0u : (1u << bitsNeeded[2]) - 1u,
            };
            for (int c = 0; c < 3; ++c)
            {
                uint32_t d = deltas[v * 3 + c];
                if (d > masks[c]) d = masks[c];
                WriteBitsLSB(out.bitstream, bitOff, d, bitsNeeded[c]);
            }
        }
        return out;
    }
}

// ============================================================================
//  Cube geometry
//
//  6 faces, each its own cluster: 4 verts, 2 tris (winding CCW from outside
//  so RAY_FLAG_CULL_BACK_FACING_TRIANGLES keeps them).
// ============================================================================
struct Face
{
    std::vector<Compressed1::float3> positions;  // 4 corners
    static constexpr uint16_t indices[6] = { 0, 3, 2, 0, 1, 3 };
};

// Generate a unit cube (halfExtent h) as 6 single-cluster faces. Each face's
// CCW-from-outside winding matches the index pattern above (the axisU x
// axisV cross product is the outward normal).
static std::vector<Face> BuildCubeFaces(float h)
{
    struct Basis { Compressed1::float3 origin, axisU, axisV; };
    const Basis bases[6] = {
        // +X (n=+X): U=+Y V=+Z
        {{ +h,-h,-h}, {0,1,0}, {0,0,1}},
        // -X (n=-X): U=+Z V=+Y
        {{ -h,-h,-h}, {0,0,1}, {0,1,0}},
        // +Y (n=+Y): U=+Z V=+X
        {{ -h,+h,-h}, {0,0,1}, {1,0,0}},
        // -Y (n=-Y): U=+X V=+Z
        {{ -h,-h,-h}, {1,0,0}, {0,0,1}},
        // +Z (n=+Z): U=+X V=+Y
        {{ -h,-h,+h}, {1,0,0}, {0,1,0}},
        // -Z (n=-Z): U=+Y V=+X
        {{ -h,-h,-h}, {0,1,0}, {1,0,0}},
    };
    const float du = 2.0f * h;
    std::vector<Face> faces(6);
    for (int f = 0; f < 6; ++f)
    {
        const auto& b = bases[f];
        for (int li = 0; li <= 1; ++li)
        for (int lj = 0; lj <= 1; ++lj)
        {
            const float u = du * (float)lj;
            const float v = du * (float)li;
            faces[f].positions.push_back({
                b.origin.x + b.axisU.x * u + b.axisV.x * v,
                b.origin.y + b.axisU.y * u + b.axisV.y * v,
                b.origin.z + b.axisU.z * u + b.axisV.z * v });
        }
    }
    return faces;
}
constexpr uint16_t Face::indices[6];

// ============================================================================
//  Shader source (runtime-compiled via dxcompiler.dll, target lib_6_10).
// ============================================================================
static const char kRaytracingHLSL[] = R"HLSL(
struct SceneCB { float4x4 viewToWorld; float4 cameraPosition; float4 misc; };

RaytracingAccelerationStructure       Scene   : register(t0);
RWTexture2D<float4>                   Output  : register(u0);
ConstantBuffer<SceneCB>               gScene  : register(b0);

struct [raypayload] Payload {
    float4 color : write(caller, closesthit, miss) : read(caller);
};
typedef BuiltInTriangleIntersectionAttributes Attribs;

[shader("raygeneration")]
void RayGen()
{
    const uint2  px      = DispatchRaysIndex().xy;
    const uint2  dim     = DispatchRaysDimensions().xy;
    const float2 ndc     = (float2(px) + 0.5) / float2(dim) * 2.0 - 1.0;
    const float2 v2d     = float2(ndc.x, -ndc.y);
    const float  aspect  = gScene.misc.x;
    const float  tanHF   = gScene.misc.y;
    const float3 dirView = normalize(float3(v2d.x * aspect * tanHF, v2d.y * tanHF, 1.0));
    const float3 dirWld  = mul((float3x3)gScene.viewToWorld, dirView);

    RayDesc r;
    r.Origin    = gScene.cameraPosition.xyz;
    r.Direction = normalize(dirWld);
    r.TMin      = 0.001;
    r.TMax      = 1000.0;

    Payload p; p.color = float4(0,0,0,1);
    TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xff, 0, 0, 0, r, p);
    Output[px] = p.color;
}

[shader("miss")]
void Miss(inout Payload p) { p.color = float4(0.40, 0.60, 0.90, 1); }

[shader("closesthit")]
void OpaqueHit(inout Payload p, in Attribs a)
{
    // Per-(cluster, primitive, instance) hash colour - each surviving
    // triangle reads as a distinct colour so missing faces are obvious.
    const uint cid  = ClusterID();
    const uint pid  = PrimitiveIndex();
    const uint inst = InstanceIndex();
    const uint h    = (cid * 2654435761u) ^ (pid * 374761393u) ^ (inst * 668265263u);
    p.color = float4(((h >>  0) & 0xFF) / 255.0,
                     ((h >>  8) & 0xFF) / 255.0,
                     ((h >> 16) & 0xFF) / 255.0,
                     1.0);
}
)HLSL";

static ComPtr<IDxcBlob> CompileShader(const char* src, size_t len)
{
    HMODULE mod = LoadLibraryW(L"dxcompiler.dll");
    if (!mod) TF(HRESULT_FROM_WIN32(GetLastError()),
                 L"LoadLibrary(dxcompiler.dll) failed - is it next to the exe?");
    using PFN_DxcCreate = HRESULT(WINAPI*)(REFCLSID, REFIID, LPVOID*);
    auto pCreate = (PFN_DxcCreate)GetProcAddress(mod, "DxcCreateInstance");

    ComPtr<IDxcCompiler3> compiler;
    pCreate(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));

    DxcBuffer buf{ src, len, DXC_CP_UTF8 };
    LPCWSTR args[] = { L"-T", L"lib_6_10", L"-HV", L"2021" };
    ComPtr<IDxcResult> result;
    TF(compiler->Compile(&buf, args, _countof(args), nullptr, IID_PPV_ARGS(&result)),
       L"IDxcCompiler3::Compile failed");

    HRESULT hr = E_FAIL;
    result->GetStatus(&hr);
    if (FAILED(hr))
    {
        ComPtr<IDxcBlobUtf8> errors;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
        Log(L"Shader compile error:\n%hs\n",
            errors ? errors->GetStringPointer() : "(no error blob)");
        TF(hr, L"Shader compile failed");
    }
    ComPtr<IDxcBlob> dxil;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&dxil), nullptr);
    return dxil;
}

// ============================================================================
//  Small D3D12 buffer helpers (one-call upload + UAV-default).
// ============================================================================
static void AllocateUploadBuffer(ID3D12Device* device, const void* src, UINT64 size,
                                 ID3D12Resource** outRes, const wchar_t* name = nullptr)
{
    auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto desc = CD3DX12_RESOURCE_DESC::Buffer(size);
    TF(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                       D3D12_RESOURCE_STATE_GENERIC_READ,
                                       nullptr, IID_PPV_ARGS(outRes)));
    if (name) (*outRes)->SetName(name);
    void* mapped = nullptr;
    (*outRes)->Map(0, nullptr, &mapped);
    memcpy(mapped, src, (size_t)size);
    (*outRes)->Unmap(0, nullptr);
}

static void AllocateUAVBuffer(ID3D12Device* device, UINT64 size,
                              ID3D12Resource** outRes,
                              D3D12_RESOURCE_STATES initialState,
                              const wchar_t* name = nullptr)
{
    auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto desc = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    TF(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                       initialState, nullptr, IID_PPV_ARGS(outRes)));
    if (name) (*outRes)->SetName(name);
}

// ============================================================================
//  Window
// ============================================================================
static constexpr UINT kWidth  = 1280;
static constexpr UINT kHeight = 720;
static HWND g_hwnd = nullptr;

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    if (msg == WM_KEYDOWN && wp == VK_ESCAPE) PostQuitMessage(0);
    if (msg == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static void CreateAppWindow(HINSTANCE inst, const wchar_t* title)
{
    WNDCLASSEXW wc = { sizeof(wc) };
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = inst;
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"DXR2_Compressed1_Repro";
    RegisterClassExW(&wc);

    RECT r = { 0, 0, (LONG)kWidth, (LONG)kHeight };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    g_hwnd = CreateWindowW(wc.lpszClassName, title, WS_OVERLAPPEDWINDOW,
                           CW_USEDEFAULT, CW_USEDEFAULT,
                           r.right - r.left, r.bottom - r.top,
                           nullptr, nullptr, inst, nullptr);
}

// ============================================================================
//  App
// ============================================================================
class App
{
public:
    enum class VertexMode { Float32_3, Compressed1 };
    VertexMode mode = VertexMode::Compressed1;     // bug-repro default
    UINT compressedBits = 12;                      // COMPRESSED1 precision

    // ---- device / swapchain ----
    static constexpr UINT kFrameCount = 2;
    ComPtr<IDXGIFactory4>                 factory;
    ComPtr<ID3D12Device5>                 device;
    ComPtr<ID3D12DeviceRaytracing2>       dxr2Device;
    ComPtr<ID3D12CommandQueue>            queue;
    ComPtr<IDXGISwapChain3>               swapChain;
    ComPtr<ID3D12Resource>                backBuffers[kFrameCount];
    ComPtr<ID3D12CommandAllocator>        cmdAlloc[kFrameCount];
    ComPtr<ID3D12GraphicsCommandList4>    cmdList;
    ComPtr<ID3D12CommandListRaytracing2>  dxr2CmdList;
    ComPtr<ID3D12Fence>                   fence;
    UINT64                                fenceVals[kFrameCount] = {};
    UINT64                                nextFenceVal = 1;
    HANDLE                                fenceEvent = nullptr;
    UINT                                  frameIndex = 0;

    // ---- output UAV ----
    ComPtr<ID3D12DescriptorHeap>          uavHeap;
    ComPtr<ID3D12Resource>                rtOutput;
    D3D12_GPU_DESCRIPTOR_HANDLE           rtOutputUav = {};

    // ---- scene ----
    std::vector<Face>                                       cubeFaces;
    std::vector<Compressed1::EncodedCluster>                encoded;
    UINT                                                    clusterCount = 0;

    // ---- AS ----
    ComPtr<ID3D12Resource>                clusterInputBuf;   // concatenated VB+IB
    ComPtr<ID3D12Resource>                clasArgsBuf;
    ComPtr<ID3D12Resource>                clasResultBuf;
    ComPtr<ID3D12Resource>                clasScratchBuf;
    ComPtr<ID3D12Resource>                clasAddrArray;
    ComPtr<ID3D12Resource>                blasArgsBuf;
    ComPtr<ID3D12Resource>                blasDestAddrBuf;
    ComPtr<ID3D12Resource>                blasStorage;
    D3D12_GPU_VIRTUAL_ADDRESS             blasGPUVA = 0;
    ComPtr<ID3D12Resource>                blasScratchBuf;
    ComPtr<ID3D12Resource>                tlasBuf;
    ComPtr<ID3D12Resource>                tlasScratchBuf;
    ComPtr<ID3D12Resource>                tlasInstanceBuf;

    // ---- raytracing pipeline ----
    ComPtr<ID3D12RootSignature>           globalRS;
    ComPtr<ID3D12RootSignature>           localRS;
    ComPtr<ID3D12StateObject>             rtPipeline;
    ComPtr<ID3D12Resource>                raygenST;
    ComPtr<ID3D12Resource>                missST;
    ComPtr<ID3D12Resource>                hitST;

    // ---- scene CB ----
    ComPtr<ID3D12Resource>                sceneCB;
    void*                                 sceneCBMapped = nullptr;

    // ------------------------------------------------------------------
    void ParseCommandLine(int argc, WCHAR** argv)
    {
        for (int i = 1; i < argc - 1; ++i)
            if (_wcsicmp(argv[i], L"--vertex-format") == 0)
            {
                if      (_wcsicmp(argv[i+1], L"float")      == 0) mode = VertexMode::Float32_3;
                else if (_wcsicmp(argv[i+1], L"compressed") == 0) mode = VertexMode::Compressed1;
            }
    }

    // ------------------------------------------------------------------
    void Init()
    {
        // Experimental features must be enabled BEFORE the first
        // D3D12CreateDevice (otherwise this returns DXGI_ERROR_SDK_COMPONENT_MISSING).
        UUID feats[] = { D3D12ExperimentalShaderModels, D3D12RaytracingExperiment };
        TF(D3D12EnableExperimentalFeatures(_countof(feats), feats, nullptr, nullptr),
           L"D3D12EnableExperimentalFeatures failed (Windows Developer Mode disabled?)");

        InitDevice();
        InitSwapChain();
        InitFenceAndCmdList();
        InitDescriptorHeapAndUavOutput();
        InitSceneCB();
        BuildScene();
        BuildAccelerationStructures();
        BuildRaytracingPipeline();
    }

    // ------------------------------------------------------------------
    void InitDevice()
    {
#ifdef _DEBUG
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer();
#endif
        TF(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));

        // Iterate hardware adapters; fall back to WARP. The experimental D3D12Core
        // we ship in bin\D3D12\ supports DXR2 on WARP, so this works headless too.
        ComPtr<IDXGIAdapter1> adapter;
        ComPtr<IDXGIFactory6> factory6;
        factory.As(&factory6);
        for (UINT i = 0;
             factory6 && factory6->EnumAdapterByGpuPreference(i,
                 DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
             ++i)
        {
            DXGI_ADAPTER_DESC1 d; adapter->GetDesc1(&d);
            if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                            __uuidof(ID3D12Device5), nullptr))) break;
        }
        if (!adapter) factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));

        TF(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
        TF(device->QueryInterface(IID_PPV_ARGS(&dxr2Device)),
           L"ID3D12DeviceRaytracing2 not available - is the experimental D3D12Core loaded?");

        D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL opts = {};
        HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL,
                                                 &opts, sizeof(opts));
        if (FAILED(hr) || !opts.ClustersAndPTLASSupported)
            TF(E_FAIL, L"ClustersAndPTLASSupported = NO on this adapter");

        DXGI_ADAPTER_DESC desc; adapter->GetDesc(&desc);
        Log(L"[device] adapter=%ls  ClustersAndPTLAS=YES\n", desc.Description);

        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        TF(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
    }

    // ------------------------------------------------------------------
    void InitSwapChain()
    {
        DXGI_SWAP_CHAIN_DESC1 scd = {};
        scd.Width       = kWidth;
        scd.Height      = kHeight;
        scd.Format      = DXGI_FORMAT_B8G8R8A8_UNORM;
        scd.SampleDesc  = { 1, 0 };
        scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scd.BufferCount = kFrameCount;
        scd.SwapEffect  = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        ComPtr<IDXGISwapChain1> sc1;
        TF(factory->CreateSwapChainForHwnd(queue.Get(), g_hwnd, &scd, nullptr, nullptr, &sc1));
        TF(sc1.As(&swapChain));
        factory->MakeWindowAssociation(g_hwnd, DXGI_MWA_NO_ALT_ENTER);
        for (UINT i = 0; i < kFrameCount; ++i)
            TF(swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])));
        frameIndex = swapChain->GetCurrentBackBufferIndex();
    }

    // ------------------------------------------------------------------
    void InitFenceAndCmdList()
    {
        for (UINT i = 0; i < kFrameCount; ++i)
            TF(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              IID_PPV_ARGS(&cmdAlloc[i])));
        TF(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                     cmdAlloc[frameIndex].Get(), nullptr,
                                     IID_PPV_ARGS(&cmdList)));
        TF(cmdList->QueryInterface(IID_PPV_ARGS(&dxr2CmdList)),
           L"ID3D12CommandListRaytracing2 QI failed");
        TF(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        // Command list begins open; close it so the per-frame Reset+record
        // path works uniformly (including the AS-build pass below).
        TF(cmdList->Close());
    }

    // ------------------------------------------------------------------
    void InitDescriptorHeapAndUavOutput()
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.NumDescriptors = 1;
        hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        TF(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&uavHeap)));

        auto outDesc = CD3DX12_RESOURCE_DESC::Tex2D(
            DXGI_FORMAT_B8G8R8A8_UNORM, kWidth, kHeight, 1, 1, 1, 0,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        TF(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &outDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&rtOutput)));
        rtOutput->SetName(L"RT output");
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(rtOutput.Get(), nullptr, &uavDesc,
                                          uavHeap->GetCPUDescriptorHandleForHeapStart());
        rtOutputUav = uavHeap->GetGPUDescriptorHandleForHeapStart();
    }

    // ------------------------------------------------------------------
    struct SceneCBData {
        XMMATRIX viewToWorld;
        XMFLOAT4 cameraPosition;
        XMFLOAT4 misc;        // .x = aspect, .y = tan(fov/2)
    };

    void InitSceneCB()
    {
        const UINT sz = (sizeof(SceneCBData) + 255) & ~255u;
        auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(sz);
        TF(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&sceneCB)));
        sceneCB->SetName(L"Scene CB");
        TF(sceneCB->Map(0, nullptr, &sceneCBMapped));

        // Static camera: eye looking at origin from in front and slightly above,
        // chosen to show 3 cube faces (+X, +Y, -Z).
        XMVECTOR eye = XMVectorSet(1.5f, 1.1f, -1.5f, 1.0f);
        XMVECTOR at  = XMVectorSet(0,    0,    0,    1.0f);
        XMVECTOR up  = XMVectorSet(0,    1,    0,    0.0f);
        XMMATRIX view = XMMatrixLookAtLH(eye, at, up);

        SceneCBData cb = {};
        cb.viewToWorld = XMMatrixInverse(nullptr, view);
        XMStoreFloat4(&cb.cameraPosition, eye);
        cb.misc.x = (float)kWidth / (float)kHeight;
        cb.misc.y = std::tan(60.0f * (XM_PI / 180.0f) * 0.5f);  // 60-deg vertical FOV
        memcpy(sceneCBMapped, &cb, sizeof(cb));
    }

    // ------------------------------------------------------------------
    void BuildScene()
    {
        cubeFaces = BuildCubeFaces(0.45f);
        clusterCount = (UINT)cubeFaces.size();

        // Pick a single scene-wide COMPRESSED1 exponent so shared cluster-edge
        // vertices quantize identically (watertight at quantization level).
        float maxExtent = 0.f;
        for (const auto& f : cubeFaces)
        {
            float mn[3] = { FLT_MAX,FLT_MAX,FLT_MAX }, mx[3] = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
            for (const auto& p : f.positions)
            {
                const float ps[3] = { p.x, p.y, p.z };
                for (int k = 0; k < 3; ++k) { mn[k] = std::min(mn[k], ps[k]); mx[k] = std::max(mx[k], ps[k]); }
            }
            for (int k = 0; k < 3; ++k) maxExtent = std::max(maxExtent, mx[k] - mn[k]);
        }
        const float minUnit = maxExtent / float((1ull << compressedBits) - 1);
        int sharedExponent = (int)std::ceil(std::log2(minUnit)) + 127;
        sharedExponent = std::clamp(sharedExponent, 1, 232);

        encoded.clear();
        encoded.reserve(cubeFaces.size());
        for (const auto& f : cubeFaces)
            encoded.push_back(Compressed1::Encode(f.positions, (int)compressedBits, sharedExponent));

        Log(L"[scene] cube: %u clusters, %u tris.  vertex format: %s\n",
            clusterCount, clusterCount * 2,
            mode == VertexMode::Compressed1 ? L"COMPRESSED1 (BUG repro - default)"
                                            : L"FLOAT32_3 (correct render)");
    }

    // ------------------------------------------------------------------
    //  AS build: upload VBs + IBs, BUILD_CLAS_FROM_TRIANGLES (implicit dest),
    //  BUILD_BLAS_FROM_CLAS (explicit dest, 1 BLAS containing all 6 CLAS),
    //  classic TLAS with 1 instance.
    // ------------------------------------------------------------------
    void BuildAccelerationStructures()
    {
        TF(cmdAlloc[frameIndex]->Reset());
        TF(cmdList->Reset(cmdAlloc[frameIndex].Get(), nullptr));

        UploadClusterInputs();
        BuildClas();
        BuildBlas();
        BuildTlas();

        TF(cmdList->Close());
        ID3D12CommandList* lists[] = { cmdList.Get() };
        queue->ExecuteCommandLists(1, lists);
        WaitForGpu();
    }

    // ------------------------------------------------------------------
    void UploadClusterInputs()
    {
        const bool useFloat = (mode == VertexMode::Float32_3);
        // FLOAT32_3 is fine at 16B; COMPRESSED1 wants generous padding -
        // some drivers prefetch past the end of a packed bitstream, and
        // back-to-back clusters can let that prefetch step into the next
        // cluster's bits.
        const size_t kAlign = useFloat ? 16 : 256;
        auto align = [](size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); };

        struct Slot { size_t vbOff, vbSize, ibOff, ibSize; };
        std::vector<Slot> slots(clusterCount);
        size_t cursor = 0;
        for (UINT i = 0; i < clusterCount; ++i)
        {
            cursor = align(cursor, kAlign);
            slots[i].vbOff  = cursor;
            slots[i].vbSize = useFloat
                ? cubeFaces[i].positions.size() * sizeof(Compressed1::float3)
                : encoded[i].TotalBytes();
            cursor += slots[i].vbSize;
            cursor = align(cursor, kAlign);
            slots[i].ibOff  = cursor;
            slots[i].ibSize = sizeof(Face::indices);
            cursor += slots[i].ibSize;
        }

        auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(align(cursor, 256));
        TF(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&clusterInputBuf)));
        clusterInputBuf->SetName(L"Cluster VB+IB");

        uint8_t* mapped = nullptr;
        TF(clusterInputBuf->Map(0, nullptr, reinterpret_cast<void**>(&mapped)));
        const D3D12_GPU_VIRTUAL_ADDRESS baseGVA = clusterInputBuf->GetGPUVirtualAddress();
        for (UINT i = 0; i < clusterCount; ++i)
        {
            if (useFloat)
            {
                memcpy(mapped + slots[i].vbOff, cubeFaces[i].positions.data(), slots[i].vbSize);
            }
            else
            {
                memcpy(mapped + slots[i].vbOff, &encoded[i].header, sizeof(encoded[i].header));
                if (!encoded[i].bitstream.empty())
                    memcpy(mapped + slots[i].vbOff + sizeof(encoded[i].header),
                           encoded[i].bitstream.data(), encoded[i].bitstream.size());
            }
            memcpy(mapped + slots[i].ibOff, Face::indices, sizeof(Face::indices));
        }
        clusterInputBuf->Unmap(0, nullptr);

        // CPU-fill BUILD_CLAS_FROM_TRIANGLES_ARGS[N] (consumed by
        // ExecuteIndirectRTASOperations below).
        std::vector<D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS> args(clusterCount);
        for (UINT i = 0; i < clusterCount; ++i)
        {
            auto& a = args[i];
            a = {};
            a.ClusterID                 = 500 + i;
            a.TriangleCount             = 2;
            a.VertexCount               = 4;
            a.BaseGeometryIndexAndFlags = (UINT)D3D12_RTAS_CLUSTERED_GEOMETRY_FLAG_OPAQUE;
            a.VertexBufferStride        = useFloat ? (UINT16)sizeof(Compressed1::float3) : 0;
            a.IndexBufferStride         = sizeof(uint16_t);
            a.PositionTruncateBitCount  = 0;
            a.VertexBuffer              = baseGVA + slots[i].vbOff;
            a.IndexBuffer               = baseGVA + slots[i].ibOff;
        }
        AllocateUploadBuffer(device.Get(), args.data(),
                             args.size() * sizeof(args[0]),
                             &clasArgsBuf, L"CLAS args");
    }

    // ------------------------------------------------------------------
    void BuildClas()
    {
        const bool useFloat = (mode == VertexMode::Float32_3);
        const UINT N = clusterCount;
        UINT maxCompSize = 0;
        if (!useFloat) for (const auto& e : encoded)
            maxCompSize = std::max(maxCompSize, (UINT)e.TotalBytes());

        D3D12_RTAS_CLUSTER_LIMITS limits = {};
        limits.MaxArgCount                                   = N;
        limits.MaxUniqueGeometryIndexAndFlagsCountPerCluster = 256;
        limits.MaxTriangleCountPerCluster                    = 2;
        limits.MaxVertexCountPerCluster                      = 4;
        limits.MaxTotalTriangleCount                         = 2 * N;
        limits.MaxTotalVertexCount                           = 4 * N;

        D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC clas = {};
        clas.ClusterLimits                    = limits;
        clas.Flags                            = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE
                                              | D3D12_RTAS_OPERATION_FLAG_ALLOW_DATA_ACCESS;
        clas.VertexFormat                     = useFloat ? D3D12_VERTEX_FORMAT_FLOAT32_3
                                                         : D3D12_VERTEX_FORMAT_COMPRESSED1;
        clas.IndexFormat                      = D3D12_INDEX_FORMAT_UINT16;
        clas.GeometryIndexAndFlagsIndexFormat = D3D12_INDEX_FORMAT_NONE;
        clas.OpacityMicromapIndexFormat       = D3D12_INDEX_FORMAT_NONE;
        if (useFloat) clas.MinPositionTruncateBitCount        = 0;
        else          clas.MaxCompressedClusterPositionsSize  = maxCompSize;
        clas.Mode = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;

        D3D12_RTAS_OPERATION_INPUTS inputs = {};
        inputs.Type                  = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
        inputs.pClusterTrianglesDesc = &clas;

        D3D12_RTAS_OPERATION_PREBUILD_INFO pre = {};
        dxr2Device->GetRTASOperationPrebuildInfo(&inputs, &pre);

        AllocateUAVBuffer(device.Get(), pre.ResultDataMaxSizeInBytes, &clasResultBuf,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, L"CLAS result");
        AllocateUAVBuffer(device.Get(), std::max<UINT64>(pre.ScratchDataSizeInBytes, 256ull),
            &clasScratchBuf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"CLAS scratch");
        AllocateUAVBuffer(device.Get(), (UINT64)N * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
            &clasAddrArray, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"CLAS addrs");

        D3D12_RTAS_BATCHED_OPERATION_DATA batched = {};
        batched.AddressResolutionFlags = D3D12_RTAS_OPERATION_ADDRESS_RESOLUTION_FLAG_NONE;
        batched.BatchResultData        = clasResultBuf->GetGPUVirtualAddress();
        batched.BatchScratchData       = clasScratchBuf->GetGPUVirtualAddress();
        batched.ResultAddressArray     = { clasAddrArray->GetGPUVirtualAddress(),
                                           sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
        batched.IndirectArgumentArray  = { clasArgsBuf->GetGPUVirtualAddress(),
                                           sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS) };

        D3D12_RTAS_OPERATION_DESC op = {};
        op.Inputs                = inputs;
        op.pBatchedOperationData = &batched;
        dxr2CmdList->ExecuteIndirectRTASOperations(1, &op,
            D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

        D3D12_RESOURCE_BARRIER bars[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(clasResultBuf.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(clasAddrArray.Get()),
        };
        cmdList->ResourceBarrier(_countof(bars), bars);
    }

    // ------------------------------------------------------------------
    void BuildBlas()
    {
        D3D12_RTAS_CLAS_INPUTS_DESC blas = {};
        blas.Flags              = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;
        blas.MaxArgCount        = 1;
        blas.Mode               = D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS;
        blas.MaxTotalClasCount  = clusterCount;
        blas.MaxClasCountPerArg = clusterCount;

        D3D12_RTAS_OPERATION_INPUTS inputs = {};
        inputs.Type      = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
        inputs.pClasDesc = &blas;

        D3D12_RTAS_OPERATION_PREBUILD_INFO pre = {};
        dxr2Device->GetRTASOperationPrebuildInfo(&inputs, &pre);

        AllocateUAVBuffer(device.Get(), pre.ResultDataMaxSizeInBytes, &blasStorage,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, L"Cluster BLAS");
        AllocateUAVBuffer(device.Get(), std::max<UINT64>(pre.ScratchDataSizeInBytes, 256ull),
            &blasScratchBuf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"BLAS scratch");
        blasGPUVA = blasStorage->GetGPUVirtualAddress();

        D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS arg = {};
        arg.ClasAddressCount  = clusterCount;
        arg.ClasAddressStride = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
        arg.ClasAddressArray  = clasAddrArray->GetGPUVirtualAddress();
        AllocateUploadBuffer(device.Get(), &arg, sizeof(arg),  &blasArgsBuf,    L"BLAS arg");
        AllocateUploadBuffer(device.Get(), &blasGPUVA, sizeof(blasGPUVA), &blasDestAddrBuf, L"BLAS dest");

        D3D12_RTAS_BATCHED_OPERATION_DATA batched = {};
        batched.BatchScratchData      = blasScratchBuf->GetGPUVirtualAddress();
        batched.ResultAddressArray    = { blasDestAddrBuf->GetGPUVirtualAddress(),
                                          sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
        batched.IndirectArgumentArray = { blasArgsBuf->GetGPUVirtualAddress(), sizeof(arg) };

        D3D12_RTAS_OPERATION_DESC op = {};
        op.Inputs                = inputs;
        op.pBatchedOperationData = &batched;
        dxr2CmdList->ExecuteIndirectRTASOperations(1, &op,
            D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

        auto bar = CD3DX12_RESOURCE_BARRIER::UAV(blasStorage.Get());
        cmdList->ResourceBarrier(1, &bar);
    }

    // ------------------------------------------------------------------
    void BuildTlas()
    {
        D3D12_RAYTRACING_INSTANCE_DESC inst = {};
        XMMATRIX m = XMMatrixIdentity();
        XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(inst.Transform), m);
        inst.InstanceID                          = 0;
        inst.InstanceMask                        = 0xFF;
        inst.InstanceContributionToHitGroupIndex = 0;
        inst.Flags                               = D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
        inst.AccelerationStructure               = blasGPUVA;
        AllocateUploadBuffer(device.Get(), &inst, sizeof(inst),
                             &tlasInstanceBuf, L"TLAS instance");

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasIn = {};
        tlasIn.Type          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlasIn.DescsLayout   = D3D12_ELEMENTS_LAYOUT_ARRAY;
        tlasIn.NumDescs      = 1;
        tlasIn.Flags         = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        tlasIn.InstanceDescs = tlasInstanceBuf->GetGPUVirtualAddress();

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pre = {};
        device->GetRaytracingAccelerationStructurePrebuildInfo(&tlasIn, &pre);

        AllocateUAVBuffer(device.Get(), pre.ResultDataMaxSizeInBytes, &tlasBuf,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, L"TLAS");
        AllocateUAVBuffer(device.Get(), std::max<UINT64>(pre.ScratchDataSizeInBytes, 256ull),
            &tlasScratchBuf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"TLAS scratch");

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd = {};
        bd.Inputs                           = tlasIn;
        bd.DestAccelerationStructureData    = tlasBuf->GetGPUVirtualAddress();
        bd.ScratchAccelerationStructureData = tlasScratchBuf->GetGPUVirtualAddress();
        cmdList->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);
        auto bar = CD3DX12_RESOURCE_BARRIER::UAV(tlasBuf.Get());
        cmdList->ResourceBarrier(1, &bar);
    }

    // ------------------------------------------------------------------
    void BuildRaytracingPipeline()
    {
        // Global root sig: t0=Scene SRV, u0=RT output UAV, b0=Scene CBV.
        {
            CD3DX12_DESCRIPTOR_RANGE uavRange;
            uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
            CD3DX12_ROOT_PARAMETER params[3];
            params[0].InitAsDescriptorTable(1, &uavRange);
            params[1].InitAsShaderResourceView(0);
            params[2].InitAsConstantBufferView(0);
            CD3DX12_ROOT_SIGNATURE_DESC d(_countof(params), params);
            ComPtr<ID3DBlob> blob, err;
            TF(D3D12SerializeRootSignature(&d, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err));
            TF(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                           IID_PPV_ARGS(&globalRS)));
        }
        // Empty local root sig (required to exist alongside global).
        {
            CD3DX12_ROOT_SIGNATURE_DESC d(0, nullptr, 0, nullptr,
                                          D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);
            ComPtr<ID3DBlob> blob, err;
            TF(D3D12SerializeRootSignature(&d, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err));
            TF(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                           IID_PPV_ARGS(&localRS)));
        }

        CD3DX12_STATE_OBJECT_DESC pipe{ D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE };
        pipe.CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>()->SetRootSignature(localRS.Get());

        ComPtr<IDxcBlob> shader = CompileShader(kRaytracingHLSL, sizeof(kRaytracingHLSL) - 1);
        D3D12_SHADER_BYTECODE bc = { shader->GetBufferPointer(), shader->GetBufferSize() };
        auto lib = pipe.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
        lib->SetDXILLibrary(&bc);
        lib->DefineExport(L"RayGen");
        lib->DefineExport(L"OpaqueHit");
        lib->DefineExport(L"Miss");

        auto hg = pipe.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
        hg->SetClosestHitShaderImport(L"OpaqueHit");
        hg->SetHitGroupExport(L"OpaqueHG");
        hg->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

        auto cfg = pipe.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
        cfg->Config(/*payload*/ 4 * sizeof(float), /*attribs*/ 2 * sizeof(float));
        pipe.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>()->SetRootSignature(globalRS.Get());

        auto pcfg = pipe.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT>();
        pcfg->Config(/*maxRecursion*/1, D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_CLUSTERED_GEOMETRY);

        TF(device->CreateStateObject(pipe, IID_PPV_ARGS(&rtPipeline)),
           L"CreateStateObject failed");

        // Shader tables: 1 record each (raygen / miss / hit-group).
        ComPtr<ID3D12StateObjectProperties> props;
        TF(rtPipeline->QueryInterface(IID_PPV_ARGS(&props)));
        const UINT idSize    = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
        const UINT recSize   = (idSize + D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT - 1)
                             & ~(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT - 1);
        auto make1 = [&](const wchar_t* exportName,
                         ComPtr<ID3D12Resource>& outRes, const wchar_t* tag)
        {
            std::vector<uint8_t> buf(recSize, 0);
            memcpy(buf.data(), props->GetShaderIdentifier(exportName), idSize);
            AllocateUploadBuffer(device.Get(), buf.data(), buf.size(), &outRes, tag);
        };
        make1(L"RayGen",    raygenST, L"raygen ST");
        make1(L"Miss",      missST,   L"miss ST");
        make1(L"OpaqueHG",  hitST,    L"hit ST");
    }

    // ------------------------------------------------------------------
    void Render()
    {
        TF(cmdAlloc[frameIndex]->Reset());
        TF(cmdList->Reset(cmdAlloc[frameIndex].Get(), nullptr));

        cmdList->SetComputeRootSignature(globalRS.Get());
        ID3D12DescriptorHeap* heaps[] = { uavHeap.Get() };
        cmdList->SetDescriptorHeaps(1, heaps);
        cmdList->SetComputeRootDescriptorTable(0, rtOutputUav);
        cmdList->SetComputeRootShaderResourceView(1, tlasBuf->GetGPUVirtualAddress());
        cmdList->SetComputeRootConstantBufferView(2, sceneCB->GetGPUVirtualAddress());
        cmdList->SetPipelineState1(rtPipeline.Get());

        D3D12_DISPATCH_RAYS_DESC drd = {};
        drd.RayGenerationShaderRecord.StartAddress = raygenST->GetGPUVirtualAddress();
        drd.RayGenerationShaderRecord.SizeInBytes  = raygenST->GetDesc().Width;
        drd.MissShaderTable.StartAddress           = missST->GetGPUVirtualAddress();
        drd.MissShaderTable.SizeInBytes            = missST->GetDesc().Width;
        drd.MissShaderTable.StrideInBytes          = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
        drd.HitGroupTable.StartAddress             = hitST->GetGPUVirtualAddress();
        drd.HitGroupTable.SizeInBytes              = hitST->GetDesc().Width;
        drd.HitGroupTable.StrideInBytes            = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
        drd.Width = kWidth; drd.Height = kHeight; drd.Depth = 1;
        cmdList->DispatchRays(&drd);

        // Copy raytracing output -> back buffer.
        D3D12_RESOURCE_BARRIER toCopy[2] = {
            CD3DX12_RESOURCE_BARRIER::Transition(rtOutput.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(backBuffers[frameIndex].Get(),
                D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST),
        };
        cmdList->ResourceBarrier(_countof(toCopy), toCopy);
        cmdList->CopyResource(backBuffers[frameIndex].Get(), rtOutput.Get());
        D3D12_RESOURCE_BARRIER toPresent[2] = {
            CD3DX12_RESOURCE_BARRIER::Transition(rtOutput.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            CD3DX12_RESOURCE_BARRIER::Transition(backBuffers[frameIndex].Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT),
        };
        cmdList->ResourceBarrier(_countof(toPresent), toPresent);

        TF(cmdList->Close());
        ID3D12CommandList* lists[] = { cmdList.Get() };
        queue->ExecuteCommandLists(1, lists);
        TF(swapChain->Present(1, 0));
        MoveToNextFrame();
    }

    // ------------------------------------------------------------------
    void MoveToNextFrame()
    {
        const UINT64 sig = nextFenceVal++;
        TF(queue->Signal(fence.Get(), sig));
        fenceVals[frameIndex] = sig;

        frameIndex = swapChain->GetCurrentBackBufferIndex();
        if (fence->GetCompletedValue() < fenceVals[frameIndex])
        {
            TF(fence->SetEventOnCompletion(fenceVals[frameIndex], fenceEvent));
            WaitForSingleObject(fenceEvent, INFINITE);
        }
    }

    void WaitForGpu()
    {
        const UINT64 sig = nextFenceVal++;
        TF(queue->Signal(fence.Get(), sig));
        if (fence->GetCompletedValue() < sig)
        {
            TF(fence->SetEventOnCompletion(sig, fenceEvent));
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        fenceVals[frameIndex] = sig;
    }

    void Shutdown()
    {
        if (sceneCB && sceneCBMapped) { sceneCB->Unmap(0, nullptr); sceneCBMapped = nullptr; }
        WaitForGpu();
        if (fenceEvent) { CloseHandle(fenceEvent); fenceEvent = nullptr; }
    }
};

// ============================================================================
//  Agility SDK loader hooks: the system d3d12.dll uses these exports to find
//  the matching D3D12Core.dll at runtime. The DLL is copied into bin\D3D12\
//  by ExperimentalD3D12.props.
//
//  The post-NVIDIA-caps-regression experimental D3D12Core loads through the
//  release-SDK loader path, not the preview one, so we export the version as
//  D3D12SDKVersion=721 rather than D3D12_PREVIEW_SDK_VERSION. (Today these
//  happen to be the same integer but route differently.) Swap back to
//  D3D12_SDK_VERSION when a real Agility SDK NuGet shipping DXR2 ships.
// ============================================================================
extern "C" { __declspec(dllexport) extern const UINT  D3D12SDKVersion = 721; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath    = ".\\D3D12\\"; }

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, LPWSTR, int nShow)
{
    int    argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);

    try
    {
        App app;
        app.ParseCommandLine(argc, argv);
        LocalFree(argv);

        const wchar_t* title = (app.mode == App::VertexMode::Compressed1)
            ? L"DXR2 COMPRESSED1 repro - 1 face visible (BUG)"
            : L"DXR2 COMPRESSED1 repro - full cube (--vertex-format float)";
        CreateAppWindow(hInst, title);

        app.Init();
        ShowWindow(g_hwnd, nShow);

        MSG msg = {};
        while (msg.message != WM_QUIT)
        {
            if (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            else
            {
                app.Render();
            }
        }
        app.Shutdown();
        return (int)msg.wParam;
    }
    catch (const std::exception& e)
    {
        Log(L"FATAL: %hs\n", e.what());
        return EXIT_FAILURE;
    }
}
