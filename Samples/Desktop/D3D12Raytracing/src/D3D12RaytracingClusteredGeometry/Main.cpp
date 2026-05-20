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

// ============================================================================
//  Log + error helpers
// ============================================================================

// Mirror to %TEMP%\compressed1_repro.log so unattended runs leave a trace.
// SAMPLE_LOG env var overrides the path (matches the parent sample's
// convention so existing tooling still works).
static std::wstring GetLogPath()
{
    WCHAR env[MAX_PATH] = {};
    DWORD n = GetEnvironmentVariableW(L"SAMPLE_LOG", env, MAX_PATH);
    if (n > 0 && n < MAX_PATH) return env;
    WCHAR tmp[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, tmp);
    std::wstring p = tmp;
    if (!p.empty() && p.back() != L'\\') p += L'\\';
    p += L"compressed1_repro.log";
    return p;
}

static void Log(const wchar_t* fmt, ...)
{
    wchar_t buf[2048];
    va_list a; va_start(a, fmt);
    _vsnwprintf_s(buf, _TRUNCATE, fmt, a);
    va_end(a);
    OutputDebugStringW(buf);
    // Append to log file (truncated on first call from InitLog()).
    FILE* f = nullptr;
    if (_wfopen_s(&f, GetLogPath().c_str(), L"ab") == 0 && f)
    {
        fputws(buf, f);
        fclose(f);
    }
}

static void InitLog()
{
    FILE* f = nullptr;
    if (_wfopen_s(&f, GetLogPath().c_str(), L"wb") == 0 && f)
    {
        unsigned char bom[2] = { 0xFF, 0xFE };   // UTF-16 LE BOM
        fwrite(bom, 1, 2, f);
        fclose(f);
    }
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
//  Basic geometry types (shared between the encoder and the cube generator).
// ============================================================================
struct float3 { float x, y, z; };

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
                                 int forcedBiasedExponent = -1)    {
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
    std::vector<float3> positions;  // 4 corners
};

// Per-face shared index buffer: 2 tris formed by quad corners (0,1,2,3) laid
// out as a row-major 2x2 grid, with CCW-from-outside winding.
static constexpr uint16_t kFaceIndices[6] = { 0, 3, 2, 0, 1, 3 };

// Generate a unit cube (halfExtent h) as 6 single-cluster faces. Each face's
// CCW-from-outside winding matches the index pattern above (the axisU x
// axisV cross product is the outward normal).
static std::vector<Face> BuildCubeFaces(float h)
{
    struct Basis { float3 origin, axisU, axisV; };
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

// ============================================================================
//  Shader source (runtime-compiled via dxcompiler.dll, target lib_6_10).
//
//  Camera is fully static (eye looking at origin from in front + slightly
//  above, 60-deg vertical FOV, 16:9 aspect) so the basis vectors are
//  hardcoded as HLSL constants - no scene constant buffer needed.
// ============================================================================
static const char kRaytracingHLSL[] = R"HLSL(
RaytracingAccelerationStructure  Scene  : register(t0);
RWTexture2D<float4>              Output : register(u0);

struct [raypayload] Payload {
    float4 color : write(caller, closesthit, miss) : read(caller);
};
typedef BuiltInTriangleIntersectionAttributes Attribs;

[shader("raygeneration")]
void RayGen()
{
    // Hardcoded LH-view basis: eye looks at the origin from (1.5, 1.1, -1.5).
    // (Constant-folded by the HLSL compiler.)
    static const float3 eye    = float3(1.5, 1.1, -1.5);
    static const float3 fwd    = normalize(float3(-1.5, -1.1, 1.5));    // at - eye
    static const float3 right  = normalize(cross(float3(0, 1, 0), fwd));
    static const float3 up     = cross(fwd, right);
    static const float  aspect = 1280.0 / 720.0;
    static const float  tanHF  = 0.57735026919;                          // tan(30 deg)

    const uint2  px  = DispatchRaysIndex().xy;
    const uint2  dim = DispatchRaysDimensions().xy;
    const float2 ndc = (float2(px) + 0.5) / float2(dim) * 2.0 - 1.0;

    RayDesc r;
    r.Origin    = eye;
    r.Direction = normalize(fwd
                          + right * ndc.x * aspect * tanHF
                          -    up * ndc.y *          tanHF);   // -ndc.y: screen-down = world-down
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
    // +1 biases inside the hash to avoid the all-zeros input producing
    // a black triangle for cluster 0, primitive 0, instance 0.
    const uint cid  = ClusterID()      + 1;
    const uint pid  = PrimitiveIndex() + 1;
    const uint inst = InstanceIndex()  + 1;
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

    // --adapter warp|hw    force a specific adapter (default: highest-perf HW)
    enum class AdapterKind { Default, Warp, Hw };
    AdapterKind adapterKind = AdapterKind::Default;

    // --diag    dump encoded COMPRESSED1 headers / a few bitstream bytes per
    //           cluster + CPU-side decode roundtrip + post-build readback of
    //           the CLAS address array
    bool diag = false;

    // ---- experimental "fix" toggles (for bisecting differences vs the
    //      passing d3d12conf BUILD_CLAS_FROM_TRIANGLES + COMPRESSED1 test) ----
    //
    //   --fix-split        CPU sync (split CLAS + BLAS into 2 cmd-list submits
    //                      with WaitForGpu between)
    //   --fix-reupload     readback the CLAS address array and re-upload via a
    //                      fresh CPU buffer (implies --fix-split)
    //   --fix-explicit     EXPLICIT_DESTINATIONS for CLAS build (CPU-known
    //                      addresses; mirrors the d3d12conf-test option)
    //   --fix-separate-vb  give every cluster its own VB+IB resource (no
    //                      concatenation, no offset math)
    bool fixSplit      = false;
    bool fixReupload   = false;
    bool fixExplicit   = false;
    bool fixSeparateVB = false;
    bool fixBlasImplicit = false;    // BUILD_BLAS_FROM_CLAS mode (default EXPLICIT)
    bool wa1BlasPerClas  = false;    // workaround: 1 BLAS per CLAS, all in TLAS
    bool wa1ClasPerBuild = false;    // workaround: 1 CLAS per BUILD_CLAS_FROM_TRIANGLES call
                                     //   (test whether multi-arg CLAS build only writes the first arg)

    // ---- bug-pinpointing toggles ----
    //   --reverse-faces    emit cube faces in reverse order (test whether the
    //                      "1 visible face" bug is "cluster 0 always wins" or
    //                      "the +X face always wins regardless of cluster idx")
    //   --rotate-faces N   rotate cube face emit order by N positions
    bool reverseFaces = false;
    int  rotateFaces  = 0;

    // ---- device / swapchain ----
    // (Flip-model swap chain requires BufferCount >= 2; we treat both buffers
    //  uniformly via swapChain->GetCurrentBackBufferIndex() inline at render
    //  time, with a full WaitForGpu after every Present.  Single allocator,
    //  no per-frame pipelining - keeps the plumbing trivial.)
    static constexpr UINT kBackBufferCount = 2;
    ComPtr<IDXGIFactory4>                 factory;
    ComPtr<ID3D12Device5>                 device;
    ComPtr<ID3D12DeviceRaytracing2>       dxr2Device;
    ComPtr<ID3D12CommandQueue>            queue;
    ComPtr<IDXGISwapChain3>               swapChain;
    ComPtr<ID3D12Resource>                backBuffers[kBackBufferCount];
    ComPtr<ID3D12CommandAllocator>        cmdAlloc;
    ComPtr<ID3D12GraphicsCommandList4>    cmdList;
    ComPtr<ID3D12CommandListRaytracing2>  dxr2CmdList;
    ComPtr<ID3D12Fence>                   fence;
    UINT64                                nextFenceVal = 1;
    HANDLE                                fenceEvent = nullptr;

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
    // Per-cluster VBs when --fix-separate-vb is on (otherwise unused; all
    // clusters share clusterInputBuf with offsets).
    std::vector<ComPtr<ID3D12Resource>>   clusterVBsSep;
    std::vector<ComPtr<ID3D12Resource>>   clusterIBsSep;
    // --fix-reupload only: readback heap + fresh CPU-uploaded copy of the
    // CLAS address array (so BUILD_BLAS_FROM_CLAS reads CPU-uploaded data
    // rather than the GPU-written UAV from the prior CLAS build).
    ComPtr<ID3D12Resource>                clasAddrReadback;
    ComPtr<ID3D12Resource>                clasAddrFresh;
    // --fix-explicit only: CPU-allocated CLAS storage slots whose addresses
    // are fed both as CLAS dests and as the BLAS-from-CLAS address array.
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> explicitClasVAs;
    ComPtr<ID3D12Resource>                explicitDestAddrUpload;
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

    // ------------------------------------------------------------------
    void ParseCommandLine(int argc, WCHAR** argv)
    {
        for (int i = 1; i < argc; ++i)
        {
            if (_wcsicmp(argv[i], L"--vertex-format") == 0 && i + 1 < argc)
            {
                if      (_wcsicmp(argv[i+1], L"float")      == 0) mode = VertexMode::Float32_3;
                else if (_wcsicmp(argv[i+1], L"compressed") == 0) mode = VertexMode::Compressed1;
                ++i;
            }
            else if (_wcsicmp(argv[i], L"--adapter") == 0 && i + 1 < argc)
            {
                if      (_wcsicmp(argv[i+1], L"warp") == 0) adapterKind = AdapterKind::Warp;
                else if (_wcsicmp(argv[i+1], L"hw")   == 0) adapterKind = AdapterKind::Hw;
                ++i;
            }
            else if (_wcsicmp(argv[i], L"--diag")             == 0) diag             = true;
            else if (_wcsicmp(argv[i], L"--fix-split")        == 0) fixSplit         = true;
            else if (_wcsicmp(argv[i], L"--fix-reupload")     == 0) { fixSplit = true; fixReupload = true; }
            else if (_wcsicmp(argv[i], L"--fix-explicit")     == 0) fixExplicit      = true;
            else if (_wcsicmp(argv[i], L"--fix-separate-vb")  == 0) fixSeparateVB    = true;
            else if (_wcsicmp(argv[i], L"--reverse-faces")    == 0) reverseFaces     = true;
            else if (_wcsicmp(argv[i], L"--rotate-faces") == 0 && i + 1 < argc) { rotateFaces = _wtoi(argv[i+1]); ++i; }
            else if (_wcsicmp(argv[i], L"--fix-blas-implicit") == 0) fixBlasImplicit = true;
            else if (_wcsicmp(argv[i], L"--wa-1blas-per-clas") == 0) { wa1BlasPerClas = true; fixSplit = true; }
            else if (_wcsicmp(argv[i], L"--wa-1clas-per-build") == 0) wa1ClasPerBuild = true;
        }
    }

    // ------------------------------------------------------------------
    void Init()
    {
        InitLog();
        Log(L"=== compressed1 min repro ===\n");
        Log(L"  mode            = %s\n",
            mode == VertexMode::Compressed1 ? L"COMPRESSED1 (default - BUG)" : L"FLOAT32_3 (correct)");
        Log(L"  adapter         = %s\n",
            adapterKind == AdapterKind::Warp ? L"WARP (forced)" :
            adapterKind == AdapterKind::Hw   ? L"HW (forced)"   : L"HW (default, fallback to WARP)");
        Log(L"  fix-split       = %d\n", (int)fixSplit);
        Log(L"  fix-reupload    = %d\n", (int)fixReupload);
        Log(L"  fix-explicit    = %d\n", (int)fixExplicit);
        Log(L"  fix-separate-vb = %d\n", (int)fixSeparateVB);
        Log(L"  diag            = %d\n", (int)diag);

        // Experimental features must be enabled BEFORE the first
        // D3D12CreateDevice (otherwise this returns DXGI_ERROR_SDK_COMPONENT_MISSING).
        UUID feats[] = { D3D12ExperimentalShaderModels, D3D12RaytracingExperiment };
        TF(D3D12EnableExperimentalFeatures(_countof(feats), feats, nullptr, nullptr),
           L"D3D12EnableExperimentalFeatures failed (Windows Developer Mode disabled?)");

        InitDevice();
        InitSwapChain();
        InitFenceAndCmdList();
        InitDescriptorHeapAndUavOutput();
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

        // Adapter pick:
        //   --adapter warp -> EnumWarpAdapter (force the experimental WARP)
        //   --adapter hw   -> first hardware adapter that supports D3D12 (no
        //                     HIGH_PERFORMANCE preference, just first hit)
        //   default        -> highest-perf hardware adapter, falling back to
        //                     WARP if none accepts D3D12CreateDevice
        ComPtr<IDXGIAdapter1> adapter;
        if (adapterKind == AdapterKind::Warp)
        {
            TF(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)),
               L"EnumWarpAdapter failed");
        }
        else
        {
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
            if (!adapter)
                TF(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)),
                   L"no HW adapter and EnumWarpAdapter failed");
        }

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
        scd.BufferCount = kBackBufferCount;
        scd.SwapEffect  = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        ComPtr<IDXGISwapChain1> sc1;
        TF(factory->CreateSwapChainForHwnd(queue.Get(), g_hwnd, &scd, nullptr, nullptr, &sc1));
        TF(sc1.As(&swapChain));
        factory->MakeWindowAssociation(g_hwnd, DXGI_MWA_NO_ALT_ENTER);
        for (UINT i = 0; i < kBackBufferCount; ++i)
            TF(swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])));
    }

    // ------------------------------------------------------------------
    void InitFenceAndCmdList()
    {
        TF(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          IID_PPV_ARGS(&cmdAlloc)));
        TF(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                     cmdAlloc.Get(), nullptr,
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
    void BuildScene()
    {
        cubeFaces = BuildCubeFaces(0.45f);
        // Optional reordering for bug-pinpointing experiments. Lets us tell
        // whether the "1 cube face visible" pattern is "cluster index 0 always
        // wins" or "the +X face always wins regardless of cluster index".
        if (rotateFaces)
        {
            const int n = (int)cubeFaces.size();
            const int r = ((rotateFaces % n) + n) % n;
            std::rotate(cubeFaces.begin(), cubeFaces.begin() + r, cubeFaces.end());
            Log(L"[scene] rotated face emit order by %d (cluster 0 = face index %d in original order)\n",
                r, r);
        }
        if (reverseFaces)
        {
            std::reverse(cubeFaces.begin(), cubeFaces.end());
            Log(L"[scene] reversed face emit order (cluster 0 = original -Z face)\n");
        }
        clusterCount = (UINT)cubeFaces.size();

        // Pick a scene-wide COMPRESSED1 exponent so cluster-edge vertices
        // quantize identically (watertight at quantization).  Geometry is a
        // fixed unit cube so the max axis-extent is known a priori.
        const float maxExtent    = 0.9f;    // = 2 * halfExtent (BuildCubeFaces(0.45f))
        const float minUnit      = maxExtent / float((1ull << compressedBits) - 1);
        const int   sharedExp    = std::clamp((int)std::ceil(std::log2(minUnit)) + 127, 1, 232);

        encoded.clear();
        encoded.reserve(cubeFaces.size());
        for (const auto& f : cubeFaces)
            encoded.push_back(Compressed1::Encode(f.positions, (int)compressedBits, sharedExp));

        Log(L"[scene] cube: %u clusters, %u tris.  vertex format: %s\n",
            clusterCount, clusterCount * 2,
            mode == VertexMode::Compressed1 ? L"COMPRESSED1 (BUG repro - default)"
                                            : L"FLOAT32_3 (correct render)");

        if (diag)
        {
            // Per-cluster COMPRESSED1 header + decode roundtrip vs original positions.
            // If the encoder is broken, max-error will be much larger than 1 quantization
            // step (~ scale = 2^(sharedExp-127)).
            const float scale = (float)std::ldexp(1.0, sharedExp - 127);
            Log(L"[diag] shared exponent = %d (biased)  scale = %.10g\n", sharedExp, scale);
            for (UINT i = 0; i < clusterCount; ++i)
            {
                const auto& h = encoded[i].header;
                const int e    = (int) (h.field0 & 0xFF);
                const int aX   = (int) ((int32_t)h.field0 >> 8);     // sign-extend top 24
                const int aY   = (int) ((int32_t)h.field1 >> 8);
                const int aZ   = (int) ((int32_t)h.field2 >> 8);
                const int xB   = (int) ((h.field1 & 0xF) + 1);
                const int yB   = (int) (((h.field1 >> 4) & 0xF) + 1);
                const int zB   = (int) ((h.field2 & 0xF) + 1);
                Log(L"[diag]   cluster %u  exp=%d  anchor=(%d,%d,%d)  bits=(%d,%d,%d)"
                    L"  bitstream=%zu B  total=%zu B\n",
                    i, e, aX, aY, aZ, xB, yB, zB,
                    encoded[i].bitstream.size(), encoded[i].TotalBytes());

                // First 4 bytes of bitstream as hex (each cluster only has 2-3 bytes since
                // 4 verts * (xB+yB+zB) bits / 8 ~= 6-8 bytes).
                const auto& bs = encoded[i].bitstream;
                wchar_t hex[64] = {};
                int n = (int)std::min<size_t>(bs.size(), 12);
                int off = 0;
                for (int k = 0; k < n; ++k)
                    off += swprintf_s(hex + off, _countof(hex) - off, L"%02X ", bs[k]);
                Log(L"[diag]              bitstream[0..%d] = %s\n", n, hex);

                // Decode roundtrip
                float maxErr = 0;
                const uint32_t nx = xB, ny = yB, nz = zB;
                uint64_t bitOff = 0;
                auto readBits = [&](uint32_t n) -> uint32_t {
                    uint32_t v = 0, r = 0;
                    while (r < n) {
                        uint32_t bytePos = (uint32_t)(bitOff / 8);
                        uint32_t bitInB  = (uint32_t)(bitOff & 7);
                        uint32_t take    = std::min<uint32_t>(8 - bitInB, n - r);
                        uint32_t mask    = (1u << take) - 1u;
                        v |= ((bs[bytePos] >> bitInB) & mask) << r;
                        r      += take;
                        bitOff += take;
                    }
                    return v;
                };
                for (size_t v = 0; v < cubeFaces[i].positions.size(); ++v)
                {
                    uint32_t dx = readBits(nx), dy = readBits(ny), dz = readBits(nz);
                    float px = (float)((double)(aX + (int32_t)dx) * (double)scale);
                    float py = (float)((double)(aY + (int32_t)dy) * (double)scale);
                    float pz = (float)((double)(aZ + (int32_t)dz) * (double)scale);
                    const auto& orig = cubeFaces[i].positions[v];
                    float err = std::max({ std::fabs(px - orig.x),
                                           std::fabs(py - orig.y),
                                           std::fabs(pz - orig.z) });
                    maxErr = std::max(maxErr, err);
                    Log(L"[diag]              v%zu: orig=(%.4f,%.4f,%.4f)  decoded=(%.4f,%.4f,%.4f)  err=%.6f\n",
                        v, orig.x, orig.y, orig.z, px, py, pz, err);
                }
                Log(L"[diag]              max-err = %.6f  (quantization scale = %.6f)\n",
                    maxErr, scale);
            }
        }
    }

    // ------------------------------------------------------------------
    //  AS build: upload VBs + IBs, BUILD_CLAS_FROM_TRIANGLES (implicit dest),
    //  BUILD_BLAS_FROM_CLAS (explicit dest, 1 BLAS containing all 6 CLAS),
    //  classic TLAS with 1 instance.
    // ------------------------------------------------------------------
    void BuildAccelerationStructures()
    {
        TF(cmdAlloc->Reset());
        TF(cmdList->Reset(cmdAlloc.Get(), nullptr));

        UploadClusterInputs();
        BuildClas();

        if (fixSplit)
        {
            // CPU-side sync between CLAS build and BLAS-from-CLAS, mirroring
            // what the d3d12conf test does (it FlushAndFinish's after CLAS
            // build before issuing BLAS-from-CLAS). Test whether the bug is in
            // the GPU-pipelined CLAS-addr -> BLAS-from-CLAS path.
            TF(cmdList->Close());
            ID3D12CommandList* lists1[] = { cmdList.Get() };
            queue->ExecuteCommandLists(1, lists1);
            WaitForGpu();

            // Always do the readback (and re-upload if fixReupload is set) so
            // the diag log + wa-1blas-per-clas variants have the data they need.
            ReadbackAndOptionallyReuploadClasAddrs();

            TF(cmdAlloc->Reset());
            TF(cmdList->Reset(cmdAlloc.Get(), nullptr));
        }

        BuildBlas();
        BuildTlas();

        TF(cmdList->Close());
        ID3D12CommandList* lists[] = { cmdList.Get() };
        queue->ExecuteCommandLists(1, lists);
        WaitForGpu();
    }

    // Helpers used by --fix-reupload / --diag only.
    void ReadbackAndOptionallyReuploadClasAddrs()
    {
        // 1. CopyResource from the GPU-written address-array UAV to a
        //    READBACK heap, then map to read CPU-side.
        TF(cmdAlloc->Reset());
        TF(cmdList->Reset(cmdAlloc.Get(), nullptr));

        if (!clasAddrReadback)
        {
            auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
            auto desc = CD3DX12_RESOURCE_DESC::Buffer(
                (UINT64)clusterCount * sizeof(D3D12_GPU_VIRTUAL_ADDRESS));
            TF(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&clasAddrReadback)));
            clasAddrReadback->SetName(L"CLAS addr readback");
        }

        auto barTo = CD3DX12_RESOURCE_BARRIER::Transition(clasAddrArray.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &barTo);
        cmdList->CopyResource(clasAddrReadback.Get(), clasAddrArray.Get());
        auto barFrom = CD3DX12_RESOURCE_BARRIER::Transition(clasAddrArray.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &barFrom);

        TF(cmdList->Close());
        ID3D12CommandList* lists[] = { cmdList.Get() };
        queue->ExecuteCommandLists(1, lists);
        WaitForGpu();

        std::vector<D3D12_GPU_VIRTUAL_ADDRESS> addrs(clusterCount);
        void* mapped = nullptr;
        TF(clasAddrReadback->Map(0, nullptr, &mapped));
        memcpy(addrs.data(), mapped, addrs.size() * sizeof(addrs[0]));
        D3D12_RANGE noWrite = {};
        clasAddrReadback->Unmap(0, &noWrite);

        Log(L"[diag] CLAS addresses (from %s):\n",
            fixExplicit ? L"explicit-dest passthrough" : L"BUILD_CLAS_FROM_TRIANGLES UAV");
        UINT zeroCount = 0;
        for (UINT i = 0; i < clusterCount; ++i)
        {
            Log(L"[diag]   cluster %u  GVA = 0x%016llX %s\n",
                i, (unsigned long long)addrs[i],
                addrs[i] == 0 ? L"  (ZERO!)" : L"");
            if (addrs[i] == 0) zeroCount++;
        }
        if (zeroCount > 0)
            Log(L"[diag]   *** %u of %u CLAS addresses are ZERO ***\n",
                zeroCount, clusterCount);
        else
            Log(L"[diag]   all %u CLAS addresses present\n", clusterCount);

        // 2. If --fix-reupload, materialize a fresh CPU-uploaded buffer that
        //    BUILD_BLAS_FROM_CLAS will read instead of the UAV.
        if (fixReupload)
        {
            AllocateUploadBuffer(device.Get(), addrs.data(),
                                 addrs.size() * sizeof(addrs[0]),
                                 &clasAddrFresh, L"CLAS addr re-uploaded");
            Log(L"[diag]   re-uploaded %u addresses to fresh CPU buffer\n", clusterCount);
        }
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

        std::vector<D3D12_GPU_VIRTUAL_ADDRESS> vbGVA(clusterCount), ibGVA(clusterCount);

        if (fixSeparateVB)
        {
            // --fix-separate-vb: each cluster owns its own VB and IB resource,
            // matching the d3d12conf test's per-cluster MakeBufferAndInit pattern.
            // This removes any "VB GVA points into a shared buffer at an offset"
            // as a variable from the bug.
            clusterVBsSep.assign(clusterCount, {});
            clusterIBsSep.assign(clusterCount, {});
            auto uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
            for (UINT i = 0; i < clusterCount; ++i)
            {
                const size_t vbSize = useFloat
                    ? cubeFaces[i].positions.size() * sizeof(float3)
                    : encoded[i].TotalBytes();
                auto vbDesc = CD3DX12_RESOURCE_DESC::Buffer(vbSize);
                TF(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &vbDesc,
                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&clusterVBsSep[i])));
                uint8_t* vbMap = nullptr;
                TF(clusterVBsSep[i]->Map(0, nullptr, reinterpret_cast<void**>(&vbMap)));
                if (useFloat) {
                    memcpy(vbMap, cubeFaces[i].positions.data(), vbSize);
                } else {
                    memcpy(vbMap, &encoded[i].header, sizeof(encoded[i].header));
                    if (!encoded[i].bitstream.empty())
                        memcpy(vbMap + sizeof(encoded[i].header),
                               encoded[i].bitstream.data(), encoded[i].bitstream.size());
                }
                clusterVBsSep[i]->Unmap(0, nullptr);
                vbGVA[i] = clusterVBsSep[i]->GetGPUVirtualAddress();

                auto ibDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(kFaceIndices));
                TF(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &ibDesc,
                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&clusterIBsSep[i])));
                uint8_t* ibMap = nullptr;
                TF(clusterIBsSep[i]->Map(0, nullptr, reinterpret_cast<void**>(&ibMap)));
                memcpy(ibMap, kFaceIndices, sizeof(kFaceIndices));
                clusterIBsSep[i]->Unmap(0, nullptr);
                ibGVA[i] = clusterIBsSep[i]->GetGPUVirtualAddress();
            }
            Log(L"[diag] --fix-separate-vb: %u per-cluster VBs + %u per-cluster IBs\n",
                clusterCount, clusterCount);
        }
        else
        {
            // Default: one shared buffer with concatenated VB+IB per cluster.
            size_t cursor = 0;
            for (UINT i = 0; i < clusterCount; ++i)
            {
                cursor = align(cursor, kAlign);
                slots[i].vbOff  = cursor;
                slots[i].vbSize = useFloat
                    ? cubeFaces[i].positions.size() * sizeof(float3)
                    : encoded[i].TotalBytes();
                cursor += slots[i].vbSize;
                cursor = align(cursor, kAlign);
                slots[i].ibOff  = cursor;
                slots[i].ibSize = sizeof(kFaceIndices);
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
                memcpy(mapped + slots[i].ibOff, kFaceIndices, sizeof(kFaceIndices));
                vbGVA[i] = baseGVA + slots[i].vbOff;
                ibGVA[i] = baseGVA + slots[i].ibOff;
            }
            clusterInputBuf->Unmap(0, nullptr);
        }

        // CPU-fill BUILD_CLAS_FROM_TRIANGLES_ARGS[N] (consumed by
        // ExecuteIndirectRTASOperations below).
        std::vector<D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS> args(clusterCount);
        for (UINT i = 0; i < clusterCount; ++i)
        {
            auto& a = args[i];
            a = {};
            a.ClusterID                 = i;
            a.TriangleCount             = 2;
            a.VertexCount               = 4;
            a.BaseGeometryIndexAndFlags = (UINT)D3D12_RTAS_CLUSTERED_GEOMETRY_FLAG_OPAQUE;
            a.VertexBufferStride        = useFloat ? (UINT16)sizeof(float3) : 0;
            a.IndexBufferStride         = sizeof(uint16_t);
            a.PositionTruncateBitCount  = 0;
            a.VertexBuffer              = vbGVA[i];
            a.IndexBuffer               = ibGVA[i];
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
        clas.Mode = fixExplicit ? D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS
                                : D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;

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
        batched.BatchScratchData       = clasScratchBuf->GetGPUVirtualAddress();
        batched.ResultAddressArray     = { clasAddrArray->GetGPUVirtualAddress(),
                                           sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
        batched.IndirectArgumentArray  = { clasArgsBuf->GetGPUVirtualAddress(),
                                           sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS) };

        if (fixExplicit)
        {
            // CPU-side slots within the CLAS result buffer (each CLAS at
            // i * pre.ResultDataMaxSizeInBytes, just like the d3d12conf test
            // at line 8492's CBLAS allocation).
            const UINT64 perClasSize = pre.ResultDataMaxSizeInBytes;
            explicitClasVAs.assign(N, 0);
            for (UINT i = 0; i < N; ++i)
                explicitClasVAs[i] = clasResultBuf->GetGPUVirtualAddress() + (UINT64)i * perClasSize;
            AllocateUploadBuffer(device.Get(), explicitClasVAs.data(),
                                 explicitClasVAs.size() * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                                 &explicitDestAddrUpload, L"CLAS explicit dest addrs");
            // In EXPLICIT mode BatchResultData is 0; the per-arg dest is the
            // pointed-to address array (see d3d12.h docs on EXPLICIT_DESTINATIONS).
            batched.BatchResultData    = 0;
            batched.ResultAddressArray = { explicitDestAddrUpload->GetGPUVirtualAddress(),
                                           sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
            Log(L"[diag] EXPLICIT_DESTINATIONS: per-CLAS size = %llu, total = %llu B\n",
                (unsigned long long)perClasSize, (unsigned long long)(perClasSize * N));
        }
        else
        {
            batched.BatchResultData    = clasResultBuf->GetGPUVirtualAddress();
        }

        D3D12_RTAS_OPERATION_DESC op = {};
        op.Inputs                = inputs;
        op.pBatchedOperationData = &batched;

        if (wa1ClasPerBuild)
        {
            // Issue clusterCount separate ExecuteIndirectRTASOperations calls,
            // each with IndirectArgumentArraySize=1 covering one arg slot. If
            // the bug is "multi-arg CLAS build only writes the first arg's
            // CLAS for COMPRESSED1", this should fix it: every call sees only
            // one arg, which is "the first" by definition.
            const D3D12_GPU_VIRTUAL_ADDRESS clasArgsBase = clasArgsBuf->GetGPUVirtualAddress();
            const UINT argStride = (UINT)sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS);
            const D3D12_GPU_VIRTUAL_ADDRESS addrArrayBase = clasAddrArray->GetGPUVirtualAddress();
            // Force IndirectArgumentArraySize=1 by overriding the per-call args
            // pointer (point each call at a different arg slot) and using
            // batched.IndirectArgumentArraySize = 1.
            batched.IndirectArgumentArraySize = 1;
            for (UINT i = 0; i < clusterCount; ++i)
            {
                batched.IndirectArgumentArray.StartAddress = clasArgsBase + (UINT64)i * argStride;
                batched.IndirectArgumentArray.StrideInBytes = argStride;
                // The driver writes one CLAS address per call; offset within
                // the global addr array so we still get a contiguous result
                // array we can feed to BUILD_BLAS_FROM_CLAS.
                batched.ResultAddressArray.StartAddress = addrArrayBase + (UINT64)i * sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
                batched.ResultAddressArray.StrideInBytes = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
                dxr2CmdList->ExecuteIndirectRTASOperations(1, &op,
                    D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
                // UAV barrier on the result/scratch/addr array between calls so
                // the next build sees this one's writes complete.
                D3D12_RESOURCE_BARRIER perBars[] = {
                    CD3DX12_RESOURCE_BARRIER::UAV(clasResultBuf.Get()),
                    CD3DX12_RESOURCE_BARRIER::UAV(clasScratchBuf.Get()),
                    CD3DX12_RESOURCE_BARRIER::UAV(clasAddrArray.Get()),
                };
                cmdList->ResourceBarrier(_countof(perBars), perBars);
            }
            Log(L"[wa-1clas-per-build] issued %u single-arg BUILD_CLAS_FROM_TRIANGLES calls\n",
                clusterCount);
        }
        else
        {
            dxr2CmdList->ExecuteIndirectRTASOperations(1, &op,
                D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
        }

        D3D12_RESOURCE_BARRIER bars[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(clasResultBuf.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(clasAddrArray.Get()),
        };
        cmdList->ResourceBarrier(_countof(bars), bars);
    }

    // ------------------------------------------------------------------
    void BuildBlas()
    {
        if (wa1BlasPerClas)
        {
            BuildBlasOnePerClas();
            return;
        }

        D3D12_RTAS_CLAS_INPUTS_DESC blas = {};
        blas.Flags              = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;
        blas.MaxArgCount        = 1;
        blas.Mode               = fixBlasImplicit
                                  ? D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS
                                  : D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS;
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
        // --fix-reupload: read CPU-uploaded addresses (mirroring the d3d12conf
        // test pattern) instead of the GPU-written UAV from the prior CLAS build.
        arg.ClasAddressArray  = fixReupload && clasAddrFresh
            ? clasAddrFresh->GetGPUVirtualAddress()
            : clasAddrArray->GetGPUVirtualAddress();
        AllocateUploadBuffer(device.Get(), &arg, sizeof(arg),  &blasArgsBuf,    L"BLAS arg");
        AllocateUploadBuffer(device.Get(), &blasGPUVA, sizeof(blasGPUVA), &blasDestAddrBuf, L"BLAS dest");

        D3D12_RTAS_BATCHED_OPERATION_DATA batched = {};
        batched.BatchScratchData      = blasScratchBuf->GetGPUVirtualAddress();
        batched.IndirectArgumentArray = { blasArgsBuf->GetGPUVirtualAddress(), sizeof(arg) };
        if (fixBlasImplicit)
        {
            // IMPLICIT: a single result-buffer base, driver picks per-arg offsets,
            // writes them to ResultAddressArray.
            batched.BatchResultData    = blasStorage->GetGPUVirtualAddress();
            // Need an output addr-array UAV for the driver to write into.
            AllocateUAVBuffer(device.Get(), sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                              &blasDestAddrBuf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              L"BLAS dest addrs (implicit)");
            batched.ResultAddressArray = { blasDestAddrBuf->GetGPUVirtualAddress(),
                                           sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
        }
        else
        {
            // EXPLICIT: per-arg destination addresses are CPU-known and uploaded.
            batched.ResultAddressArray = { blasDestAddrBuf->GetGPUVirtualAddress(),
                                           sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
        }

        D3D12_RTAS_OPERATION_DESC op = {};
        op.Inputs                = inputs;
        op.pBatchedOperationData = &batched;
        dxr2CmdList->ExecuteIndirectRTASOperations(1, &op,
            D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

        auto bar = CD3DX12_RESOURCE_BARRIER::UAV(blasStorage.Get());
        cmdList->ResourceBarrier(1, &bar);
    }

    // Workaround mode: build clusterCount separate BLASes, each containing
    // exactly 1 CLAS. (Tests the hypothesis "NVIDIA BLAS-from-CLAS only
    // consumes the first ClasAddressArray entry for COMPRESSED1".)
    std::vector<ComPtr<ID3D12Resource>>          waBlasStores;
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS>       waBlasGVAs;

    void BuildBlasOnePerClas()
    {
        // Caller (BuildAccelerationStructures) has already done split + readback
        // (we force-set fixSplit when wa1BlasPerClas in ParseCommandLine).
        // cmdList is already open + empty here.

        // Re-read the addresses from clasAddrReadback (populated by the split-
        // mode CLAS readback).
        void* mapped = nullptr;
        TF(clasAddrReadback->Map(0, nullptr, &mapped));
        std::vector<D3D12_GPU_VIRTUAL_ADDRESS> clasGVAs(clusterCount);
        memcpy(clasGVAs.data(), mapped, clasGVAs.size() * sizeof(clasGVAs[0]));
        D3D12_RANGE noWrite = {};
        clasAddrReadback->Unmap(0, &noWrite);

        D3D12_RTAS_CLAS_INPUTS_DESC blas = {};
        blas.Flags              = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;
        blas.MaxArgCount        = 1;
        blas.Mode               = D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS;
        blas.MaxTotalClasCount  = 1;
        blas.MaxClasCountPerArg = 1;

        D3D12_RTAS_OPERATION_INPUTS inputs = {};
        inputs.Type      = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
        inputs.pClasDesc = &blas;

        D3D12_RTAS_OPERATION_PREBUILD_INFO pre = {};
        dxr2Device->GetRTASOperationPrebuildInfo(&inputs, &pre);

        AllocateUAVBuffer(device.Get(), std::max<UINT64>(pre.ScratchDataSizeInBytes, 256ull),
            &blasScratchBuf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"BLAS scratch (per-cluster)");

        waBlasStores.assign(clusterCount, {});
        waBlasGVAs.assign(clusterCount, 0);
        for (UINT i = 0; i < clusterCount; ++i)
        {
            AllocateUAVBuffer(device.Get(), pre.ResultDataMaxSizeInBytes, &waBlasStores[i],
                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, L"Per-cluster BLAS");
            waBlasGVAs[i] = waBlasStores[i]->GetGPUVirtualAddress();

            // Per-arg CLAS-address-array buffer (single CLAS).
            ComPtr<ID3D12Resource> clasAddrBuf;
            AllocateUploadBuffer(device.Get(), &clasGVAs[i], sizeof(clasGVAs[i]),
                                 &clasAddrBuf, L"single CLAS addr");
            // Per-arg BLAS-from-CLAS args.
            D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS arg = {};
            arg.ClasAddressCount  = 1;
            arg.ClasAddressStride = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
            arg.ClasAddressArray  = clasAddrBuf->GetGPUVirtualAddress();
            ComPtr<ID3D12Resource> argBuf;
            AllocateUploadBuffer(device.Get(), &arg, sizeof(arg), &argBuf, L"single BLAS arg");
            ComPtr<ID3D12Resource> destBuf;
            AllocateUploadBuffer(device.Get(), &waBlasGVAs[i], sizeof(waBlasGVAs[i]),
                                 &destBuf, L"single BLAS dest");

            D3D12_RTAS_BATCHED_OPERATION_DATA batched = {};
            batched.BatchScratchData      = blasScratchBuf->GetGPUVirtualAddress();
            batched.ResultAddressArray    = { destBuf->GetGPUVirtualAddress(), sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
            batched.IndirectArgumentArray = { argBuf->GetGPUVirtualAddress(), sizeof(arg) };

            D3D12_RTAS_OPERATION_DESC op = {};
            op.Inputs                = inputs;
            op.pBatchedOperationData = &batched;
            dxr2CmdList->ExecuteIndirectRTASOperations(1, &op,
                D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

            // Scratch is shared across all 6 per-cluster BLAS builds; need a
            // UAV barrier between them so the next iteration sees the prior
            // build complete + scratch free for reuse.
            D3D12_RESOURCE_BARRIER scratchBar = CD3DX12_RESOURCE_BARRIER::UAV(blasScratchBuf.Get());
            cmdList->ResourceBarrier(1, &scratchBar);

            // Keep the small upload buffers alive across the loop iteration
            // until execute+wait; need to add them to a holder.
            waUploadKeepalive.push_back(clasAddrBuf);
            waUploadKeepalive.push_back(argBuf);
            waUploadKeepalive.push_back(destBuf);
        }

        // UAV barrier on all per-cluster BLAS storage.
        std::vector<D3D12_RESOURCE_BARRIER> bars;
        for (auto& b : waBlasStores)
            bars.push_back(CD3DX12_RESOURCE_BARRIER::UAV(b.Get()));
        cmdList->ResourceBarrier((UINT)bars.size(), bars.data());

        TF(cmdList->Close());
        ID3D12CommandList* lists[] = { cmdList.Get() };
        queue->ExecuteCommandLists(1, lists);
        WaitForGpu();

        TF(cmdAlloc->Reset());
        TF(cmdList->Reset(cmdAlloc.Get(), nullptr));

        Log(L"[wa-1blas-per-clas] built %u single-CLAS BLASes\n", clusterCount);
    }

    std::vector<ComPtr<ID3D12Resource>> waUploadKeepalive;

    // ------------------------------------------------------------------
    void BuildTlas()
    {
        const UINT numInstances = wa1BlasPerClas ? clusterCount : 1u;
        std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances(numInstances);
        for (UINT i = 0; i < numInstances; ++i)
        {
            auto& inst = instances[i];
            inst = {};
            inst.Transform[0][0] = 1.0f;
            inst.Transform[1][1] = 1.0f;
            inst.Transform[2][2] = 1.0f;
            inst.InstanceID                          = i;
            inst.InstanceMask                        = 0xFF;
            inst.InstanceContributionToHitGroupIndex = 0;
            inst.Flags                               = D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
            inst.AccelerationStructure               = wa1BlasPerClas ? waBlasGVAs[i] : blasGPUVA;
        }
        AllocateUploadBuffer(device.Get(), instances.data(),
                             instances.size() * sizeof(instances[0]),
                             &tlasInstanceBuf, L"TLAS instance");

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasIn = {};
        tlasIn.Type          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlasIn.DescsLayout   = D3D12_ELEMENTS_LAYOUT_ARRAY;
        tlasIn.NumDescs      = numInstances;
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
        // Global root sig: u0 = RT output UAV (descriptor table), t0 = TLAS SRV.
        // (No CBV - camera is hardcoded in the shader.)
        {
            CD3DX12_DESCRIPTOR_RANGE uavRange;
            uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
            CD3DX12_ROOT_PARAMETER params[2];
            params[0].InitAsDescriptorTable(1, &uavRange);
            params[1].InitAsShaderResourceView(0);
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
        const UINT bbIdx = swapChain->GetCurrentBackBufferIndex();
        auto*      bb    = backBuffers[bbIdx].Get();

        TF(cmdAlloc->Reset());
        TF(cmdList->Reset(cmdAlloc.Get(), nullptr));

        cmdList->SetComputeRootSignature(globalRS.Get());
        ID3D12DescriptorHeap* heaps[] = { uavHeap.Get() };
        cmdList->SetDescriptorHeaps(1, heaps);
        cmdList->SetComputeRootDescriptorTable(0, rtOutputUav);
        cmdList->SetComputeRootShaderResourceView(1, tlasBuf->GetGPUVirtualAddress());
        cmdList->SetPipelineState1(rtPipeline.Get());

        D3D12_DISPATCH_RAYS_DESC drd = {};
        drd.RayGenerationShaderRecord = { raygenST->GetGPUVirtualAddress(),
                                          raygenST->GetDesc().Width };
        drd.MissShaderTable           = { missST->GetGPUVirtualAddress(),
                                          missST->GetDesc().Width,
                                          D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES };
        drd.HitGroupTable             = { hitST->GetGPUVirtualAddress(),
                                          hitST->GetDesc().Width,
                                          D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES };
        drd.Width = kWidth; drd.Height = kHeight; drd.Depth = 1;
        cmdList->DispatchRays(&drd);

        // Copy raytracing output -> back buffer.
        D3D12_RESOURCE_BARRIER toCopy[2] = {
            CD3DX12_RESOURCE_BARRIER::Transition(rtOutput.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(bb,
                D3D12_RESOURCE_STATE_PRESENT,          D3D12_RESOURCE_STATE_COPY_DEST),
        };
        cmdList->ResourceBarrier(2, toCopy);
        cmdList->CopyResource(bb, rtOutput.Get());
        D3D12_RESOURCE_BARRIER toPresent[2] = {
            CD3DX12_RESOURCE_BARRIER::Transition(rtOutput.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            CD3DX12_RESOURCE_BARRIER::Transition(bb,
                D3D12_RESOURCE_STATE_COPY_DEST,   D3D12_RESOURCE_STATE_PRESENT),
        };
        cmdList->ResourceBarrier(2, toPresent);

        TF(cmdList->Close());
        ID3D12CommandList* lists[] = { cmdList.Get() };
        queue->ExecuteCommandLists(1, lists);
        TF(swapChain->Present(1, 0));
        WaitForGpu();   // simple synchronous loop - no CPU/GPU pipelining
    }

    // ------------------------------------------------------------------
    void WaitForGpu()
    {
        const UINT64 sig = nextFenceVal++;
        TF(queue->Signal(fence.Get(), sig));
        if (fence->GetCompletedValue() < sig)
        {
            TF(fence->SetEventOnCompletion(sig, fenceEvent));
            WaitForSingleObject(fenceEvent, INFINITE);
        }
    }

    void Shutdown()
    {
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
