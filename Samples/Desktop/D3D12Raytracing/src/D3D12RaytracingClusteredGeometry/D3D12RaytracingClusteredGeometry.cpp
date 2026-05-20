//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include "stdafx.h"
#include "D3D12RaytracingClusteredGeometry.h"
#include "DirectXRaytracingHelper.h"
#include "CompiledShaders\\Raytracing.hlsl.h"
#include "CompiledShaders\\FillBlasFromClasArgs.hlsl.h"
#include "CompiledShaders\\FillClasFromTrianglesArgs.hlsl.h"
#include "CompiledShaders\\FillClusterTemplateArgs.hlsl.h"

#include <DirectXMath.h>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <set>

using namespace std;
using namespace DX;
using namespace DirectX;
using Microsoft::WRL::ComPtr;

// ==== Shader entry-point names (must match Raytracing.hlsl) ====
const wchar_t* D3D12RaytracingClusteredGeometry::c_raygenName            = L"RayGen";
const wchar_t* D3D12RaytracingClusteredGeometry::c_opaqueClosestHitName  = L"OpaqueHit";
const wchar_t* D3D12RaytracingClusteredGeometry::c_missName              = L"Miss";
const wchar_t* D3D12RaytracingClusteredGeometry::c_opaqueHitGroupName    = L"OpaqueHitGroup";

// =====================================================================================
// Construction + command-line + lifecycle
// =====================================================================================
D3D12RaytracingClusteredGeometry::D3D12RaytracingClusteredGeometry(UINT width, UINT height, std::wstring name)
    : DXSample(width, height, name)
{
    UpdateForSizeChange(width, height);
}

void D3D12RaytracingClusteredGeometry::ParseCommandLineArgs(_In_reads_(argc) WCHAR* argv[], int argc)
{
    // Minimal CLI for COMPRESSED1 repro: only --vertex-format remains.
    // The bug is visible interactively -- just run the exe and look.
    DXSample::ParseCommandLineArgs(argv, argc);
    for (int i = 1; i < argc; i++)
    {
        if (_wcsicmp(argv[i], L"--vertex-format") == 0 && i + 1 < argc)
        {
            if      (_wcsicmp(argv[i+1], L"float")      == 0) m_vertexMode = VertexMode::Float32_3;
            else if (_wcsicmp(argv[i+1], L"compressed") == 0) m_vertexMode = VertexMode::Compressed1;
            i += 1;
        }
    }
}

void D3D12RaytracingClusteredGeometry::OnInit()
{
    // ---- DXR2 experimental-features opt-in ----
    // The runtime requires D3D12RaytracingExperiment to be enabled BEFORE any
    // D3D12CreateDevice call in this process. Without it, ClustersAndPTLAS
    // GetRTASOperationPrebuildInfo() calls silently return zero sizes even when
    // the cap reports YES. (D3D12ExperimentalShaderModels is needed for the
    // SM 6.10 raygen/closesthit shaders.)
    {
        UUID experimentalFeatures[] = { D3D12ExperimentalShaderModels, D3D12RaytracingExperiment };
        HRESULT hrExp = D3D12EnableExperimentalFeatures(_countof(experimentalFeatures),
                                                        experimentalFeatures, nullptr, nullptr);
        SampleLog::LogF(L"OnInit: D3D12EnableExperimentalFeatures -> hr=0x%08X (%s)\n",
                        (unsigned)hrExp, SUCCEEDED(hrExp) ? L"OK" : L"FAIL");
        ThrowIfFailed(hrExp,
            L"D3D12EnableExperimentalFeatures failed. Is Windows Developer Mode enabled?\n");
    }

    SampleLog::Write(L"OnInit: creating DeviceResources\n");
    // 5th arg = device-resources options.  0 here means VSYNC ON
    // (Present(1, 0) inside DeviceResources::Present).  Tearing is visible at
    // low framerates without vsync -- this scene's clusters can dip to <60
    // fps under heavy reflection/refraction bounces, so we lock to refresh.
    // If you want VRR / unlocked framerate to measure perf, swap this to
    // DeviceResources::c_RequireTearingSupport (Present(0, ALLOW_TEARING)).
    m_deviceResources = std::make_unique<DeviceResources>(
        DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_UNKNOWN,
        FrameCount,
        D3D_FEATURE_LEVEL_11_0,
        /*options*/0,
        m_adapterIDoverride);
    m_deviceResources->RegisterDeviceNotify(this);
    m_deviceResources->SetWindow(Win32Application::GetHwnd(), m_width, m_height);
    m_deviceResources->InitializeDXGIAdapter();

    SampleLog::Write(L"OnInit: checking DXR support\n");
    {
        ComPtr<ID3D12Device> testDevice;
        HRESULT hrCreate = D3D12CreateDevice(m_deviceResources->GetAdapter(),
                                             D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&testDevice));
        SampleLog::LogF(L"  D3D12CreateDevice(adapter, FL11_0) -> hr=0x%08X (%s)\n",
                        (unsigned)hrCreate, SUCCEEDED(hrCreate) ? L"OK" : L"FAIL");
        ThrowIfFailed(hrCreate);
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5 = {};
        HRESULT hrFeat = testDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5));
        SampleLog::LogF(L"  CheckFeatureSupport(OPTIONS5) hr=0x%08X, RaytracingTier=0x%X\n",
                        (unsigned)hrFeat, (unsigned)opts5.RaytracingTier);
        ThrowIfFalse(SUCCEEDED(hrFeat) && opts5.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED,
            L"ERROR: DirectX Raytracing tier 1.0+ is required.\n\n");
    }

    m_deviceResources->CreateDeviceResources();
    m_deviceResources->CreateWindowSizeDependentResources();
    CreateDeviceDependentResources();
    SampleLog::Write(L"OnInit: complete\n");
}

void D3D12RaytracingClusteredGeometry::CreateDeviceDependentResources()
{
    auto device      = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();

    ThrowIfFailed(device->QueryInterface(IID_PPV_ARGS(&m_dxrDevice)),
        L"ERROR: ID3D12Device5 not available.\n");
    ThrowIfFailed(commandList->QueryInterface(IID_PPV_ARGS(&m_dxrCommandList)),
        L"ERROR: ID3D12GraphicsCommandList4 not available.\n");
    ThrowIfFailed(device->QueryInterface(IID_PPV_ARGS(&m_dxr2Device)),
        L"ERROR: ID3D12DeviceRaytracing2 not available - is the experimental D3D12Core loaded?\n");
    ThrowIfFailed(commandList->QueryInterface(IID_PPV_ARGS(&m_dxr2CommandList)),
        L"ERROR: ID3D12CommandListRaytracing2 not available.\n");

    QueryDXR2Support();
    BuildScene();
    CreateFillBlasArgsPipeline();
    CreateFillClasTriArgsPipeline();
    BuildAccelerationStructures();
    CreateRaytracingPipelineAndShaderTables();
    CreateDescriptorHeapAndRaytracingOutput();
}

void D3D12RaytracingClusteredGeometry::QueryDXR2Support()
{
    auto device = m_deviceResources->GetD3DDevice();
    D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL opts = {};
    HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL, &opts, sizeof(opts));
    m_clustersAndPtlasSupported = SUCCEEDED(hr) && opts.ClustersAndPTLASSupported;

    D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5 = {};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5));

    SampleLog::LogF(L"\n[DXR2] adapter: %s\n", m_deviceResources->GetAdapterDescription());
    SampleLog::LogF(L"  D3D12_RAYTRACING_TIER: 0x%X\n", (unsigned)opts5.RaytracingTier);
    SampleLog::LogF(L"  ClustersAndPTLASSupported: %s   (hr=0x%08X)\n",
                    m_clustersAndPtlasSupported ? L"YES" : L"NO", (unsigned)hr);

    // Resolve the auto-default for m_aaSamplesPerPixel now that the adapter
    // description is known.  Sentinel 0 in the header means "not set by
    // CLI"; an explicit --aa-samples N takes precedence and skips this
    // branch.  WARP / Basic Render -> 1 (4x on a software rasterizer
    // pushes a 1280x720 frame to seconds-per-frame); HW -> 4.
    if (m_aaSamplesPerPixel == 0)
    {
        const wchar_t* desc = m_deviceResources->GetAdapterDescription();
        const bool isSoftware = (wcsstr(desc, L"WARP") != nullptr) ||
                                (wcsstr(desc, L"Basic Render") != nullptr);
        m_aaSamplesPerPixel = isSoftware ? 1u : 4u;
        SampleLog::LogF(L"  auto AA samples-per-pixel: %u  (%ls adapter)\n",
                        m_aaSamplesPerPixel, isSoftware ? L"software" : L"hardware");
    }
    // Same auto-resolve for the per-frame timing snapshot window.  60
    // samples on HW (~1 s at 60 fps), 5 samples on WARP (~30-50 s at
    // 6-10 s/frame).  Without this, the PER-FRAME overlay line shows
    // "recalculating..." for 10+ minutes on WARP after each toggle.
    if (m_pfSnapTargetCount == 0)
    {
        const wchar_t* desc = m_deviceResources->GetAdapterDescription();
        const bool isSoftware = (wcsstr(desc, L"WARP") != nullptr) ||
                                (wcsstr(desc, L"Basic Render") != nullptr);
        m_pfSnapTargetCount = isSoftware ? 5 : 60;
        SampleLog::LogF(L"  auto per-frame snap samples: %d  (%ls adapter)\n",
                        m_pfSnapTargetCount, isSoftware ? L"software" : L"hardware");
    }

    // Wall-clock FPS rolling-average window size.  Same adapter-aware
    // sentinel pattern.  60-frame window on HW (~0.5-1 s, smooths vsync
    // jitter), 3-frame window on WARP (frames are seconds long, a
    // large window would lag behind state changes by minutes).
    if (m_frameTimeWindow == 0)
    {
        const wchar_t* desc = m_deviceResources->GetAdapterDescription();
        const bool isSoftware = (wcsstr(desc, L"WARP") != nullptr) ||
                                (wcsstr(desc, L"Basic Render") != nullptr);
        m_frameTimeWindow = isSoftware ? 3u : 60u;
        m_frameTimeRing.assign(m_frameTimeWindow, 0.0);
        m_frameTimeRingIdx   = 0;
        m_frameTimeRingCount = 0;
        m_frameTimeRingSum   = 0.0;
        SampleLog::LogF(L"  auto FPS rolling window: %u frames  (%ls adapter)\n",
                        m_frameTimeWindow, isSoftware ? L"software" : L"hardware");
    }

    if (!m_clustersAndPtlasSupported)
    {
        // Hardware doesn't support clusters -- lock the sample into
        // Traditional (DXR1) mode.  The [T] toggle becomes a no-op, the
        // overlay shows a "(locked, clusters unsupported)" hint, and the
        // window title flags the fallback so it's obvious before the
        // overlay even renders.  All cluster-path init / per-frame work
        // is gated on m_clustersAndPtlasSupported below.
        m_geometryMode = GeometryMode::Traditional;
        SampleLog::Write(L"  -> clusters unsupported; falling back to traditional BLAS, [T] toggle disabled.\n");
        SetCustomWindowText(L"clusters not supported on this adapter -- traditional BLAS fallback");
    }
}

// =====================================================================================
// Scene construction
//
// Multiple procedurally-generated objects, each becoming one Cluster BLAS:
//   - 4 spheres of varying subdivisions (different cluster counts per BLAS)
//   - 1 torus
//   - 1 subdivided cube
// All clusters use **spatial-tile** decomposition - small contiguous patches
// of the surface (meshlet-style), not long latitude bands.
// =====================================================================================
void D3D12RaytracingClusteredGeometry::BuildScene()
{
    // ===== MINIMAL COMPRESSED1 REPRO SCENE =====
    // A single cube object: 6 face clusters of 4 verts / 2 tris each.
    // No spheres, torus, klein bottle, floor, animated ball, or
    // mixed-material regions -- everything the bug-isolation work
    // confirmed was not necessary to trigger the corruption.
    ClusterObject obj;
    obj.mesh = ProceduralGeometry::GenerateCubeSpatialTiles(
        /*halfExtent  */ 0.45f,
        /*faceSubdiv  */ 1,
        /*tileSize    */ 1,
        /*firstClusterID*/ 500);

    obj.worldPos      = XMFLOAT3(1.275f, 0.0f, -2.208f);  // hex slot 5 at R=2.55
    obj.worldScale    = 1.0f;
    obj.worldRotEuler = XMFLOAT3(0, 0, 0);
    obj.instanceID    = 5;
    obj.checker       = CheckerConfig{};            // disabled (no per-cluster material override)
    obj.surfTintMul   = 1.0f;
    obj.refrTintMul   = 0.50f;
    obj.reflTintMul   = 1.08f;
    obj.nonOrientable = false;
    obj.globalClusterStart = 0;
    obj.clusterCount       = (UINT)obj.mesh.clusters.size();

    m_totalClusterCount  = obj.clusterCount;
    m_totalTriangleCount = obj.mesh.totalTriangles;
    m_objects.push_back(std::move(obj));

    SampleLog::LogF(L"\n[scene] cube-only: %u clusters, %u tris\n",
                    m_totalClusterCount, m_totalTriangleCount);

    SampleLog::LogF(L"[vertex format] %s\n",
                    m_vertexMode == VertexMode::Compressed1
                        ? L"COMPRESSED1 (shared-exponent quantized)"
                        : L"FLOAT32_3 (no quantization)");

    // Re-encode every cluster's positions into the active vertex format.
    EncodeCompressedClusters();
}


// ---------------------------------------------------------------------------------
// Re-encode every cluster's positions into Compressed1 blobs using the live
// m_compressedBitsPerComponent.  Called once from BuildScene at startup and
// again from RebuildStaticAccelerationStructures when the user cycles the
// precision slider ('[' / ']' in COMPRESSED1 mode).
//
// We always populate obj.encoded even in FLOAT32_3 mode so the 'v' toggle
// can flip vertex format without re-running scene build.  Cost is pure CPU
// and ~50 ms one-time at scene scale; re-encode is comparable.
// ---------------------------------------------------------------------------------
void D3D12RaytracingClusteredGeometry::EncodeCompressedClusters()
{
    const UINT bitsPerComp = m_compressedBitsPerComponent;

    // Pick a single shared compressed1 exponent across the WHOLE scene (all
    // clusters in all objects).  Same-grid quantization keeps adjacent
    // clusters' shared-edge vertices bit-identical -> watertight at the
    // quantization level (BVH-leaf precision is a separate matter).
    int sharedExponent;
    {
        float maxExtent = 0.f;
        for (const auto& obj : m_objects)
        for (const auto& c   : obj.mesh.clusters)
        {
            float mn[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX };
            float mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX };
            for (const auto& p : c.positions)
            {
                const float ps[3] = { p.x, p.y, p.z };
                for (int k = 0; k < 3; ++k) { mn[k] = std::min(mn[k], ps[k]); mx[k] = std::max(mx[k], ps[k]); }
            }
            for (int k = 0; k < 3; ++k) maxExtent = std::max(maxExtent, mx[k] - mn[k]);
        }
        const float minUnit = maxExtent / float((1ull << bitsPerComp) - 1);
        sharedExponent = (int)std::ceil(std::log2(minUnit)) + 127;
        if (sharedExponent < 1)   sharedExponent = 1;
        if (sharedExponent > 232) sharedExponent = 232;
        SampleLog::LogF(L"[compressed1] %u bits/comp; shared exponent (biased): %d  (unit=%.6f)\n",
                        bitsPerComp, sharedExponent, std::ldexp(1.0f, sharedExponent - 127));
    }

    size_t totalCompressedBytes = 0, totalUncompressedBytes = 0;
    float  maxRoundtripError = 0.f;
    for (auto& obj : m_objects)
    {
        obj.encoded.clear();                                 // re-encode wipes any prior blobs
        obj.encoded.reserve(obj.mesh.clusters.size());
        for (const auto& c : obj.mesh.clusters)
        {
            auto enc = Compressed1::Encode(c.positions, (int)bitsPerComp, sharedExponent);
            auto decoded = Compressed1::Decode(enc);
            for (size_t i = 0; i < c.positions.size(); ++i)
            {
                float dx = decoded[i].x - c.positions[i].x;
                float dy = decoded[i].y - c.positions[i].y;
                float dz = decoded[i].z - c.positions[i].z;
                maxRoundtripError = std::max(maxRoundtripError, std::sqrt(dx*dx + dy*dy + dz*dz));
            }
            totalCompressedBytes   += enc.TotalBytes();
            totalUncompressedBytes += c.positions.size() * sizeof(ProceduralGeometry::float3);
            obj.encoded.push_back(std::move(enc));
        }

        // DEBUG (COMPRESSED1 isolation): for the cube, dump per-cluster
        // header + raw bytes + source/decoded positions so we can verify
        // the encoder output against the spec by eye.  WARP renders this
        // correctly using the same bytes, so if encoder matches spec
        // -> the NVIDIA-driver decoder is the bug.
        if (obj.instanceID == 5 /*cube*/)
        {
            SampleLog::LogF(L"\n=== [COMPRESSED1 diag] cube (instanceID=5): %u clusters ===\n",
                            (UINT)obj.encoded.size());
            for (size_t ci = 0; ci < obj.encoded.size(); ++ci)
            {
                const auto& enc = obj.encoded[ci];
                const auto& src = obj.mesh.clusters[ci].positions;
                const UINT exp_b = (UINT)DECODE_D3D12_COMPRESSED1_EXPONENT(enc.header);
                const int  xA = (int)DECODE_D3D12_COMPRESSED1_X_ANCHOR(enc.header);
                const int  yA = (int)DECODE_D3D12_COMPRESSED1_Y_ANCHOR(enc.header);
                const int  zA = (int)DECODE_D3D12_COMPRESSED1_Z_ANCHOR(enc.header);
                const UINT xB = (UINT)DECODE_D3D12_COMPRESSED1_X_BITS(enc.header) + 1u;
                const UINT yB = (UINT)DECODE_D3D12_COMPRESSED1_Y_BITS(enc.header) + 1u;
                const UINT zB = (UINT)DECODE_D3D12_COMPRESSED1_Z_BITS(enc.header) + 1u;
                const double unit = std::ldexp(1.0, (int)exp_b - 127);
                SampleLog::LogF(
                    L"  cluster[%u]  verts=%u  header f0=0x%08X f1=0x%08X f2=0x%08X\n"
                    L"               exp(biased)=%u (unit=%g)  anchor=(%d,%d,%d)  bits=(%u,%u,%u)\n"
                    L"               bitstream %zu bytes, total %zu bytes\n",
                    (UINT)ci, enc.vertexCount,
                    (UINT)enc.header.field0, (UINT)enc.header.field1, (UINT)enc.header.field2,
                    exp_b, unit, xA, yA, zA, xB, yB, zB,
                    enc.bitstream.size(), enc.TotalBytes());

                // Hex dump: full header (12 B) then bitstream
                wchar_t hex[3*32 + 1] = {};
                const uint8_t* hp = reinterpret_cast<const uint8_t*>(&enc.header);
                int pos = 0;
                for (int b = 0; b < 12; ++b)  { swprintf_s(hex + pos, 4, L"%02X ", hp[b]);     pos += 3; }
                SampleLog::LogF(L"               hdr bytes:  %s\n", hex);
                pos = 0;
                const size_t streamBytes = std::min<size_t>(enc.bitstream.size(), 32);
                for (size_t b = 0; b < streamBytes; ++b) { swprintf_s(hex + pos, 4, L"%02X ", enc.bitstream[b]); pos += 3; }
                SampleLog::LogF(L"               stream(%zub): %s\n", streamBytes, hex);

                auto dec = Compressed1::Decode(enc);
                for (size_t vi = 0; vi < src.size(); ++vi)
                {
                    const auto& s = src[vi];
                    const auto& d = dec[vi];
                    // Reconstruct the per-axis quantized integer = anchor + delta
                    // (== what the spec's `I` is); helpful for verifying by hand
                    // that "I = round(src/unit)" matches the dump's delta + anchor.
                    const long long Ix = (long long)std::llround((double)s.x / unit);
                    const long long Iy = (long long)std::llround((double)s.y / unit);
                    const long long Iz = (long long)std::llround((double)s.z / unit);
                    SampleLog::LogF(
                        L"               v[%zu] src=(%+.6f,%+.6f,%+.6f) dec=(%+.6f,%+.6f,%+.6f) "
                        L"I=(%lld,%lld,%lld) delta=(%lld,%lld,%lld)\n",
                        vi, s.x, s.y, s.z, d.x, d.y, d.z,
                        Ix, Iy, Iz, Ix - xA, Iy - yA, Iz - zA);
                }
            }
        }
    }
    SampleLog::LogF(L"[compressed1] %u clusters @ %u bits/comp -> %zu bytes (vs %zu uncompressed = %.2fx); "
                    L"worst-case roundtrip error %.6f units\n",
                    m_totalClusterCount, bitsPerComp,
                    totalCompressedBytes, totalUncompressedBytes,
                    (double)totalUncompressedBytes / (double)totalCompressedBytes,
                    maxRoundtripError);
}



// =====================================================================================
// Acceleration structure build orchestration.
// =====================================================================================
void D3D12RaytracingClusteredGeometry::BuildAccelerationStructures()
{
    auto commandList      = m_deviceResources->GetCommandList();
    auto commandAllocator = m_deviceResources->GetCommandAllocator();

    ThrowIfFailed(commandAllocator->Reset());
    ThrowIfFailed(commandList->Reset(commandAllocator, nullptr));

    UploadClusterInputs();
    BuildClasIndirect();
    BuildBlasFromClasIndirect();
    BuildTlasClassic();

    m_deviceResources->ExecuteCommandList();
    m_deviceResources->WaitForGpu();
}

// Skip maximising on software adapters (per-frame cost is multi-second on WARP).
bool D3D12RaytracingClusteredGeometry::ShouldMaximizeWindowOnLaunch() const
{
    const wchar_t* desc = m_deviceResources->GetAdapterDescription();
    return !(wcsstr(desc, L"WARP") || wcsstr(desc, L"Basic Render"));
}

// ---------------------------------------------------------------------------------
// Runtime tear-down + rebuild of the STATIC half of the AS pipeline (CLAS,
// BLAS-from-CLAS, TLAS).  Animated object's per-frame INSTANTIATE + BLAS
// rebuild uses its own buffers + scratch and is intentionally untouched -
// it continues to render correctly across the rebuild because:
//   1. m_animatedObject.blasGPUVA is stable (EXPLICIT_DESTINATIONS).
//   2. The TLAS instance desc array we rebuild references that same GVA.
//
// Drives the 'a' (cycle CLAS alloc mode) and 'v' (cycle vertex format)
// keyboard shortcuts in OnKeyDown.  Per-frame cost is one full GPU flush
// per press; perfectly fine for a tech-demo toggle but not something you
// would hook into a hot path in production code.
// ---------------------------------------------------------------------------------


// ---------------------------------------------------------------------------------
// Concatenate per-cluster vertex blob + index buffer + per-cluster build args
// into a single upload buffer. Layout (clusters in object order, all flattened):
// =====================================================================================
// Per-instance material assignment + upload to a structured buffer.
//
// 9 slots, indexed by InstanceID() in HLSL.  Every object in this scene
// has reflectivity > 0 (the user wanted everything mirror-tinged); a
// subset additionally has translucency or refractivity to demo any-hit
// stochastic transparency / Snell-law refraction.  Stacking is allowed -
// e.g., the Klein bottle is BOTH partly reflective AND frosted.
//
//   0  large sphere       diffuse + soft 20% reflection                (matte plastic)
//   1  medium sphere      85% mirror reflection                        (chrome)
//   2  small sphere       40% reflection + 50% stochastic translucency (frosted mirror)
//   3  smallest sphere    35% reflection                               (warm metal)
//   4  torus              25% reflection                               (brushed brass)
//   5  cube               60% reflection                               (polished steel)
//   6  floor              15% reflection                               (wet stone)
//   7  ANIMATED sphere    20% reflection + 70% refraction (IOR 1.5)    (animated glass)
//   8  Klein bottle       30% reflection + 55% stochastic translucency (frosted purple)
//
// Per-instance flags are derived from the floats:
//   translucency==0 && refractivity==0  -> FORCE_OPAQUE (any-hit skipped)
//   otherwise                            -> NO opaque flag (any-hit fires)
// (A 100% reflective surface is still fully OPAQUE to ray traversal -
//  reflection is composed in closesthit, not in any-hit.)
// =====================================================================================






// =====================================================================================
// BuildClusterMetadata - generic per-cluster metadata pass.
//
// Walks every cluster of every object (including the animated sphere) and
// emits one ClusterMeta entry per cluster ID.  All per-object material /
// colour decisions live in the ClusterObject::checker config and the per-
// object tint multipliers - the GPU shader is fully data-driven and has
// ZERO per-instance / per-cid-range branches.
//
// The fields populated come straight from CPU-side scene data:
//   - colorIndex       = cluster.matchedColorCid  (set by generator;
//                                                  matched top tile for
//                                                  slab bottom/wall)
//   - flags            = cluster.flags            (INTERIOR_SURFACE etc.)
//   - overrideRefl/Refr/Ior = picked by parity (cluster.gridU+gridV)&1
//                             from obj.checker.{even,odd}Parity
//   - baseColorScale   = same (from CheckerOverride.baseColorScale)
//   - surfTintMul / refrTintMul / reflTintMul = per-object knobs
// =====================================================================================




//
//   [aligned 16] cluster[0].vertex_blob
//   [aligned 16] cluster[0].index_buffer
//   ...
//   [aligned 16] cluster[N-1].vertex_blob
//   [aligned 16] cluster[N-1].index_buffer
//   [aligned 16] BUILD_CLAS_FROM_TRIANGLES_ARGS[0..N-1]
// ---------------------------------------------------------------------------------
void D3D12RaytracingClusteredGeometry::UploadClusterInputs()
{
    auto device = m_deviceResources->GetD3DDevice();
    // Use a generous per-cluster alignment for COMPRESSED1 vertex data; some
    // drivers appear to read past the spec'd end of a cluster's compressed
    // blob (probably prefetch), so leaving real padding between clusters
    // avoids stepping into the next cluster's bits. 16 bytes is fine for the
    // FLOAT32_3 path which has a fixed per-vertex stride.
    const bool       useFloat = (m_vertexMode == VertexMode::Float32_3);
    const size_t     kAlign   = useFloat ? 16 : 256;
    auto alignTo = [](size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); };

    auto vbSizeForCluster = [&](const ClusterObject& obj, size_t i) -> size_t
    {
        if (useFloat)
            return obj.mesh.clusters[i].positions.size() * sizeof(ProceduralGeometry::float3);
        else
            return obj.encoded[i].TotalBytes();
    };

    // DEBUG (NVIDIA COMPRESSED1 isolation): conformance test uses
    // D3D12_INDEX_FORMAT_UINT16 unconditionally for cluster indices,
    // sample previously used UINT8 (uint8_t cluster.indices arrays).
    // The size scaling is applied here so each cluster's IB region is
    // 2 bytes per index (matches IBStride=2 in FillClasFromTrianglesArgs.hlsl)
    // and the upload loop below widens uint8_t -> uint16_t before memcpy.
    auto ibSizeForCluster = [&](const ClusterObject& obj, size_t i) -> size_t
    {
        return obj.mesh.clusters[i].indices.size() * sizeof(uint16_t);
    };

    struct Slot { size_t vbOffset, vbSize, ibOffset, ibSize; };
    std::vector<Slot> slots(m_totalClusterCount);
    size_t cursor = 0;
    UINT gIdx = 0;
    for (const auto& obj : m_objects)
    for (size_t i = 0; i < obj.mesh.clusters.size(); ++i, ++gIdx)
    {
        const auto& src = obj.mesh.clusters[i];
        cursor = alignTo(cursor, kAlign);
        slots[gIdx].vbOffset = cursor;
        slots[gIdx].vbSize   = vbSizeForCluster(obj, i);
        cursor += slots[gIdx].vbSize;
        cursor = alignTo(cursor, kAlign);
        slots[gIdx].ibOffset = cursor;
        slots[gIdx].ibSize   = ibSizeForCluster(obj, i);
        cursor += slots[gIdx].ibSize;
    }
    // Layout the per-cluster vertex+index slots in the upload buffer.  The
    // args used to live in this same buffer at `argsOffset` -- now they
    // live in a separate DEFAULT-heap UAV (m_clasArgsBuffer below) so a
    // compute shader can write them GPU-side.  m_clusterInputBuffer is
    // strictly vertex+index data now.
    const size_t totalDataSize = alignTo(cursor, 256);
    SampleLog::LogF(L"[upload] cluster input buffer: %zu bytes (vert+idx; args now in separate UAV)\n",
                    totalDataSize);

    auto uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto bufDesc    = CD3DX12_RESOURCE_DESC::Buffer(totalDataSize);
    ThrowIfFailed(device->CreateCommittedResource(
        &uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_clusterInputBuffer)));
    m_clusterInputBuffer->SetName(L"ClusterInputBuffer (vertex+index data only)");

    uint8_t* mapped = nullptr;
    CD3DX12_RANGE noRead(0, 0);
    ThrowIfFailed(m_clusterInputBuffer->Map(0, &noRead, reinterpret_cast<void**>(&mapped)));
    const D3D12_GPU_VIRTUAL_ADDRESS baseGPUVA = m_clusterInputBuffer->GetGPUVirtualAddress();

    // Per-cluster vertex + index data.
    gIdx = 0;
    for (const auto& obj : m_objects)
    for (size_t i = 0; i < obj.mesh.clusters.size(); ++i, ++gIdx)
    {
        const auto& src = obj.mesh.clusters[i];
        if (useFloat)
        {
            // float32_3: just copy the positions array. Same byte-exact value
            // for shared edge vertices across adjacent clusters -> watertight
            // to the vertex-precision level.
            memcpy(mapped + slots[gIdx].vbOffset,
                   src.positions.data(),
                   src.positions.size() * sizeof(ProceduralGeometry::float3));
        }
        else
        {
            // compressed1: encoded header + bitstream blob.
            const auto& enc = obj.encoded[i];
            memcpy(mapped + slots[gIdx].vbOffset, &enc.header, sizeof(enc.header));
            if (!enc.bitstream.empty())
                memcpy(mapped + slots[gIdx].vbOffset + sizeof(enc.header),
                       enc.bitstream.data(), enc.bitstream.size());
        }
        if (!src.indices.empty())
        {
            // Widen uint8 cluster indices to uint16 for the
            // experimental UINT16 IndexFormat path (NVIDIA COMPRESSED1
            // isolation -- conformance test never uses UINT8 for
            // cluster indices and may have an undocumented
            // UINT8+COMPRESSED1 driver bug).
            uint16_t* dst16 = reinterpret_cast<uint16_t*>(mapped + slots[gIdx].ibOffset);
            for (size_t k = 0; k < src.indices.size(); ++k)
                dst16[k] = static_cast<uint16_t>(src.indices[k]);
        }
    }
    m_clusterInputBuffer->Unmap(0, nullptr);

    // ------------------------------------------------------------------
    // Per-cluster metadata for the FillClasFromTrianglesArgs CS.
    // 36 bytes/cluster: see ClasArgsMeta layout in FillClasFromTrianglesArgs.hlsl.
    //   { clusterID, triCount, vertCount, vbOff, ibOff, opaqueFlag,
    //     matRegionIdx, vbGvaLo, vbGvaHi }
    // matRegionIdx becomes BaseGeometryIndex in the CLAS (upper 24 bits
    // of BaseGeometryIndexAndFlags), so GeometryIndex() in the closest-
    // hit returns this per-cluster value -- the same identifier the
    // traditional path's per-material-region geom-desc layout produces.
    //
    // vbGvaLo/Hi: per-cluster absolute GPU VA for the VertexBuffer.
    // In default (kPerClusterVbExperiment==false OR not COMPRESSED1) it
    // is just (baseGpuVa + vbOff) so the CS sees the same value
    // whether it adds the offset itself or trusts the meta -- the
    // hypothesis-test path leaves the CS using GVA-from-meta
    // unconditionally to minimise control-flow changes.
    // ------------------------------------------------------------------
    {
        struct ClasArgsMeta {
            UINT clusterID, triCount, vertCount, vbOff, ibOff, opaqueFlag, matRegionIdx;
        };
        static_assert(sizeof(ClasArgsMeta) == 28, "must match the HLSL load offsets");

        std::vector<ClasArgsMeta> meta(m_totalClusterCount);
        gIdx = 0;
        for (const auto& obj : m_objects)
        for (size_t i = 0; i < obj.mesh.clusters.size(); ++i, ++gIdx)
        {
            const auto& src = obj.mesh.clusters[i];
            meta[gIdx].clusterID    = src.clusterID;
            meta[gIdx].triCount     = (UINT)(src.indices.size() / 3);
            meta[gIdx].vertCount    = (UINT)src.positions.size();
            meta[gIdx].vbOff        = (UINT)slots[gIdx].vbOffset;
            meta[gIdx].ibOff        = (UINT)slots[gIdx].ibOffset;
            meta[gIdx].opaqueFlag   = (UINT)D3D12_RTAS_CLUSTERED_GEOMETRY_FLAG_OPAQUE;
            meta[gIdx].matRegionIdx = 0;
        }
        AllocateUploadBuffer(device, meta.data(),
                             meta.size() * sizeof(ClasArgsMeta),
                             &m_clasArgsMetaBuffer, L"Cluster args metadata");
    }

    // GPU-written args buffer (DEFAULT/UAV).
    {
        const UINT64 argsBytes = (UINT64)m_totalClusterCount * sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS);
        AllocateUAVBuffer(device, argsBytes,
            &m_clasArgsBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            L"CLAS-from-triangles args (GPU-filled)");
    }

    // Dispatch FillClasFromTrianglesArgs.
    {
        auto cmdList = m_deviceResources->GetCommandList();
        cmdList->SetComputeRootSignature(m_fillClasTriArgsRS.Get());
        cmdList->SetPipelineState(m_fillClasTriArgsPSO.Get());
        struct {
            UINT baseLo, baseHi;
            UINT count;
            UINT vbStride;
            UINT posTruncBits;
        } cb;
        cb.baseLo       = (UINT)(baseGPUVA & 0xFFFFFFFFu);
        cb.baseHi       = (UINT)(baseGPUVA >> 32);
        cb.count        = m_totalClusterCount;
        cb.vbStride     = useFloat ? (UINT)sizeof(ProceduralGeometry::float3) : 0u;
        cb.posTruncBits = useFloat ? m_positionTruncateBits : 0u;
        cmdList->SetComputeRoot32BitConstants(0, sizeof(cb) / 4, &cb, 0);
        cmdList->SetComputeRootShaderResourceView(1, m_clasArgsMetaBuffer->GetGPUVirtualAddress());
        cmdList->SetComputeRootUnorderedAccessView(2, m_clasArgsBuffer->GetGPUVirtualAddress());
        const UINT groups = (m_totalClusterCount + 63) / 64;
        cmdList->Dispatch(groups, 1, 1);
        D3D12_RESOURCE_BARRIER uav =
            CD3DX12_RESOURCE_BARRIER::UAV(m_clasArgsBuffer.Get());
        cmdList->ResourceBarrier(1, &uav);
    }

    m_clasArgsArrayGPUVA = m_clasArgsBuffer->GetGPUVirtualAddress();
    m_clasArgsStride     = (UINT)sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS);
}

// =====================================================================================
// CLAS allocation strategies.
//
// All three modes share the same per-cluster inputs (vertex/index buffers,
// triangle counts, args array, etc.). They differ only in HOW the per-cluster
// output CLAS bytes get placed in GPU memory:
//
//   Implicit  - one IMPLICIT_DESTINATIONS build; allocate worst-case storage
//               from prebuild.ResultDataMaxSizeInBytes. Simplest path. Wastes
//               memory when actual per-cluster size << worst case (typical
//               5x over-alloc for this scene on NVIDIA).
//
//   GetSizes  - two passes:
//                 1. MODE_GET_SIZES: no result-buffer alloc, driver writes
//                    per-cluster bytes-needed into a UAV size array.
//                 2. CPU readback of sizes, sum to get the exact total,
//                    compute per-cluster destination offsets, allocate an
//                    exact-sized result buffer.
//                 3. MODE_EXPLICIT_DESTINATIONS build into that buffer.
//               No over-allocation at any point. Cost: one extra GPU pass +
//               a CPU stall on the size readback.
//
//   Compact   - two passes, opposite tradeoff:
//                 1. MODE_IMPLICIT_DESTINATIONS build (worst-case alloc),
//                    with ResultSizeArray attached so per-cluster actual
//                    sizes are written as a side effect of the build.
//                 2. CPU readback of sizes -> compacted total.
//                 3. MOVE_CLUSTER_OBJECTS in MODE_IMPLICIT_DESTINATIONS into
//                    a tightly-sized compacted result buffer.
//                 4. Old (worst-case) result buffer is released.
//               Peak GPU memory = worst-case + compacted (briefly, during
//               step 3). Final = compacted. No CPU-blocking required if the
//               size info is consumed by a GPU shader -> only useful if you
//               want EXACT post-build alloc on a single GPU timeline.
//
// All three end with m_clasResultBuffer, m_clasAddressArray, m_clasSizeArray
// in their canonical post-CLAS state so the subsequent BLAS-from-CLAS build
// can proceed identically. The captured stats (m_clasMemStats) let the
// user see the tradeoff in numbers via the [CLAS mem] log line.
// =====================================================================================
void D3D12RaytracingClusteredGeometry::BuildClasIndirect()
{
    m_clasMemStats = ClasMemStats{};
    BuildClasImplicit();
}

// ---------------------------------------------------------------------------------
// Build the shared D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC + ClusterLimits.
// All three modes need the same input description; only their Mode field and
// the surrounding batched-op-data setup differ. Keeping this in one helper
// avoids drift between the three paths.
// ---------------------------------------------------------------------------------
static void BuildSharedClusterTrianglesInputs(
    const D3D12RaytracingClusteredGeometry& self,
    UINT N, bool useFloat, UINT maxTris, UINT maxVerts, UINT totalTris, UINT totalVerts,
    UINT maxCompressedSize,
    D3D12_RTAS_CLUSTER_LIMITS& outLimits,
    D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC& outClasDesc)
{
    (void)self;
    outLimits = {};
    outLimits.MaxArgCount                                   = N;
    outLimits.MaxGeometryIndexValue                         = 0;
    // DEBUG: conformance test uses 256 here.  Sample previously used 1
    // (every cluster has only ONE geom-index slot since the per-cluster
    // GeometryIndexAndFlagsArray is null).  Testing whether NVIDIA's
    // optimization path for count=1 has a COMPRESSED1 bug.
    outLimits.MaxUniqueGeometryIndexAndFlagsCountPerCluster = 256;
    outLimits.MaxTriangleCountPerCluster                    = maxTris;
    outLimits.MaxVertexCountPerCluster                      = maxVerts;
    outLimits.MaxTotalTriangleCount                         = totalTris;
    outLimits.MaxTotalVertexCount                           = totalVerts;
    outLimits.MaxOpacityMicromapIndicesPerCluster           = 0;

    outClasDesc = {};
    outClasDesc.ClusterLimits                       = outLimits;
    // FAST_TRACE: optimise for trace performance over build speed (the typical
    // static-asset choice; use FAST_OPERATION instead for streaming rebuilds).
    // ALLOW_DATA_ACCESS: required for the TriangleObjectPositions() HLSL
    // intrinsic the closest-hit shader uses to compute per-hit normals - the
    // driver stores the cluster's source positions in/alongside the BVH so the
    // intrinsic can read them back. Per-cluster ClusterFlags can override with
    // D3D12_RTAS_CLUSTER_OPERATION_CLAS_FLAG_DISALLOW_DATA_ACCESS to opt some
    // clusters out (we don't).
    outClasDesc.Flags                               = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE | D3D12_RTAS_OPERATION_FLAG_ALLOW_DATA_ACCESS;
    outClasDesc.VertexFormat                        = useFloat ? D3D12_VERTEX_FORMAT_FLOAT32_3
                                                               : D3D12_VERTEX_FORMAT_COMPRESSED1;
    outClasDesc.IndexFormat                         = D3D12_INDEX_FORMAT_UINT16;
    outClasDesc.GeometryIndexAndFlagsIndexFormat    = D3D12_INDEX_FORMAT_NONE;
    outClasDesc.OpacityMicromapIndexFormat          = D3D12_INDEX_FORMAT_NONE;
    // The last field is a union: in COMPRESSED1 mode it's
    // MaxCompressedClusterPositionsSize (the max compressed-blob bytes per
    // cluster), in FLOAT32_3 mode it's MinPositionTruncateBitCount (the
    // floor on per-cluster mantissa-truncation; per-cluster
    // PositionTruncateBitCount in the args struct must be >= this).
    if (useFloat)
        outClasDesc.MinPositionTruncateBitCount = self.PositionTruncateBits();
    else
        outClasDesc.MaxCompressedClusterPositionsSize = maxCompressedSize;
    // .Mode set by each caller.
}

// ---------------------------------------------------------------------------------
// Implicit mode (default). Single IMPLICIT_DESTINATIONS build covering all
// clusters. Allocates the worst-case result-buffer size from prebuild info;
// per-cluster GVAs land in m_clasAddressArray; per-cluster sizes land in
// m_clasSizeArray (as a side effect of the build, useful for stats).
// ---------------------------------------------------------------------------------
void D3D12RaytracingClusteredGeometry::BuildClasImplicit()
{
    auto device = m_deviceResources->GetD3DDevice();
    const UINT N = m_totalClusterCount;
    const bool useFloat = (m_vertexMode == VertexMode::Float32_3);

    UINT maxTris = 0, maxVerts = 0, totalTris = 0, totalVerts = 0;
    UINT maxCompressedSize = 0;
    for (const auto& obj : m_objects)
    for (size_t i = 0; i < obj.mesh.clusters.size(); ++i)
    {
        UINT t = (UINT)(obj.mesh.clusters[i].indices.size() / 3);
        UINT v = (UINT)obj.mesh.clusters[i].positions.size();
        maxTris  = std::max(maxTris,  t);  totalTris  += t;
        maxVerts = std::max(maxVerts, v);  totalVerts += v;
        if (!useFloat)
            maxCompressedSize = std::max(maxCompressedSize, (UINT)obj.encoded[i].TotalBytes());
    }

    D3D12_RTAS_CLUSTER_LIMITS limits;
    D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC clasDesc;
    BuildSharedClusterTrianglesInputs(*this, N, useFloat, maxTris, maxVerts,
                                      totalTris, totalVerts, maxCompressedSize,
                                      limits, clasDesc);
    clasDesc.Mode = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;

    D3D12_RTAS_OPERATION_INPUTS opInputs = {};
    opInputs.Type                  = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
    opInputs.pClusterTrianglesDesc = &clasDesc;

    D3D12_RTAS_OPERATION_PREBUILD_INFO prebuild = {};
    m_dxr2Device->GetRTASOperationPrebuildInfo(&opInputs, &prebuild);
    SampleLog::LogF(L"[CLAS prebuild] %u clusters: result max=%llu bytes, scratch=%llu bytes\n",
                    N,
                    (unsigned long long)prebuild.ResultDataMaxSizeInBytes,
                    (unsigned long long)prebuild.ScratchDataSizeInBytes);

    m_clasMemStats.resultPrebuildMax  = prebuild.ResultDataMaxSizeInBytes;
    m_clasMemStats.resultInitialBytes = prebuild.ResultDataMaxSizeInBytes;
    m_clasMemStats.resultFinalBytes   = prebuild.ResultDataMaxSizeInBytes;
    m_clasMemStats.peakResidentBytes  = prebuild.ResultDataMaxSizeInBytes;
    m_clasMemStats.scratchBytesPhase1 = prebuild.ScratchDataSizeInBytes;

    AllocateUAVBuffer(device, prebuild.ResultDataMaxSizeInBytes,
                      &m_clasResultBuffer, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                      L"All-cluster CLAS result buffer (implicit)");
    AllocateUAVBuffer(device, std::max<UINT64>(prebuild.ScratchDataSizeInBytes, 256ull),
                      &m_clasScratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS scratch");
    AllocateUAVBuffer(device, (UINT64)N * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                      &m_clasAddressArray, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS address array");
    AllocateUAVBuffer(device, (UINT64)N * sizeof(UINT64),
                      &m_clasSizeArray, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS size array");

    D3D12_RTAS_BATCHED_OPERATION_DATA batched = {};
    batched.AddressResolutionFlags    = D3D12_RTAS_OPERATION_ADDRESS_RESOLUTION_FLAG_NONE;
    batched.BatchResultData           = m_clasResultBuffer->GetGPUVirtualAddress();
    batched.BatchScratchData          = m_clasScratchBuffer->GetGPUVirtualAddress();
    batched.ResultAddressArray        = { m_clasAddressArray->GetGPUVirtualAddress(), sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    batched.ResultSizeArray           = { m_clasSizeArray->GetGPUVirtualAddress(),    sizeof(UINT64) };
    batched.IndirectArgumentArray     = { m_clasArgsArrayGPUVA, m_clasArgsStride };
    batched.IndirectArgumentArraySize = 0;
    batched.ToolsInfo                 = 0;

    D3D12_RTAS_OPERATION_DESC opDesc = {};
    opDesc.Inputs                = opInputs;
    opDesc.pBatchedOperationData = &batched;

    const auto t0 = std::chrono::steady_clock::now();
    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDesc, D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    D3D12_RESOURCE_BARRIER barriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasResultBuffer.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasAddressArray.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasSizeArray.Get()),
    };
    m_dxrCommandList->ResourceBarrier(_countof(barriers), barriers);
    const auto t1 = std::chrono::steady_clock::now();
    m_clasMemStats.cpuWallMsPhase1 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Read back the per-cluster sizes the driver wrote into m_clasSizeArray
    // as a side effect of the build, so the [CLAS mem] stats line can show
    // the actual irreducible storage (sum_actual). This is a one-time CPU
    // stall - production code that picks implicit mode would skip it. The
    // sample does it so users can see the over-allocation at a glance.
    auto cmdList = m_deviceResources->GetCommandList();
    ComPtr<ID3D12Resource> sizesReadback;
    {
        auto rbHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto rbDesc = CD3DX12_RESOURCE_DESC::Buffer((UINT64)N * sizeof(UINT64));
        ThrowIfFailed(device->CreateCommittedResource(&rbHeap, D3D12_HEAP_FLAG_NONE, &rbDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&sizesReadback)));
        auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(m_clasSizeArray.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &toCopy);
        cmdList->CopyBufferRegion(sizesReadback.Get(), 0, m_clasSizeArray.Get(), 0,
            (UINT64)N * sizeof(UINT64));
        auto fromCopy = CD3DX12_RESOURCE_BARRIER::Transition(m_clasSizeArray.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &fromCopy);
    }
    m_deviceResources->ExecuteCommandList();
    m_deviceResources->WaitForGpu();
    {
        void* mapped = nullptr;
        D3D12_RANGE rr = { 0, (SIZE_T)N * sizeof(UINT64) };
        ThrowIfFailed(sizesReadback->Map(0, &rr, &mapped));
        auto* sizes = reinterpret_cast<const UINT64*>(mapped);
        UINT64 sum = 0;
        for (UINT i = 0; i < N; ++i) sum += sizes[i];
        D3D12_RANGE noWrite = {0, 0};
        sizesReadback->Unmap(0, &noWrite);
        m_clasMemStats.sumActualBytes = sum;
    }
    auto cmdAlloc = m_deviceResources->GetCommandAllocator();
    ThrowIfFailed(cmdAlloc->Reset());
    ThrowIfFailed(cmdList->Reset(cmdAlloc, nullptr));
}

// ---------------------------------------------------------------------------------
// GetSizes mode. Two-pass: first a GET_SIZES pass to learn per-cluster bytes,
// then an EXPLICIT_DESTINATIONS build into an exact-sized result buffer.
// CPU stalls between passes on the size-array readback.
// ---------------------------------------------------------------------------------


// ---------------------------------------------------------------------------------
// Compact mode. Single IMPLICIT build (worst-case alloc) -- with the size
// array attached so we get per-cluster actual sizes for free -- then a CPU
// readback + MOVE_CLUSTER_OBJECTS in IMPLICIT mode that compacts the live
// CLAS into a tightly-sized buffer. After the move the old (worst-case)
// result buffer is released; what remains is the compacted buffer.
// ---------------------------------------------------------------------------------


// ---------------------------------------------------------------------------------
// One BLAS per object, all built in a single ExecuteIndirectRTASOperations call.
// Each per-object BLAS arg points at its slice of the global CLAS-address array
// via (m_clasAddressArray.GPUVA + obj.globalClusterStart * 8).
//
// EXPLICIT_DESTINATIONS mode so the BLAS GPU VAs end up in CPU-known buffers
// (one per object) - we bake them straight into the TLAS instance descs.
// ---------------------------------------------------------------------------------
void D3D12RaytracingClusteredGeometry::BuildBlasFromClasIndirect()
{
    auto device = m_deviceResources->GetD3DDevice();
    const UINT  N_obj  = (UINT)m_objects.size();

    // Find per-result max BLAS size + worst-case CLAS count per BLAS for limits.
    UINT maxClasPerArg = 0;
    UINT totalClas     = 0;
    for (const auto& obj : m_objects)
    {
        maxClasPerArg = std::max(maxClasPerArg, obj.clusterCount);
        totalClas    += obj.clusterCount;
    }

    D3D12_RTAS_CLAS_INPUTS_DESC blasDesc = {};
    blasDesc.Flags              = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;  // ALLOW_DATA_ACCESS is NOT permitted here - it's a per-CLAS property set at the CLAS-from-triangles build above, and the BLAS-from-CLAS must read it consistently across all referenced CLAS
    blasDesc.MaxArgCount        = N_obj;
    blasDesc.Mode               = D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS;
    blasDesc.MaxTotalClasCount  = totalClas;
    blasDesc.MaxClasCountPerArg = maxClasPerArg;

    D3D12_RTAS_OPERATION_INPUTS opInputs = {};
    opInputs.Type      = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
    opInputs.pClasDesc = &blasDesc;

    D3D12_RTAS_OPERATION_PREBUILD_INFO prebuild = {};
    m_dxr2Device->GetRTASOperationPrebuildInfo(&opInputs, &prebuild);
    SampleLog::LogF(L"[BLAS prebuild] %u objects: per-result max=%llu bytes, scratch=%llu bytes\n",
                    N_obj,
                    (unsigned long long)prebuild.ResultDataMaxSizeInBytes,
                    (unsigned long long)prebuild.ScratchDataSizeInBytes);

    // One BLAS storage buffer per object (each at its own GPU VA).
    for (auto& obj : m_objects)
    {
        AllocateUAVBuffer(device, prebuild.ResultDataMaxSizeInBytes, &obj.blasStorage,
                          D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                          L"Cluster BLAS storage");
        obj.blasGPUVA = obj.blasStorage->GetGPUVirtualAddress();
    }
    AllocateUAVBuffer(device, std::max<UINT64>(prebuild.ScratchDataSizeInBytes, 256ull),
                      &m_blasScratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"Cluster BLAS scratch");

    // Build args array: one BUILD_BLAS_FROM_CLAS_ARGS per object, each
    // referencing a sub-range of the global CLAS-address array.  GPU-filled
    // by FillBlasFromClasArgs CS (shared with the animated path).  CPU
    // prepares a small per-object {count, gvaLo, gvaHi} input buffer.
    {
        struct BlasArgsMeta { UINT count, gvaLo, gvaHi; };
        static_assert(sizeof(BlasArgsMeta) == 12, "must match HLSL Load3 stride");

        const D3D12_GPU_VIRTUAL_ADDRESS clasArrayGPUVA = m_clasAddressArray->GetGPUVirtualAddress();
        std::vector<BlasArgsMeta> meta(N_obj);
        for (UINT i = 0; i < N_obj; ++i)
        {
            const D3D12_GPU_VIRTUAL_ADDRESS perObjArrayGva = clasArrayGPUVA
                + (UINT64)m_objects[i].globalClusterStart * sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
            meta[i].count = m_objects[i].clusterCount;
            meta[i].gvaLo = (UINT)(perObjArrayGva & 0xFFFFFFFFu);
            meta[i].gvaHi = (UINT)(perObjArrayGva >> 32);
        }
        AllocateUploadBuffer(device, meta.data(), meta.size() * sizeof(BlasArgsMeta),
                             &m_blasArgsMeta, L"BLAS args metadata (multi-object)");

        AllocateUAVBuffer(device,
            (UINT64)N_obj * sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS),
            &m_blasArgsBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            L"BLAS args (multi-object, GPU-filled)");

        auto cmdList = m_deviceResources->GetCommandList();
        cmdList->SetComputeRootSignature(m_fillBlasArgsRS.Get());
        cmdList->SetPipelineState(m_fillBlasArgsPSO.Get());
        const UINT cb[2] = { N_obj, (UINT)sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
        cmdList->SetComputeRoot32BitConstants(0, 2, cb, 0);
        cmdList->SetComputeRootShaderResourceView(1, m_blasArgsMeta->GetGPUVirtualAddress());
        cmdList->SetComputeRootUnorderedAccessView(2, m_blasArgsBuffer->GetGPUVirtualAddress());
        const UINT groups = (N_obj + 63) / 64;
        cmdList->Dispatch(groups, 1, 1);
        D3D12_RESOURCE_BARRIER uav =
            CD3DX12_RESOURCE_BARRIER::UAV(m_blasArgsBuffer.Get());
        cmdList->ResourceBarrier(1, &uav);
    }
    {
        // Explicit destination addresses: one entry per object = its m_blasGPUVA.
        std::vector<D3D12_GPU_VIRTUAL_ADDRESS> destAddrs(N_obj);
        for (UINT i = 0; i < N_obj; ++i) destAddrs[i] = m_objects[i].blasGPUVA;
        AllocateUploadBuffer(device, destAddrs.data(), destAddrs.size() * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                             &m_blasResultAddrBuffer, L"BLAS dest addrs (multi-object)");
    }

    D3D12_RTAS_BATCHED_OPERATION_DATA batched = {};
    batched.AddressResolutionFlags    = D3D12_RTAS_OPERATION_ADDRESS_RESOLUTION_FLAG_NONE;
    batched.BatchResultData           = 0;
    batched.BatchScratchData          = m_blasScratchBuffer->GetGPUVirtualAddress();
    batched.ResultAddressArray        = { m_blasResultAddrBuffer->GetGPUVirtualAddress(), sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    batched.ResultSizeArray           = { 0, 0 };
    batched.IndirectArgumentArray     = { m_blasArgsBuffer->GetGPUVirtualAddress(), sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS) };
    batched.IndirectArgumentArraySize = 0;

    D3D12_RTAS_OPERATION_DESC opDesc = {};
    opDesc.Inputs                = opInputs;
    opDesc.pBatchedOperationData = &batched;

    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDesc, D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

    // UAV barrier per BLAS so subsequent TLAS build sees the writes.
    std::vector<D3D12_RESOURCE_BARRIER> uavBarriers;
    uavBarriers.reserve(N_obj);
    for (auto& obj : m_objects)
        uavBarriers.push_back(CD3DX12_RESOURCE_BARRIER::UAV(obj.blasStorage.Get()));
    m_dxrCommandList->ResourceBarrier((UINT)uavBarriers.size(), uavBarriers.data());
}

// =====================================================================================
// Per-frame static-AS rebuilds (driven by m_staticRebuildMode = [R] key)
// =====================================================================================
//
// Re-issues the corresponding RTAS op against the buffers built once at init
// (no reallocations).  Used purely to demonstrate the per-frame cost of an
// LOD-style AS-churn workload even though the inputs in this sample don't
// actually change between frames.
//
// IMPORTANT: these run on the same command list as everything else.  Caller
// is responsible for ordering relative to UpdateAnimatedObjectPerFrame +
// RebuildTlasPerFrame (the static rebuild must happen BEFORE the TLAS
// rebuild, so the TLAS sees the freshly-rebuilt BLAS contents -- BLAS GVAs
// stay constant via EXPLICIT_DESTINATIONS so TLAS instance descs are still
// valid).
// ---------------------------------------------------------------------------------

// =====================================================================================
// Traditional BLAS build path -- one classic BLAS per static object, with
// one D3D12_RAYTRACING_GEOMETRY_DESC per cluster inside that BLAS (so
// GeometryIndex() at hit time recovers the cluster index, matching the
// cluster path's ClusterID()).  Used when m_geometryMode == Traditional.
// TLAS hands these BLAS GVAs out via obj.tradBlasGPUVA.
//
// ------------------------------------------------------------------------
// WHY ONE GEOMETRY DESC PER CLUSTER (READ THIS BEFORE COPYING):
//
// The per-cluster geometry-desc layout we use here is a SAMPLE-ONLY
// choice driven by the demo's apples-to-apples comparison goal.  We
// want the traditional path's closest-hit to be able to recover the
// same per-cluster identity the cluster path gets from ClusterID(),
// so we lay one geometry desc per cluster inside the per-object BLAS
// and feed g_perInstanceFirstCid[InstanceIndex()] + GeometryIndex()
// back into the SAME g_clusterMeta / g_clusterNormals / etc. tables.
// That gets you visually identical output between the two paths with
// the smallest possible code delta, which is what makes the memory
// + build-cost numbers in the overlay directly comparable.
//
// A real app shipping the traditional path would almost certainly
// NOT subdivide a mesh into ~50 geometry descs per object just to
// preserve cluster scope -- it costs measurable BLAS bytes (the
// BVH carries per-geometry split metadata) and build scratch (see
// the overlay numbers).  The natural unit a real app picks is
// MATERIALS: one geometry desc per material region of the mesh,
// because that's the boundary at which the closest-hit needs to
// branch.  Material count is typically O(few) per object instead
// of O(hundreds) of clusters, so the BVH overhead is small.
//
// Worth noting: even in the cluster path, a real app would also
// usually use multiple geometry descs at the TLAS-instance level
// to express materials -- cluster boundaries are placed for spatial
// coherence and traversal efficiency, NOT material coherence, so a
// single cluster will routinely straddle multiple materials in
// real meshes.  The cluster path gives you finer-than-material
// granularity (per-cluster overrides, per-cluster opacity flags,
// per-cluster precision) AS WELL AS material-driven geometries --
// the two layers compose; they're not alternatives.  Our demo
// scene happens to be one-material-per-object so we don't exercise
// that, but the API supports it cleanly.
// ------------------------------------------------------------------------
//
// Two allocation strategies, cycled via [A] in traditional mode (mirrors
// the cluster path's [A] CLAS alloc-mode toggle):
//
//   Implicit: one-shot.  Prebuild reports worst-case size, allocate that
//             buffer, build into it, done.  obj.tradBlasActualBytes ==
//             obj.tradBlasResultBytes.
//
//   Compact:  two-pass.
//             Pass 1: build with ALLOW_COMPACTION into a worst-case
//             temp BLAS + emit POSTBUILD_INFO_COMPACTED_SIZE.
//             GPU flush, CPU readback of per-BLAS actual sizes.
//             Pass 2: allocate tight compacted buffers per object,
//             CopyRaytracingAccelerationStructure(COPY_MODE_COMPACT)
//             into them, drop the worst-case temp buffers.
//             obj.tradBlasActualBytes is the compacted size; the
//             persistent obj.tradBlasStorage IS the compacted buffer
//             (worst-case temp is freed at end of build).
// =====================================================================================


// =====================================================================================
// Traditional-BLAS cid recovery tables.
//
// Cluster path's closest-hit uses ClusterID() (DXR2 intrinsic) so the
// cid is "free" at hit time.  Traditional path doesn't have that --
// GeometryIndex() returns the material-region slot now (one geom desc
// per material region), and PrimitiveIndex() returns the triangle slot
// within that geom.  Neither directly identifies the source cluster.
//
// So we precompute two small tables:
//
//   m_tradTriToCidBuffer: flat per-static-triangle uint = cluster ID
//                         of the cluster that triangle came from.
//                         Indexed by (geomTriBase + PrimitiveIndex()).
//
//   m_tradGeomTriBaseBuffer: per-(InstanceIdx, GeometryIdx) uint =
//                            first index into m_tradTriToCidBuffer for
//                            this instance's GeomIdx-th material region.
//                            Flat 2D layout with kMaxGeomsPerInstance
//                            entries per instance (padded; unused slots
//                            are 0).
//
// At hit time the traditional shader does
//   uint geomBase = g_tradGeomTriBase[InstanceIndex()*MaxGeoms + GeomIdx];
//   uint cid      = g_tradTriToCid[geomBase + PrimitiveIndex()];
// and then everything downstream (cluster-meta lookup, per-vertex
// normals, cluster colour palette, etc.) is identical to the cluster
// path.  Cost is one uint per scene triangle + a few hundred bytes
// for the per-geom base table -- negligible.
// =====================================================================================


// =====================================================================================
// Per-(InstanceIdx, GeometryIdx) -> material-slot lookup table.
//
// Replaces the in-shader `g_materials[InstanceID()]` lookup with one
// that's keyed on both the TLAS instance AND the per-region GeometryIndex().
// For single-region objects the table just has one meaningful entry per
// instance (set to obj.instanceID, so the closesthit gets the same
// material it used to).  For multi-region objects (e.g. the mixed-
// material small sphere with chrome upper / glass lower hemispheres)
// each region gets its own entry pointing at whichever g_materials[]
// slot the region should render as.
//
// Flat 2D layout: uint per (InstIdx, GeomIdx), padded to
// kMaxGeomsPerInstance entries per instance.  Unused slots are 0.
// =====================================================================================








// ---------------------------------------------------------------------------------
// Classic TLAS holding one instance per object (unaffected by the cluster path -
// from the TLAS's perspective, a Cluster BLAS is just a BLAS).
// ---------------------------------------------------------------------------------
// =====================================================================================
// ANIMATED OBJECT - cluster templates path
//
// Demonstrates the second half of the DXR2 cluster build API:
//
//   ┌─ once at init ─────────────────────────────────────────────────────────┐
//   │ BuildAnimatedObjectSetup()                                             │
//   │  1. Generate a sphere mesh; compute "hint" positions = max-deformation │
//   │     envelope (rest * 1.15). The driver uses these to size the cluster  │
//   │     template BVHs so per-frame instantiated positions fit.             │
//   │  2. Upload hint vertex blob + index data + per-cluster                 │
//   │     BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES_ARGS into one upload buffer │
//   │  3. ExecuteIndirectRTASOperations(BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES)│
//   │     → templateResultBuffer holds N "positionless" template CLAS, their │
//   │       GPU VAs land in templateAddressArray                             │
//   │  4. Allocate the per-frame resources (vertex upload, instantiate args  │
//   │     upload, CLAS result/scratch/addr, BLAS storage/scratch/args)       │
//   │  5. Set the BLAS GPU VA - this never changes; only the BVH inside it   │
//   │     gets overwritten every frame                                       │
//   └────────────────────────────────────────────────────────────────────────┘
//
//   ┌─ every frame ──────────────────────────────────────────────────────────┐
//   │ UpdateAnimatedObjectPerFrame()                                         │
//   │  1. CPU: compute new vertex positions (pulsating sphere); memcpy into  │
//   │     the persistently-mapped per-frame upload buffer                    │
//   │  2. ExecuteIndirectRTASOperations(INSTANTIATE_CLUSTER_TEMPLATES)       │
//   │     consumes (template GVA, new vertex slice GVA) per cluster          │
//   │     and writes fresh CLAS into perFrameClasResultBuffer                │
//   │  3. ExecuteIndirectRTASOperations(BUILD_BLAS_FROM_CLAS) → rewrites     │
//   │     the BLAS contents at the existing BLAS GVA (EXPLICIT_DESTINATIONS) │
//   │  TLAS rebuild happens elsewhere - it has to be re-run to pick up the   │
//   │  new BLAS root bounds even though the BLAS GVA is unchanged.           │
//   └────────────────────────────────────────────────────────────────────────┘
//
// CLUSTER ID OFFSET: the animated object's clusters get unique IDs by setting
// ClusterIdOffset=800 in the per-cluster instantiate args. So the shader
// sees the same ClusterColor() palette but at a different region of the
// rainbow than the 6 static objects (which use ClusterIDs 0..505).
// =====================================================================================


// =====================================================================================
// One-shot INSTANTIATE_CLUSTER_TEMPLATES with ResultSizeArray hooked up so we
// can read the per-cluster CLAS leaf sizes the driver actually emitted.  This
// is invoked ONCE at the end of BuildAnimatedObjectSetup (which itself runs
// at init + on every config change), so the steady-state per-frame INSTANTIATE
// in UpdateAnimatedObjectPerFrame stays free of readback overhead.
// Stalls the GPU briefly while it copies the size UAV into a readback buffer.
// =====================================================================================
// =====================================================================================
// Traditional-mode addendum to BuildAnimatedObjectSetup: builds the flat
// per-object IB (cluster-major, with each cluster's local indices biased
// by its vertex offset in the per-frame VB so a single
// D3D12_RAYTRACING_GEOMETRY_DESC covers the whole ball) and allocates
// the per-frame-rebuilt BLAS storage + scratch.  Run ONCE per
// m_animatedObjectEnabled transition; subsequent frames just call
// UpdateAnimatedTradPerFrame to rebuild/refit into tradBlasStorage.
//
// Why a flat IB rather than per-cluster geom descs?  The animated ball
// renders as a single material (the cluster path's checker pattern
// comes from per-cluster ClusterMeta overrides, not from per-geom-desc
// material slots), so we don't need per-region geom-desc routing here.
// ONE geom desc keeps the BLAS build scratch small and the BVH topology
// compact.  If the animated ball ever grows multi-material regions,
// mirror the static-trad path: split clusters by matRegionIdx, emit one
// geom desc per region, populate matching tri-to-cid lookup entries.
// =====================================================================================


// =====================================================================================
// Per-frame trad-mode animated update.  Mirrors UpdateAnimatedObjectPerFrame's
// shape but builds a DXR1 BLAS from perFrameVertexBuffer + tradIndexBuffer
// instead of going through INSTANTIATE_CLUSTER_TEMPLATES + BLAS_FROM_CLAS.
//
// Two operating modes selected by m_traditionalAnimMode (toggled by [F]):
//   Rebuild: full BuildRaytracingAccelerationStructure each frame with
//            PREFER_FAST_TRACE.  Slower but produces the highest-quality
//            BVH each frame.
//   Refit:   BuildRaytracingAccelerationStructure with PERFORM_UPDATE flag,
//            reusing the existing BVH topology.  ~4-10x cheaper but BVH
//            quality drifts as vertices wobble away from the topology
//            cached on the source build -- works because the animation
//            stays within the rest-pose envelope (kEnvelopeScale).
// =====================================================================================




// =====================================================================================
// Walk all the live buffers + m_clasMemStats and copy the displayed numbers
// into m_overlayStats so the per-frame overlay can read from a snapshot rather
// than poking GetDesc().Width on every frame.  Called from
// RebuildStaticAccelerationStructures (after both static and animated paths
// have finished).  Per-frame timing values are *not* captured here - they're
// snapped by Tick() a few frames later once the ring buffer has refilled.
// =====================================================================================
// =====================================================================================
// Stash the CURRENT overlay snapshot into m_overlayStatsPrev (the delta-colour
// baseline).  Must be called BEFORE any of the build code mutates m_overlayStats
// for a config-change rebuild -- otherwise the rebuild path would stash an
// already-half-updated snapshot as prev, killing the delta colouring.
//
// Specifically: MeasureAnimatedClasBytesOneShot runs INSIDE BuildAnimatedObjectSetup
// and writes s.animatedPerFrameClasActualBytes mid-rebuild.  If we stash s -> prev
// AFTER that write (the old combined-Capture* behaviour), prev gets the NEW
// actual-CLAS value and the delta is always zero -> the per-frame-CLAS-actual
// number never lights up green/red even though it changed visibly on screen.
// Splitting stash + refresh fixes this: the rebuild path stashes prev at the
// top (s is still the OLD state at that point) and the refresh runs at the
// bottom (s gets the NEW state to display).
// =====================================================================================


// =====================================================================================
// Snapshot m_overlayStats from the current live GPU resource state.  Re-reads
// every displayed value EXCEPT animatedPerFrameClasActualBytes (which is set by
// MeasureAnimatedClasBytesOneShot at end-of-build and persists across calls).
// Also arms the 5s delta-colour fade and the "recalculating..." per-frame
// settle flag.  Does NOT stash prev -- that must already have been done by
// either CaptureOverlayStatsSnapshot (one-shot callers) or by an earlier
// StashOverlayStatsAsPrev (rebuild callers).
// =====================================================================================


// =====================================================================================
// Convenience wrapper: stash prev, then refresh current.  Use this from any
// caller whose code path does NOT mutate m_overlayStats between the prev-stash
// and the refresh.  Rebuild paths (RebuildStaticAccelerationStructures) MUST
// instead bracket the build with StashOverlayStatsAsPrev() at the top and
// RefreshOverlayStatsCurrent() at the bottom -- otherwise the
// MeasureAnimatedClasBytesOneShot write inside the build steals prev.
// =====================================================================================


// =====================================================================================
// Per-frame work for the animated object. Called from DoRender BEFORE the TLAS
// rebuild + DispatchRays. Total work: 1 memcpy (positions) + 2 batched
// ExecuteIndirectRTASOperations calls + 2 UAV barriers.
// =====================================================================================


void D3D12RaytracingClusteredGeometry::BuildTlasClassic()
{
    auto device = m_deviceResources->GetD3DDevice();
    const UINT N_total = (UINT)m_objects.size();

    // Minimal TLAS for cube-only repro: one instance, opaque hit group,
    // backface-cull-honouring (no double-sided), force-opaque (no any-hit).
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances(N_total);
    for (UINT i = 0; i < N_total; ++i)
    {
        const auto& obj = m_objects[i];
        XMMATRIX rot = XMMatrixRotationRollPitchYaw(obj.worldRotEuler.x,
                                                    obj.worldRotEuler.y,
                                                    obj.worldRotEuler.z);
        XMMATRIX m = XMMatrixScaling(obj.worldScale, obj.worldScale, obj.worldScale)
                   * rot
                   * XMMatrixTranslation(obj.worldPos.x, obj.worldPos.y, obj.worldPos.z);
        XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(instances[i].Transform), m);
        instances[i].InstanceID                          = obj.instanceID;
        instances[i].InstanceMask                        = 0xFF;
        instances[i].InstanceContributionToHitGroupIndex = 0;
        instances[i].Flags                               = D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
        instances[i].AccelerationStructure               = obj.blasGPUVA;
    }
    AllocateUploadBuffer(device, instances.data(), instances.size() * sizeof(instances[0]),
                         &m_tlasInstanceDescs, L"TLAS instance descs");

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.NumDescs       = N_total;
    tlasInputs.Flags          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    tlasInputs.InstanceDescs  = m_tlasInstanceDescs->GetGPUVirtualAddress();

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild = {};
    m_dxrDevice->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &prebuild);
    SampleLog::LogF(L"[TLAS prebuild] %u instances: result=%llu, scratch=%llu bytes\n",
                    N_total,
                    (unsigned long long)prebuild.ResultDataMaxSizeInBytes,
                    (unsigned long long)prebuild.ScratchDataSizeInBytes);

    AllocateUAVBuffer(device, prebuild.ResultDataMaxSizeInBytes, &m_tlasBuffer,
                      D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, L"TLAS storage");
    AllocateUAVBuffer(device, std::max<UINT64>(prebuild.ScratchDataSizeInBytes, 256ull),
                      &m_tlasScratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"TLAS scratch");

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.Inputs                             = tlasInputs;
    buildDesc.DestAccelerationStructureData      = m_tlasBuffer->GetGPUVirtualAddress();
    buildDesc.ScratchAccelerationStructureData   = m_tlasScratchBuffer->GetGPUVirtualAddress();
    m_dxrCommandList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_tlasBuffer.Get());
    m_dxrCommandList->ResourceBarrier(1, &barrier);
}

// ---------------------------------------------------------------------------------
// TLAS rebuild for per-frame use. Same inputs (instance descs buffer) as the
// initial build - only the TLAS BVH itself needs refreshing because the
// animated BLAS contents have changed (its GVA is stable). Buffers are reused;
// nothing reallocated. Single batched command + UAV barrier.
// ---------------------------------------------------------------------------------
void D3D12RaytracingClusteredGeometry::RebuildTlasPerFrame()
{
    if (!m_tlasBuffer || !m_tlasScratchBuffer || !m_tlasInstanceDescs) return;
    const UINT N_total = (UINT)m_objects.size()
        + (m_animatedObjectEnabled ? 1u : 0u);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.NumDescs       = N_total;
    tlasInputs.Flags          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    tlasInputs.InstanceDescs  = m_tlasInstanceDescs->GetGPUVirtualAddress();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.Inputs                             = tlasInputs;
    buildDesc.DestAccelerationStructureData      = m_tlasBuffer->GetGPUVirtualAddress();
    buildDesc.ScratchAccelerationStructureData   = m_tlasScratchBuffer->GetGPUVirtualAddress();
    m_dxrCommandList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_tlasBuffer.Get());
    m_dxrCommandList->ResourceBarrier(1, &barrier);
}



// ---------------------------------------------------------------------------------
// Read back the AS build timestamps from m_buildQueryReadback and compute
// per-operation wall-clock times. Called once after init (post-WaitForGpu).
// ---------------------------------------------------------------------------------


// Title bar intentionally left as the default static "<app name>" set at
// window-creation time.  All per-frame stats live in the on-screen overlay
// (RenderUI()) which has no width limit and supports colour-coded keybinds.
// =====================================================================================
// Raytracing pipeline + shader tables (unchanged from milestone 2).
// =====================================================================================
void D3D12RaytracingClusteredGeometry::CreateRaytracingPipelineAndShaderTables()
{
    auto device = m_deviceResources->GetD3DDevice();
    {
        CD3DX12_DESCRIPTOR_RANGE uavRange;
        uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
        CD3DX12_ROOT_PARAMETER params[GlobalRootSig::Count];
        params[GlobalRootSig::OutputUAVSlot].InitAsDescriptorTable(1, &uavRange);
        params[GlobalRootSig::AccelerationStructureSlot].InitAsShaderResourceView(0);
        params[GlobalRootSig::SceneCBVSlot].InitAsConstantBufferView(0);
        CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params);
        ComPtr<ID3DBlob> blob, err;
        ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err),
            err ? (wchar_t*)err->GetBufferPointer() : L"D3D12SerializeRootSignature failed\n");
        ThrowIfFailed(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
            IID_PPV_ARGS(&m_globalRootSignature)));
        m_globalRootSignature->SetName(L"Global root signature");
    }

    // Optional empty local root signature. The HLK IndirectBuild test creates
    // one of these alongside the global RS - leaving it in here for parity
    // even though our shaders don't take per-shader records. Harmless on
    // adapters that don't need it.
    SampleLog::Write(L"  >>> Create empty local root signature\n");
    {
        CD3DX12_ROOT_SIGNATURE_DESC localDesc(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);
        ComPtr<ID3DBlob> blob, err;
        ThrowIfFailed(D3D12SerializeRootSignature(&localDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err),
            err ? (wchar_t*)err->GetBufferPointer() : L"D3D12SerializeRootSignature(local) failed\n");
        ThrowIfFailed(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
            IID_PPV_ARGS(&m_localRootSignature)));
        m_localRootSignature->SetName(L"Empty local root signature");
    }

    CD3DX12_STATE_OBJECT_DESC pipeline{ D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE };

    pipeline.CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>()
        ->SetRootSignature(m_localRootSignature.Get());

    auto lib = pipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE((void*)g_pRaytracing, ARRAYSIZE(g_pRaytracing));
    lib->SetDXILLibrary(&libdxil);
    lib->DefineExport(c_raygenName);
    lib->DefineExport(c_opaqueClosestHitName);
    lib->DefineExport(c_missName);

    // Single OPAQUE hit group: just a closesthit; no any-hit (TLAS uses
    // FLAG_FORCE_OPAQUE so any-hit dispatch is skipped at traversal time
    // even if we'd bound one).  No shadow / glass hit groups -- the
    // minimal shader fires only primary rays.
    auto opaqueHG = pipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    opaqueHG->SetClosestHitShaderImport(c_opaqueClosestHitName);
    opaqueHG->SetHitGroupExport(c_opaqueHitGroupName);
    opaqueHG->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

    auto shaderConfig = pipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
    shaderConfig->Config(/*payload*/ 4 * sizeof(float) + 2 * sizeof(uint),
                         /*attribs*/ 2 * sizeof(float));
    auto globalRS = pipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
    globalRS->SetRootSignature(m_globalRootSignature.Get());

    // ALLOW_CLUSTERED_GEOMETRY is the DXR2 opt-in for tracing a BLAS built
    // from CLAS.  MaxRecursionDepth = 1 (raygen + 1 trace = primary rays only).
    auto pipelineConfig = pipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT>();
    pipelineConfig->Config(1, D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_CLUSTERED_GEOMETRY);

    SampleLog::Write(L"  >>> CreateStateObject\n");
    HRESULT hrCSO = m_dxrDevice->CreateStateObject(pipeline, IID_PPV_ARGS(&m_dxrStateObject));
    SampleLog::LogF(L"  CreateStateObject -> hr=0x%08X\n", (unsigned)hrCSO);
    ThrowIfFailed(hrCSO, L"CreateStateObject failed\n");
    m_dxrStateObject->SetName(L"RT pipeline (cluster-aware)");

    SampleLog::Write(L"  >>> Build shader tables\n");
    ComPtr<ID3D12StateObjectProperties> props;
    ThrowIfFailed(m_dxrStateObject->QueryInterface(IID_PPV_ARGS(&props)));
    void* rgID        = props->GetShaderIdentifier(c_raygenName);
    void* missID      = props->GetShaderIdentifier(c_missName);
    void* opaqueHgID  = props->GetShaderIdentifier(c_opaqueHitGroupName);
    const UINT idSize     = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    const UINT recordSize = Align(idSize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);

    auto makeTable1 = [&](void* shaderID, ComPtr<ID3D12Resource>& outTable, const wchar_t* name)
    {
        std::vector<uint8_t> data(recordSize, 0);
        memcpy(data.data(), shaderID, idSize);
        AllocateUploadBuffer(device, data.data(), data.size(), &outTable, name);
    };
    makeTable1(rgID,       m_rayGenShaderTable,   L"raygen shader table");
    makeTable1(missID,     m_missShaderTable,     L"miss shader table");
    makeTable1(opaqueHgID, m_hitGroupShaderTable, L"hit-group shader table (opaque primary)");
}
// ---------------------------------------------------------------------------------
// DirectXTK SpriteBatch + SpriteFont setup.  Creates the GraphicsMemory ring
// allocator that DirectXTK uses for per-frame upload data, builds a sprite
// batch PSO that matches the back-buffer + depth-buffer format pair, and
// loads SegoeUI_24.spritefont (deployed by the .vcxproj alongside the exe).
// The 24 pt atlas is exactly 2x the visual target -- combined with kScale=0.5
// below, that gives a clean 2:1 bilinear downsample (every destination pixel
// = perfect average of its 2x2 source texels).  Any other ratio (e.g. 48 pt /
// kScale=0.265 we briefly tried) drops source texels under bilinear filtering
// and aliases on thin glyph strokes.
//
// The font texture SRV is placed into slot 1 of our shader-visible descriptor
// heap (slot 0 = the raytracing-output UAV).  AllocateDescriptor() hands
// these out monotonically; the heap was bumped to 8 slots so we have plenty
// of room for the font + future overlays.
//
// MUST be called AFTER CreateDescriptorHeapAndRaytracingOutput so that the
// shared descriptor heap exists when we ask it for slot 1.
// ---------------------------------------------------------------------------------


// ---------------------------------------------------------------------------------
// Compute pipeline for the per-frame GPU ball animation pass.
//
// Root signature is intentionally tiny: 2 dwords of inline constants
// (time + vertex count), one raw SRV (rest positions), one raw UAV
// (animated positions).  All bindings are root-direct -- no descriptor
// heap involvement -- so this pass is self-contained and doesn't need
// to share descriptor slots with the raytracing path.
//
// PSO is built from the dxc-compiled cs_6_6 bytecode embedded via
// AnimateBall.hlsl.h.  Called once at init from CreateDeviceDependentResources.
// ---------------------------------------------------------------------------------


// =====================================================================
// Build the compute pipeline that GPU-fills the per-cluster
// INSTANTIATE_CLUSTER_TEMPLATES_ARGS array for the animated object.
// See FillInstantiateArgs.hlsl for the shader contract; this just wires
// the root signature parameters in the same slot order the shader
// expects:
//   b0 = root constants (5 dwords: pfVbLo, pfVbHi, clusterCount,
//                                  clusterIdOffset, vertexStride)
//   u1 = templateGvas   (root UAV;  shader reads only)
//   t0 = vertexOffsets  (root SRV)
//   u0 = argsOut        (root UAV)
//
// 4 root parameters total = 5 + 2 + 2 + 2 = 11 dwords, well under the
// 64-DWORD root-sig budget.
// =====================================================================



// =====================================================================
// Tiny helper used by the four args-fill pipeline builders below.
// Each pipeline has exactly the same C++ shape: a root sig built from
// a small N-element rootParams array, no static samplers, no flags,
// + a single CS PSO referencing the supplied bytecode.  Factoring this
// out keeps each Create*Pipeline body focused on its actual schema.
//
// DESIGN NOTE -- separate CSes vs ubershader:
//   We dispatch one CS per args-buffer type (FillClasFromTrianglesArgs,
//   FillClusterTemplateArgs, FillBlasFromClasArgs, FillMoveClusterArgs,
//   FillInstantiateArgs).  This is the simplest, most readable layout
//   for a sample but it's NOT the cheapest possible.  Alternatives a
//   production renderer might use:
//
//     (a) Ubershader / mega-kernel.  Combine all 6 fills into a single
//         CS that branches on SV_GroupID -- different thread groups
//         handle different tasks.  One Dispatch() instead of six.
//         Saves command-processor overhead (~10s of us across the suite)
//         and improves SM occupancy: our smallest fill (FillBlasArgs at
//         8 entries = 1 group) barely uses any of the GPU's 100+ SMs;
//         combined dispatches fit the GPU better.  Costs: one big root
//         sig binds ALL inputs/outputs (root-param-budget pressure),
//         shader source gets larger.
//
//     (b) Async-compute queue.  Submit the independent fills to a
//         separate compute queue so they overlap with the graphics+RTAS
//         queue's work.  Requires queue-fence sync but lets the GPU
//         schedule fills in unused SM headroom while ExecuteIndirect
//         RTAS ops are issuing.
//
//     (c) ExecuteIndirect with N dispatch records.  One submit, driver
//         may schedule records concurrently (vendor-dependent).
//
//   For this sample the args fills run only at init / on config-change,
//   total cost << 1 ms, so simplicity wins.  In a real LOD-driven
//   renderer that re-emits args every frame, (a) or (b) would matter.
// =====================================================================
static void BuildArgsFillPipeline(
    ID3D12Device*                       device,
    const D3D12_ROOT_PARAMETER*         rootParams,
    UINT                                rootParamCount,
    const void*                         csBytecode,
    SIZE_T                              csByteSize,
    const wchar_t*                      rsName,
    const wchar_t*                      psoName,
    ComPtr<ID3D12RootSignature>&        outRS,
    ComPtr<ID3D12PipelineState>&        outPSO)
{
    CD3DX12_ROOT_SIGNATURE_DESC rsDesc;
    rsDesc.Init(rootParamCount, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    Microsoft::WRL::ComPtr<ID3DBlob> serialized, error;
    HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                             &serialized, &error);
    if (FAILED(hr))
    {
        if (error) SampleLog::LogF(L"[fill-args] root sig '%ls' serialize failed: %hs\n",
                                   rsName, (const char*)error->GetBufferPointer());
        ThrowIfFailed(hr);
    }
    ThrowIfFailed(device->CreateRootSignature(0, serialized->GetBufferPointer(),
                                              serialized->GetBufferSize(),
                                              IID_PPV_ARGS(&outRS)));
    outRS->SetName(rsName);

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = outRS.Get();
    psoDesc.CS             = CD3DX12_SHADER_BYTECODE(csBytecode, csByteSize);
    ThrowIfFailed(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&outPSO)));
    outPSO->SetName(psoName);
}

// FillMoveClusterArgs: b0 + u1 (src GVAs) + u0 (out args).


// FillBlasFromClasArgs: b0 + t0 (meta) + u0 (out args).
void D3D12RaytracingClusteredGeometry::CreateFillBlasArgsPipeline()
{
    auto device = m_deviceResources->GetD3DDevice();
    CD3DX12_ROOT_PARAMETER rp[3] = {};
    rp[0].InitAsConstants(2, /*shaderReg*/0);                  // b0 = { entryCount, addrStride }
    rp[1].InitAsShaderResourceView(/*shaderReg*/0);            // t0 = inputs
    rp[2].InitAsUnorderedAccessView(/*shaderReg*/0);           // u0 = argsOut
    BuildArgsFillPipeline(device, rp, _countof(rp),
        g_pFillBlasFromClasArgs, ARRAYSIZE(g_pFillBlasFromClasArgs),
        L"FillBlasFromClasArgs root sig", L"FillBlasFromClasArgs compute PSO",
        m_fillBlasArgsRS, m_fillBlasArgsPSO);
    SampleLog::LogF(L"[fill-args-cs] FillBlasFromClasArgs PSO built (%zu bytes)\n",
                    (size_t)ARRAYSIZE(g_pFillBlasFromClasArgs));
}

// FillClasFromTrianglesArgs: b0 (5 dwords) + t0 (meta) + u0 (out args).
void D3D12RaytracingClusteredGeometry::CreateFillClasTriArgsPipeline()
{
    auto device = m_deviceResources->GetD3DDevice();
    CD3DX12_ROOT_PARAMETER rp[3] = {};
    rp[0].InitAsConstants(5, /*shaderReg*/0);                  // b0
    rp[1].InitAsShaderResourceView(/*shaderReg*/0);            // t0 = meta
    rp[2].InitAsUnorderedAccessView(/*shaderReg*/0);           // u0 = argsOut
    BuildArgsFillPipeline(device, rp, _countof(rp),
        g_pFillClasFromTrianglesArgs, ARRAYSIZE(g_pFillClasFromTrianglesArgs),
        L"FillClasFromTrianglesArgs root sig", L"FillClasFromTrianglesArgs compute PSO",
        m_fillClasTriArgsRS, m_fillClasTriArgsPSO);
    SampleLog::LogF(L"[fill-args-cs] FillClasFromTrianglesArgs PSO built (%zu bytes)\n",
                    (size_t)ARRAYSIZE(g_pFillClasFromTrianglesArgs));
}

// FillClusterTemplateArgs: b0 (4 dwords) + t0 (meta) + u0 (out args).




// ---------------------------------------------------------------------------------
// Per-frame overlay text.  Called from DoRender() AFTER the raytraced output
// has been copied into the back buffer and the back buffer is transitioned
// back to RENDER_TARGET state.  The sprite batch rebinds the back-buffer RTV
// (no depth attachment -- text is depth-disabled by default) and draws our
// stats block + key bindings on top of the ray-traced image.
//
// Layout:
//   line 0   adapter name (white, larger weight via DrawString w/ 1.1x scale)
//   line 1   BLAS / CLAS / triangle counts
//   line 2   CLAS memory: total + avg/cluster + scratch
//   line 3   active vertex format
//   line 4   active CLAS alloc mode
//   line 5   per-frame rebuild timing (anim + TLAS)
//   line 6   FPS
//   line 7+  key-binding hints, with the hotkey character coloured yellow
//
// Pixel coordinates are top-left origin; we leave a 24-px inset from the
// window's top-left corner.
// ---------------------------------------------------------------------------------




void D3D12RaytracingClusteredGeometry::CreateDescriptorHeapAndRaytracingOutput()
{
    auto device = m_deviceResources->GetD3DDevice();

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = 8;    // slot 0 = RT-output UAV, slot 1 = font SRV (DirectXTK), spare = 6
    heapDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_descriptorHeap)));
    m_descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    auto bbDesc = m_deviceResources->GetRenderTarget()->GetDesc();
    auto outDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        m_deviceResources->GetBackBufferFormat(), bbDesc.Width, bbDesc.Height, 1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    ThrowIfFailed(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
        &outDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_raytracingOutput)));
    m_raytracingOutput->SetName(L"Raytracing output");

    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
    UINT idx = AllocateDescriptor(&cpuHandle);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(m_raytracingOutput.Get(), nullptr, &uavDesc, cpuHandle);
    m_raytracingOutputUAV = CD3DX12_GPU_DESCRIPTOR_HANDLE(
        m_descriptorHeap->GetGPUDescriptorHandleForHeapStart(), idx, m_descriptorSize);

    auto upload = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto cbufDesc = CD3DX12_RESOURCE_DESC::Buffer(Align((UINT)sizeof(SceneConstantBuffer), 256));
    ThrowIfFailed(device->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE,
        &cbufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_sceneCB)));
    m_sceneCB->SetName(L"Scene CB");
    CD3DX12_RANGE noRead(0, 0);
    ThrowIfFailed(m_sceneCB->Map(0, &noRead, reinterpret_cast<void**>(&m_sceneCBMapped)));
}

UINT D3D12RaytracingClusteredGeometry::AllocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE* outCpu)
{
    UINT idx = m_descriptorsAllocated++;
    *outCpu = CD3DX12_CPU_DESCRIPTOR_HANDLE(
        m_descriptorHeap->GetCPUDescriptorHandleForHeapStart(), idx, m_descriptorSize);
    return idx;
}

// =====================================================================================
// Time-based camera orbit. Uses m_animSeconds (accumulated wall-clock) so the
// pan rate is independent of frame rate, and pauses cleanly with the 'P' key.
// =====================================================================================
void D3D12RaytracingClusteredGeometry::UpdateSceneConstantBuffer()
{
    if (!m_sceneCBMapped) return;

    // Slowly orbit around the scene center.  Two independent cycles so the
    // camera traces a Lissajous-style path that doesn't repeat for tens of
    // minutes:
    //   - YAW (pan):   2 pi / 30 s   per cycle  -- full revolution every 30 s.
    //   - DOLLY (z):   2 pi / 19 s   per cycle  -- radius pulses INWARD only.
    // 30 and 19 are coprime (19 is prime), so the eye never re-visits the
    // same (yaw, radius) tuple within a reasonable session.  The dolly uses a
    // (1 - cos)/2 envelope rather than a bare sin so the camera NEVER goes
    // farther back than the rest position -- it only swings closer to the
    // scene, framing interior cluster detail at the inner peak.  Range:
    // [kBaseRadius - kDollyAmp, kBaseRadius].
    const double t           = m_animSeconds;
    const float  angle       = float(t * (2.0 * M_PI / 30.0));
    constexpr float kBaseRadius  = 6.5f;     // outermost orbit distance (= rest position)
    constexpr float kDollyAmp    = 3.0f;     // inward swing magnitude; radius sweeps 3.5..6.5
    constexpr float kDollyPeriod = 19.0f;    // seconds; coprime with 30s pan period (full in-and-out every 19 s)
    const float  dollyPhase  = float(t * (2.0 * M_PI / kDollyPeriod));
    const float  radius      = kBaseRadius
                             - kDollyAmp * 0.5f * (1.0f - std::cos(dollyPhase));
    const float  height      = 1.5f;                                 // eye lifted (was 0.8) - higher vantage
    XMVECTOR eye = XMVectorSet(radius * std::sin(angle), height,
                               -radius * std::cos(angle), 1.0f);
    XMVECTOR at  = XMVectorSet(0.5f, 0.0f, 0.0f, 1.0f);   // look-at LOWERED (was 1.2) - camera now pitches DOWN noticeably -> floor reads as the dominant surface, sky shrinks to a band at the top, slab gets seen more from above
    XMVECTOR up  = XMVectorSet(0, 1, 0, 0);

    XMMATRIX view = XMMatrixLookAtLH(eye, at, up);
    XMMATRIX viewToWorld = XMMatrixInverse(nullptr, view);

    SceneConstantBuffer cb = {};
    // NB: XMMATRIX is row-major, HLSL constant-buffer matrices default to
    // column-major packing. Storing the XM matrix directly (without transpose)
    // makes HLSL see XM_M^T, and `mul(M, v_col)` in HLSL then computes
    // XM_M^T * v_col, which is the correct view->world transform of a column
    // direction vector. Transposing here AND using `mul(M, v)` in HLSL would
    // double-flip the convention and put the camera basis in the wrong place
    // (cardinal-axis orbit positions end up pointing AWAY from the scene).
    cb.viewToWorld = viewToWorld;
    XMStoreFloat4(&cb.cameraPosition, eye);
    cb.miscParams.x = (float)m_width / (float)m_height;
    cb.miscParams.y = std::tan(60.0f * (XM_PI / 180.0f) * 0.5f);   // 60deg vertical FOV
    cb.miscParams.z = (float)m_aaSamplesPerPixel;                  // raygen sample count (1/2/4)
    cb.miscParams.w = m_clusterTint;                               // 0..1 cluster-rainbow tint blend

    // Runtime knobs the shader reads each TraceRay.  See SceneConstantBuffer
    // in RaytracingHlslCompat.h for the slot reservations.
    cb.runtimeParams.x = ReflectionBounces();                      // computed from m_bounceSlider
    cb.runtimeParams.y = RefractionBounces();                      // ditto (= refl or refl+2 with clamps)
    cb.runtimeParams.z = IsTraditional() ? 1u : 0u;                // 1 -> closest-hit uses per-instance lookups instead of ClusterID
    // Sun in upper-back-right. Direction TO the light, normalized. .w is the
    // ambient floor: even fully-shadowed pixels get this fraction of base
    // colour so the scene reads instead of going pitch black.
    XMVECTOR sunDir = XMVector3Normalize(XMVectorSet(0.45f, 0.75f, 0.50f, 0.0f));
    XMStoreFloat4(&cb.lightDir, sunDir);
    cb.lightDir.w = 0.15f;                                          // ambient floor (0.15 - middle setting; tried 0.08 punchy, reverted)
    memcpy(m_sceneCBMapped, &cb, sizeof(cb));
}

// =====================================================================================
// Per-frame
// =====================================================================================
void D3D12RaytracingClusteredGeometry::OnUpdate()
{
    m_timer.Tick();
    if (!m_animPaused)
        m_animSeconds += m_timer.GetElapsedSeconds();
    UpdateSceneConstantBuffer();

    // Wall-clock frame-time measurement for the overlay's FPS / ms-per-frame
    // line.  Independent of StepTimer's 100 ms cap (see m_frameTimeRing
    // header comment).  Rolling average over m_frameTimeWindow samples so
    // the displayed value doesn't jitter frame-to-frame (HW) and doesn't
    // lag minutes behind state changes (WARP); window size auto-resolved
    // in OnInit per adapter.  First frame has no prior timestamp so we
    // just seed m_lastFrameWallTime; rolling buffer fills over the next
    // N frames.
    const auto now = std::chrono::steady_clock::now();
    if (m_lastFrameWallTime.time_since_epoch().count() != 0 && m_frameTimeWindow > 0)
    {
        const double dt = std::chrono::duration<double>(now - m_lastFrameWallTime).count();
        // Maintain rolling sum: subtract the slot we're about to overwrite,
        // add the new dt.  Until the ring fills, the overwritten slot is
        // 0 so we just accumulate.
        m_frameTimeRingSum -= m_frameTimeRing[m_frameTimeRingIdx];
        m_frameTimeRing[m_frameTimeRingIdx] = dt;
        m_frameTimeRingSum += dt;
        m_frameTimeRingIdx = (m_frameTimeRingIdx + 1) % m_frameTimeWindow;
        if (m_frameTimeRingCount < m_frameTimeWindow) ++m_frameTimeRingCount;
    }
    m_lastFrameWallTime = now;
}

void D3D12RaytracingClusteredGeometry::OnRender()
{
    if (!m_deviceResources->IsWindowVisible()) return;
    DoRender();
    ++m_framesRendered;
}

void D3D12RaytracingClusteredGeometry::DoRender()
{
    m_deviceResources->Prepare();
    auto cl  = m_deviceResources->GetCommandList();
    auto cl4 = m_dxrCommandList.Get();

    // ---- Per-frame AS work + timestamp queries -----------------------------
    // Read back the slot that was written N-frames ago first (data is safely
    // past GPU completion - we read the slot we're about to overwrite). Skip
    // until we've captured at least kPerFrameRingSlots samples to avoid
    // reading uninitialised heap data.
    cl4->SetComputeRootSignature(m_globalRootSignature.Get());
    ID3D12DescriptorHeap* heaps[] = { m_descriptorHeap.Get() };
    cl4->SetDescriptorHeaps(_countof(heaps), heaps);
    cl4->SetComputeRootDescriptorTable(GlobalRootSig::OutputUAVSlot, m_raytracingOutputUAV);
    cl4->SetComputeRootShaderResourceView(GlobalRootSig::AccelerationStructureSlot,
        m_tlasBuffer->GetGPUVirtualAddress());
    cl4->SetComputeRootConstantBufferView(GlobalRootSig::SceneCBVSlot, m_sceneCB->GetGPUVirtualAddress());
    cl4->SetPipelineState1(m_dxrStateObject.Get());

    auto bbDesc = m_deviceResources->GetRenderTarget()->GetDesc();
    D3D12_DISPATCH_RAYS_DESC drd = {};
    drd.RayGenerationShaderRecord.StartAddress = m_rayGenShaderTable->GetGPUVirtualAddress();
    drd.RayGenerationShaderRecord.SizeInBytes  = m_rayGenShaderTable->GetDesc().Width;
    drd.MissShaderTable.StartAddress           = m_missShaderTable->GetGPUVirtualAddress();
    drd.MissShaderTable.SizeInBytes            = m_missShaderTable->GetDesc().Width;
    // Stride = single record size (32 B), not whole-table width. The miss
    // table now has TWO records (primary + shadow); each TraceRay's
    // MissShaderIndex selects which one by stepping `stride` bytes in.
    drd.MissShaderTable.StrideInBytes          = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    drd.HitGroupTable.StartAddress             = m_hitGroupShaderTable->GetGPUVirtualAddress();
    drd.HitGroupTable.SizeInBytes              = m_hitGroupShaderTable->GetDesc().Width;
    drd.HitGroupTable.StrideInBytes            = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    drd.Width  = (UINT)bbDesc.Width;
    drd.Height = (UINT)bbDesc.Height;
    drd.Depth  = 1;
    cl4->DispatchRays(&drd);

    D3D12_RESOURCE_BARRIER toCopy[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_raytracingOutput.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(m_deviceResources->GetRenderTarget(),
            D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST),
    };
    cl->ResourceBarrier(_countof(toCopy), toCopy);
    cl->CopyResource(m_deviceResources->GetRenderTarget(), m_raytracingOutput.Get());
    D3D12_RESOURCE_BARRIER toRtAndPresent[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_raytracingOutput.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        CD3DX12_RESOURCE_BARRIER::Transition(m_deviceResources->GetRenderTarget(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET),
    };
    cl->ResourceBarrier(_countof(toRtAndPresent), toRtAndPresent);

    // Paint on-screen overlay text (stats, key bindings).  The back buffer is
    // now in RENDER_TARGET; SpriteBatch binds it as an RTV, draws text, and
    // we hand off to Present() which transitions to PRESENT internally.
m_deviceResources->Present();

    // DirectXTK: tag per-frame upload pages with the queue's current fence value
    // AFTER the cmd list has been executed (Present did ExecuteCommandList).  This
    // is the canonical placement -- calling Commit before Present would signal the
    // fence on a queue position before our text-draw commands and the ring
    // allocator could reclaim live upload pages.
    if (m_graphicsMemory) m_graphicsMemory->Commit(m_deviceResources->GetCommandQueue());
}

// =====================================================================================
// Lifecycle
//
// The close-hang fix: the persistent map on m_sceneCB plus the swap chain in
// fullscreen state (occasionally) was making the implicit cleanup wait. Explicit
// Unmap + WaitForGpu BEFORE any ComPtr release is the safe order.
// =====================================================================================
void D3D12RaytracingClusteredGeometry::OnSizeChanged(UINT width, UINT height, bool minimized)
{
    if (!m_deviceResources->WindowSizeChanged(width, height, minimized)) return;
    UpdateForSizeChange(width, height);

    // The DXR output UAV is sized to match the back buffer.  After
    // WindowSizeChanged resizes the swap chain, the UAV is the OLD size
    // and the per-frame `CopyResource(m_raytracingOutput -> back buffer)`
    // hangs (size mismatch).  Recreate the UAV at the new size and rebind
    // it into descriptor slot 0 (where CreateDescriptorHeapAndRaytracing-
    // Output originally placed it - first AllocateDescriptor call).
    auto device = m_deviceResources->GetD3DDevice();
    auto bbDesc = m_deviceResources->GetRenderTarget()->GetDesc();
    if (m_raytracingOutput &&
        (bbDesc.Width  != m_raytracingOutput->GetDesc().Width ||
         bbDesc.Height != m_raytracingOutput->GetDesc().Height))
    {
        m_raytracingOutput.Reset();
        auto outDesc = CD3DX12_RESOURCE_DESC::Tex2D(
            m_deviceResources->GetBackBufferFormat(), bbDesc.Width, bbDesc.Height, 1, 1, 1, 0,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        ThrowIfFailed(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
            &outDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_raytracingOutput)));
        m_raytracingOutput->SetName(L"Raytracing output");

        // Overwrite descriptor heap slot 0 in place (don't AllocateDescriptor
        // again - that would leak the slot and bump m_descriptorsAllocated).
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(
            m_descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 0, m_descriptorSize);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(m_raytracingOutput.Get(), nullptr, &uavDesc, cpuHandle);
    }
}

void D3D12RaytracingClusteredGeometry::OnDestroy()
{
    if (m_deviceResources) m_deviceResources->WaitForGpu();
    if (m_sceneCB && m_sceneCBMapped)
    {
        m_sceneCB->Unmap(0, nullptr);
        m_sceneCBMapped = nullptr;
    }
    // ComPtrs reset in declaration order (which is dependency-ordered in the
    // header: leaf resources first, root device last) when the object goes out
    // of scope.  Nothing else worth explicitly releasing for the minimal repro.
}



void D3D12RaytracingClusteredGeometry::OnDeviceLost()
{
    OnDestroy();
}

void D3D12RaytracingClusteredGeometry::OnDeviceRestored()
{
    CreateDeviceDependentResources();
}

// =====================================================================================
// Screenshot capture
// =====================================================================================





// OnKeyDown is a virtual override from DXSample base.  Headless repro
// has no interactive keys, so this is just a no-op stub.
void D3D12RaytracingClusteredGeometry::OnKeyDown(UINT8) {}
