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
#include "CompiledShaders\\AnimateBall.hlsl.h"
#include "CompiledShaders\\FillInstantiateArgs.hlsl.h"
#include "CompiledShaders\\FillMoveClusterArgs.hlsl.h"
#include "CompiledShaders\\FillBlasFromClasArgs.hlsl.h"
#include "CompiledShaders\\FillClasFromTrianglesArgs.hlsl.h"
#include "CompiledShaders\\FillClusterTemplateArgs.hlsl.h"
#include "SceneData.h"
#include "MaterialData.h"

#include <DirectXMath.h>
#include <algorithm>
#include <chrono>
#include <cmath>

using namespace std;
using namespace DX;
using namespace DirectX;
using Microsoft::WRL::ComPtr;

// ==== Shader entry-point names (must match Raytracing.hlsl) ====
const wchar_t* D3D12RaytracingClusteredGeometry::c_raygenName            = L"RayGen";
// Two specialised closesthit shaders + two primary hit groups:
//   OpaqueHitGroup binds OpaqueHit, NO any-hit (any-hit dispatch is
//   skipped entirely for opaque-only instances - significantly cheaper
//   for chrome / pure-matte traversals).
//   GlassHitGroup binds GlassHit + GlassAnyHit (anyhit handles
//   stochastic translucency; closesthit handles Fresnel + refraction).
const wchar_t* D3D12RaytracingClusteredGeometry::c_opaqueClosestHitName  = L"OpaqueHit";
const wchar_t* D3D12RaytracingClusteredGeometry::c_glassClosestHitName   = L"GlassHit";
const wchar_t* D3D12RaytracingClusteredGeometry::c_glassAnyHitName       = L"GlassAnyHit";
const wchar_t* D3D12RaytracingClusteredGeometry::c_missName              = L"Miss";
const wchar_t* D3D12RaytracingClusteredGeometry::c_shadowMissName        = L"ShadowMiss";
const wchar_t* D3D12RaytracingClusteredGeometry::c_opaqueHitGroupName    = L"OpaqueHitGroup";
const wchar_t* D3D12RaytracingClusteredGeometry::c_glassHitGroupName     = L"GlassHitGroup";
const wchar_t* D3D12RaytracingClusteredGeometry::c_shadowHitGroupName    = L"ShadowHitGroup";

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
    DXSample::ParseCommandLineArgs(argv, argc);
    for (int i = 1; i < argc; i++)
    {
        if (_wcsicmp(argv[i], L"--screenshot") == 0 && i + 2 < argc)
        {
            m_screenshotFrame = _wtoi(argv[i + 1]);
            m_screenshotPath  = argv[i + 2];
            i += 2;
        }
        else if (_wcsicmp(argv[i], L"--screenshot-at") == 0 && i + 2 < argc)
        {
            // --screenshot-at <seconds> <path> : jump m_animSeconds to <seconds>
            // on frame 0 and capture once the swap chain has warmed up. Useful
            // for verifying the time-based camera orbit at known angles
            // (orbit period is 30s).
            m_screenshotAtSeconds = _wtof(argv[i + 1]);
            m_screenshotPath      = argv[i + 2];
            i += 2;
        }
        else if (_wcsicmp(argv[i], L"--vertex-format") == 0 && i + 1 < argc)
        {
            if      (_wcsicmp(argv[i+1], L"float")      == 0) m_vertexMode = VertexMode::Float32_3;
            else if (_wcsicmp(argv[i+1], L"compressed") == 0) m_vertexMode = VertexMode::Compressed1;
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--clas-alloc") == 0 && i + 1 < argc)
        {
            // CLAS memory-allocation strategy. See ClasAllocMode in the header for
            // the contract of each mode; the [CLAS mem] log line at init time
            // reports the stats that change between them.
            if      (_wcsicmp(argv[i+1], L"implicit")  == 0) m_clasAllocMode = ClasAllocMode::Implicit;
            else if (_wcsicmp(argv[i+1], L"get-sizes") == 0) m_clasAllocMode = ClasAllocMode::GetSizes;
            else if (_wcsicmp(argv[i+1], L"compact")   == 0) m_clasAllocMode = ClasAllocMode::Compact;
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--position-truncate") == 0 && i + 1 < argc)
        {
            // FLOAT32_3 mode only - clamp the per-vertex position mantissa to
            // 23-N bits by zeroing the low N. Range 0 (full precision, default)
            // to 22 (only sign+exponent kept). Sweet spot for sub-mm scenes is
            // 8-12. Ignored under --vertex-format compressed.
            int n = _wtoi(argv[i+1]);
            if (n < 0)  n = 0;
            if (n > 22) n = 22;
            m_positionTruncateBits = (UINT)n;
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--compressed-bits") == 0 && i + 1 < argc)
        {
            // COMPRESSED1 mode only - bits per component for the shared-exponent
            // quantizer. Range 1 (extreme - 1 bit each axis) to 16 (max).
            // Same value the [/] runtime slider drives.  Ignored under
            // --vertex-format float.
            int n = _wtoi(argv[i+1]);
            if (n < 1)  n = 1;
            if (n > 16) n = 16;
            m_compressedBitsPerComponent = (UINT)n;
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--aa-samples") == 0 && i + 1 < argc)
        {
            // Anti-aliasing samples per pixel.  The raygen shader traces N
            // jittered primary rays per pixel and averages.  N must be 1,
            // 2, or 4 (clamped to nearest valid).  Default is 4.
            int n = _wtoi(argv[i+1]);
            if (n <= 1) n = 1;
            else if (n <= 2) n = 2;
            else n = 4;
            m_aaSamplesPerPixel = (UINT)n;
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--cluster-tint") == 0 && i + 1 < argc)
        {
            // Cluster-rainbow tint blend in [0..1].  0 = pure material
            // colour; 1 = original "cluster rainbow dominates everything"
            // look.  Default 0.3 leaves a visible hint of cluster
            // boundaries without overpowering the material palette.
            float t = (float)_wtof(argv[i+1]);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            m_clusterTint = t;
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

    // Register an info-queue callback so D3D12 debug-layer messages get
    // mirrored into our SampleLog file before any break-on-severity fires.
    // Without this, the debug layer calls __debugbreak() and (if no debugger
    // is attached) the process exits with STATUS_BREAKPOINT (0xC0000005-ish)
    // without us ever seeing WHY - which makes WARP and other layered failures
    // very hard to diagnose. The callback runs in-process before the break.
    //
    // Also mute id=1328 (CREATERESOURCE_STATE_IGNORED) - this is a benign
    // D3D12 quirk: buffer resources are always created in COMMON state
    // regardless of pInitialState, and the debug layer reminds you per
    // CreateCommittedResource. Filtering it via SetMessageFilter() leaves
    // every other warning category active, so we still see real bugs.
    {
        ComPtr<ID3D12InfoQueue> infoQueue0;
        if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&infoQueue0))))
        {
            D3D12_MESSAGE_ID denied[] = {
                D3D12_MESSAGE_ID_CREATERESOURCE_STATE_IGNORED, // 1328
            };
            D3D12_INFO_QUEUE_FILTER filter = {};
            filter.DenyList.NumIDs  = _countof(denied);
            filter.DenyList.pIDList = denied;
            HRESULT hrFilter = infoQueue0->AddStorageFilterEntries(&filter);
            SampleLog::LogF(L"InfoQueue: muted id=1328 hr=0x%08X\n", (unsigned)hrFilter);
        }

        ComPtr<ID3D12InfoQueue1> infoQueue1;
        if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&infoQueue1))))
        {
            DWORD cookie = 0;
            HRESULT hrReg = infoQueue1->RegisterMessageCallback(
                [](D3D12_MESSAGE_CATEGORY /*cat*/,
                   D3D12_MESSAGE_SEVERITY sev,
                   D3D12_MESSAGE_ID id,
                   LPCSTR desc, void* /*ctx*/)
                {
                    // Mute the persistent benign id=1328 in the callback path
                    // too (storage-filter doesn't reach the callback).
                    if (id == D3D12_MESSAGE_ID_CREATERESOURCE_STATE_IGNORED) return;
                    const wchar_t* sevStr =
                        sev == D3D12_MESSAGE_SEVERITY_CORRUPTION ? L"CORRUPTION" :
                        sev == D3D12_MESSAGE_SEVERITY_ERROR      ? L"ERROR" :
                        sev == D3D12_MESSAGE_SEVERITY_WARNING    ? L"WARNING" :
                        sev == D3D12_MESSAGE_SEVERITY_INFO       ? L"INFO" :
                                                                   L"MESSAGE";
                    // Convert UTF8 description to wide for SampleLog.
                    int wlen = MultiByteToWideChar(CP_UTF8, 0, desc, -1, nullptr, 0);
                    std::wstring wdesc(wlen > 0 ? wlen - 1 : 0, L'\0');
                    if (wlen > 0)
                        MultiByteToWideChar(CP_UTF8, 0, desc, -1, wdesc.data(), wlen);
                    SampleLog::LogF(L"[D3D12 %s id=%d] %s\n", sevStr, (int)id, wdesc.c_str());
                },
                D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                nullptr, &cookie);
            SampleLog::LogF(L"InfoQueue1 callback registered: hr=0x%08X cookie=%u\n",
                            (unsigned)hrReg, (unsigned)cookie);
        }
        else
        {
            SampleLog::Write(L"InfoQueue1 not available - D3D12 messages won't be mirrored to log\n");
        }
    }

    QueryDXR2Support();
    SampleLog::Write(L">>> BuildScene\n");
    BuildScene();
    SampleLog::Write(L">>> BuildMaterials\n");
    BuildMaterials();
    // CreateAnimationComputePipeline must run BEFORE BuildAccelerationStructures
    // because the latter invokes UpdateAnimatedObjectPerFrame on the init path,
    // which dispatches the AnimateBall compute shader -- needs the PSO + RS.
    SampleLog::Write(L">>> CreateAnimationComputePipeline\n");
    CreateAnimationComputePipeline();
    SampleLog::Write(L">>> CreateFillInstantiateArgsPipeline\n");
    CreateFillInstantiateArgsPipeline();
    SampleLog::Write(L">>> CreateFillMoveArgsPipeline\n");
    CreateFillMoveArgsPipeline();
    SampleLog::Write(L">>> CreateFillBlasArgsPipeline\n");
    CreateFillBlasArgsPipeline();
    SampleLog::Write(L">>> CreateFillClasTriArgsPipeline\n");
    CreateFillClasTriArgsPipeline();
    SampleLog::Write(L">>> CreateFillTemplateArgsPipeline\n");
    CreateFillTemplateArgsPipeline();
    SampleLog::Write(L">>> BuildAccelerationStructures\n");
    BuildAccelerationStructures();
    SampleLog::Write(L">>> BuildClusterShaderSideBuffers\n");
    BuildClusterShaderSideBuffers();
    SampleLog::Write(L">>> BuildClusterMetadata\n");
    BuildClusterMetadata();
    SampleLog::Write(L">>> CreateRaytracingPipelineAndShaderTables\n");
    CreateRaytracingPipelineAndShaderTables();
    SampleLog::Write(L">>> CreateDescriptorHeapAndRaytracingOutput\n");
    CreateDescriptorHeapAndRaytracingOutput();
    SampleLog::Write(L">>> CreateUIFont\n");
    CreateUIFont();
    SampleLog::Write(L">>> CreateDeviceDependentResources DONE\n");
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

    if (!m_clustersAndPtlasSupported)
    {
        // Cluster builds will hard-fault later on this adapter.  Pop a
        // user-visible dialog with the suggested workaround and exit
        // cleanly rather than letting the app silently die mid-init.
        const wchar_t* message =
            L"This sample requires a D3D12 adapter that reports "
            L"ClustersAndPTLASSupported=YES (D3D12 Raytracing Tier 1.4 "
            L"DXR2 cluster feature).\n\n"
            L"The current adapter does not support it.\n\n"
            L"You can run the sample on the WARP software adapter by "
            L"enabling it globally with the developer-mode command:\n\n"
            L"    d3dconfig device force-warp=true\n\n"
            L"(re-run d3dconfig with force-warp=false to switch back).\n\n"
            L"WARP renders correctly but is much slower than real hardware.";
        MessageBoxW(Win32Application::GetHwnd(),
                    message,
                    L"D3D12RaytracingClusteredGeometry: unsupported adapter",
                    MB_ICONERROR | MB_OK);
        ExitProcess(1);
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
    // (COMPRESSED1 precision used to be a constexpr 12 here; promoted to the
    // m_compressedBitsPerComponent member so the '[' / ']' keys can cycle it
    // live in COMPRESSED1 mode.  EncodeCompressedClusters() reads the live
    // member, so re-calling it after a slider change re-encodes everything.)
    // GENERIC scene-build pass.  Iterates over the pure-data scene
    // definition (SceneData::BuildSceneDefinition()) and dispatches to
    // the right ProceduralGeometry generator per ObjectSpec.  All art /
    // material / placement / per-cluster-checker config lives in
    // SceneData.cpp - this function has zero hardcoded geometry numbers
    // and zero per-object branches outside the GenKind switch.
    using SD = SceneData::ObjectSpec;
    const auto scene = SceneData::BuildSceneDefinition();

    auto genMesh = [](const SD& s) -> ProceduralGeometry::Mesh
    {
        switch (s.kind)
        {
        case SceneData::GenKind::UVSphere:
            return ProceduralGeometry::GenerateUVSphereSpatialTiles(
                s.args.sphereRadius, s.args.sphereNumLat, s.args.sphereNumLong,
                s.args.sphereTileLat, s.args.sphereTileLong, s.firstClusterID);
        case SceneData::GenKind::Torus:
            return ProceduralGeometry::GenerateTorusSpatialTiles(
                s.args.torusMajor, s.args.torusMinor,
                s.args.torusRingSegs, s.args.torusSideSegs,
                s.args.torusTileRing, s.args.torusTileSide, s.firstClusterID);
        case SceneData::GenKind::Cube:
            return ProceduralGeometry::GenerateCubeSpatialTiles(
                s.args.cubeHalfExtent, s.args.cubeFaceSubdiv,
                s.args.cubeTileSize, s.firstClusterID);
        case SceneData::GenKind::Slab:
            return ProceduralGeometry::GeneratePlaneSpatialTiles(
                s.args.slabHalfSizeU, s.args.slabHalfSizeV,
                s.args.slabTilesU, s.args.slabTilesV,
                s.args.slabTileQuadsU, s.args.slabTileQuadsV,
                s.firstClusterID, s.args.slabThickness);
        case SceneData::GenKind::Klein:
            return ProceduralGeometry::GenerateKleinBottleSpatialTiles(
                s.args.kleinScale, s.args.kleinNumU, s.args.kleinNumV,
                s.args.kleinTileUSize, s.args.kleinTileVSize, s.firstClusterID);
        }
        // Unreachable - all enum cases handled above.
        return ProceduralGeometry::Mesh{};
    };

    for (const SD& s : scene)
    {
        ClusterObject obj;
        obj.mesh          = genMesh(s);
        obj.worldPos      = s.pos;
        obj.worldScale    = s.scale;
        obj.worldRotEuler = s.rotEuler;
        obj.instanceID    = s.instanceID;
        obj.checker       = s.checker;
        obj.surfTintMul   = s.surfTintMul;
        obj.refrTintMul   = s.refrTintMul;
        obj.reflTintMul   = s.reflTintMul;
        obj.nonOrientable = s.nonOrientable;
        m_objects.push_back(std::move(obj));
    }

    // Determine per-cluster offsets in the global cluster array (used by the
    // BLAS-from-CLAS builds to slice the global CLAS-address array per-object).
    UINT runningOffset = 0;
    for (auto& obj : m_objects)
    {
        obj.globalClusterStart = runningOffset;
        obj.clusterCount       = (UINT)obj.mesh.clusters.size();
        runningOffset         += obj.clusterCount;
    }
    m_totalClusterCount = runningOffset;
    m_totalTriangleCount = 0;
    for (const auto& obj : m_objects)
        m_totalTriangleCount += obj.mesh.totalTriangles;
    // m_animatedObject.mesh.totalTriangles is added later (after animated
    // mesh is generated in BuildAnimatedObjectSetup).

    SampleLog::LogF(L"\n[scene] %zu objects, %u total clusters\n",
                    m_objects.size(), m_totalClusterCount);
    UINT objIdx = 0;
    for (const auto& obj : m_objects)
    {
        SampleLog::LogF(L"  object[%u]: %u clusters, %u tris, %u verts at (%.2f,%.2f,%.2f) scale %.2f\n",
                        objIdx++, obj.clusterCount, obj.mesh.totalTriangles, obj.mesh.totalVertices,
                        obj.worldPos.x, obj.worldPos.y, obj.worldPos.z, obj.worldScale);
    }

    SampleLog::LogF(L"\n[vertex format] %s\n",
                    m_vertexMode == VertexMode::Compressed1 ? L"COMPRESSED1 (shared-exponent quantized)"
                                                            : L"FLOAT32_3 (no quantization)");

    // ============================================================================
    // COMPRESSED1 STATIC PATH - NVIDIA DRIVER BUG (open as of 2026-05-15)
    // ----------------------------------------------------------------------------
    // SUMMARY: The exact same compressed1 byte stream produced by this sample's
    // encoder renders correctly on experimental WARP and incorrectly on NVIDIA
    // (RTX 4090, D3D12Core 1.10 preview, agility SDK 722). On NVIDIA, the cube
    // renders cleanly but sphere/torus clusters are mangled: one cluster appears
    // as a stretched "tail" reaching well beyond the object's bounds, an
    // adjacent cluster goes missing, the rest of the scene renders correctly.
    //
    // EVIDENCE:
    //   1. Force-warp=true: all 7 objects (animated sphere + 4 static spheres +
    //      torus + cube) render pixel-equivalent to the FLOAT32_3 path.
    //   2. Force-warp=false (NVIDIA): same input bytes, same args -> broken.
    //   3. CPU-side Compressed1::Decode is bit-exact (max error ~0.0002 units,
    //      sub-quantization-step).
    //   4. Byte-for-byte cluster dumps via DUMP_COMPRESSED1_DIAG match the
    //      d3d12conf reference encoder's header layout and bitstream packing
    //      (see Compressed1.h header for the side-by-side derivation).
    //   5. Breakage on NVIDIA persists across every variable I tried:
    //        - 8 / 12 / 16 bits/axis
    //        - uniform vs per-axis bit counts
    //        - 16-byte vs 256-byte vertex-buffer alignment
    //        - positive-only anchors (mesh shifted to +x +y +z)
    //        - MaxCompressedClusterPositionsSize exact vs 4x oversize
    //        - UPLOAD heap vs DEFAULT heap for the vertex buffer
    //
    // CONCLUSION: The bug is in NVIDIA's COMPRESSED1 BVH-build implementation,
    // not in this sample. Filed as: <TODO bug-tracker link>. Until resolved,
    // the sample defaults to FLOAT32_3 (VertexMode::Float32_3 in the header);
    // pass --vertex-format compressed to exercise the broken path against a
    // future NVIDIA driver update.
    // ============================================================================
    EncodeCompressedClusters();
    // FLOAT32_3 path uses obj.mesh.clusters[i].positions directly at upload
    // time; obj.rawPositions is unused and intentionally left empty.  Log
    // the equivalent byte count so the user can compare paths at a glance.
    {
        size_t totalBytes = 0;
        for (const auto& obj : m_objects)
            for (const auto& c : obj.mesh.clusters)
                totalBytes += c.positions.size() * sizeof(ProceduralGeometry::float3);
        SampleLog::LogF(L"[float32_3] %u clusters: %zu bytes total\n",
                        m_totalClusterCount, totalBytes);
    }
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
    auto device           = m_deviceResources->GetD3DDevice();
    auto commandList      = m_deviceResources->GetCommandList();
    auto commandAllocator = m_deviceResources->GetCommandAllocator();
    auto commandQueue     = m_deviceResources->GetCommandQueue();

    // ---- Create the build-timestamp query heap + readback buffer. We straddle
    // each ExecuteIndirectRTASOperations / classic TLAS build with two EndQuery
    // calls so we can report per-operation wall-clock times to the title bar. ----
    {
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type     = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count    = kBuildTimestampCount;
        qhd.NodeMask = 0;
        ThrowIfFailed(device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&m_buildQueryHeap)));
        m_buildQueryHeap->SetName(L"AS build timestamp query heap");

        auto rbHeap  = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto bufDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(UINT64) * kBuildTimestampCount);
        ThrowIfFailed(device->CreateCommittedResource(&rbHeap, D3D12_HEAP_FLAG_NONE,
            &bufDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_buildQueryReadback)));
        m_buildQueryReadback->SetName(L"AS build timestamp readback");

        ThrowIfFailed(commandQueue->GetTimestampFrequency(&m_timestampFrequency));
        SampleLog::LogF(L"[stats] timestamp frequency: %llu ticks/sec\n",
                        (unsigned long long)m_timestampFrequency);
    }

    // Per-frame timestamp heap + readback. 3 ring slots * 6 timestamps each.
    // Per-frame writes to one slot, resolves to its matching range in the
    // readback buffer; CPU reads from the slot ~3 frames behind the write
    // (safe past GPU completion).
    {
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type  = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = kPerFrameTsPerSlot * kPerFrameRingSlots;
        ThrowIfFailed(device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&m_pfQueryHeap)));
        m_pfQueryHeap->SetName(L"AS per-frame timestamp heap");

        auto rbHeap  = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto bufDesc = CD3DX12_RESOURCE_DESC::Buffer(
            sizeof(UINT64) * kPerFrameTsPerSlot * kPerFrameRingSlots);
        ThrowIfFailed(device->CreateCommittedResource(&rbHeap, D3D12_HEAP_FLAG_NONE,
            &bufDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_pfQueryReadback)));
        m_pfQueryReadback->SetName(L"AS per-frame timestamp readback");
    }

    ThrowIfFailed(commandAllocator->Reset());
    ThrowIfFailed(commandList->Reset(commandAllocator, nullptr));

    UploadClusterInputs();

    // Stamp slot 0 (before CLAS build).
    commandList->EndQuery(m_buildQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    BuildClasIndirect();
    // Stamp slot 1 (after CLAS UAV barrier).
    commandList->EndQuery(m_buildQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);

    // Stamp slot 2 (before BLAS build).
    commandList->EndQuery(m_buildQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
    BuildBlasFromClasIndirect();
    // Stamp slot 3 (after BLAS UAV barrier).
    commandList->EndQuery(m_buildQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);

    // Animated object: build cluster templates once, then do a first
    // per-frame instantiate+BLAS so the TLAS instance points at valid BVH
    // contents from frame 0. (Subsequent frames re-run UpdateAnimatedObjectPerFrame
    // from DoRender.) NOTE: BuildAnimatedObjectSetup itself flushes the cmd
    // list partway through (CPU readback of the template GVA array), so it
    // must run BEFORE the per-build EndQuery pair below or the resolve at
    // bottom would resolve mid-flushed timestamps.
    BuildAnimatedObjectSetup();
    UpdateAnimatedObjectPerFrame();

    // Stamp slot 4 (before TLAS build).
    commandList->EndQuery(m_buildQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 4);
    BuildTlasClassic();
    // Stamp slot 5 (after TLAS UAV barrier).
    commandList->EndQuery(m_buildQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 5);

    // Resolve into the readback buffer. NumQueries=6 because we only wrote slots 0..5
    // (the heap has spare capacity for future per-frame timestamps).
    commandList->ResolveQueryData(m_buildQueryHeap.Get(),
        D3D12_QUERY_TYPE_TIMESTAMP, 0, 6,
        m_buildQueryReadback.Get(), 0);

    m_deviceResources->ExecuteCommandList();
    m_deviceResources->WaitForGpu();

    ReadBuildTimestamps();
    DumpClusterStatsAsync();

    // Initial snapshot of overlay stats so the first frame of rendering shows
    // the correct numbers (subsequent config changes refresh via the
    // CaptureOverlayStatsSnapshot call at the bottom of
    // RebuildStaticAccelerationStructures).
    CaptureOverlayStatsSnapshot();
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
void D3D12RaytracingClusteredGeometry::RebuildStaticAccelerationStructures(const wchar_t* reason)
{
    SampleLog::LogF(L"\n>>> RebuildStaticAccelerationStructures (%s)\n", reason ? reason : L"?");

    auto commandList      = m_deviceResources->GetCommandList();
    auto commandAllocator = m_deviceResources->GetCommandAllocator();

    // 1) Flush the GPU so the in-flight frame finishes reading from CLAS/BLAS/TLAS
    //    before we tear them down.  ComPtr<>::Reset() releases the underlying
    //    ID3D12Resource - if the GPU is still touching it the runtime errors out.
    m_deviceResources->WaitForGpu();

    // 2) Drop all CLAS-related GPU resources.  Each mode's BuildClas* will
    //    reallocate these via ComPtr assignment (which releases stale slots).
    //    Explicitly clearing here makes the tear-down explicit and is the
    //    correct hygiene for the Compact mode (which keeps m_clasMoveArgsBuffer
    //    alive across function returns).
    m_clasResultBuffer.Reset();
    m_clasScratchBuffer.Reset();
    m_clasAddressArray.Reset();
    m_clasSizeArray.Reset();
    m_clasMoveArgsBuffer.Reset();
    m_clasArgsBuffer.Reset();
    m_clasArgsMetaBuffer.Reset();
    m_clasArgsArrayGPUVA = 0;
    m_clasArgsStride     = 0;
    m_totalClasBytes     = 0;
    m_clasMemStats       = ClasMemStats{};

    // 3) Drop static-object BLAS storage.  Each ClusterObject's blasStorage
    //    will be reallocated by BuildBlasFromClasIndirect.  The animated
    //    object's BLAS is owned by m_animatedObject and intentionally
    //    untouched (its GVA must remain valid for the TLAS instance desc).
    for (auto& obj : m_objects)
    {
        obj.blasStorage.Reset();
        obj.blasGPUVA = 0;
    }
    m_blasScratchBuffer.Reset();
    m_blasArgsBuffer.Reset();
    m_blasArgsMeta.Reset();
    m_blasResultAddrBuffer.Reset();

    // 4) Drop TLAS storage.  Per-frame rebuild path keys off these, so they
    //    MUST exist before the next OnRender; BuildTlasClassic recreates them.
    m_tlasBuffer.Reset();
    m_tlasScratchBuffer.Reset();
    m_tlasInstanceDescs.Reset();

    // 5) Reset the command list/allocator and re-run the static build chain.
    //    UploadClusterInputs reads the live m_vertexMode; BuildClasIndirect
    //    reads the live m_clasAllocMode - so cycling either field via the
    //    keyboard before this call is sufficient.  EncodeCompressedClusters
    //    is unconditional so the 'v' toggle to COMPRESSED1 and the '[' / ']'
    //    precision slider both pick up the latest m_compressedBitsPerComponent.
    ThrowIfFailed(commandAllocator->Reset());
    ThrowIfFailed(commandList->Reset(commandAllocator, nullptr));

    EncodeCompressedClusters();
    UploadClusterInputs();
    BuildClasIndirect();
    BuildBlasFromClasIndirect();
    // Animated path also picks up the new precision: BuildAnimatedObjectSetup's
    // per-cluster TrianglesArgs.PositionTruncateBitCount comes straight from
    // m_positionTruncateBits, so rebuilding the templates is what makes the
    // slider visibly affect the showpiece ball.  Setup does its own internal
    // flush+wait for the template-GVA readback; cmd list is reset afterwards
    // and we continue recording into the fresh list for the TLAS rebuild.
    if (m_animatedObjectEnabled)
    {
        BuildAnimatedObjectSetup();
        UpdateAnimatedObjectPerFrame();
    }
    BuildTlasClassic();

    m_deviceResources->ExecuteCommandList();
    m_deviceResources->WaitForGpu();

    // 6) Refresh the stats that feed the title bar.  m_totalClasBytes is the
    //    live result buffer size; the per-build CPU-wall stats already landed
    //    in m_clasMemStats via the relevant BuildClas* function.
    if (m_clasResultBuffer) m_totalClasBytes = m_clasResultBuffer->GetDesc().Width;

    SampleLog::LogF(L"<<< RebuildStaticAccelerationStructures done.  CLAS=%.1f MB, scratch=%.1f MB\n",
                    m_totalClasBytes / (1024.0 * 1024.0),
                    m_clasMemStats.scratchBytesPhase1 / (1024.0 * 1024.0));

    // Snapshot all the displayed numbers (except per-frame ms) into
    // m_overlayStats so the overlay reads from a cached struct instead of
    // poking GetDesc().Width every frame.  Per-frame timing gets snapped
    // a few frames later, once the ring buffer refills - see Tick().
    CaptureOverlayStatsSnapshot();
}

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
void D3D12RaytracingClusteredGeometry::BuildMaterials()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Pure copy from the data table.  The actual MaterialDesc values
    // live in MaterialData.cpp - this engine code never touches the
    // baseColor / refl / refr / ior numbers directly.
    std::copy(MaterialData::kMaterials.begin(),
              MaterialData::kMaterials.end(),
              m_materials.begin());

    AllocateUploadBuffer(device, m_materials.data(),
                         m_materials.size() * sizeof(MaterialDesc),
                         &m_materialsBuffer, L"Per-instance materials");

    SampleLog::LogF(L"[materials] %u slots loaded from MaterialData::kMaterials\n",
                    (unsigned)m_materials.size());
}


void D3D12RaytracingClusteredGeometry::BuildClusterShaderSideBuffers()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Animated sphere clusters are reachable via templated INSTANTIATE with
    // ClusterIdOffset = 800 (see BuildAnimatedObjectSetup), so the GPU-side
    // ClusterID() seen on hits is template_id + 800.  Mirror that offset
    // here when registering the animated mesh's clusters.
    const UINT kAnimatedClusterIdOffset = 800;

    // Pass 1: figure out the offset-table size.
    UINT maxClusterID = 0;
    for (const auto& obj : m_objects)
        for (const auto& c : obj.mesh.clusters)
            maxClusterID = std::max(maxClusterID, c.clusterID);
    if (m_animatedObjectEnabled)
        for (const auto& c : m_animatedObject.mesh.clusters)
            maxClusterID = std::max(maxClusterID, c.clusterID + kAnimatedClusterIdOffset);

    const UINT offsetTableSize = maxClusterID + 1;

    std::vector<XMFLOAT4> normals;            // 16 bytes per normal (.w padding) to match HLSL's
                                              // ByteAddressBuffer.Load3((off.x + i_k) * 16) stride.
                                              // See Raytracing.hlsl declaration of g_clusterNormals.
    std::vector<UINT>     indices;
    std::vector<XMUINT2>  offsets(offsetTableSize, XMUINT2(0u, 0u));

    // Helper: append one cluster's data and stamp its offset entry.
    auto appendCluster = [&](const ProceduralGeometry::Cluster& c, UINT cidForOffsetTable)
    {
        offsets[cidForOffsetTable] = XMUINT2((UINT)normals.size(), (UINT)indices.size());
        for (const auto& n : c.normals)
            normals.push_back(XMFLOAT4(n.x, n.y, n.z, 0.0f));
        for (uint8_t i : c.indices)
            indices.push_back((UINT)i);
    };

    for (const auto& obj : m_objects)
        for (const auto& c : obj.mesh.clusters)
            appendCluster(c, c.clusterID);

    if (m_animatedObjectEnabled)
        for (const auto& c : m_animatedObject.mesh.clusters)
            appendCluster(c, c.clusterID + kAnimatedClusterIdOffset);

    AllocateUploadBuffer(device, normals.data(), normals.size() * sizeof(DirectX::XMFLOAT4),
                         &m_clusterNormalsBuffer, L"Cluster vertex normals (per-vertex side channel)");
    AllocateUploadBuffer(device, indices.data(), indices.size() * sizeof(UINT),
                         &m_clusterIndicesBuffer, L"Cluster indices (uint32-widened side channel)");
    AllocateUploadBuffer(device, offsets.data(), offsets.size() * sizeof(XMUINT2),
                         &m_clusterOffsetsBuffer, L"Cluster vert/idx offset table");

    m_clusterNormalsCount = (UINT)normals.size();
    m_clusterIndicesCount = (UINT)indices.size();
    m_clusterOffsetsCount = (UINT)offsets.size();

    SampleLog::LogF(L"[normals] side channel: %u vertex normals, %u indices, %u offset slots (max cid=%u, ~%u KB total)\n",
                    m_clusterNormalsCount, m_clusterIndicesCount, m_clusterOffsetsCount,
                    maxClusterID,
                    (UINT)((normals.size() * sizeof(XMFLOAT3) +
                            indices.size() * sizeof(UINT) +
                            offsets.size() * sizeof(XMUINT2)) / 1024));
}


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
void D3D12RaytracingClusteredGeometry::BuildClusterMetadata()
{
    auto device = m_deviceResources->GetD3DDevice();

    const UINT kAnimatedClusterIdOffset = 800;

    // Pass 1: same size discovery as BuildClusterShaderSideBuffers.
    UINT maxClusterID = 0;
    for (const auto& obj : m_objects)
        for (const auto& c : obj.mesh.clusters)
            maxClusterID = std::max(maxClusterID, c.clusterID);
    if (m_animatedObjectEnabled)
        for (const auto& c : m_animatedObject.mesh.clusters)
            maxClusterID = std::max(maxClusterID, c.clusterID + kAnimatedClusterIdOffset);

    const UINT metaCount = maxClusterID + 1;
    std::vector<ClusterMeta> meta(metaCount, ClusterMeta{});

    // Initialize all slots with neutral defaults: no override, colorIndex=cid,
    // baseColorScale=1, tint mul defaults that reproduce the legacy look.
    for (UINT i = 0; i < metaCount; ++i)
    {
        meta[i].colorIndex     = i;
        meta[i].flags          = 0u;
        meta[i].overrideRefl   = -1.0f;
        meta[i].overrideRefr   = -1.0f;
        meta[i].overrideIor    = -1.0f;
        meta[i].baseColorScale = 1.0f;
        meta[i].surfTintMul    = 1.0f;
        meta[i].refrTintMul    = 0.50f;
        meta[i].reflTintMul    = 1.08f;   // = 0.70 / 0.65 (default clusterTint)
    }

    // Generic per-cluster fill - works on any object type that has a
    // mesh + optional checker config + tint multipliers (since the
    // animated sphere uses a different storage class than ClusterObject
    // but otherwise carries the same per-cluster info on its mesh).
    auto fillFromMesh = [&](const ProceduralGeometry::Mesh& mesh,
                            UINT cidOffset,
                            const CheckerConfig& checker,
                            float surfTintMul, float refrTintMul, float reflTintMul)
    {
        for (const auto& c : mesh.clusters)
        {
            const UINT id = c.clusterID + cidOffset;
            ClusterMeta& m = meta[id];

            m.colorIndex     = c.matchedColorCid + cidOffset;
            m.flags          = c.flags;
            m.surfTintMul    = surfTintMul;
            m.refrTintMul    = refrTintMul;
            m.reflTintMul    = reflTintMul;

            if (checker.enabled)
            {
                const bool isOdd = ((c.gridU + c.gridV) & 1u) != 0u;
                const auto& side = isOdd ? checker.oddParity
                                         : checker.evenParity;
                m.overrideRefl   = side.overrideRefl;
                m.overrideRefr   = side.overrideRefr;
                m.overrideIor    = side.overrideIor;
                m.baseColorScale = side.baseColorScale;
            }
        }
    };

    for (const auto& obj : m_objects)
        fillFromMesh(obj.mesh, 0u, obj.checker,
                     obj.surfTintMul, obj.refrTintMul, obj.reflTintMul);
    if (m_animatedObjectEnabled)
    {
        // Animated object: alternating cluster checker - even parity
        // is OPAQUE SHINY (chrome-like; refl=0.95, refr=0), odd parity
        // keeps the baseline refractive glass.  Plus cranked tint
        // multipliers so the refractive clusters' cluster colours read
        // prominently.
        CheckerConfig checker;
        checker.enabled = true;
        checker.evenParity.overrideRefl = 0.95f;   // chrome-ish reflectivity
        checker.evenParity.overrideRefr = 0.0f;    // opaque (no refraction)
        checker.evenParity.baseColorScale = 1.0f;
        checker.oddParity.overrideRefl  = 0.0f;    // translucent clusters: no reflection
        checker.oddParity.overrideRefr  = -1.0f;   // keep baseline refractivity (0.78)
        checker.oddParity.overrideIor   = -1.0f;   // keep baseline ior (2.4)
        fillFromMesh(m_animatedObject.mesh, kAnimatedClusterIdOffset,
                     checker, /*surf*/1.0f, /*refr*/0.75f, /*refl*/1.20f);
    }

    AllocateUploadBuffer(device, meta.data(), meta.size() * sizeof(ClusterMeta),
                         &m_clusterMetaBuffer, L"Per-cluster GENERIC metadata (ClusterMeta[])");
    m_clusterMetaCount = (UINT)meta.size();

    SampleLog::LogF(L"[clustermeta] %u entries (~%u KB)\n",
                    m_clusterMetaCount,
                    (UINT)((meta.size() * sizeof(ClusterMeta)) / 1024));
}



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
        slots[gIdx].ibSize   = src.indices.size();
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
            memcpy(mapped + slots[gIdx].ibOffset, src.indices.data(), src.indices.size());
    }
    m_clusterInputBuffer->Unmap(0, nullptr);

    // ------------------------------------------------------------------
    // Per-cluster metadata for the FillClasFromTrianglesArgs CS.
    // 24 bytes/cluster: see ClasArgsMeta layout in FillClasFromTrianglesArgs.hlsl.
    //   { clusterID, triCount, vertCount, vbOff, ibOff, opaqueFlag }
    // CPU computes this from mesh + material data once; CS expands it
    // into the 80-byte D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS
    // every time we (re)build.
    // ------------------------------------------------------------------
    {
        struct ClasArgsMeta {
            UINT clusterID, triCount, vertCount, vbOff, ibOff, opaqueFlag;
        };
        static_assert(sizeof(ClasArgsMeta) == 24, "must match the HLSL load offsets");

        std::vector<ClasArgsMeta> meta(m_totalClusterCount);
        gIdx = 0;
        for (const auto& obj : m_objects)
        for (size_t i = 0; i < obj.mesh.clusters.size(); ++i, ++gIdx)
        {
            const auto& src = obj.mesh.clusters[i];
            const auto& mat = m_materials[obj.instanceID];
            const bool isOpaqueLike = (mat.translucency == 0.0f) && (mat.refractivity == 0.0f);
            meta[gIdx].clusterID  = src.clusterID;
            meta[gIdx].triCount   = (UINT)(src.indices.size() / 3);
            meta[gIdx].vertCount  = (UINT)src.positions.size();
            meta[gIdx].vbOff      = (UINT)slots[gIdx].vbOffset;
            meta[gIdx].ibOff      = (UINT)slots[gIdx].ibOffset;
            meta[gIdx].opaqueFlag = isOpaqueLike
                ? (UINT)D3D12_RTAS_CLUSTERED_GEOMETRY_FLAG_OPAQUE
                : 0u;
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
    SampleLog::LogF(L"[CLAS alloc-mode] %s; position-truncate-bits=%u (FLOAT32_3 only)\n",
                    ClasAllocModeName(), m_positionTruncateBits);
    m_clasMemStats = ClasMemStats{};        // reset stats for this run
    switch (m_clasAllocMode)
    {
    case ClasAllocMode::Implicit: BuildClasImplicit(); break;
    case ClasAllocMode::GetSizes: BuildClasGetSizes(); break;
    case ClasAllocMode::Compact:  BuildClasCompact();  break;
    }
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
    outLimits.MaxUniqueGeometryIndexAndFlagsCountPerCluster = 1;
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
    outClasDesc.IndexFormat                         = D3D12_INDEX_FORMAT_UINT8;
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
void D3D12RaytracingClusteredGeometry::BuildClasGetSizes()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto cmdList = m_deviceResources->GetCommandList();
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
    D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC clasDescBase;
    BuildSharedClusterTrianglesInputs(*this, N, useFloat, maxTris, maxVerts,
                                      totalTris, totalVerts, maxCompressedSize,
                                      limits, clasDescBase);

    // ---- Phase 1: GET_SIZES pass ----
    // No result buffer needed; driver writes per-cluster bytes-needed into a
    // UAV size buffer. m_clasSizeArray is reused as the destination.
    auto clasDescSizes = clasDescBase;
    clasDescSizes.Mode = D3D12_RTAS_OPERATION_MODE_GET_SIZES;

    D3D12_RTAS_OPERATION_INPUTS opInputsSizes = {};
    opInputsSizes.Type                  = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
    opInputsSizes.pClusterTrianglesDesc = &clasDescSizes;

    D3D12_RTAS_OPERATION_PREBUILD_INFO prebuildSizes = {};
    m_dxr2Device->GetRTASOperationPrebuildInfo(&opInputsSizes, &prebuildSizes);
    SampleLog::LogF(L"[CLAS prebuild get-sizes] result max=%llu bytes, scratch=%llu bytes\n",
                    (unsigned long long)prebuildSizes.ResultDataMaxSizeInBytes,
                    (unsigned long long)prebuildSizes.ScratchDataSizeInBytes);

    // Worst-case result alloc for stats reference - we DON'T allocate it.
    D3D12_RTAS_OPERATION_INPUTS opInputsImplicitProbe = opInputsSizes;
    auto clasDescProbe = clasDescBase;
    clasDescProbe.Mode = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
    opInputsImplicitProbe.pClusterTrianglesDesc = &clasDescProbe;
    D3D12_RTAS_OPERATION_PREBUILD_INFO prebuildImplicitProbe = {};
    m_dxr2Device->GetRTASOperationPrebuildInfo(&opInputsImplicitProbe, &prebuildImplicitProbe);
    m_clasMemStats.resultPrebuildMax  = prebuildImplicitProbe.ResultDataMaxSizeInBytes;
    m_clasMemStats.scratchBytesPhase1 = prebuildSizes.ScratchDataSizeInBytes;

    AllocateUAVBuffer(device, std::max<UINT64>(prebuildSizes.ScratchDataSizeInBytes, 256ull),
                      &m_clasScratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS scratch (get-sizes phase)");
    AllocateUAVBuffer(device, (UINT64)N * sizeof(UINT64),
                      &m_clasSizeArray, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS size array");

    D3D12_RTAS_BATCHED_OPERATION_DATA batchedSizes = {};
    batchedSizes.BatchScratchData      = m_clasScratchBuffer->GetGPUVirtualAddress();
    batchedSizes.ResultSizeArray       = { m_clasSizeArray->GetGPUVirtualAddress(), sizeof(UINT64) };
    batchedSizes.IndirectArgumentArray = { m_clasArgsArrayGPUVA, m_clasArgsStride };

    D3D12_RTAS_OPERATION_DESC opDescSizes = {};
    opDescSizes.Inputs                = opInputsSizes;
    opDescSizes.pBatchedOperationData = &batchedSizes;

    const auto p1_t0 = std::chrono::steady_clock::now();
    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDescSizes,
        D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    auto sizesUav = CD3DX12_RESOURCE_BARRIER::UAV(m_clasSizeArray.Get());
    m_dxrCommandList->ResourceBarrier(1, &sizesUav);

    // Copy size array to a readback heap so the CPU can sum them.
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
    const auto p1_t1 = std::chrono::steady_clock::now();
    m_clasMemStats.cpuWallMsPhase1 = std::chrono::duration<double, std::milli>(p1_t1 - p1_t0).count();

    // Sum sizes + compute per-cluster destination offsets in the about-to-be-
    // allocated exact-sized result buffer. ACCELERATION_STRUCTURE alignment
    // (256B) must be respected for every cluster's start address.
    constexpr UINT64 kClasAlign = D3D12_RAYTRACING_CLAS_BYTE_ALIGNMENT;
    auto alignUp = [](UINT64 x, UINT64 a) { return (x + (a - 1)) & ~(a - 1); };

    void* mapped = nullptr;
    D3D12_RANGE rr = { 0, (SIZE_T)N * sizeof(UINT64) };
    ThrowIfFailed(sizesReadback->Map(0, &rr, &mapped));
    auto* sizes = reinterpret_cast<const UINT64*>(mapped);

    std::vector<UINT64> destOffsets(N);
    UINT64 sumActual = 0, packedTotal = 0;
    for (UINT i = 0; i < N; ++i)
    {
        destOffsets[i] = packedTotal;
        packedTotal   += alignUp(sizes[i], kClasAlign);
        sumActual     += sizes[i];
    }
    D3D12_RANGE noWrite = { 0, 0 };
    sizesReadback->Unmap(0, &noWrite);
    SampleLog::LogF(L"[CLAS get-sizes] sum actual=%llu bytes, packed (256B-aligned)=%llu bytes\n",
                    (unsigned long long)sumActual, (unsigned long long)packedTotal);
    m_clasMemStats.sumActualBytes  = sumActual;
    m_clasMemStats.resultInitialBytes = packedTotal;
    m_clasMemStats.resultFinalBytes   = packedTotal;
    m_clasMemStats.peakResidentBytes  = packedTotal;

    // ---- Phase 2: EXPLICIT_DESTINATIONS build into exact-sized buffer ----
    auto commandAllocator = m_deviceResources->GetCommandAllocator();
    ThrowIfFailed(commandAllocator->Reset());
    ThrowIfFailed(cmdList->Reset(commandAllocator, nullptr));

    AllocateUAVBuffer(device, packedTotal,
                      &m_clasResultBuffer, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                      L"All-cluster CLAS result buffer (get-sizes exact)");

    // Per-cluster destination GVAs go into m_clasAddressArray. EXPLICIT mode
    // reads from this buffer; the driver places each result at the address we
    // specify here. After the build, m_clasAddressArray's contents (which we
    // just wrote on CPU) are the canonical post-build per-cluster GVAs that
    // the BLAS-from-CLAS step needs.
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> destAddrs(N);
    const D3D12_GPU_VIRTUAL_ADDRESS baseGVA = m_clasResultBuffer->GetGPUVirtualAddress();
    for (UINT i = 0; i < N; ++i) destAddrs[i] = baseGVA + destOffsets[i];

    AllocateUploadBuffer(device, destAddrs.data(),
                         destAddrs.size() * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                         &m_clasAddressArray, L"CLAS address array (explicit destinations)");

    auto clasDescBuild = clasDescBase;
    clasDescBuild.Mode = D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS;

    D3D12_RTAS_OPERATION_INPUTS opInputsBuild = {};
    opInputsBuild.Type                  = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
    opInputsBuild.pClusterTrianglesDesc = &clasDescBuild;

    D3D12_RTAS_OPERATION_PREBUILD_INFO prebuildBuild = {};
    m_dxr2Device->GetRTASOperationPrebuildInfo(&opInputsBuild, &prebuildBuild);
    SampleLog::LogF(L"[CLAS prebuild explicit-build] scratch=%llu bytes\n",
                    (unsigned long long)prebuildBuild.ScratchDataSizeInBytes);
    m_clasMemStats.scratchBytesPhase2 = prebuildBuild.ScratchDataSizeInBytes;

    AllocateUAVBuffer(device, std::max<UINT64>(prebuildBuild.ScratchDataSizeInBytes, 256ull),
                      &m_clasScratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS scratch (explicit-build phase)");

    D3D12_RTAS_BATCHED_OPERATION_DATA batchedBuild = {};
    batchedBuild.BatchScratchData      = m_clasScratchBuffer->GetGPUVirtualAddress();
    batchedBuild.ResultAddressArray    = { m_clasAddressArray->GetGPUVirtualAddress(), sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    batchedBuild.ResultSizeArray       = { m_clasSizeArray->GetGPUVirtualAddress(),    sizeof(UINT64) };
    batchedBuild.IndirectArgumentArray = { m_clasArgsArrayGPUVA, m_clasArgsStride };

    D3D12_RTAS_OPERATION_DESC opDescBuild = {};
    opDescBuild.Inputs                = opInputsBuild;
    opDescBuild.pBatchedOperationData = &batchedBuild;

    const auto p2_t0 = std::chrono::steady_clock::now();
    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDescBuild,
        D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    // NB: m_clasAddressArray was allocated as an UPLOAD-heap buffer in this
    // mode (read-only INPUT to the EXPLICIT_DESTINATIONS build), so it has no
    // UAV bind flags and must NOT be UAV-barriered. Only the result/size
    // buffers get the barrier.
    D3D12_RESOURCE_BARRIER barriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasResultBuffer.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasSizeArray.Get()),
    };
    m_dxrCommandList->ResourceBarrier(_countof(barriers), barriers);
    const auto p2_t1 = std::chrono::steady_clock::now();
    m_clasMemStats.cpuWallMsPhase2 = std::chrono::duration<double, std::milli>(p2_t1 - p2_t0).count();
}

// ---------------------------------------------------------------------------------
// Compact mode. Single IMPLICIT build (worst-case alloc) -- with the size
// array attached so we get per-cluster actual sizes for free -- then a CPU
// readback + MOVE_CLUSTER_OBJECTS in IMPLICIT mode that compacts the live
// CLAS into a tightly-sized buffer. After the move the old (worst-case)
// result buffer is released; what remains is the compacted buffer.
// ---------------------------------------------------------------------------------
void D3D12RaytracingClusteredGeometry::BuildClasCompact()
{
    auto device  = m_deviceResources->GetD3DDevice();
    auto cmdList = m_deviceResources->GetCommandList();
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
    D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC clasDescBase;
    BuildSharedClusterTrianglesInputs(*this, N, useFloat, maxTris, maxVerts,
                                      totalTris, totalVerts, maxCompressedSize,
                                      limits, clasDescBase);
    clasDescBase.Mode = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;

    D3D12_RTAS_OPERATION_INPUTS opInputs = {};
    opInputs.Type                  = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
    opInputs.pClusterTrianglesDesc = &clasDescBase;

    D3D12_RTAS_OPERATION_PREBUILD_INFO prebuild = {};
    m_dxr2Device->GetRTASOperationPrebuildInfo(&opInputs, &prebuild);
    SampleLog::LogF(L"[CLAS prebuild compact phase1] result max=%llu, scratch=%llu bytes\n",
                    (unsigned long long)prebuild.ResultDataMaxSizeInBytes,
                    (unsigned long long)prebuild.ScratchDataSizeInBytes);
    m_clasMemStats.resultPrebuildMax  = prebuild.ResultDataMaxSizeInBytes;
    m_clasMemStats.resultInitialBytes = prebuild.ResultDataMaxSizeInBytes;
    m_clasMemStats.scratchBytesPhase1 = prebuild.ScratchDataSizeInBytes;

    // Phase 1 buffers - this is the implicit/worst-case build.
    ComPtr<ID3D12Resource> phase1ResultBuffer;
    AllocateUAVBuffer(device, prebuild.ResultDataMaxSizeInBytes,
                      &phase1ResultBuffer, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                      L"CLAS phase1 (worst-case) result buffer");
    AllocateUAVBuffer(device, std::max<UINT64>(prebuild.ScratchDataSizeInBytes, 256ull),
                      &m_clasScratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS scratch (compact phase1)");
    ComPtr<ID3D12Resource> phase1AddressArray;
    AllocateUAVBuffer(device, (UINT64)N * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                      &phase1AddressArray, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS phase1 address array");
    AllocateUAVBuffer(device, (UINT64)N * sizeof(UINT64),
                      &m_clasSizeArray, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS size array");

    D3D12_RTAS_BATCHED_OPERATION_DATA batched = {};
    batched.BatchResultData           = phase1ResultBuffer->GetGPUVirtualAddress();
    batched.BatchScratchData          = m_clasScratchBuffer->GetGPUVirtualAddress();
    batched.ResultAddressArray        = { phase1AddressArray->GetGPUVirtualAddress(), sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    batched.ResultSizeArray           = { m_clasSizeArray->GetGPUVirtualAddress(),    sizeof(UINT64) };
    batched.IndirectArgumentArray     = { m_clasArgsArrayGPUVA, m_clasArgsStride };

    D3D12_RTAS_OPERATION_DESC opDesc = {};
    opDesc.Inputs                = opInputs;
    opDesc.pBatchedOperationData = &batched;

    const auto p1_t0 = std::chrono::steady_clock::now();
    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDesc,
        D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

    D3D12_RESOURCE_BARRIER barriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(phase1ResultBuffer.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(phase1AddressArray.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasSizeArray.Get()),
    };
    m_dxrCommandList->ResourceBarrier(_countof(barriers), barriers);

    // Read sizes back so we know how big the compacted buffer needs to be.
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
    const auto p1_t1 = std::chrono::steady_clock::now();
    m_clasMemStats.cpuWallMsPhase1 = std::chrono::duration<double, std::milli>(p1_t1 - p1_t0).count();

    // Sum sizes -> compacted total. The move op packs sequentially with 256B
    // alignment per element.
    constexpr UINT64 kClasAlign = D3D12_RAYTRACING_CLAS_BYTE_ALIGNMENT;
    auto alignUp = [](UINT64 x, UINT64 a) { return (x + (a - 1)) & ~(a - 1); };

    void* mapped = nullptr;
    D3D12_RANGE rr = { 0, (SIZE_T)N * sizeof(UINT64) };
    ThrowIfFailed(sizesReadback->Map(0, &rr, &mapped));
    auto* sizes = reinterpret_cast<const UINT64*>(mapped);
    UINT64 sumActual = 0, packedTotal = 0;
    for (UINT i = 0; i < N; ++i)
    {
        sumActual   += sizes[i];
        packedTotal += alignUp(sizes[i], kClasAlign);
    }
    D3D12_RANGE noWrite = { 0, 0 };
    sizesReadback->Unmap(0, &noWrite);
    SampleLog::LogF(L"[CLAS compact] sum actual=%llu bytes, packed (256B-aligned)=%llu bytes (vs worst-case %llu)\n",
                    (unsigned long long)sumActual, (unsigned long long)packedTotal,
                    (unsigned long long)prebuild.ResultDataMaxSizeInBytes);
    m_clasMemStats.sumActualBytes  = sumActual;
    m_clasMemStats.resultFinalBytes = packedTotal;
    // Peak = worst-case + compacted (both live during MOVE).
    m_clasMemStats.peakResidentBytes =
        prebuild.ResultDataMaxSizeInBytes + packedTotal;

    // ---- Phase 2: MOVE_CLUSTER_OBJECTS in IMPLICIT_DESTINATIONS ----
    // Pack into a fresh tightly-sized buffer; driver writes new per-cluster
    // addresses into a new m_clasAddressArray. After the move, phase1 buffers
    // are released - only the compacted result remains live.
    auto commandAllocator = m_deviceResources->GetCommandAllocator();
    ThrowIfFailed(commandAllocator->Reset());
    ThrowIfFailed(cmdList->Reset(commandAllocator, nullptr));

    // Compacted result buffer + new address array (output of MOVE).
    AllocateUAVBuffer(device, packedTotal,
                      &m_clasResultBuffer, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                      L"All-cluster CLAS result buffer (compacted)");
    AllocateUAVBuffer(device, (UINT64)N * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                      &m_clasAddressArray, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS address array (post-compact)");

    // MOVE_CLUSTER_OBJECTS args (one u64 GVA per cluster) are GPU-filled by
    // FillMoveClusterArgs CS from phase1AddressArray.  Previously the CPU
    // round-tripped these addresses through a readback + upload; the CS
    // copies them GPU-to-GPU.  m_clasMoveArgsBuffer is a member so it
    // outlives this function call (the queued cmd list reads from it until
    // BuildAccelerationStructures flushes at the bottom).
    AllocateUAVBuffer(device, (UINT64)N * sizeof(D3D12_RTAS_OPERATION_MOVE_CLUSTER_OBJECTS_ARGS),
                      &m_clasMoveArgsBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS move args (GPU-filled)");
    {
        cmdList->SetComputeRootSignature(m_fillMoveArgsRS.Get());
        cmdList->SetPipelineState(m_fillMoveArgsPSO.Get());
        cmdList->SetComputeRoot32BitConstants(0, 1, &N, 0);
        cmdList->SetComputeRootUnorderedAccessView(1, phase1AddressArray->GetGPUVirtualAddress());
        cmdList->SetComputeRootUnorderedAccessView(2, m_clasMoveArgsBuffer->GetGPUVirtualAddress());
        const UINT groups = (N + 63) / 64;
        cmdList->Dispatch(groups, 1, 1);
        D3D12_RESOURCE_BARRIER uav =
            CD3DX12_RESOURCE_BARRIER::UAV(m_clasMoveArgsBuffer.Get());
        cmdList->ResourceBarrier(1, &uav);
    }

    D3D12_RTAS_CLUSTER_MOVES_DESC movesDesc = {};
    movesDesc.MaxArgCount    = N;
    movesDesc.Mode           = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
    movesDesc.Type           = D3D12_RTAS_MOVE_OPERATION_TYPE_CLUSTER_LEVEL_ACCELERATION_STRUCTURE;
    movesDesc.MaxBytesMoved  = packedTotal;
    movesDesc.Flags          = D3D12_RTAS_CLUSTER_MOVE_OPERATION_FLAG_NONE;

    D3D12_RTAS_OPERATION_INPUTS moveInputs = {};
    moveInputs.Type              = D3D12_RTAS_OPERATION_TYPE_MOVE_CLUSTER_OBJECTS;
    moveInputs.pClusterMovesDesc = &movesDesc;

    D3D12_RTAS_OPERATION_PREBUILD_INFO movePrebuild = {};
    m_dxr2Device->GetRTASOperationPrebuildInfo(&moveInputs, &movePrebuild);
    SampleLog::LogF(L"[CLAS prebuild compact phase2 (move)] result max=%llu, scratch=%llu bytes\n",
                    (unsigned long long)movePrebuild.ResultDataMaxSizeInBytes,
                    (unsigned long long)movePrebuild.ScratchDataSizeInBytes);
    m_clasMemStats.scratchBytesPhase2 = movePrebuild.ScratchDataSizeInBytes;

    AllocateUAVBuffer(device, std::max<UINT64>(movePrebuild.ScratchDataSizeInBytes, 256ull),
                      &m_clasScratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS scratch (compact move)");

    D3D12_RTAS_BATCHED_OPERATION_DATA batchedMove = {};
    batchedMove.BatchResultData       = m_clasResultBuffer->GetGPUVirtualAddress();
    batchedMove.BatchScratchData      = m_clasScratchBuffer->GetGPUVirtualAddress();
    batchedMove.ResultAddressArray    = { m_clasAddressArray->GetGPUVirtualAddress(), sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    batchedMove.ResultSizeArray       = { m_clasSizeArray->GetGPUVirtualAddress(),    sizeof(UINT64) };
    batchedMove.IndirectArgumentArray = { m_clasMoveArgsBuffer->GetGPUVirtualAddress(),
                                          sizeof(D3D12_RTAS_OPERATION_MOVE_CLUSTER_OBJECTS_ARGS) };

    D3D12_RTAS_OPERATION_DESC moveOpDesc = {};
    moveOpDesc.Inputs                = moveInputs;
    moveOpDesc.pBatchedOperationData = &batchedMove;

    const auto p2_t0 = std::chrono::steady_clock::now();
    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &moveOpDesc,
        D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    D3D12_RESOURCE_BARRIER moveBarriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasResultBuffer.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasAddressArray.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasSizeArray.Get()),
    };
    m_dxrCommandList->ResourceBarrier(_countof(moveBarriers), moveBarriers);

    // Flush + wait the move op before this function returns. Without this,
    // the queued move work + the still-OPEN cmd list interact badly with the
    // next mid-flush in BuildAnimatedObjectSetup (we get an
    // ID3D12CommandAllocator::Reset error - "previous executions associated
    // with the allocator have not completed"), even with a UAV barrier
    // between the move and the BLAS-from-CLAS that consumes its output.
    // Treating the compact-pass as a self-contained init step (build+move+
    // wait) is the natural shape and matches how a streaming asset pipeline
    // would issue compactions anyway. Cost: roughly 1ms wall-clock on this
    // hardware - this is the price of doing exact-fit allocation under a
    // post-build compaction model.
    m_deviceResources->ExecuteCommandList();
    m_deviceResources->WaitForGpu();
    auto cmdAlloc = m_deviceResources->GetCommandAllocator();
    ThrowIfFailed(cmdAlloc->Reset());
    ThrowIfFailed(cmdList->Reset(cmdAlloc, nullptr));

    const auto p2_t1 = std::chrono::steady_clock::now();
    m_clasMemStats.cpuWallMsPhase2 = std::chrono::duration<double, std::milli>(p2_t1 - p2_t0).count();

    // phase1ResultBuffer + phase1AddressArray go out of scope here; their
    // ComPtrs drop their last ref; the GPU is done with them (we just waited).
    // Final memory cost is m_clasResultBuffer (compacted) + scratch + addr/size.
}

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
void D3D12RaytracingClusteredGeometry::BuildAnimatedObjectSetup()
{
    auto device      = m_deviceResources->GetD3DDevice();
    auto& obj        = m_animatedObject;

    // ------------------------------------------------------------------
    // 1) Mesh + hints. Sphere with spatial cluster tiles. Hints = rest * envelopeScale
    //    so animation can squash/stretch within [1/envelope, envelope] without
    //    leaving the cluster's pre-built BVH bounds.
    //
    //    INVARIANT: kEnvelopeScale MUST stay above (1 + kAnimWobbleAmp) -- the
    //    template's per-cluster AABB and (for a future COMPRESSED1
    //    instantiation path) the Compressed1TemplateHeader quantization range
    //    are both seeded from these hint positions.  If the animation worst-
    //    case scale (1 + amp) overflows kEnvelopeScale, FLOAT32_3 per-frame
    //    INSTANTIATE will clip against InstantiationBoundingBoxLimit and
    //    COMPRESSED1 INSTANTIATE will saturate at the header's max
    //    representable value -- both silently corrupt geometry.  The
    //    HLSL-side wobble amplitude lives in AnimateBall.hlsl::kWobbleAmp and
    //    is mirrored here ONLY for this assert.  Keep the two in sync.
    // ------------------------------------------------------------------
    constexpr float kRestRadius    = 1.00f;       // central showpiece ball, bigger
    constexpr float kEnvelopeScale = 1.18f;       // hint sphere radius = rest * 1.18
    constexpr float kAnimWobbleAmp = 0.08f;       // MUST match AnimateBall.hlsl::kWobbleAmp
    static_assert(1.0f + kAnimWobbleAmp < kEnvelopeScale,
        "kEnvelopeScale must leave headroom over the animation's max radial "
        "displacement (1 + kAnimWobbleAmp).  See AnimateBall.hlsl::kWobbleAmp -- "
        "if you raise the wobble amplitude there, bump kEnvelopeScale here.");
    // 2x cluster count per dim (192 -> 768 clusters: 24 lat x 32 long)
    // while KEEPING per-cluster tris count at 48 - resolution doubled
    // in each dim (48x96 -> 96x192), tile size kept (tileLat 4, tileLong 6).
    obj.mesh = ProceduralGeometry::GenerateUVSphereSpatialTiles(
        kRestRadius, /*numLat*/96, /*numLong*/192, /*tileLat*/4, /*tileLong*/6, 0);
    obj.clusterCount         = (UINT)obj.mesh.clusters.size();
    obj.worldPos             = XMFLOAT3(0.0f, 1.10f, 0.0f);        // hovers above the hex group, dropped lower so refractions through it pick up the floor + objects below
    obj.worldScale           = 1.0f;
    obj.instanceID           = 7;       // matches NUM_MATERIAL_SLOTS-1; this is the refractive glass slot
    obj.vertexBufferStride   = (UINT)sizeof(XMFLOAT3);

    // Per-cluster hint positions = rest position scaled outward by envelopeScale.
    obj.hintPositions.resize(obj.clusterCount);
    obj.totalVertexCount = 0;
    for (UINT c = 0; c < obj.clusterCount; ++c)
    {
        const auto& src = obj.mesh.clusters[c];
        obj.hintPositions[c].resize(src.positions.size());
        for (size_t v = 0; v < src.positions.size(); ++v)
        {
            obj.hintPositions[c][v] = {
                src.positions[v].x * kEnvelopeScale,
                src.positions[v].y * kEnvelopeScale,
                src.positions[v].z * kEnvelopeScale,
            };
        }
        obj.totalVertexCount += (UINT)src.positions.size();
        obj.maxVertsPerCluster = std::max(obj.maxVertsPerCluster, (UINT)src.positions.size());
        obj.maxTrisPerCluster  = std::max(obj.maxTrisPerCluster,  (UINT)(src.indices.size() / 3));
    }
    SampleLog::LogF(L"\n[animated] sphere mesh: %u clusters, %u verts, %u tris (envelope x%.2f)\n",
                    obj.clusterCount, obj.totalVertexCount,
                    obj.mesh.totalTriangles, kEnvelopeScale);

    // Roll the animated mesh's triangles into the scene-wide total now
    // that the mesh has been generated; the title bar reads from this.
    // Recompute from scratch each time because BuildAnimatedObjectSetup is
    // now re-invoked by RebuildStaticAccelerationStructures on A/V/[/]
    // toggles -- a naive '+=' would keep stacking the animated tri count.
    m_totalTriangleCount = obj.mesh.totalTriangles;
    for (const auto& o : m_objects)
        m_totalTriangleCount += o.mesh.totalTriangles;

    // ------------------------------------------------------------------
    // 2) Upload buffer: hint vertex data + index data + template build args.
    //    Layout (all aligned to 16):
    //        [cluster 0 hint vertices][cluster 0 indices] ... [cluster N-1 ...]
    //        [per-cluster D3D12_RTAS_OPERATION_BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES_ARGS]
    // ------------------------------------------------------------------
    constexpr size_t kAlign = 16;
    auto alignTo = [](size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); };

    struct Slot { size_t vbOffset, vbSize, ibOffset, ibSize; };
    std::vector<Slot> slots(obj.clusterCount);
    size_t cursor = 0;
    for (UINT c = 0; c < obj.clusterCount; ++c)
    {
        const auto& src = obj.mesh.clusters[c];
        cursor = alignTo(cursor, kAlign);
        slots[c].vbOffset = cursor;
        slots[c].vbSize   = src.positions.size() * sizeof(XMFLOAT3);
        cursor += slots[c].vbSize;
        cursor = alignTo(cursor, kAlign);
        slots[c].ibOffset = cursor;
        slots[c].ibSize   = src.indices.size();
        cursor += slots[c].ibSize;
    }
    cursor = alignTo(cursor, kAlign);
    const size_t argsOffset = cursor;
    const size_t argStride  = sizeof(D3D12_RTAS_OPERATION_BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES_ARGS);
    cursor += obj.clusterCount * argStride;
    // Layout: vertex+index data lives in obj.templateInputBuffer (upload heap,
    // CPU-filled here).  Args live in a separate DEFAULT/UAV buffer
    // (obj.templateArgsBuffer below) filled by FillClusterTemplateArgs CS --
    // mirrors the static-CLAS-args refactor.
    const size_t totalDataSize = alignTo(cursor, 256);

    auto uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto bufDesc    = CD3DX12_RESOURCE_DESC::Buffer(totalDataSize);
    ThrowIfFailed(device->CreateCommittedResource(
        &uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&obj.templateInputBuffer)));
    obj.templateInputBuffer->SetName(L"Animated: cluster-template input buffer (vert+idx)");

    uint8_t* mapped = nullptr;
    CD3DX12_RANGE noRead(0, 0);
    ThrowIfFailed(obj.templateInputBuffer->Map(0, &noRead, reinterpret_cast<void**>(&mapped)));
    const D3D12_GPU_VIRTUAL_ADDRESS templateInputBaseGPUVA =
        obj.templateInputBuffer->GetGPUVirtualAddress();

    for (UINT c = 0; c < obj.clusterCount; ++c)
    {
        memcpy(mapped + slots[c].vbOffset, obj.hintPositions[c].data(), slots[c].vbSize);
        memcpy(mapped + slots[c].ibOffset, obj.mesh.clusters[c].indices.data(), slots[c].ibSize);
    }
    obj.templateInputBuffer->Unmap(0, nullptr);

    // Per-cluster metadata for FillClusterTemplateArgs CS.  16 bytes/cluster:
    //   { triCount, vertCount, vbOff, ibOff }
    // (ClusterID is implicit = thread index; opaque flag always 0 for
    // animated; stride is hardcoded to 12 in the shader.)
    {
        struct TplArgsMeta { UINT triCount, vertCount, vbOff, ibOff; };
        static_assert(sizeof(TplArgsMeta) == 16, "must match HLSL Load4 stride");

        std::vector<TplArgsMeta> meta(obj.clusterCount);
        for (UINT c = 0; c < obj.clusterCount; ++c)
        {
            const auto& src = obj.mesh.clusters[c];
            meta[c].triCount  = (UINT)(src.indices.size() / 3);
            meta[c].vertCount = (UINT)src.positions.size();
            meta[c].vbOff     = (UINT)slots[c].vbOffset;
            meta[c].ibOff     = (UINT)slots[c].ibOffset;
        }
        AllocateUploadBuffer(device, meta.data(), meta.size() * sizeof(TplArgsMeta),
            &obj.templateMetaBuffer, L"Animated: template args metadata");
    }

    // GPU-written args buffer (DEFAULT/UAV).
    {
        const UINT64 argsBytes = (UINT64)obj.clusterCount * sizeof(D3D12_RTAS_OPERATION_BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES_ARGS);
        AllocateUAVBuffer(device, argsBytes,
            &obj.templateArgsBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            L"Animated: template args (GPU-filled)");
    }

    // Dispatch FillClusterTemplateArgs.
    {
        auto cmdList = m_deviceResources->GetCommandList();
        cmdList->SetComputeRootSignature(m_fillTemplateArgsRS.Get());
        cmdList->SetPipelineState(m_fillTemplateArgsPSO.Get());
        struct {
            UINT baseLo, baseHi;
            UINT count;
            UINT posTruncBits;
        } cb;
        cb.baseLo       = (UINT)(templateInputBaseGPUVA & 0xFFFFFFFFu);
        cb.baseHi       = (UINT)(templateInputBaseGPUVA >> 32);
        cb.count        = obj.clusterCount;
        cb.posTruncBits = m_positionTruncateBits;
        cmdList->SetComputeRoot32BitConstants(0, sizeof(cb) / 4, &cb, 0);
        cmdList->SetComputeRootShaderResourceView(1, obj.templateMetaBuffer->GetGPUVirtualAddress());
        cmdList->SetComputeRootUnorderedAccessView(2, obj.templateArgsBuffer->GetGPUVirtualAddress());
        const UINT groups = (obj.clusterCount + 63) / 64;
        cmdList->Dispatch(groups, 1, 1);
        D3D12_RESOURCE_BARRIER uav =
            CD3DX12_RESOURCE_BARRIER::UAV(obj.templateArgsBuffer.Get());
        cmdList->ResourceBarrier(1, &uav);
    }

    obj.templateArgsArrayGPUVA = obj.templateArgsBuffer->GetGPUVirtualAddress();
    obj.templateArgsStride     = (UINT)sizeof(D3D12_RTAS_OPERATION_BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES_ARGS);

    // ------------------------------------------------------------------
    // 3) BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES. We compute prebuild info,
    //    allocate the result+scratch+address buffers, fire one batched op for
    //    all N clusters in IMPLICIT_DESTINATIONS mode.
    // ------------------------------------------------------------------
    D3D12_RTAS_CLUSTER_LIMITS limits = {};
    limits.MaxArgCount                                   = obj.clusterCount;
    limits.MaxGeometryIndexValue                         = 0;
    limits.MaxUniqueGeometryIndexAndFlagsCountPerCluster = 1;
    limits.MaxTriangleCountPerCluster                    = obj.maxTrisPerCluster;
    limits.MaxVertexCountPerCluster                      = obj.maxVertsPerCluster;
    limits.MaxTotalTriangleCount                         = obj.mesh.totalTriangles;
    limits.MaxTotalVertexCount                           = obj.totalVertexCount;

    D3D12_RTAS_CLUSTER_TEMPLATE_TRIANGLES_INPUTS_DESC tplDesc = {};
    tplDesc.ClusterLimits             = limits;
    tplDesc.Flags                     = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE | D3D12_RTAS_OPERATION_FLAG_ALLOW_DATA_ACCESS;
    tplDesc.Mode                      = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
    tplDesc.VertexHintFormat          = D3D12_VERTEX_FORMAT_FLOAT32_3;
    tplDesc.VertexInstantiationFormat = D3D12_VERTEX_FORMAT_FLOAT32_3;
    tplDesc.IndexFormat               = D3D12_INDEX_FORMAT_UINT8;
    tplDesc.GeometryIndexAndFlagsIndexFormat = D3D12_INDEX_FORMAT_NONE;
    tplDesc.OpacityMicromapIndexFormat       = D3D12_INDEX_FORMAT_NONE;

    D3D12_RTAS_OPERATION_INPUTS opInputsTpl = {};
    opInputsTpl.Type                          = D3D12_RTAS_OPERATION_TYPE_BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES;
    opInputsTpl.pClusterTemplateTrianglesDesc = &tplDesc;

    D3D12_RTAS_OPERATION_PREBUILD_INFO prebuildTpl = {};
    m_dxr2Device->GetRTASOperationPrebuildInfo(&opInputsTpl, &prebuildTpl);
    SampleLog::LogF(L"[animated template prebuild] %u clusters: result=%llu, scratch=%llu bytes\n",
                    obj.clusterCount,
                    (unsigned long long)prebuildTpl.ResultDataMaxSizeInBytes,
                    (unsigned long long)prebuildTpl.ScratchDataSizeInBytes);

    AllocateUAVBuffer(device, prebuildTpl.ResultDataMaxSizeInBytes,
                      &obj.templateResultBuffer, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                      L"Animated: cluster templates result buffer");
    AllocateUAVBuffer(device, std::max<UINT64>(prebuildTpl.ScratchDataSizeInBytes, 256ull),
                      &obj.templateScratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"Animated: cluster templates scratch");
    AllocateUAVBuffer(device, (UINT64)obj.clusterCount * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                      &obj.templateAddressArray, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"Animated: cluster templates address array");

    D3D12_RTAS_BATCHED_OPERATION_DATA batchedTpl = {};
    batchedTpl.BatchResultData       = obj.templateResultBuffer->GetGPUVirtualAddress();
    batchedTpl.BatchScratchData      = obj.templateScratchBuffer->GetGPUVirtualAddress();
    batchedTpl.ResultAddressArray    = { obj.templateAddressArray->GetGPUVirtualAddress(),
                                         sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    batchedTpl.IndirectArgumentArray = { obj.templateArgsArrayGPUVA, obj.templateArgsStride };

    D3D12_RTAS_OPERATION_DESC opDescTpl = {};
    opDescTpl.Inputs                = opInputsTpl;
    opDescTpl.pBatchedOperationData = &batchedTpl;
    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDescTpl,
        D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

    D3D12_RESOURCE_BARRIER tplBarriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(obj.templateResultBuffer.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(obj.templateAddressArray.Get()),
    };
    m_dxrCommandList->ResourceBarrier(_countof(tplBarriers), tplBarriers);

    // ------------------------------------------------------------------
    // 4) Per-frame resources (allocated once, contents rewritten every frame).
    //    perFrameVertexBuffer  - GPU-only DEFAULT-heap UAV.  Written by
    //                            AnimateBall.cs each frame, read by
    //                            INSTANTIATE_CLUSTER_TEMPLATES.  Cycles
    //                            UAV <-> NON_PIXEL_SHADER_RESOURCE inside
    //                            UpdateAnimatedObjectPerFrame.
    //    restPositionsBuffer   - DEFAULT-heap SRV containing the rest-pose
    //                            positions, uploaded once below.
    //    perFrameInstArgsBuffer- one INSTANTIATE_CLUSTER_TEMPLATES_ARGS per cluster
    //    perFrameClasResultBuffer + scratch + addresses (IMPLICIT_DESTINATIONS)
    //    blasStorage + scratch + args  (EXPLICIT_DESTINATIONS for fixed GPU VA)
    // ------------------------------------------------------------------
    const size_t vbBytes = (size_t)obj.totalVertexCount * sizeof(XMFLOAT3);
    {
        // GPU-only animated vertex buffer (UAV; the compute shader writes;
        // INSTANTIATE reads).  Starts in UNORDERED_ACCESS so the first frame's
        // compute pass works without an initial state transition.
        auto defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        auto vbDesc = CD3DX12_RESOURCE_DESC::Buffer(alignTo(vbBytes, 256),
                                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ThrowIfFailed(device->CreateCommittedResource(
            &defaultHeap, D3D12_HEAP_FLAG_NONE, &vbDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&obj.perFrameVertexBuffer)));
        obj.perFrameVertexBuffer->SetName(L"Animated: per-frame vertex buffer (UAV, compute-written)");
    }
    {
        // Rest-pose positions: upload once from CPU into a DEFAULT-heap buffer.
        // Read by AnimateBall.cs via a raw SRV bound as a root descriptor.
        // Done as a self-contained ExecuteCommandList+WaitForGpu pair so the
        // upload staging buffer can die at the end of this block (the existing
        // template-GVA readback flush further down would also do, but a
        // self-contained upload reads more straightforwardly).
        auto defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        auto restDesc = CD3DX12_RESOURCE_DESC::Buffer(alignTo(vbBytes, 256));
        ThrowIfFailed(device->CreateCommittedResource(
            &defaultHeap, D3D12_HEAP_FLAG_NONE, &restDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&obj.restPositionsBuffer)));
        obj.restPositionsBuffer->SetName(L"Animated: rest-pose positions (SRV, compute-read)");

        Microsoft::WRL::ComPtr<ID3D12Resource> staging;
        auto stageDesc = CD3DX12_RESOURCE_DESC::Buffer(alignTo(vbBytes, 256));
        ThrowIfFailed(device->CreateCommittedResource(
            &uploadHeap, D3D12_HEAP_FLAG_NONE, &stageDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&staging)));
        staging->SetName(L"Animated: rest-pose staging upload");

        XMFLOAT3* mapped = nullptr;
        ThrowIfFailed(staging->Map(0, &noRead, reinterpret_cast<void**>(&mapped)));
        for (UINT c = 0; c < obj.clusterCount; ++c)
        {
            const auto& src = obj.mesh.clusters[c];
            for (size_t v = 0; v < src.positions.size(); ++v)
            {
                mapped->x = src.positions[v].x;
                mapped->y = src.positions[v].y;
                mapped->z = src.positions[v].z;
                ++mapped;
            }
        }
        staging->Unmap(0, nullptr);

        auto cl = m_deviceResources->GetCommandList();
        cl->CopyBufferRegion(obj.restPositionsBuffer.Get(), 0, staging.Get(), 0, vbBytes);
        auto toSrv = CD3DX12_RESOURCE_BARRIER::Transition(obj.restPositionsBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        cl->ResourceBarrier(1, &toSrv);
        m_deviceResources->ExecuteCommandList();
        m_deviceResources->WaitForGpu();
        // Re-open the command list for the template-GVA readback that follows.
        auto allocator = m_deviceResources->GetCommandAllocator();
        ThrowIfFailed(allocator->Reset());
        ThrowIfFailed(cl->Reset(allocator, nullptr));
        // staging goes out of scope here -- safe, GPU has consumed it.
    }
    // perFrameInstArgsBuffer: per-cluster INSTANTIATE_CLUSTER_TEMPLATES_ARGS.
    // GPU-written via FillInstantiateArgs CS below (no CPU mapping, no
    // CPU-side readback of template GVAs).  Stays in UNORDERED_ACCESS so
    // CS writes + INSTANTIATE reads are separated by a UAV barrier.
    {
        const size_t instArgsBytes = obj.clusterCount * sizeof(D3D12_RTAS_OPERATION_INSTANTIATE_CLUSTER_TEMPLATES_ARGS);
        AllocateUAVBuffer(device, alignTo(instArgsBytes, 256),
            &obj.perFrameInstArgsBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            L"Animated: per-frame instantiate args (GPU-filled)");
    }

    // vertexOffsetArray: per-cluster byte offset into perFrameVertexBuffer.
    // CPU-computed once (cumulative prefix sum of per-cluster vertex sizes)
    // and uploaded; the fill CS reads from it to compute each cluster's
    // VertexBuffer.StartAddress = pfVbGpuVa + vertexOffsets[c].  The
    // template-GVAs input is obj.templateAddressArray (already on GPU --
    // we don't have to round-trip it through CPU readback any more).
    {
        std::vector<UINT> offsets(obj.clusterCount);
        UINT cursor = 0;
        for (UINT c = 0; c < obj.clusterCount; ++c)
        {
            offsets[c] = cursor;
            cursor += (UINT)(obj.mesh.clusters[c].positions.size() * sizeof(XMFLOAT3));
        }
        AllocateUploadBuffer(device, offsets.data(), offsets.size() * sizeof(UINT),
            &obj.vertexOffsetArray, L"Animated: per-cluster VB byte offsets");
    }

    // Dispatch the GPU args-fill CS.  obj.templateAddressArray is still in
    // UNORDERED_ACCESS from the BUILD_CLUSTER_TEMPLATES op above and the
    // UAV barrier we emitted right after that op is sufficient to make its
    // contents visible to this dispatch (we bind it as RWByteAddressBuffer
    // and only read).  After the dispatch we leave a UAV barrier on
    // perFrameInstArgsBuffer so the downstream INSTANTIATE_CLUSTER_TEMPLATES
    // op sees the written args.
    {
        auto cmdList = m_deviceResources->GetCommandList();
        cmdList->SetComputeRootSignature(m_fillInstArgsRS.Get());
        cmdList->SetPipelineState(m_fillInstArgsPSO.Get());

        const D3D12_GPU_VIRTUAL_ADDRESS pfVbGPUVA = obj.perFrameVertexBuffer->GetGPUVirtualAddress();
        struct {
            UINT pfVbLo, pfVbHi;
            UINT clusterCount;
            INT  clusterIdOffset;
            UINT vertexStride;
        } cb;
        cb.pfVbLo           = (UINT)(pfVbGPUVA & 0xFFFFFFFFu);
        cb.pfVbHi           = (UINT)(pfVbGPUVA >> 32);
        cb.clusterCount     = obj.clusterCount;
        cb.clusterIdOffset  = 800;   // matches the CPU baseline used before
        cb.vertexStride     = (UINT)sizeof(XMFLOAT3);
        cmdList->SetComputeRoot32BitConstants(0, sizeof(cb) / 4, &cb, 0);

        // u1 = templateGvas (input as RWByteAddressBuffer; UAV barrier above
        // already separates it from the producing template build).
        cmdList->SetComputeRootUnorderedAccessView(1, obj.templateAddressArray->GetGPUVirtualAddress());
        // t0 = vertexOffsets (upload heap, always GENERIC_READ, includes
        // NON_PIXEL_SHADER_RESOURCE).
        cmdList->SetComputeRootShaderResourceView(2, obj.vertexOffsetArray->GetGPUVirtualAddress());
        // u0 = argsOut (UAV).
        cmdList->SetComputeRootUnorderedAccessView(3, obj.perFrameInstArgsBuffer->GetGPUVirtualAddress());

        const UINT groups = (obj.clusterCount + 63) / 64;
        cmdList->Dispatch(groups, 1, 1);

        D3D12_RESOURCE_BARRIER argsUav =
            CD3DX12_RESOURCE_BARRIER::UAV(obj.perFrameInstArgsBuffer.Get());
        cmdList->ResourceBarrier(1, &argsUav);
    }

    // ------------------------------------------------------------------
    // 5) Per-frame CLAS + BLAS storage. Sized via prebuild info.
    // ------------------------------------------------------------------
    D3D12_RTAS_INSTANTIATE_CLUSTER_TEMPLATE_INPUTS_DESC instDesc = {};
    instDesc.ClusterLimits     = limits;
    instDesc.Flags             = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE | D3D12_RTAS_OPERATION_FLAG_ALLOW_DATA_ACCESS;
    instDesc.Mode              = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
    instDesc.VertexSourceFormat = D3D12_VERTEX_FORMAT_FLOAT32_3;

    D3D12_RTAS_OPERATION_INPUTS opInputsInst = {};
    opInputsInst.Type                            = D3D12_RTAS_OPERATION_TYPE_INSTANTIATE_CLUSTER_TEMPLATES;
    opInputsInst.pInstantiateClusterTemplateDesc = &instDesc;

    D3D12_RTAS_OPERATION_PREBUILD_INFO prebuildInst = {};
    m_dxr2Device->GetRTASOperationPrebuildInfo(&opInputsInst, &prebuildInst);
    SampleLog::LogF(L"[animated instantiate prebuild] result=%llu, scratch=%llu bytes\n",
                    (unsigned long long)prebuildInst.ResultDataMaxSizeInBytes,
                    (unsigned long long)prebuildInst.ScratchDataSizeInBytes);

    AllocateUAVBuffer(device, prebuildInst.ResultDataMaxSizeInBytes,
                      &obj.perFrameClasResultBuffer, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                      L"Animated: per-frame CLAS result");
    AllocateUAVBuffer(device, std::max<UINT64>(prebuildInst.ScratchDataSizeInBytes, 256ull),
                      &obj.perFrameClasScratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"Animated: per-frame CLAS scratch");
    AllocateUAVBuffer(device, (UINT64)obj.clusterCount * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                      &obj.perFrameClasAddressArray, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"Animated: per-frame CLAS address array");
    // Per-cluster output-size buffer for the one-shot measurement INSTANTIATE
    // (only written by MeasureAnimatedClasBytesOneShot at the end of this
    // setup function -- the regular per-frame INSTANTIATE skips ResultSizeArray
    // so there's zero per-frame readback cost).
    AllocateUAVBuffer(device, (UINT64)obj.clusterCount * sizeof(UINT64),
                      &obj.perFrameClasSizesBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"Animated: per-frame CLAS sizes (measurement)");
    {
        const UINT64 rbBytes = (UINT64)obj.clusterCount * sizeof(UINT64);
        auto rbDesc = CD3DX12_RESOURCE_DESC::Buffer(rbBytes);
        auto rbHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        ThrowIfFailed(device->CreateCommittedResource(
            &rbHeap, D3D12_HEAP_FLAG_NONE, &rbDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&obj.perFrameClasSizesReadback)));
        obj.perFrameClasSizesReadback->SetName(L"Animated: per-frame CLAS sizes readback");
    }

    // BLAS-from-CLAS prebuild & alloc (single BLAS, EXPLICIT_DESTINATIONS).
    D3D12_RTAS_CLAS_INPUTS_DESC blasDesc = {};
    blasDesc.Flags              = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;  // ALLOW_DATA_ACCESS is NOT permitted here - it's a per-CLAS property set at the CLAS-from-triangles build above, and the BLAS-from-CLAS must read it consistently across all referenced CLAS
    blasDesc.MaxArgCount        = 1;
    blasDesc.Mode               = D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS;
    blasDesc.MaxTotalClasCount  = obj.clusterCount;
    blasDesc.MaxClasCountPerArg = obj.clusterCount;

    D3D12_RTAS_OPERATION_INPUTS opInputsBlas = {};
    opInputsBlas.Type      = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
    opInputsBlas.pClasDesc = &blasDesc;
    D3D12_RTAS_OPERATION_PREBUILD_INFO prebuildBlas = {};
    m_dxr2Device->GetRTASOperationPrebuildInfo(&opInputsBlas, &prebuildBlas);
    SampleLog::LogF(L"[animated BLAS prebuild] result=%llu, scratch=%llu bytes\n",
                    (unsigned long long)prebuildBlas.ResultDataMaxSizeInBytes,
                    (unsigned long long)prebuildBlas.ScratchDataSizeInBytes);

    AllocateUAVBuffer(device, prebuildBlas.ResultDataMaxSizeInBytes,
                      &obj.blasStorage, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                      L"Animated: BLAS storage");
    AllocateUAVBuffer(device, std::max<UINT64>(prebuildBlas.ScratchDataSizeInBytes, 256ull),
                      &obj.blasScratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"Animated: BLAS scratch");
    obj.blasGPUVA = obj.blasStorage->GetGPUVirtualAddress();

    // BLAS-from-CLAS args (1 entry, never changes).  GPU-filled by
    // FillBlasFromClasArgs CS (same shader as the static path -- the only
    // difference is dispatch count and the input metadata buffer).
    {
        struct BlasArgsMeta { UINT count, gvaLo, gvaHi; };
        const D3D12_GPU_VIRTUAL_ADDRESS clasArrayGva = obj.perFrameClasAddressArray->GetGPUVirtualAddress();
        BlasArgsMeta meta = {
            obj.clusterCount,
            (UINT)(clasArrayGva & 0xFFFFFFFFu),
            (UINT)(clasArrayGva >> 32),
        };
        AllocateUploadBuffer(device, &meta, sizeof(meta),
            &obj.blasArgsMeta, L"Animated: BLAS args metadata");

        AllocateUAVBuffer(device,
            sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS),
            &obj.blasArgsBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            L"Animated: BLAS args (GPU-filled)");

        auto cmdList = m_deviceResources->GetCommandList();
        cmdList->SetComputeRootSignature(m_fillBlasArgsRS.Get());
        cmdList->SetPipelineState(m_fillBlasArgsPSO.Get());
        const UINT cb[2] = { 1u, (UINT)sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
        cmdList->SetComputeRoot32BitConstants(0, 2, cb, 0);
        cmdList->SetComputeRootShaderResourceView(1, obj.blasArgsMeta->GetGPUVirtualAddress());
        cmdList->SetComputeRootUnorderedAccessView(2, obj.blasArgsBuffer->GetGPUVirtualAddress());
        cmdList->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER uav =
            CD3DX12_RESOURCE_BARRIER::UAV(obj.blasArgsBuffer.Get());
        cmdList->ResourceBarrier(1, &uav);
    }
    {
        // BLAS dest-addrs array (1 entry = our fixed BLAS storage GVA).
        AllocateUploadBuffer(device, &obj.blasGPUVA, sizeof(obj.blasGPUVA),
            &obj.blasResultAddrBuffer, L"Animated: BLAS dest addrs");
    }

    m_animatedObjectEnabled = true;
    SampleLog::LogF(L"[animated] setup complete: BLAS at GVA 0x%llx, %u clusters per frame\n",
                    (unsigned long long)obj.blasGPUVA, obj.clusterCount);

    // One-shot per-cluster CLAS size readback - runs synchronously, populates
    // m_overlayStats.animatedPerFrameClasActualBytes.  Costs ~1ms of extra
    // setup work per config change; zero per-frame cost.
    MeasureAnimatedClasBytesOneShot();
}

// =====================================================================================
// One-shot INSTANTIATE_CLUSTER_TEMPLATES with ResultSizeArray hooked up so we
// can read the per-cluster CLAS leaf sizes the driver actually emitted.  This
// is invoked ONCE at the end of BuildAnimatedObjectSetup (which itself runs
// at init + on every config change), so the steady-state per-frame INSTANTIATE
// in UpdateAnimatedObjectPerFrame stays free of readback overhead.
// Stalls the GPU briefly while it copies the size UAV into a readback buffer.
// =====================================================================================
void D3D12RaytracingClusteredGeometry::MeasureAnimatedClasBytesOneShot()
{
    if (!m_animatedObjectEnabled) return;
    auto& obj = m_animatedObject;
    auto device = m_deviceResources->GetD3DDevice();
    auto cmdList = m_deviceResources->GetCommandList();

    // BuildAnimatedObjectSetup left the cmd list OPEN and the perFrameVertexBuffer
    // in UNORDERED_ACCESS state (compute UAV).  The animated compute pass that
    // populates positions hasn't run yet (it runs every frame in
    // UpdateAnimatedObjectPerFrame) -- so the source VB still holds zeros from
    // its initial allocation.  That's fine for *sizing* purposes: INSTANTIATE
    // produces fixed-shape CLAS leaves regardless of vertex values, so the size
    // it reports for an all-zeros input matches the size for any other input.

    // Need to transition perFrameVertexBuffer UAV -> NON_PIXEL_SHADER_RESOURCE
    // before INSTANTIATE, then back to UAV after (matching the per-frame state
    // cycle in UpdateAnimatedObjectPerFrame).
    D3D12_RESOURCE_BARRIER toRead =
        CD3DX12_RESOURCE_BARRIER::Transition(obj.perFrameVertexBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmdList->ResourceBarrier(1, &toRead);

    D3D12_RTAS_CLUSTER_LIMITS limits = {};
    limits.MaxArgCount                                   = obj.clusterCount;
    limits.MaxUniqueGeometryIndexAndFlagsCountPerCluster = 1;
    limits.MaxTriangleCountPerCluster                    = obj.maxTrisPerCluster;
    limits.MaxVertexCountPerCluster                      = obj.maxVertsPerCluster;
    limits.MaxTotalTriangleCount                         = obj.mesh.totalTriangles;
    limits.MaxTotalVertexCount                           = obj.totalVertexCount;

    D3D12_RTAS_INSTANTIATE_CLUSTER_TEMPLATE_INPUTS_DESC instDesc = {};
    instDesc.ClusterLimits      = limits;
    instDesc.Flags              = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE | D3D12_RTAS_OPERATION_FLAG_ALLOW_DATA_ACCESS;
    instDesc.Mode               = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
    instDesc.VertexSourceFormat = D3D12_VERTEX_FORMAT_FLOAT32_3;

    D3D12_RTAS_OPERATION_INPUTS opInputsInst = {};
    opInputsInst.Type                            = D3D12_RTAS_OPERATION_TYPE_INSTANTIATE_CLUSTER_TEMPLATES;
    opInputsInst.pInstantiateClusterTemplateDesc = &instDesc;

    // Same as per-frame INSTANTIATE but with ResultSizeArray populated.
    D3D12_RTAS_BATCHED_OPERATION_DATA batchedInst = {};
    batchedInst.BatchResultData       = obj.perFrameClasResultBuffer->GetGPUVirtualAddress();
    batchedInst.BatchScratchData      = obj.perFrameClasScratchBuffer->GetGPUVirtualAddress();
    batchedInst.ResultAddressArray    = { obj.perFrameClasAddressArray->GetGPUVirtualAddress(),
                                          sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    batchedInst.ResultSizeArray       = { obj.perFrameClasSizesBuffer->GetGPUVirtualAddress(),
                                          sizeof(UINT64) };
    batchedInst.IndirectArgumentArray = { obj.perFrameInstArgsBuffer->GetGPUVirtualAddress(),
                                          sizeof(D3D12_RTAS_OPERATION_INSTANTIATE_CLUSTER_TEMPLATES_ARGS) };

    D3D12_RTAS_OPERATION_DESC opDescInst = {};
    opDescInst.Inputs                = opInputsInst;
    opDescInst.pBatchedOperationData = &batchedInst;
    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDescInst,
        D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

    D3D12_RESOURCE_BARRIER instUav[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(obj.perFrameClasResultBuffer.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(obj.perFrameClasSizesBuffer.Get()),
    };
    cmdList->ResourceBarrier(_countof(instUav), instUav);

    // Copy size UAV -> readback.
    D3D12_RESOURCE_BARRIER sizesToCopy =
        CD3DX12_RESOURCE_BARRIER::Transition(obj.perFrameClasSizesBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmdList->ResourceBarrier(1, &sizesToCopy);
    cmdList->CopyBufferRegion(obj.perFrameClasSizesReadback.Get(), 0,
                              obj.perFrameClasSizesBuffer.Get(), 0,
                              (UINT64)obj.clusterCount * sizeof(UINT64));
    D3D12_RESOURCE_BARRIER sizesBackToUav =
        CD3DX12_RESOURCE_BARRIER::Transition(obj.perFrameClasSizesBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmdList->ResourceBarrier(1, &sizesBackToUav);

    // Restore perFrameVertexBuffer state so the next per-frame compute pass
    // finds it as UAV (matches what UpdateAnimatedObjectPerFrame expects).
    D3D12_RESOURCE_BARRIER backToUav =
        CD3DX12_RESOURCE_BARRIER::Transition(obj.perFrameVertexBuffer.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmdList->ResourceBarrier(1, &backToUav);

    // Synchronous flush + wait so we can map the readback below.
    m_deviceResources->ExecuteCommandList();
    m_deviceResources->WaitForGpu();
    auto allocator = m_deviceResources->GetCommandAllocator();
    ThrowIfFailed(allocator->Reset());
    ThrowIfFailed(cmdList->Reset(allocator, nullptr));

    // Map + sum + store.
    UINT64 sumActual = 0;
    void* mapped = nullptr;
    D3D12_RANGE rdRange = { 0, (size_t)obj.clusterCount * sizeof(UINT64) };
    ThrowIfFailed(obj.perFrameClasSizesReadback->Map(0, &rdRange, &mapped));
    const UINT64* sizes = reinterpret_cast<const UINT64*>(mapped);
    UINT64 minSz = UINT64_MAX, maxSz = 0;
    for (UINT c = 0; c < obj.clusterCount; ++c)
    {
        sumActual += sizes[c];
        minSz = std::min(minSz, sizes[c]);
        maxSz = std::max(maxSz, sizes[c]);
    }
    D3D12_RANGE noWrite = { 0, 0 };
    obj.perFrameClasSizesReadback->Unmap(0, &noWrite);

    m_overlayStats.animatedPerFrameClasActualBytes = sumActual;
    SampleLog::LogF(L"[animated CLAS measure] %u clusters: bytes min=%llu max=%llu mean=%llu total=%llu\n",
                    obj.clusterCount,
                    (unsigned long long)minSz, (unsigned long long)maxSz,
                    obj.clusterCount > 0 ? (unsigned long long)(sumActual / obj.clusterCount) : 0ull,
                    (unsigned long long)sumActual);
}

// =====================================================================================
// Walk all the live buffers + m_clasMemStats and copy the displayed numbers
// into m_overlayStats so the per-frame overlay can read from a snapshot rather
// than poking GetDesc().Width on every frame.  Called from
// RebuildStaticAccelerationStructures (after both static and animated paths
// have finished).  Per-frame timing values are *not* captured here - they're
// snapped by Tick() a few frames later once the ring buffer has refilled.
// =====================================================================================
void D3D12RaytracingClusteredGeometry::CaptureOverlayStatsSnapshot()
{
    auto sizeOf = [](const Microsoft::WRL::ComPtr<ID3D12Resource>& r) -> UINT64 {
        return r ? r->GetDesc().Width : 0ull;
    };
    auto& s = m_overlayStats;

    // Stash the previous snapshot so the overlay can colour numbers red/green
    // when this toggle changes them.  Every capture EXCEPT the very first
    // (init) saves current as prev -- so colours always reflect "what
    // changed vs the immediately previous toggle".  The 5 s timer below
    // applies to the fade-out only, not to which values we compare against.
    const auto now    = std::chrono::steady_clock::now();
    const bool isInit = (s.staticClasAllocBytes == 0);   // m_overlayStats default-initialised
    if (!isInit)
    {
        m_overlayStatsPrev    = s;
        m_overlayStatsHasPrev = true;
    }

    // Static
    s.staticClasAllocBytes   = m_totalClasBytes;
    s.staticClasActualBytes  = m_clasMemStats.sumActualBytes;
    s.staticClasScratchBytes = m_clasMemStats.scratchBytesPhase1;
    UINT64 blasSum = 0;
    for (const auto& obj : m_objects)
        if (obj.blasStorage) blasSum += obj.blasStorage->GetDesc().Width;
    s.staticBlasTotalBytes   = blasSum;

    // Animated  (animatedPerFrameClasActualBytes is set by
    // MeasureAnimatedClasBytesOneShot and we leave it alone here).
    if (m_animatedObjectEnabled)
    {
        const auto& a = m_animatedObject;
        s.animatedTemplateBytes            = sizeOf(a.templateResultBuffer);
        s.animatedPerFrameClasAllocBytes   = sizeOf(a.perFrameClasResultBuffer);
        s.animatedPerFrameClasScratchBytes = sizeOf(a.perFrameClasScratchBuffer);
        s.animatedBlasBytes                = sizeOf(a.blasStorage);
    }
    else
    {
        s.animatedTemplateBytes = s.animatedPerFrameClasAllocBytes =
        s.animatedPerFrameClasScratchBytes = s.animatedBlasBytes =
        s.animatedPerFrameClasActualBytes = 0;
    }

    s.tlasBytes            = sizeOf(m_tlasBuffer);
    s.totalClusterCount    = m_totalClusterCount;
    s.totalTriangleCount   = m_totalTriangleCount;

    // Arm timing capture for a few frames out so the ring buffer can refill
    // with post-rebuild samples before we snap pfAnimRebuildMs / pfTlasRebuildMs.
    // kPerFrameRingSlots + 2 gives a small margin.
    s.pfTimingValid            = false;
    m_overlayStatsArmCountdown = (INT)kPerFrameRingSlots + 2;

    // Refresh the delta-colour fade window: 5 seconds from NOW.  Each toggle
    // installs a fresh window (and a fresh prev above), so the user sees
    // immediate red/green vs the immediately previous toggle, fading back to
    // subtle 5 s after the LAST toggle.  Skipped on init -- there's nothing
    // meaningful to compare against yet, so we leave deltaUntil at min() so
    // the first real toggle's colours show without waiting for the fake-init
    // window to drain.
    if (!isInit)
        m_overlayStatsDeltaUntil = now + std::chrono::seconds(5);
}

// =====================================================================================
// Per-frame work for the animated object. Called from DoRender BEFORE the TLAS
// rebuild + DispatchRays. Total work: 1 memcpy (positions) + 2 batched
// ExecuteIndirectRTASOperations calls + 2 UAV barriers.
// =====================================================================================
void D3D12RaytracingClusteredGeometry::UpdateAnimatedObjectPerFrame(UINT pfTimestampBase)
{
    if (!m_animatedObjectEnabled) return;
    auto& obj = m_animatedObject;

    auto cl = m_deviceResources->GetCommandList();

    // ------------------------------------------------------------------
    // 1) GPU compute pass: AnimateBall.cs reads obj.restPositionsBuffer
    //    (SRV) and writes the deformed positions into
    //    obj.perFrameVertexBuffer (UAV).  The compute root signature is
    //    intentionally tiny (root constants + raw SRV + raw UAV) so we
    //    don't need to touch the descriptor heap here -- the raytracing
    //    pipeline's heap binding from the previous DispatchRays still
    //    holds (and Dispatch() doesn't read from it).
    //
    //    Per-frame CPU work is now ~5 D3D12 method calls: set RS+PSO,
    //    push 2 dwords, 2 SetComputeRoot*View, 1 Dispatch, 1 barrier.
    // ------------------------------------------------------------------
    cl->SetComputeRootSignature(m_animComputeRS.Get());
    cl->SetPipelineState(m_animComputePSO.Get());
    struct { float t; UINT vertexCount; } params = {
        (float)m_animSeconds, obj.totalVertexCount,
    };
    cl->SetComputeRoot32BitConstants(0, 2, &params, 0);
    cl->SetComputeRootShaderResourceView (1, obj.restPositionsBuffer ->GetGPUVirtualAddress());
    cl->SetComputeRootUnorderedAccessView(2, obj.perFrameVertexBuffer->GetGPUVirtualAddress());
    const UINT kThreadsPerGroup = 64;
    const UINT groups = (obj.totalVertexCount + kThreadsPerGroup - 1) / kThreadsPerGroup;
    cl->Dispatch(groups, 1, 1);

    // UAV barrier + transition to NON_PIXEL_SHADER_RESOURCE so INSTANTIATE
    // sees the just-written positions.  After INSTANTIATE we transition back
    // to UAV for the next frame's compute pass.
    {
        D3D12_RESOURCE_BARRIER toRead[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(obj.perFrameVertexBuffer.Get()),
            CD3DX12_RESOURCE_BARRIER::Transition(obj.perFrameVertexBuffer.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        };
        cl->ResourceBarrier(_countof(toRead), toRead);
    }
    // ------------------------------------------------------------------
    // 2) INSTANTIATE_CLUSTER_TEMPLATES - one batched op, N args. The args
    //    were pre-filled at setup time (ClusterTemplate + VertexBuffer slice
    //    per cluster); only their *contents* (the positions) change per frame.
    // ------------------------------------------------------------------
    D3D12_RTAS_CLUSTER_LIMITS limits = {};
    limits.MaxArgCount                                   = obj.clusterCount;
    limits.MaxUniqueGeometryIndexAndFlagsCountPerCluster = 1;
    limits.MaxTriangleCountPerCluster                    = obj.maxTrisPerCluster;
    limits.MaxVertexCountPerCluster                      = obj.maxVertsPerCluster;
    limits.MaxTotalTriangleCount                         = obj.mesh.totalTriangles;
    limits.MaxTotalVertexCount                           = obj.totalVertexCount;

    D3D12_RTAS_INSTANTIATE_CLUSTER_TEMPLATE_INPUTS_DESC instDesc = {};
    instDesc.ClusterLimits      = limits;
    instDesc.Flags              = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE | D3D12_RTAS_OPERATION_FLAG_ALLOW_DATA_ACCESS;
    instDesc.Mode               = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
    instDesc.VertexSourceFormat = D3D12_VERTEX_FORMAT_FLOAT32_3;

    D3D12_RTAS_OPERATION_INPUTS opInputsInst = {};
    opInputsInst.Type                            = D3D12_RTAS_OPERATION_TYPE_INSTANTIATE_CLUSTER_TEMPLATES;
    opInputsInst.pInstantiateClusterTemplateDesc = &instDesc;

    D3D12_RTAS_BATCHED_OPERATION_DATA batchedInst = {};
    batchedInst.BatchResultData       = obj.perFrameClasResultBuffer->GetGPUVirtualAddress();
    batchedInst.BatchScratchData      = obj.perFrameClasScratchBuffer->GetGPUVirtualAddress();
    batchedInst.ResultAddressArray    = { obj.perFrameClasAddressArray->GetGPUVirtualAddress(),
                                          sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    batchedInst.IndirectArgumentArray = { obj.perFrameInstArgsBuffer->GetGPUVirtualAddress(),
                                          sizeof(D3D12_RTAS_OPERATION_INSTANTIATE_CLUSTER_TEMPLATES_ARGS) };

    D3D12_RTAS_OPERATION_DESC opDescInst = {};
    opDescInst.Inputs                = opInputsInst;
    opDescInst.pBatchedOperationData = &batchedInst;
    // Bracket JUST the INSTANTIATE op with a timestamp pair so the overlay
    // can attribute time changes to vertex format / precision (which only
    // affect INSTANTIATE, not BLAS-from-CLAS).  The barriers immediately
    // after are intentionally OUTSIDE the bracket -- those are zero-cost
    // pipeline ordering edges, not work.
    auto cl4_ts = m_dxrCommandList.Get();
    if (pfTimestampBase != UINT_MAX)
        cl4_ts->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, pfTimestampBase + 0);
    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDescInst,
        D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    if (pfTimestampBase != UINT_MAX)
        cl4_ts->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, pfTimestampBase + 1);

    D3D12_RESOURCE_BARRIER instBarriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(obj.perFrameClasResultBuffer.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(obj.perFrameClasAddressArray.Get()),
        // INSTANTIATE consumed perFrameVertexBuffer (NON_PIXEL_SHADER_RESOURCE);
        // flip back to UAV for next frame's compute pass.  Doing this RIGHT
        // after the INSTANTIATE op keeps the buffer ready for the next
        // dispatch without an extra mid-frame transition.
        CD3DX12_RESOURCE_BARRIER::Transition(obj.perFrameVertexBuffer.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    };
    m_dxrCommandList->ResourceBarrier(_countof(instBarriers), instBarriers);

    // ------------------------------------------------------------------
    // 3) BUILD_BLAS_FROM_CLAS - 1 arg, EXPLICIT_DESTINATIONS so the BLAS
    //    storage VA stays fixed across frames (the BLAS contents get
    //    overwritten in place; the TLAS instance that points at this VA
    //    just needs a rebuild/refit to pick up the new root bounds).
    // ------------------------------------------------------------------
    D3D12_RTAS_CLAS_INPUTS_DESC blasDesc = {};
    blasDesc.Flags              = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;  // ALLOW_DATA_ACCESS is NOT permitted here - it's a per-CLAS property set at the CLAS-from-triangles build above, and the BLAS-from-CLAS must read it consistently across all referenced CLAS
    blasDesc.MaxArgCount        = 1;
    blasDesc.Mode               = D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS;
    blasDesc.MaxTotalClasCount  = obj.clusterCount;
    blasDesc.MaxClasCountPerArg = obj.clusterCount;

    D3D12_RTAS_OPERATION_INPUTS opInputsBlas = {};
    opInputsBlas.Type      = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
    opInputsBlas.pClasDesc = &blasDesc;

    D3D12_RTAS_BATCHED_OPERATION_DATA batchedBlas = {};
    batchedBlas.BatchScratchData      = obj.blasScratchBuffer->GetGPUVirtualAddress();
    batchedBlas.ResultAddressArray    = { obj.blasResultAddrBuffer->GetGPUVirtualAddress(),
                                          sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    batchedBlas.IndirectArgumentArray = { obj.blasArgsBuffer->GetGPUVirtualAddress(),
                                          sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS) };

    D3D12_RTAS_OPERATION_DESC opDescBlas = {};
    opDescBlas.Inputs                = opInputsBlas;
    opDescBlas.pBatchedOperationData = &batchedBlas;
    // Bracket the BLAS-from-CLAS op separately so the overlay reports it
    // alongside (but distinct from) INSTANTIATE.  This is the "baseline" cost
    // that shouldn't move much when the precision slider changes.
    if (pfTimestampBase != UINT_MAX)
        cl4_ts->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, pfTimestampBase + 2);
    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDescBlas,
        D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    if (pfTimestampBase != UINT_MAX)
        cl4_ts->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, pfTimestampBase + 3);

    auto blasBarrier = CD3DX12_RESOURCE_BARRIER::UAV(obj.blasStorage.Get());
    m_dxrCommandList->ResourceBarrier(1, &blasBarrier);
}

void D3D12RaytracingClusteredGeometry::BuildTlasClassic()
{
    auto device = m_deviceResources->GetD3DDevice();
    const UINT N_static = (UINT)m_objects.size();
    const UINT N_anim   = m_animatedObjectEnabled ? 1 : 0;
    const UINT N_total  = N_static + N_anim;

    // Build the full instance-desc array. The animated object's BLAS lives at
    // a fixed GVA (EXPLICIT_DESTINATIONS), so this array is good for the life
    // of the sample - only the TLAS BVH itself needs per-frame rebuild to
    // pick up the animated BLAS's new root bounds.
    // Per-instance flags + hit-group pick.
    //
    // FORCE_OPAQUE: set when neither baseline material nor any per-cluster
    // checker override introduces translucency or refractivity - then the
    // any-hit dispatch is skipped at traversal time.
    //
    // HIT GROUP: opaque-only instances use OpaqueHitGroup (no any-hit
    // shader bound at all - cheapest possible closesthit path).
    // Anything that may refract or translucently-reject goes to GlassHit-
    // Group (closesthit handles full Fresnel + refraction; any-hit handles
    // stochastic translucency).
    //
    // The two decisions share the same predicate ("does this instance
    // EVER need refraction or translucency?") - if no, OpaqueHitGroup +
    // FORCE_OPAQUE; if yes, GlassHitGroup + NONE.
    auto needsGlass = [](const MaterialDesc& mat, const ClusterObject& obj) -> bool
    {
        if (mat.refractivity > 0.0f || mat.translucency > 0.0f)
            return true;
        if (obj.checker.enabled)
        {
            // Per-cluster checker override may introduce refractivity on
            // a subset of clusters (e.g. sphere2: matte baseline + glass
            // odd-parity tiles).  -1 sentinel = no override (keep baseline).
            if (obj.checker.evenParity.overrideRefr > 0.0f) return true;
            if (obj.checker.oddParity.overrideRefr  > 0.0f) return true;
        }
        return false;
    };
    // Matches the hit-group shader table layout in
    // CreateRaytracingPipelineAndShaderTables: opaque block at offset 0,
    // glass block at offset 2 (2 records per block: primary + shadow).
    constexpr UINT kHitGroupContribOpaque = 0;
    constexpr UINT kHitGroupContribGlass  = 2;

    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances(N_total);
    UINT nOpaque = 0, nGlass = 0;
    for (UINT i = 0; i < N_static; ++i)
    {
        const auto& obj = m_objects[i];
        XMMATRIX rot = XMMatrixRotationRollPitchYaw(obj.worldRotEuler.x,
                                                    obj.worldRotEuler.y,
                                                    obj.worldRotEuler.z);
        XMMATRIX m = XMMatrixScaling(obj.worldScale, obj.worldScale, obj.worldScale)
                   * rot
                   * XMMatrixTranslation(obj.worldPos.x, obj.worldPos.y, obj.worldPos.z);
        XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(instances[i].Transform), m);
        instances[i].InstanceID  = obj.instanceID;
        instances[i].InstanceMask= 0xFF;
        const bool isGlass = needsGlass(m_materials[obj.instanceID], obj);
        instances[i].InstanceContributionToHitGroupIndex =
            isGlass ? kHitGroupContribGlass : kHitGroupContribOpaque;
        instances[i].Flags = isGlass ? D3D12_RAYTRACING_INSTANCE_FLAG_NONE
                                     : D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
        // Non-orientable / self-intersecting surfaces need double-sided
        // traversal: their triangle winding doesn't have a consistent
        // global "outward" (Klein bottle's classic problem), and culling
        // any "back-facing" triangles leaves visible holes along the
        // orientation seam.  This per-instance flag overrides the ray's
        // RAY_FLAG_CULL_BACK_FACING_TRIANGLES so only THIS instance pays
        // the cost; orientable meshes (sphere, torus, cube, slab) keep
        // their fast single-sided traversal.
        if (m_objects[i].nonOrientable)
            instances[i].Flags |= D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
        instances[i].AccelerationStructure = obj.blasGPUVA;
        (isGlass ? nGlass : nOpaque)++;
    }
    if (m_animatedObjectEnabled)
    {
        const auto& a = m_animatedObject;
        XMMATRIX m = XMMatrixScaling(a.worldScale, a.worldScale, a.worldScale)
                   * XMMatrixTranslation(a.worldPos.x, a.worldPos.y, a.worldPos.z);
        XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(instances[N_static].Transform), m);
        instances[N_static].InstanceID  = a.instanceID;
        instances[N_static].InstanceMask= 0xFF;
        // Animated object has no ClusterObject scene config yet; treat by
        // material only (currently refractive glass -> GlassHitGroup).
        const auto& aMat = m_materials[a.instanceID];
        const bool isGlass = (aMat.refractivity > 0.0f) || (aMat.translucency > 0.0f);
        instances[N_static].InstanceContributionToHitGroupIndex =
            isGlass ? kHitGroupContribGlass : kHitGroupContribOpaque;
        instances[N_static].Flags = isGlass ? D3D12_RAYTRACING_INSTANCE_FLAG_NONE
                                            : D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
        instances[N_static].AccelerationStructure = a.blasGPUVA;
        (isGlass ? nGlass : nOpaque)++;
    }
    SampleLog::LogF(L"[hitgroups] %u opaque-instance(s) -> OpaqueHitGroup (no any-hit), "
                    L"%u glass-instance(s) -> GlassHitGroup\n", nOpaque, nGlass);
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
    SampleLog::LogF(L"[TLAS prebuild] %u instances (%u static + %u animated): result=%llu, scratch=%llu bytes\n",
                    N_total, N_static, N_anim,
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
    const UINT N_total = (UINT)m_objects.size() + (m_animatedObjectEnabled ? 1u : 0u);

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

void D3D12RaytracingClusteredGeometry::DumpClusterStatsAsync()
{
    auto device       = m_deviceResources->GetD3DDevice();
    auto commandQueue = m_deviceResources->GetCommandQueue();
    const UINT N = m_totalClusterCount;

    auto rbHeap  = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto bufDesc = CD3DX12_RESOURCE_DESC::Buffer((UINT64)N * sizeof(UINT64));
    ComPtr<ID3D12Resource> readback;
    ThrowIfFailed(device->CreateCommittedResource(&rbHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)));

    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)));
    ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&cl)));
    auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(m_clasSizeArray.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->ResourceBarrier(1, &toCopy);
    cl->CopyBufferRegion(readback.Get(), 0, m_clasSizeArray.Get(), 0, (UINT64)N * sizeof(UINT64));
    auto fromCopy = CD3DX12_RESOURCE_BARRIER::Transition(m_clasSizeArray.Get(),
        D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cl->ResourceBarrier(1, &fromCopy);
    ThrowIfFailed(cl->Close());
    ID3D12CommandList* cls[] = { cl.Get() };
    commandQueue->ExecuteCommandLists(1, cls);
    m_deviceResources->WaitForGpu();

    void* p = nullptr;
    D3D12_RANGE rng = { 0, (SIZE_T)N * sizeof(UINT64) };
    ThrowIfFailed(readback->Map(0, &rng, &p));
    const UINT64* sizes = reinterpret_cast<const UINT64*>(p);
    UINT64 total = 0, mn = UINT64_MAX, mx = 0;
    for (UINT i = 0; i < N; ++i) { total += sizes[i]; mn = std::min(mn, sizes[i]); mx = std::max(mx, sizes[i]); }
    D3D12_RANGE noWrite = { 0, 0 };
    readback->Unmap(0, &noWrite);

    SampleLog::LogF(L"[CLAS stats] %u clusters: bytes min=%llu max=%llu mean=%llu total=%llu\n",
                    N, (unsigned long long)mn, (unsigned long long)mx,
                    (unsigned long long)(total / N), (unsigned long long)total);
}

// ---------------------------------------------------------------------------------
// Read back the AS build timestamps from m_buildQueryReadback and compute
// per-operation wall-clock times. Called once after init (post-WaitForGpu).
// ---------------------------------------------------------------------------------
void D3D12RaytracingClusteredGeometry::ReadBuildTimestamps()
{
    // Record CLAS storage total for the title bar.
    m_totalClasBytes = 0;
    if (m_clasResultBuffer) m_totalClasBytes = m_clasResultBuffer->GetDesc().Width;

    // Compute total BLAS allocation size.
    m_totalBlasBytes = 0;
    for (const auto& obj : m_objects)
        if (obj.blasStorage)
            m_totalBlasBytes += obj.blasStorage->GetDesc().Width;

    UINT64* ts = nullptr;
    D3D12_RANGE r = { 0, sizeof(UINT64) * kBuildTimestampCount };
    ThrowIfFailed(m_buildQueryReadback->Map(0, &r, reinterpret_cast<void**>(&ts)));
    const double inv = 1000.0 / (double)m_timestampFrequency;     // ticks -> ms
    m_clasBuildMs  = (ts[1] - ts[0]) * inv;
    m_blasBuildMs  = (ts[3] - ts[2]) * inv;
    m_tlasBuildMs  = (ts[5] - ts[4]) * inv;
    m_totalBuildMs = (ts[5] - ts[0]) * inv;
    D3D12_RANGE noWrite = { 0, 0 };
    m_buildQueryReadback->Unmap(0, &noWrite);

    SampleLog::LogF(L"[build wall-clock] CLAS=%.3f ms, BLAS=%.3f ms, TLAS=%.3f ms (total %.3f ms)\n",
                    m_clasBuildMs, m_blasBuildMs, m_tlasBuildMs, m_totalBuildMs);
    SampleLog::LogF(L"[storage] CLAS: %llu bytes  BLAS: %llu bytes\n",
                    (unsigned long long)m_totalClasBytes,
                    (unsigned long long)m_totalBlasBytes);

    // Per-mode CLAS memory stats. Use [CLAS mem] tag so the table is grep-able.
    // Notes:
    //  - sumActualBytes is the irreducible CLAS storage (= sum of per-cluster
    //    sizes the driver picked). It's the same for all three modes; the
    //    modes differ only in how much we *reserved* on top of that.
    //  - resultPrebuildMax is the worst-case alloc the driver would ask for
    //    in IMPLICIT mode (used as the reference for "wasted" bytes).
    //  - peakResidentBytes is the maximum live result-buffer memory at any
    //    instant during the build flow.
    const auto& s = m_clasMemStats;
    const double overheadVsActual = s.sumActualBytes
        ? 100.0 * (double)((double)s.resultFinalBytes - (double)s.sumActualBytes) / (double)s.sumActualBytes
        : 0.0;
    const double savingsVsImplicit = s.resultPrebuildMax
        ? 100.0 * (double)((double)s.resultPrebuildMax - (double)s.resultFinalBytes) / (double)s.resultPrebuildMax
        : 0.0;
    SampleLog::LogF(
        L"[CLAS mem mode=%s] prebuild_max=%llu  initial=%llu  final=%llu  peak=%llu  "
        L"sum_actual=%llu  scratch=(%llu, %llu)  cpu_phase_ms=(%.3f, %.3f)  "
        L"final_overhead_vs_actual=%+.1f%%  savings_vs_implicit=%+.1f%%\n",
        ClasAllocModeName(),
        (unsigned long long)s.resultPrebuildMax,
        (unsigned long long)s.resultInitialBytes,
        (unsigned long long)s.resultFinalBytes,
        (unsigned long long)s.peakResidentBytes,
        (unsigned long long)s.sumActualBytes,
        (unsigned long long)s.scratchBytesPhase1,
        (unsigned long long)s.scratchBytesPhase2,
        s.cpuWallMsPhase1, s.cpuWallMsPhase2,
        overheadVsActual, savingsVsImplicit);

    // Discoverability hint: if the user is on the simplest path (implicit)
    // AND the over-allocation is actually painful (>2x), point them at the
    // alternative modes. Most users won't read all of d3d12.h, so the sample
    // itself should help connect "you allocated 1.2 MB" to "you only used
    // 220 KB; here's how to fix it".
    if (m_clasAllocMode == ClasAllocMode::Implicit
        && s.sumActualBytes > 0
        && s.resultFinalBytes > s.sumActualBytes * 2)
    {
        const double ratio = (double)s.resultFinalBytes / (double)s.sumActualBytes;
        SampleLog::LogF(
            L"[CLAS mem hint] implicit mode allocated %.1fx the bytes actually "
            L"used (%llu / %llu). Re-run with --clas-alloc get-sizes for an "
            L"exact-fit alloc, or --clas-alloc compact for build-then-compact "
            L"semantics. See readme.md for the tradeoff table.\n",
            ratio,
            (unsigned long long)s.resultFinalBytes,
            (unsigned long long)s.sumActualBytes);
    }
}

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
        // Per-instance materials live in a structured buffer at t1, indexed by
        // InstanceID() in the closesthit/anyhit shaders.
        params[GlobalRootSig::MaterialsSRVSlot].InitAsShaderResourceView(1);
        // Per-cluster shader-side buffers for smooth normals.  DXR2 cluster
        // geometry's CLAS only takes positions in its vertex buffer, so per-
        // vertex normals (and the index buffer, so we can find which 3
        // vertices a triangle hit references) travel through 3 separate
        // structured buffers indexed by ClusterID() + PrimitiveIndex().
        params[GlobalRootSig::ClusterNormalsSRVSlot].InitAsShaderResourceView(2);
        params[GlobalRootSig::ClusterIndicesSRVSlot].InitAsShaderResourceView(3);
        params[GlobalRootSig::ClusterOffsetsSRVSlot].InitAsShaderResourceView(4);
        // Per-cluster GENERIC metadata buffer.  Replaces ALL hardcoded
        // per-instance / per-cid-range branches in the closesthit.  See
        // RaytracingHlslCompat.h ClusterMeta.
        params[GlobalRootSig::ClusterMetaSRVSlot].InitAsShaderResourceView(5);
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
    lib->DefineExport(c_glassClosestHitName);
    lib->DefineExport(c_glassAnyHitName);
    lib->DefineExport(c_missName);
    lib->DefineExport(c_shadowMissName);

    // ---- OPAQUE primary hit group --------------------------------------
    // Bound to instances whose material chain (baseline + every per-cluster
    // checker override) keeps refractivity = 0 and translucency = 0.
    // Currently: only sphere0 (chrome).
    //
    // NO any-hit shader binding - any-hit dispatch is skipped entirely
    // by the runtime for traversals on this hit group.  Combined with the
    // FORCE_OPAQUE TLAS instance flag this is the cheapest possible
    // closesthit path: every triangle hit goes straight to OpaqueHit.
    auto opaqueHG = pipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    opaqueHG->SetClosestHitShaderImport(c_opaqueClosestHitName);
    opaqueHG->SetHitGroupExport(c_opaqueHitGroupName);
    opaqueHG->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

    // ---- GLASS primary hit group ---------------------------------------
    // Bound to any instance whose material chain includes refraction OR
    // translucency at the baseline OR via per-cluster checker overrides.
    // Currently: spheres 1/2/3, torus, cube, floor, animated, Klein.
    //
    // any-hit = GlassAnyHit (stochastic translucency reject).  Closest-
    // hit = GlassHit (full Fresnel + refraction + reflection composition).
    auto glassHG = pipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    glassHG->SetClosestHitShaderImport(c_glassClosestHitName);
    glassHG->SetAnyHitShaderImport(c_glassAnyHitName);
    glassHG->SetHitGroupExport(c_glassHitGroupName);
    glassHG->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

    // Shadow hit group: empty (no closest-hit, no any-hit). Shadow rays use
    // SKIP_CLOSEST_HIT_SHADER + FORCE_OPAQUE so neither shader can run on a
    // hit. The hit group still has to exist for ray-contribution-index 1.
    auto shadowHitGroup = pipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    shadowHitGroup->SetHitGroupExport(c_shadowHitGroupName);
    shadowHitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

    // Payload: max(Payload(float4), ShadowPayload(bool)) -> 16 bytes is enough.
    auto shaderConfig = pipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
    shaderConfig->Config(/*payload*/ 4 * sizeof(float) + 2 * sizeof(uint), /*attribs*/ 2 * sizeof(float));   // Payload = float4 color + uint depth + uint inGlass
    auto globalRS = pipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
    globalRS->SetRootSignature(m_globalRootSignature.Get());

    // ALLOW_CLUSTERED_GEOMETRY is the DXR2 opt-in for the shader to traverse a
    // BLAS built from CLAS. Without it, hits on a Cluster BLAS are undefined.
    auto pipelineConfig = pipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT>();
    // MaxRecursionDepth = 8: budget for cascading-refraction paths now that
    // EVERY object in the scene is a glass variant.  A primary ray that
    // pierces several glass volumes in sequence can chain refractions to
    // depth 4-5 before bottoming out, plus 1 shadow ray per surface hit.
    // Worst-case path:
    //   raygen TraceRay     -> recursion 1 (primary closesthit on glass A)
    //   refract-in glass A   -> recursion 2 (back-face of A)
    //   refract-out into air -> recursion 3 (front-face of glass B)
    //   refract-in glass B   -> recursion 4 (back-face of B)
    //   refract-out          -> recursion 5 (next surface)
    //   reflection bounce    -> recursion 6 (mirror-reflect off some glass)
    //   shadow ray           -> recursion 7 (LEAF; SKIP_CLOSEST_HIT)
    //   safety               -> recursion 8 (unused headroom)
    // The closesthit gates refraction at myDepth <= 4 to stay well under
    // this budget while still letting 4+ glass volumes compose visually.
    pipelineConfig->Config(16, D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_CLUSTERED_GEOMETRY);

    SampleLog::Write(L"  >>> CreateStateObject\n");
    HRESULT hrCSO = m_dxrDevice->CreateStateObject(pipeline, IID_PPV_ARGS(&m_dxrStateObject));
    SampleLog::LogF(L"  CreateStateObject -> hr=0x%08X\n", (unsigned)hrCSO);
    ThrowIfFailed(hrCSO, L"CreateStateObject failed\n");
    m_dxrStateObject->SetName(L"RT pipeline (cluster-aware)");

    SampleLog::Write(L"  >>> Build shader tables\n");
    ComPtr<ID3D12StateObjectProperties> props;
    ThrowIfFailed(m_dxrStateObject->QueryInterface(IID_PPV_ARGS(&props)));
    void* rgID            = props->GetShaderIdentifier(c_raygenName);
    void* missID          = props->GetShaderIdentifier(c_missName);
    void* shadowMissID    = props->GetShaderIdentifier(c_shadowMissName);
    void* opaqueHgID      = props->GetShaderIdentifier(c_opaqueHitGroupName);
    void* glassHgID       = props->GetShaderIdentifier(c_glassHitGroupName);
    void* shadowHgID      = props->GetShaderIdentifier(c_shadowHitGroupName);
    const UINT idSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    const UINT recordSize = Align(idSize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);

    // Single-record table (raygen).
    auto makeTable1 = [&](void* shaderID, ComPtr<ID3D12Resource>& outTable, const wchar_t* name)
    {
        std::vector<uint8_t> data(recordSize, 0);
        memcpy(data.data(), shaderID, idSize);
        AllocateUploadBuffer(device, data.data(), data.size(), &outTable, name);
    };
    // Two-record table (miss).
    auto makeTable2 = [&](void* shaderID0, void* shaderID1,
                          ComPtr<ID3D12Resource>& outTable, const wchar_t* name)
    {
        std::vector<uint8_t> data(recordSize * 2, 0);
        memcpy(data.data() + 0,          shaderID0, idSize);
        memcpy(data.data() + recordSize, shaderID1, idSize);
        AllocateUploadBuffer(device, data.data(), data.size(), &outTable, name);
    };
    // ---- HIT-GROUP shader table layout ----
    // 4 records, 2 contiguous (primary + shadow) blocks per material kind:
    //
    //   [0] OpaqueHitGroup    <-- primary for OPAQUE instances (InstanceContrib=0)
    //   [1] ShadowHitGroup    <-- shadow  for OPAQUE instances (RayContrib=1)
    //   [2] GlassHitGroup     <-- primary for GLASS  instances (InstanceContrib=2)
    //   [3] ShadowHitGroup    <-- shadow  for GLASS  instances (RayContrib=1)
    //
    // Per-instance InstanceContributionToHitGroupIndex picks the block;
    // shadow rays add RayContributionToHitGroupIndex=1 within the block.
    // The shadow records share the same dummy hit-group identifier but
    // physically live at two distinct table indices so the same RayContrib=1
    // offset works from either block start.
    auto makeTable4 = [&](void* r0, void* r1, void* r2, void* r3,
                          ComPtr<ID3D12Resource>& outTable, const wchar_t* name)
    {
        std::vector<uint8_t> data(recordSize * 4, 0);
        memcpy(data.data() + 0 * recordSize, r0, idSize);
        memcpy(data.data() + 1 * recordSize, r1, idSize);
        memcpy(data.data() + 2 * recordSize, r2, idSize);
        memcpy(data.data() + 3 * recordSize, r3, idSize);
        AllocateUploadBuffer(device, data.data(), data.size(), &outTable, name);
    };

    makeTable1(rgID,                    m_rayGenShaderTable,   L"raygen shader table");
    makeTable2(missID, shadowMissID,    m_missShaderTable,     L"miss shader table (primary + shadow)");
    makeTable4(opaqueHgID, shadowHgID,
               glassHgID,  shadowHgID,
               m_hitGroupShaderTable,
               L"hit-group shader table (opaque-primary, shadow, glass-primary, shadow)");
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
void D3D12RaytracingClusteredGeometry::CreateUIFont()
{
    using namespace DirectX;
    auto device = m_deviceResources->GetD3DDevice();

    m_graphicsMemory = std::make_unique<GraphicsMemory>(device);

    // SpriteBatch PSO needs to match the back-buffer format pair (RTV format
    // + DSV format).  Depth buffer is present on this sample's DeviceResources
    // but the overlay itself disables depth read/write via SpriteBatch's
    // default state -- the format just has to match the bound DSV slot at
    // draw time, even if we render with no depth.
    ResourceUploadBatch resourceUpload(device);
    resourceUpload.Begin();
    {
        RenderTargetState rtState(m_deviceResources->GetBackBufferFormat(),
                                  m_deviceResources->GetDepthBufferFormat());
        SpriteBatchPipelineStateDescription pd(rtState);
        m_spriteBatch = std::make_unique<SpriteBatch>(device, resourceUpload, pd);
    }
    auto uploadFinished = resourceUpload.End(m_deviceResources->GetCommandQueue());
    uploadFinished.wait();

    // Reserve slot 1 of our shared descriptor heap for the font texture SRV.
    // SpriteFont's constructor takes the CPU + GPU descriptor handles where
    // it should write the SRV.
    D3D12_CPU_DESCRIPTOR_HANDLE fontCpu;
    UINT fontSlot = AllocateDescriptor(&fontCpu);
    D3D12_GPU_DESCRIPTOR_HANDLE fontGpu = CD3DX12_GPU_DESCRIPTOR_HANDLE(
        m_descriptorHeap->GetGPUDescriptorHandleForHeapStart(), fontSlot, m_descriptorSize);

    {
        ResourceUploadBatch fontUpload(device);
        fontUpload.Begin();
        m_uiFont = std::make_unique<SpriteFont>(device, fontUpload,
            L"SegoeUI_24.spritefont",
            fontCpu, fontGpu);
        // Defensive: if we ever try to render a character that isn't in the
        // sprite font (e.g. a non-ASCII codepoint baked into a log message
        // by mistake), SpriteFont::DrawString throws std::runtime_error which
        // propagates out of WndProc -> STATUS_FATAL_USER_CALLBACK_EXCEPTION.
        // Setting a default glyph turns that crash into a visible '?' instead.
        m_uiFont->SetDefaultCharacter(L'?');
        auto finished = fontUpload.End(m_deviceResources->GetCommandQueue());
        finished.wait();
    }
    SampleLog::LogF(L"[ui] SpriteFont loaded (descriptor heap slot %u); "
                    L"line spacing = %.1f px\n",
                    fontSlot, m_uiFont->GetLineSpacing());
}

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
void D3D12RaytracingClusteredGeometry::CreateAnimationComputePipeline()
{
    auto device = m_deviceResources->GetD3DDevice();

    CD3DX12_ROOT_PARAMETER rootParams[3] = {};
    rootParams[0].InitAsConstants(2, /*shaderReg*/0);                  // b0 = { float t, uint vertexCount }
    rootParams[1].InitAsShaderResourceView(0);                         // t0 = ByteAddressBuffer rest positions
    rootParams[2].InitAsUnorderedAccessView(0);                        // u0 = RWByteAddressBuffer anim positions

    CD3DX12_ROOT_SIGNATURE_DESC rsDesc;
    rsDesc.Init(_countof(rootParams), rootParams, 0, nullptr,
                D3D12_ROOT_SIGNATURE_FLAG_NONE);

    Microsoft::WRL::ComPtr<ID3DBlob> serialized, error;
    HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                             &serialized, &error);
    if (FAILED(hr))
    {
        if (error) SampleLog::LogF(L"[anim-cs] root sig serialize failed: %hs\n",
                                   (const char*)error->GetBufferPointer());
        ThrowIfFailed(hr);
    }
    ThrowIfFailed(device->CreateRootSignature(0, serialized->GetBufferPointer(),
                                              serialized->GetBufferSize(),
                                              IID_PPV_ARGS(&m_animComputeRS)));
    m_animComputeRS->SetName(L"Animate ball compute root sig");

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_animComputeRS.Get();
    psoDesc.CS             = CD3DX12_SHADER_BYTECODE((void*)g_pAnimateBall, ARRAYSIZE(g_pAnimateBall));
    ThrowIfFailed(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_animComputePSO)));
    m_animComputePSO->SetName(L"Animate ball compute PSO");

    SampleLog::LogF(L"[anim-cs] compute PSO built (%zu bytes of CS bytecode)\n",
                    (size_t)ARRAYSIZE(g_pAnimateBall));
}

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
void D3D12RaytracingClusteredGeometry::CreateFillInstantiateArgsPipeline()
{
    auto device = m_deviceResources->GetD3DDevice();

    CD3DX12_ROOT_PARAMETER rootParams[4] = {};
    rootParams[0].InitAsConstants(5, /*shaderReg*/0);                  // b0
    rootParams[1].InitAsUnorderedAccessView(/*shaderReg*/1);           // u1 = templateGvas
    rootParams[2].InitAsShaderResourceView(/*shaderReg*/0);            // t0 = vertexOffsets
    rootParams[3].InitAsUnorderedAccessView(/*shaderReg*/0);           // u0 = argsOut

    CD3DX12_ROOT_SIGNATURE_DESC rsDesc;
    rsDesc.Init(_countof(rootParams), rootParams, 0, nullptr,
                D3D12_ROOT_SIGNATURE_FLAG_NONE);

    Microsoft::WRL::ComPtr<ID3DBlob> serialized, error;
    HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                             &serialized, &error);
    if (FAILED(hr))
    {
        if (error) SampleLog::LogF(L"[fill-args-cs] root sig serialize failed: %hs\n",
                                   (const char*)error->GetBufferPointer());
        ThrowIfFailed(hr);
    }
    ThrowIfFailed(device->CreateRootSignature(0, serialized->GetBufferPointer(),
                                              serialized->GetBufferSize(),
                                              IID_PPV_ARGS(&m_fillInstArgsRS)));
    m_fillInstArgsRS->SetName(L"FillInstantiateArgs root sig");

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_fillInstArgsRS.Get();
    psoDesc.CS             = CD3DX12_SHADER_BYTECODE((void*)g_pFillInstantiateArgs, ARRAYSIZE(g_pFillInstantiateArgs));
    ThrowIfFailed(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_fillInstArgsPSO)));
    m_fillInstArgsPSO->SetName(L"FillInstantiateArgs compute PSO");

    SampleLog::LogF(L"[fill-args-cs] compute PSO built (%zu bytes of CS bytecode)\n",
                    (size_t)ARRAYSIZE(g_pFillInstantiateArgs));
}


// =====================================================================
// Tiny helper used by the four args-fill pipeline builders below.
// Each pipeline has exactly the same C++ shape: a root sig built from
// a small N-element rootParams array, no static samplers, no flags,
// + a single CS PSO referencing the supplied bytecode.  Factoring this
// out keeps each Create*Pipeline body focused on its actual schema.
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
void D3D12RaytracingClusteredGeometry::CreateFillMoveArgsPipeline()
{
    auto device = m_deviceResources->GetD3DDevice();
    CD3DX12_ROOT_PARAMETER rp[3] = {};
    rp[0].InitAsConstants(1, /*shaderReg*/0);                  // b0 = { clusterCount }
    rp[1].InitAsUnorderedAccessView(/*shaderReg*/1);           // u1 = srcAddrs
    rp[2].InitAsUnorderedAccessView(/*shaderReg*/0);           // u0 = argsOut
    BuildArgsFillPipeline(device, rp, _countof(rp),
        g_pFillMoveClusterArgs, ARRAYSIZE(g_pFillMoveClusterArgs),
        L"FillMoveClusterArgs root sig", L"FillMoveClusterArgs compute PSO",
        m_fillMoveArgsRS, m_fillMoveArgsPSO);
    SampleLog::LogF(L"[fill-args-cs] FillMoveClusterArgs PSO built (%zu bytes)\n",
                    (size_t)ARRAYSIZE(g_pFillMoveClusterArgs));
}

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
void D3D12RaytracingClusteredGeometry::CreateFillTemplateArgsPipeline()
{
    auto device = m_deviceResources->GetD3DDevice();
    CD3DX12_ROOT_PARAMETER rp[3] = {};
    rp[0].InitAsConstants(4, /*shaderReg*/0);                  // b0
    rp[1].InitAsShaderResourceView(/*shaderReg*/0);            // t0 = meta
    rp[2].InitAsUnorderedAccessView(/*shaderReg*/0);           // u0 = argsOut
    BuildArgsFillPipeline(device, rp, _countof(rp),
        g_pFillClusterTemplateArgs, ARRAYSIZE(g_pFillClusterTemplateArgs),
        L"FillClusterTemplateArgs root sig", L"FillClusterTemplateArgs compute PSO",
        m_fillTemplateArgsRS, m_fillTemplateArgsPSO);
    SampleLog::LogF(L"[fill-args-cs] FillClusterTemplateArgs PSO built (%zu bytes)\n",
                    (size_t)ARRAYSIZE(g_pFillClusterTemplateArgs));
}



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
void D3D12RaytracingClusteredGeometry::RenderUI()
{
    using namespace DirectX;
    if (!m_spriteBatch || !m_uiFont) return;

    auto commandList = m_deviceResources->GetCommandList();
    auto viewport    = m_deviceResources->GetScreenViewport();
    auto scissor     = m_deviceResources->GetScissorRect();

    // Bind the back buffer as render target (no depth -- text doesn't write Z).
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = m_deviceResources->GetRenderTargetView();
    commandList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    commandList->RSSetViewports(1, &viewport);
    commandList->RSSetScissorRects(1, &scissor);

    // Re-bind the shader-visible descriptor heap so the SpriteBatch shader
    // can sample the font texture SRV from slot 1.  (Compute path bound it
    // earlier; OMSetRenderTargets doesn't disturb it but the SetDescriptorHeaps
    // call is documented to be safe to repeat per-pass and PIX captures look
    // cleaner with the explicit bind.)
    ID3D12DescriptorHeap* heaps[] = { m_descriptorHeap.Get() };
    commandList->SetDescriptorHeaps(_countof(heaps), heaps);

    m_spriteBatch->SetViewport(viewport);
    m_spriteBatch->Begin(commandList);

    // Render at exactly 0.5x of the 24 pt atlas -- the only fractional kScale
    // bilinear filtering downsamples cleanly (each destination pixel = perfect
    // 2x2 average of source texels).  Any other ratio drops source data and
    // aliases on thin glyph strokes; we tried 48 pt + 0.265x and the moire on
    // small digits was visible.  If you change the visual size, regen the
    // spritefont at exactly 2x the new target pt and KEEP kScale = 0.5.
    const float    kScale  = 0.5f;
    const float    kLineH  = m_uiFont->GetLineSpacing() * kScale;
    XMFLOAT2       pos{ 24.0f, 18.0f };
    const XMFLOAT2 kOrigin { 0.0f, 0.0f };
    // Body text colours.  White on this scene's sky/hex/floor palette --
    // kSubtle is the "no change" colour for stat rows (delta-coloured numbers
    // tint away from this to green/red when they move).
    const XMVECTOR kWhite  = XMVectorSet(1.00f, 1.00f, 1.00f, 1);
    const XMVECTOR kSubtle = XMVectorSet(0.92f, 0.94f, 0.97f, 1);
    const XMVECTOR kAccent = XMVectorSet(0.40f, 0.15f, 0.65f, 1);   // dark purple for section headers
    const XMVECTOR kHotkey = XMVectorSet(1.00f, 0.90f, 0.15f, 1);   // bright yellow for the hotkey char
    // Delta-colouring: every per-config-change number compares to its
    // previous-snapshot value and tints itself.
    //   green = went down (smaller = "better" for memory/time)
    //   red   = went up   ("worse")
    //   subtle (unchanged) = no prev or no change
    const XMVECTOR kGreen  = XMVectorSet(0.40f, 1.00f, 0.40f, 1);   // bright green delta
    const XMVECTOR kRed    = XMVectorSet(1.00f, 0.45f, 0.45f, 1);   // bright red delta

    // Local helper -- DrawString with the global scale factor baked in.
    auto draw = [&](const wchar_t* s, XMFLOAT2 p, FXMVECTOR colour, float relScale = 1.0f) {
        m_uiFont->DrawString(m_spriteBatch.get(), s, p, colour,
                             /*rotation*/0.0f, kOrigin, kScale * relScale);
    };
    // MeasureString returns the size at scale 1; cursor advances need scaling
    // by kScale to land glyphs flush with each other.
    // ignoreWhitespace=false is CRITICAL for drawSeg below: trailing spaces in
    // a label segment must contribute to the advance, else "CLAS " + "0.66"
    // composes as "CLAS0.66" with the label and value glued together.
    auto measureX = [&](const wchar_t* s) {
        return XMVectorGetX(m_uiFont->MeasureString(s, /*ignoreWhitespace*/false)) * kScale;
    };
    // Segment draw: writes `text` at the cursor and advances the cursor's X.
    // Used to compose a stat line out of subtle-prefix / coloured-number /
    // subtle-suffix pieces without needing a fully tagged-text renderer.
    // `cursor` is mutated in place (x advances, y untouched).
    auto drawSeg = [&](const wchar_t* text, XMFLOAT2& cursor, FXMVECTOR colour) {
        m_uiFont->DrawString(m_spriteBatch.get(), text, cursor, colour,
                             /*rotation*/0.0f, kOrigin, kScale);
        cursor.x += measureX(text);
    };
    // Delta-colour pickers.  Return green/red/subtle based on cur vs prev.
    // Two gates:
    //   1. m_overlayStatsHasPrev -- no baseline yet (first capture).
    //   2. m_overlayStatsDeltaUntil -- the 5s post-toggle window during which
    //      colours are shown.  After it expires we fall back to subtle so the
    //      screen doesn't permanently glow red/green long after the user
    //      finished iterating.
    // Plus a NOISE THRESHOLD: tiny changes (rounding drift, EMA wobble) get
    // suppressed -- only a |cur - prev| / |prev| >= 2 % relative change
    // triggers a colour, so the screen doesn't flash for sub-MB / sub-us
    // differences that don't matter.  2 % was chosen so that going 11.00 KB/cl
    // -> 11.21 KB/cl still reads as "same"; 11.00 -> 11.23 lights up.
    constexpr double kDeltaPctThreshold = 0.02;
    const bool deltaActive = m_overlayStatsHasPrev
        && (std::chrono::steady_clock::now() < m_overlayStatsDeltaUntil);
    auto deltaColour = [&](double cur, double prev) -> XMVECTOR {
        if (!deltaActive) return kSubtle;
        const double base = std::max(std::abs(prev), 1e-9);
        const double rel  = std::abs(cur - prev) / base;
        if (rel < kDeltaPctThreshold) return kSubtle;
        return (cur < prev) ? kGreen : kRed;
    };
    auto deltaColourU = [&](UINT64 cur, UINT64 prev) -> XMVECTOR {
        if (!deltaActive) return kSubtle;
        const double base = std::max((double)prev, 1.0);
        const double diff = (cur > prev) ? (double)(cur - prev) : (double)(prev - cur);
        if (diff / base < kDeltaPctThreshold) return kSubtle;
        return (cur < prev) ? kGreen : kRed;
    };
    // Convenience: print a value into a small buffer + return a wchar_t* so
    // drawSeg can consume it inline.  Each fmt* call is the SAME format string
    // so cur and prev are formatted identically (avoids "0.49 vs 0.495"
    // false-equal artifacts when the difference is below the displayed precision).
    auto fmt2 = [](wchar_t* dst, size_t cch, const wchar_t* f, double v) {
        swprintf_s(dst, cch, f, v); return dst;
    };
    wchar_t fnum[64];   // scratch for formatted numbers

    wchar_t buf[256];

    // Line 0: adapter name (slightly bigger than the body lines).
    swprintf_s(buf, L"Adapter: %s", m_deviceResources->GetAdapterDescription());
    draw(buf, pos, kWhite, /*relScale*/1.10f);
    pos.y += kLineH * 1.6f;

    // Triangle-count short form.
    auto formatTris = [](UINT n, wchar_t* out, size_t cch) {
        if      (n >= 1000000u) swprintf_s(out, cch, L"%.1fM", n / 1.0e6);
        else if (n >= 1000u)    swprintf_s(out, cch, L"%.1fK", n / 1.0e3);
        else                    swprintf_s(out, cch, L"%u",    n);
    };
    // ----- All numeric stats below read from m_overlayStats (snapshotted on
    //       config change) so the per-frame overhead of this overlay is just
    //       a few swprintf_s calls + the SpriteBatch draws.  Only FPS is
    //       computed live.  See OverlayStats / CaptureOverlayStatsSnapshot
    //       in D3D12RaytracingClusteredGeometry.h/.cpp for the snap logic.
    //
    // Layout convention:
    //   - Blue (kAccent)  = section headers + top-level counts.
    //   - White (kSubtle) = stat rows underneath, indented two spaces.
    // That's the only colour-meaning rule.  Don't sprinkle blue on data rows.
    const auto& s = m_overlayStats;
    const float kSectionGap = kLineH * 0.4f;

    // ----- SCENE-WIDE counts (blue header + indented detail, matching the
    //       rest of the layout).  Includes BOTH static and animated paths so
    //       the numbers add up consistently -- m_totalClusterCount is
    //       static-only (used by the build code) but the user-facing scene
    // ----- SCENE: scene-wide counts.  No delta colouring -- cluster/tri
    //       counts are scene properties, not "better/worse" knobs.
    wchar_t trisBuf[16];
    formatTris(s.totalTriangleCount, trisBuf, _countof(trisBuf));
    const UINT animClusters = m_animatedObjectEnabled ? m_animatedObject.clusterCount : 0u;
    const UINT sceneBlasCount    = (UINT)m_objects.size() + (m_animatedObjectEnabled ? 1u : 0u);
    const UINT sceneClusterCount = s.totalClusterCount + animClusters;
    draw(L"SCENE:", pos, kAccent);
    pos.y += kLineH;
    swprintf_s(buf, L"  %u BLAS  /  %u CLAS  /  %s tris",
               sceneBlasCount, sceneClusterCount, trisBuf);
    draw(buf, pos, kSubtle);
    pos.y += kLineH + kSectionGap;

    // ----- STATIC section -----
    // NOTE on precision-vs-memory: staticClasAllocBytes is the RESULT BUFFER
    // ALLOCATION, not the bytes the driver actually emitted into it.  In
    // Implicit-dest mode the buffer is sized for worst-case (max possible
    // per-cluster BVH leaf) and is *insensitive* to PositionTruncateBitCount
    // / compressed1 bits-per-component.  In GetSizes / Compact modes the
    // buffer is sized after a GetSizes probe returns per-cluster actual
    // sizes, so the slider visibly shrinks/grows it.  Either way we show
    // staticClasActualBytes alongside so the precision effect is visible.
    draw(L"STATIC:", pos, kAccent);
    pos.y += kLineH;
    {
        const auto& p = m_overlayStatsPrev;
        const double allocMb     = s.staticClasAllocBytes   / (1024.0 * 1024.0);
        const double prevAllocMb = p.staticClasAllocBytes   / (1024.0 * 1024.0);
        const double actualMb    = s.staticClasActualBytes  / (1024.0 * 1024.0);
        const double prevActMb   = p.staticClasActualBytes  / (1024.0 * 1024.0);
        const double scratchMb   = s.staticClasScratchBytes / (1024.0 * 1024.0);
        const double prevScrMb   = p.staticClasScratchBytes / (1024.0 * 1024.0);
        const double avgKb       = (s.totalClusterCount > 0)
            ? (double)s.staticClasAllocBytes / (double)s.totalClusterCount / 1024.0 : 0.0;
        const double prevAvgKb   = (p.totalClusterCount > 0)
            ? (double)p.staticClasAllocBytes / (double)p.totalClusterCount / 1024.0 : 0.0;

        XMFLOAT2 c = pos;
        drawSeg(L"  CLAS ", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", allocMb), c, deltaColour(allocMb, prevAllocMb));
        drawSeg(L" MB alloc  (", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", actualMb), c, deltaColour(actualMb, prevActMb));
        drawSeg(L" MB actual, avg ", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", avgKb), c, deltaColour(avgKb, prevAvgKb));
        drawSeg(L" KB/cl)", c, kSubtle);
        pos.y += kLineH;

        c = pos;
        drawSeg(L"  scratch ", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", scratchMb), c, deltaColour(scratchMb, prevScrMb));
        drawSeg(L" MB", c, kSubtle);
        pos.y += kLineH;
    }
    pos.y += kSectionGap;

    // ----- ANIMATED section -----
    if (m_animatedObjectEnabled)
    {
        const auto& a = m_animatedObject;
        const auto& p = m_overlayStatsPrev;
        draw(L"ANIMATED:", pos, kAccent);
        pos.y += kLineH;
        swprintf_s(buf, L"  %u clusters / %u verts / %u tris   (template + per-frame template instantiate path)",
                   a.clusterCount, a.totalVertexCount, a.mesh.totalTriangles);
        draw(buf, pos, kSubtle); pos.y += kLineH;

        const double tplMb     = s.animatedTemplateBytes            / (1024.0 * 1024.0);
        const double prevTplMb = p.animatedTemplateBytes            / (1024.0 * 1024.0);
        const double pfMb      = s.animatedPerFrameClasAllocBytes   / (1024.0 * 1024.0);
        const double prevPfMb  = p.animatedPerFrameClasAllocBytes   / (1024.0 * 1024.0);
        const double pfActMb   = s.animatedPerFrameClasActualBytes  / (1024.0 * 1024.0);
        const double prevPfAct = p.animatedPerFrameClasActualBytes  / (1024.0 * 1024.0);
        const double pfScrMb   = s.animatedPerFrameClasScratchBytes / (1024.0 * 1024.0);
        const double prevPfScr = p.animatedPerFrameClasScratchBytes / (1024.0 * 1024.0);
        const double aBlasMb   = s.animatedBlasBytes                / (1024.0 * 1024.0);
        const double prevABlas = p.animatedBlasBytes                / (1024.0 * 1024.0);

        XMFLOAT2 c = pos;
        drawSeg(L"  templates ", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", tplMb), c, deltaColour(tplMb, prevTplMb));
        drawSeg(L" MB", c, kSubtle);
        pos.y += kLineH;

        c = pos;
        drawSeg(L"  per-frame CLAS ", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", pfMb), c, deltaColour(pfMb, prevPfMb));
        drawSeg(L" MB alloc  (", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", pfActMb), c, deltaColour(pfActMb, prevPfAct));
        drawSeg(L" MB actual)  + scratch ", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", pfScrMb), c, deltaColour(pfScrMb, prevPfScr));
        drawSeg(L" MB", c, kSubtle);
        pos.y += kLineH;

        c = pos;
        drawSeg(L"  BLAS ", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", aBlasMb), c, deltaColour(aBlasMb, prevABlas));
        drawSeg(L" MB", c, kSubtle);
        pos.y += kLineH;
        pos.y += kSectionGap;
    }

    // ----- TOTAL section -----
    {
        const auto& p = m_overlayStatsPrev;
        const UINT64 animatedTotal     = s.animatedTemplateBytes
                                       + s.animatedPerFrameClasAllocBytes
                                       + s.animatedBlasBytes;
        const UINT64 prevAnimatedTotal = p.animatedTemplateBytes
                                       + p.animatedPerFrameClasAllocBytes
                                       + p.animatedBlasBytes;
        const UINT64 grandTotal        = s.staticClasAllocBytes + s.staticBlasTotalBytes
                                       + animatedTotal + s.tlasBytes;
        const UINT64 prevGrandTotal    = p.staticClasAllocBytes + p.staticBlasTotalBytes
                                       + prevAnimatedTotal + p.tlasBytes;
        const UINT64 staticTotal       = s.staticClasAllocBytes + s.staticBlasTotalBytes;
        const UINT64 prevStaticTotal   = p.staticClasAllocBytes + p.staticBlasTotalBytes;

        draw(L"TOTAL:", pos, kAccent);
        pos.y += kLineH;

        const double grandMb = grandTotal / (1024.0 * 1024.0);
        const double statMb  = staticTotal / (1024.0 * 1024.0);
        const double animMb  = animatedTotal / (1024.0 * 1024.0);
        const double tlasMb  = s.tlasBytes / (1024.0 * 1024.0);
        XMFLOAT2 c = pos;
        drawSeg(L"  AS memory ", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", grandMb), c, deltaColourU(grandTotal, prevGrandTotal));
        drawSeg(L" MB   (static ", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", statMb), c, deltaColourU(staticTotal, prevStaticTotal));
        drawSeg(L" + animated ", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", animMb), c, deltaColourU(animatedTotal, prevAnimatedTotal));
        drawSeg(L" + TLAS ", c, kSubtle);
        drawSeg(fmt2(fnum, _countof(fnum), L"%.2f", tlasMb), c, deltaColourU(s.tlasBytes, p.tlasBytes));
        drawSeg(L")", c, kSubtle);
        pos.y += kLineH;
        pos.y += kSectionGap;
    }

    // ----- PER-FRAME section -----
    // Times are snapped from the EMA a few frames after each config change so
    // the displayed value belongs to the current config (not a smear of pre +
    // post-rebuild samples).  Splitting INSTANTIATE vs BLAS vs TLAS is what
    // makes the precision / vertex-format sweep meaningful: changing
    // precision should move ONLY the INSTANTIATE column.
    if (m_animatedObjectEnabled)
    {
        const auto& p = m_overlayStatsPrev;
        draw(L"PER-FRAME:", pos, kAccent);
        pos.y += kLineH;
        if (s.pfTimingValid)
        {
            const double totalMs     = s.pfInstantiateMs + s.pfBlasRebuildMs + s.pfTlasRebuildMs;
            const double prevTotalMs = p.pfInstantiateMs + p.pfBlasRebuildMs + p.pfTlasRebuildMs;
            // Only colour-compare the per-frame timings if prev had valid
            // timing too (else we'd compare against a stale 0.0 every time).
            const bool   prevValid   = p.pfTimingValid;
            auto pfPick = [&](double cur, double prev) -> XMVECTOR {
                if (!deltaActive || !prevValid) return kSubtle;
                const double base = std::max(std::abs(prev), 1e-9);
                const double rel  = std::abs(cur - prev) / base;
                if (rel < kDeltaPctThreshold) return kSubtle;
                return (cur < prev) ? kGreen : kRed;
            };
            // Auto-picked unit so the user always knows what scale they're
            // looking at -- "0.073" is ambiguous (ms? us? s?), "73 us" /
            // "0.073 ms" / "1.5 s" are not.  Switch us<->ms at the 1 ms mark
            // and ms<->s at the 1000 ms mark, both round numbers.
            //
            // NOTE: we deliberately use the ASCII "us" rather than the U+00B5
            // micro-sign because MakeSpriteFont's default character range only
            // covers ASCII 32-126 -- the spritefont file we ship has NO glyph
            // for the micro-sign and SpriteFont::DrawString throws std::runtime_error
            // ("character not in the font") when it sees one, which propagates
            // out of WndProc as STATUS_FATAL_USER_CALLBACK_EXCEPTION (0xC000041D).
            // Regenerating the spritefont with /CharacterRegion would also work
            // but ASCII "us" is universally readable and avoids the asset rebuild.
            auto fmtTime = [](wchar_t* dst, size_t cch, double ms) -> wchar_t* {
                if (ms >= 1000.0)      swprintf_s(dst, cch, L"%.3f s",  ms / 1000.0);
                else if (ms >= 1.0)    swprintf_s(dst, cch, L"%.3f ms", ms);
                else                   swprintf_s(dst, cch, L"%.1f us", ms * 1000.0);
                return dst;
            };

            XMFLOAT2 c = pos;
            drawSeg(L"  rebuild ", c, kSubtle);
            drawSeg(fmtTime(fnum, _countof(fnum), totalMs), c, pfPick(totalMs, prevTotalMs));
            drawSeg(L"   (template instantiate ", c, kSubtle);
            drawSeg(fmtTime(fnum, _countof(fnum), s.pfInstantiateMs), c, pfPick(s.pfInstantiateMs, p.pfInstantiateMs));
            drawSeg(L"  +  BLAS ", c, kSubtle);
            drawSeg(fmtTime(fnum, _countof(fnum), s.pfBlasRebuildMs), c, pfPick(s.pfBlasRebuildMs, p.pfBlasRebuildMs));
            drawSeg(L"  +  TLAS ", c, kSubtle);
            drawSeg(fmtTime(fnum, _countof(fnum), s.pfTlasRebuildMs), c, pfPick(s.pfTlasRebuildMs, p.pfTlasRebuildMs));
            drawSeg(L")", c, kSubtle);
        }
        else
        {
            draw(L"  measuring...", pos, kSubtle);
        }
        pos.y += kLineH;
        pos.y += kSectionGap;
    }

    // ----- FPS section.  The ONLY number that's truly live - we don't
    //       snapshot it because the whole point of FPS is the instantaneous
    //       value the user can watch fluctuate.  Section header in blue +
    //       indented value in white, matching the convention above.
    draw(L"FPS:", pos, kAccent);
    pos.y += kLineH;
    swprintf_s(buf, L"  %u", (unsigned)m_timer.GetFramesPerSecond());
    draw(buf, pos, kSubtle);
    pos.y += kLineH * 1.4f;

    // ----- Interactive controls.  Key prefix coloured amber, label + current
    // value in white -- each line owns its own value so there's no need to
    // hunt up the screen for "what is it set to right now?".  The "(-/+)"
    // hint is omitted because ',' and '.' / '[' and ']' are visually paired
    // keys -- you can tell from the prefix which way each one moves.
    auto drawKeyLine = [&](const wchar_t* keyPrefix, const wchar_t* tail) {
        XMFLOAT2 p = pos;
        draw(keyPrefix, p, kHotkey);
        p.x += measureX(keyPrefix);
        draw(tail, p, kWhite);
        pos.y += kLineH;
    };

    wchar_t kbuf[256];

    swprintf_s(kbuf, L"   CLAS alloc mode:  %s", ClasAllocModeName());
    drawKeyLine(L"[A]", kbuf);

    swprintf_s(kbuf, L"   vertex format:    %s",
               m_vertexMode == VertexMode::Float32_3 ? L"FLOAT32_3" : L"COMPRESSED1");
    drawKeyLine(L"[V]", kbuf);

    if (m_vertexMode == VertexMode::Float32_3)
        swprintf_s(kbuf, L"   cluster precision: %u bits/component   (32-bit float, PositionTruncateBitCount=%u)",
                   32u - m_positionTruncateBits, m_positionTruncateBits);
    else
        swprintf_s(kbuf, L"   cluster precision: %u bits/component   (shared exponent)",
                   m_compressedBitsPerComponent);
    drawKeyLine(L"[ ]", kbuf);

    swprintf_s(kbuf, L"   ray bounces:      reflection %u   refraction %u",
               ReflectionBounces(), RefractionBounces());
    drawKeyLine(L", .", kbuf);

    swprintf_s(kbuf, L"   animation:        %s", m_animPaused ? L"PAUSED" : L"playing");
    drawKeyLine(L"[P]", kbuf);

    m_spriteBatch->End();
}



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
}

void D3D12RaytracingClusteredGeometry::OnRender()
{
    if (!m_deviceResources->IsWindowVisible()) return;

    // For --screenshot-at <seconds>, advance the animation clock straight to
    // the requested timestamp on the first rendered frame (so the orbit camera
    // is at the right angle when we capture).
    if (m_screenshotAtSeconds >= 0 && !m_screenshotTaken && m_framesRendered == 0)
    {
        m_animSeconds = m_screenshotAtSeconds;
        UpdateSceneConstantBuffer();
    }

    DoRender();
    ++m_framesRendered;

    bool capture = false;
    if (m_screenshotFrame >= 0 && (UINT)m_screenshotFrame < m_framesRendered && !m_screenshotTaken)
        capture = true;
    // Wait a few frames for swap chain warm-up before time-based capture too.
    // Need at least kPerFrameRingSlots * 2 + 5 frames to also get stable
    // per-frame timestamp EMA reads in the log on shutdown (otherwise the
    // ring buffer hasn't filled yet).
    if (m_screenshotAtSeconds >= 0 && m_framesRendered >= 15 && !m_screenshotTaken)
        capture = true;

    if (capture)
    {
        m_screenshotTaken = true;
        CaptureBackBufferToFile(m_screenshotPath);
        PostQuitMessage(0);
    }
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
    if (m_animatedObjectEnabled)
    {
        if (m_pfFramesCaptured >= kPerFrameRingSlots)
        {
            // The slot we're about to write was last written N frames ago and
            // resolved on the GPU before this frame was even submitted.
            UINT readSlot = m_pfWriteSlot;
            UINT64 ts[kPerFrameTsPerSlot] = {};
            CD3DX12_RANGE readRange(readSlot * kPerFrameTsPerSlot * sizeof(UINT64),
                                   (readSlot + 1) * kPerFrameTsPerSlot * sizeof(UINT64));
            void* mapped = nullptr;
            if (SUCCEEDED(m_pfQueryReadback->Map(0, &readRange, &mapped)))
            {
                memcpy(ts, (uint8_t*)mapped + readSlot * kPerFrameTsPerSlot * sizeof(UINT64),
                       sizeof(ts));
                D3D12_RANGE noWrite = { 0, 0 };
                m_pfQueryReadback->Unmap(0, &noWrite);

                const double freq = (double)m_timestampFrequency;
                // 3 op pairs: (0,1)=INSTANTIATE, (2,3)=BLAS, (4,5)=TLAS.
                const double instMs = (ts[1] >= ts[0]) ? (double)(ts[1] - ts[0]) * 1000.0 / freq : 0.0;
                const double blasMs = (ts[3] >= ts[2]) ? (double)(ts[3] - ts[2]) * 1000.0 / freq : 0.0;
                const double tlasMs = (ts[5] >= ts[4]) ? (double)(ts[5] - ts[4]) * 1000.0 / freq : 0.0;
                // EMA, alpha=0.1 for a sub-second smoothing window.
                constexpr double a = 0.1;
                m_pfInstantiateMs = m_pfInstantiateMs * (1.0 - a) + instMs * a;
                m_pfBlasRebuildMs = m_pfBlasRebuildMs * (1.0 - a) + blasMs * a;
                m_pfTlasRebuildMs = m_pfTlasRebuildMs * (1.0 - a) + tlasMs * a;

                // Snap timing into m_overlayStats once the ring has refilled
                // post-rebuild.  After CaptureOverlayStatsSnapshot arms the
                // countdown to (kPerFrameRingSlots+2), we decrement here once
                // per frame in which we successfully read a sample; when it
                // hits 0 the EMAs reflect post-rebuild timings and we lock
                // them in as the displayed values until the next rebuild.
                if (m_overlayStatsArmCountdown > 0)
                {
                    --m_overlayStatsArmCountdown;
                    if (m_overlayStatsArmCountdown == 0)
                    {
                        m_overlayStats.pfInstantiateMs = m_pfInstantiateMs;
                        m_overlayStats.pfBlasRebuildMs = m_pfBlasRebuildMs;
                        m_overlayStats.pfTlasRebuildMs = m_pfTlasRebuildMs;
                        m_overlayStats.pfTimingValid   = true;
                    }
                }
            }
        }

        // Record this frame's per-frame work.  UpdateAnimatedObjectPerFrame
        // emits timestamp pairs at base+0/1 (INSTANTIATE) and base+2/3 (BLAS);
        // we bracket the TLAS rebuild here at base+4/5.
        const UINT base = m_pfWriteSlot * kPerFrameTsPerSlot;
        UpdateAnimatedObjectPerFrame(base);

        cl4->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, base + 4);
        RebuildTlasPerFrame();
        cl4->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, base + 5);

        cl->ResolveQueryData(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
            base, kPerFrameTsPerSlot, m_pfQueryReadback.Get(),
            base * sizeof(UINT64));

        m_pfWriteSlot = (m_pfWriteSlot + 1) % kPerFrameRingSlots;
        if (m_pfFramesCaptured < kPerFrameRingSlots * 2)
            ++m_pfFramesCaptured;
    }
    cl4->SetComputeRootSignature(m_globalRootSignature.Get());
    ID3D12DescriptorHeap* heaps[] = { m_descriptorHeap.Get() };
    cl4->SetDescriptorHeaps(_countof(heaps), heaps);
    cl4->SetComputeRootDescriptorTable(GlobalRootSig::OutputUAVSlot, m_raytracingOutputUAV);
    cl4->SetComputeRootShaderResourceView(GlobalRootSig::AccelerationStructureSlot,
        m_tlasBuffer->GetGPUVirtualAddress());
    cl4->SetComputeRootConstantBufferView(GlobalRootSig::SceneCBVSlot, m_sceneCB->GetGPUVirtualAddress());
    cl4->SetComputeRootShaderResourceView(GlobalRootSig::MaterialsSRVSlot,
        m_materialsBuffer->GetGPUVirtualAddress());
    cl4->SetComputeRootShaderResourceView(GlobalRootSig::ClusterNormalsSRVSlot,
        m_clusterNormalsBuffer->GetGPUVirtualAddress());
    cl4->SetComputeRootShaderResourceView(GlobalRootSig::ClusterIndicesSRVSlot,
        m_clusterIndicesBuffer->GetGPUVirtualAddress());
    cl4->SetComputeRootShaderResourceView(GlobalRootSig::ClusterOffsetsSRVSlot,
        m_clusterOffsetsBuffer->GetGPUVirtualAddress());
    cl4->SetComputeRootShaderResourceView(GlobalRootSig::ClusterMetaSRVSlot,
        m_clusterMetaBuffer->GetGPUVirtualAddress());
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
    RenderUI();

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
    SampleLog::Write(L"OnDestroy: starting shutdown\n");
    if (m_animatedObjectEnabled && m_pfFramesCaptured >= kPerFrameRingSlots)
    {
        SampleLog::LogF(L"[per-frame wall-clock, EMA] INSTANTIATE=%.4f ms  BLAS=%.4f ms  TLAS=%.4f ms  total=%.4f ms\n",
                        m_pfInstantiateMs, m_pfBlasRebuildMs, m_pfTlasRebuildMs,
                        m_pfInstantiateMs + m_pfBlasRebuildMs + m_pfTlasRebuildMs);
    }

    if (m_deviceResources)
    {
        // Make sure the GPU is idle before tearing down resources it still
        // references (BLASes, scratch, etc).
        m_deviceResources->WaitForGpu();
    }

    // Explicit unmap of the persistently-mapped CB. Letting the implicit
    // ComPtr destructor handle this CAN make swap chain / dxgi shutdown hang
    // intermittently on the experimental runtime.
    if (m_sceneCB && m_sceneCBMapped)
    {
        m_sceneCB->Unmap(0, nullptr);
        m_sceneCBMapped = nullptr;
    }

    // Animated object: release all its GPU resources.  Args buffers are
    // now DEFAULT-heap UAVs (GPU-filled), no maps to release.
    {
        auto& a = m_animatedObject;
        a.templateInputBuffer.Reset();
        a.templateMetaBuffer.Reset();
        a.templateArgsBuffer.Reset();
        a.templateResultBuffer.Reset();
        a.templateScratchBuffer.Reset();
        a.templateAddressArray.Reset();
        a.restPositionsBuffer.Reset();
        a.perFrameVertexBuffer.Reset();
        a.perFrameInstArgsBuffer.Reset();
        a.vertexOffsetArray.Reset();
        a.perFrameClasResultBuffer.Reset();
        a.perFrameClasScratchBuffer.Reset();
        a.perFrameClasAddressArray.Reset();
        a.blasStorage.Reset();
        a.blasScratchBuffer.Reset();
        a.blasArgsBuffer.Reset();
        a.blasArgsMeta.Reset();
        a.blasResultAddrBuffer.Reset();
        m_animatedObjectEnabled = false;
    }

    // Drop our refs to D3D12 objects in dependency order so destruction is
    // deterministic. (Without this the order is the member-decl order, which
    // would release the device first - DXGI doesn't love that.)
    m_objects.clear();
    m_clusterInputBuffer.Reset();
    m_clasArgsMetaBuffer.Reset();
    m_clasArgsBuffer.Reset();
    m_clasResultBuffer.Reset();
    m_clasScratchBuffer.Reset();
    m_clasAddressArray.Reset();
    m_clasSizeArray.Reset();
    m_clasMoveArgsBuffer.Reset();
    m_blasScratchBuffer.Reset();
    m_blasArgsBuffer.Reset();
    m_blasArgsMeta.Reset();
    m_blasResultAddrBuffer.Reset();
    m_tlasBuffer.Reset();
    m_tlasScratchBuffer.Reset();
    m_tlasInstanceDescs.Reset();
    m_dxrStateObject.Reset();
    m_globalRootSignature.Reset();
    m_rayGenShaderTable.Reset();
    m_missShaderTable.Reset();
    m_hitGroupShaderTable.Reset();
    m_animComputePSO.Reset();
    m_animComputeRS.Reset();
    m_fillInstArgsPSO.Reset();
    m_fillInstArgsRS.Reset();
    m_fillMoveArgsPSO.Reset();
    m_fillMoveArgsRS.Reset();
    m_fillBlasArgsPSO.Reset();
    m_fillBlasArgsRS.Reset();
    m_fillClasTriArgsPSO.Reset();
    m_fillClasTriArgsRS.Reset();
    m_fillTemplateArgsPSO.Reset();
    m_fillTemplateArgsRS.Reset();
    m_descriptorHeap.Reset();
    m_raytracingOutput.Reset();
    m_sceneCB.Reset();
    // DirectXTK UI resources (must be torn down before the device that
    // owns their backing GPU resources is released).
    m_uiFont.reset();
    m_spriteBatch.reset();
    m_graphicsMemory.reset();
    m_dxr2CommandList.Reset();
    m_dxr2Device.Reset();
    m_dxrCommandList.Reset();
    m_dxrDevice.Reset();

    SampleLog::Write(L"OnDestroy: done\n");
}

void D3D12RaytracingClusteredGeometry::OnKeyDown(UINT8 key)
{
    // ----- A/V/P primary toggles -----
    if (key == 'P' || key == 'p')
    {
        m_animPaused = !m_animPaused;
        SampleLog::LogF(L"[input] animation %s\n", m_animPaused ? L"PAUSED" : L"resumed");
    }
    else if (key == 'A' || key == 'a')
    {
        // Cycle CLAS allocation strategy: Implicit -> GetSizes -> Compact -> ...
        // Triggers a full GPU flush + rebuild of the static AS pipeline.
        switch (m_clasAllocMode)
        {
        case ClasAllocMode::Implicit: m_clasAllocMode = ClasAllocMode::GetSizes; break;
        case ClasAllocMode::GetSizes: m_clasAllocMode = ClasAllocMode::Compact;  break;
        case ClasAllocMode::Compact:  m_clasAllocMode = ClasAllocMode::Implicit; break;
        }
        SampleLog::LogF(L"[input] CLAS alloc mode -> %s\n", ClasAllocModeName());
        RebuildStaticAccelerationStructures(L"alloc-mode toggle");
    }
    else if (key == 'V' || key == 'v')
    {
        // Cycle vertex format: FLOAT32_3 <-> COMPRESSED1.
        // NOTE: COMPRESSED1 currently exposes an NVIDIA driver bug (see the
        // long comment in BuildScene above); on RTX hardware the second
        // cycle may render geometry artifacts on some clusters.  WARP
        // ("d3dconfig device force-warp=true") renders both paths cleanly.
        m_vertexMode = (m_vertexMode == VertexMode::Float32_3)
                     ? VertexMode::Compressed1
                     : VertexMode::Float32_3;
        SampleLog::LogF(L"[input] vertex format -> %s\n",
                        m_vertexMode == VertexMode::Compressed1 ? L"COMPRESSED1" : L"FLOAT32_3");
        RebuildStaticAccelerationStructures(L"vertex-format toggle");
    }
    // ----- ',' / '.' = bounce-depth slider.  See m_bounceSlider in the
    //   header for the canonical mapping table.  Slider direction has a
    //   single meaning at every position -- moving '.' bumps the higher
    //   value first (refraction) and once it saturates at 5, starts
    //   bumping the lower one (reflection).  Moving ',' is the mirror
    //   image.  Going up and back down ALWAYS lands at the same (refl,
    //   refr) pair the slider passed through on the way up -- so the
    //   default +2 gap is restored automatically.  No rebuild needed.
    else if (key == VK_OEM_COMMA)            // ','
    {
        if (m_bounceSlider <= kBounceSliderMin) return;
        --m_bounceSlider;
        SampleLog::LogF(L"[input] bounce slider %d -> refl %u  refr %u\n",
                        m_bounceSlider, ReflectionBounces(), RefractionBounces());
    }
    else if (key == VK_OEM_PERIOD)           // '.'
    {
        if (m_bounceSlider >= kBounceSliderMax) return;
        ++m_bounceSlider;
        SampleLog::LogF(L"[input] bounce slider %d -> refl %u  refr %u\n",
                        m_bounceSlider, ReflectionBounces(), RefractionBounces());
    }
    // ----- '[' / ']' = per-cluster precision slider.  Direction is the same
    //   in both modes: '[' -> LESS precision, ']' -> MORE precision.
    //   Internally:
    //     FLOAT32_3   -> m_positionTruncateBits in [0, 23].  '[' increments
    //                    (truncates more bits); ']' decrements (keeps more).
    //                    23 is float32's mantissa width -- truncating more
    //                    than that just zeros the whole mantissa.
    //     COMPRESSED1 -> m_compressedBitsPerComponent in [1, 16].  '['
    //                    decrements; ']' increments.  16 is the per-axis
    //                    cap in the D3D12 COMPRESSED1 encoding; 1 is the
    //                    minimum (0 would divide-by-zero in our encoder).
    //   Either change triggers a full static-AS rebuild (CLAS is re-encoded
    //   in the COMPRESSED1 case via EncodeCompressedClusters inside Rebuild).
    else if (key == VK_OEM_4 || key == VK_OEM_6)
    {
        const bool wantLess = (key == VK_OEM_4);
        if (m_vertexMode == VertexMode::Float32_3)
        {
            const UINT prev = m_positionTruncateBits;
            if (wantLess)  m_positionTruncateBits = (prev >= 23u) ? prev : prev + 1u;
            else           m_positionTruncateBits = (prev == 0u)  ? 0u  : prev - 1u;
            if (m_positionTruncateBits == prev) return;
            SampleLog::LogF(L"[input] position truncate bits -> %u  (%u bits kept)\n",
                            m_positionTruncateBits, 32u - m_positionTruncateBits);
        }
        else
        {
            const UINT prev = m_compressedBitsPerComponent;
            if (wantLess)  m_compressedBitsPerComponent = (prev <= 1u)  ? 1u  : prev - 1u;
            else           m_compressedBitsPerComponent = (prev >= 16u) ? prev : prev + 1u;
            if (m_compressedBitsPerComponent == prev) return;
            SampleLog::LogF(L"[input] compressed1 bits/component -> %u\n", m_compressedBitsPerComponent);
        }
        RebuildStaticAccelerationStructures(L"precision slider");
    }
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
void D3D12RaytracingClusteredGeometry::CaptureBackBufferToFile(const std::wstring& path)
{
    auto device       = m_deviceResources->GetD3DDevice();
    auto commandQueue = m_deviceResources->GetCommandQueue();

    UINT capturedBackBufferIndex = m_deviceResources->GetPreviousFrameIndex();
    auto swapChain = m_deviceResources->GetSwapChain();
    ComPtr<ID3D12Resource> backBuffer;
    ThrowIfFailed(swapChain->GetBuffer(capturedBackBufferIndex, IID_PPV_ARGS(&backBuffer)));

    auto rtDesc = backBuffer->GetDesc();
    const UINT width  = (UINT)rtDesc.Width;
    const UINT height = (UINT)rtDesc.Height;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    UINT64 totalBytes = 0; UINT rowCount = 0; UINT64 rowSizeInBytes = 0;
    device->GetCopyableFootprints(&rtDesc, 0, 1, 0, &footprint, &rowCount, &rowSizeInBytes, &totalBytes);

    auto heapPropsRB = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto bufDesc     = CD3DX12_RESOURCE_DESC::Buffer(totalBytes);
    ComPtr<ID3D12Resource> readback;
    ThrowIfFailed(device->CreateCommittedResource(&heapPropsRB, D3D12_HEAP_FLAG_NONE,
        &bufDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)));

    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)));
    ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&cl)));
    {
        auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer.Get(),
            D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cl->ResourceBarrier(1, &toCopy);
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = readback.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = backBuffer.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex = 0;
        cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        auto toPresent = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
        cl->ResourceBarrier(1, &toPresent);
    }
    ThrowIfFailed(cl->Close());
    ID3D12CommandList* cls[] = { cl.Get() };
    commandQueue->ExecuteCommandLists(1, cls);
    m_deviceResources->WaitForGpu();

    void* mappedRaw = nullptr;
    D3D12_RANGE readRange = { 0, (SIZE_T)totalBytes };
    ThrowIfFailed(readback->Map(0, &readRange, &mappedRaw));
    HRESULT hrSave = SaveBGRAToPng(path, width, height,
        (const uint8_t*)mappedRaw + footprint.Offset, footprint.Footprint.RowPitch);
    D3D12_RANGE writeRange = { 0, 0 };
    readback->Unmap(0, &writeRange);

    SampleLog::LogF(L"[screenshot] %s %ux%u -> %s\n",
        SUCCEEDED(hrSave) ? L"wrote" : L"FAILED to write", width, height, path.c_str());
}

HRESULT D3D12RaytracingClusteredGeometry::SaveBGRAToPng(const std::wstring& path,
                                                       UINT width, UINT height,
                                                       const uint8_t* data,
                                                       UINT rowPitchBytes)
{
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    bool initedCom = SUCCEEDED(hr);
    auto cleanup = [&](HRESULT result) -> HRESULT { if (initedCom) CoUninitialize(); return result; };
    ComPtr<IWICImagingFactory> factory;
    if (FAILED(hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))) return cleanup(hr);
    ComPtr<IWICStream> stream;
    if (FAILED(hr = factory->CreateStream(&stream))) return cleanup(hr);
    if (FAILED(hr = stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE))) return cleanup(hr);
    ComPtr<IWICBitmapEncoder> encoder;
    if (FAILED(hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder))) return cleanup(hr);
    if (FAILED(hr = encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache))) return cleanup(hr);
    ComPtr<IWICBitmapFrameEncode> frame;
    ComPtr<IPropertyBag2> props;
    if (FAILED(hr = encoder->CreateNewFrame(&frame, &props))) return cleanup(hr);
    if (FAILED(hr = frame->Initialize(props.Get()))) return cleanup(hr);
    if (FAILED(hr = frame->SetSize(width, height))) return cleanup(hr);
    WICPixelFormatGUID pf = GUID_WICPixelFormat32bppBGRA;
    if (FAILED(hr = frame->SetPixelFormat(&pf))) return cleanup(hr);
    const UINT tightStride = width * 4;
    if (rowPitchBytes == tightStride)
    {
        if (FAILED(hr = frame->WritePixels(height, tightStride, tightStride * height,
                                           const_cast<BYTE*>(data)))) return cleanup(hr);
    }
    else
    {
        std::vector<uint8_t> packed((size_t)tightStride * height);
        for (UINT y = 0; y < height; ++y)
            memcpy(packed.data() + (size_t)y * tightStride,
                   data + (size_t)y * rowPitchBytes, tightStride);
        if (FAILED(hr = frame->WritePixels(height, tightStride, tightStride * height,
                                           packed.data()))) return cleanup(hr);
    }
    if (FAILED(hr = frame->Commit())) return cleanup(hr);
    if (FAILED(hr = encoder->Commit())) return cleanup(hr);
    return cleanup(S_OK);
}

