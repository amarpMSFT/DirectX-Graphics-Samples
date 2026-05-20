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

// =====================================================================
// COMPRESSED1-bug isolation switch.
// Set to one of the kIsolate* constants below to mask all TLAS
// instances except the named one (and optionally the floor for context
// reflections).  Used to narrow the NVIDIA-driver COMPRESSED1 rendering
// corruption (works in WARP, broken on HW) down to a single object.
// kIsolateNone = normal scene (every instance visible).
//
// Implementation: instances whose `obj.instanceID` doesn't match the
// isolated id (and the animated ball if it isn't the target) get
// `InstanceMask = 0` so the primary-ray `TraceRay(InclusionMask=0xff)`
// AND skips them.  Geometry, BLAS, CLAS, etc. are still built -- only
// traversal is suppressed -- so the AS-storage stats stay representative.
// =====================================================================
namespace DebugIsolate {
    constexpr UINT kIsolateNone     = 0xFFFFFFFFu;
    constexpr UINT kIsolateSphere0  = 0;
    constexpr UINT kIsolateSphere1  = 1;
    constexpr UINT kIsolateSphere2  = 2;
    constexpr UINT kIsolateSphere3  = 3;
    constexpr UINT kIsolateTorus    = 4;
    constexpr UINT kIsolateCube     = 5;
    constexpr UINT kIsolateFloor    = 6;
    constexpr UINT kIsolateKlein    = 7;
    constexpr UINT kIsolateAnimated = 99;   // synthetic id for the animated ball

    // Which object to isolate.  Set to kIsolateNone for the full scene.
    constexpr UINT kIsolateTarget   = kIsolateCube;
    // Also keep the floor visible (useful so the isolated object has
    // something to reflect / cast shadows on).  Ignored when
    // kIsolateTarget == kIsolateNone.
    constexpr bool kKeepFloor       = false;

    // ---- Hypothesis test for the NVIDIA COMPRESSED1 corruption ----
    // The non-template COMPRESSED1 conformance test (c:\experimental\src\
    // D3D12Conf\Raytracing\IndirectBuild.cpp ~line 2974) creates a
    // dedicated D3D12 resource per cluster for the COMPRESSED1 vertex
    // data (cluster's VertexBuffer GVA points at OFFSET 0 of its own
    // resource).  The sample, for memory efficiency, concatenates all
    // clusters into one shared m_clusterInputBuffer and lets each
    // cluster's VertexBuffer GVA be base+vbOff with offset > 0.
    //
    // When kPerClusterVbExperiment is true AND m_vertexMode is
    // COMPRESSED1, UploadClusterInputs additionally allocates a tiny
    // dedicated UPLOAD-heap resource for each cluster's compressed
    // vertex blob.  The CLAS-args meta struct is extended (28 -> 36
    // bytes) with the per-cluster VB GVA; FillClasFromTrianglesArgs
    // uses that GVA directly instead of base+vbOff.  IndexBuffer keeps
    // the shared-buffer offset (the experiment isolates VB only).  If
    // rendering becomes correct, the bug is NVIDIA-driver mis-decoding
    // COMPRESSED1 when VertexBuffer GVA is non-zero within a larger
    // resource.
    constexpr bool kPerClusterVbExperiment = true;
}
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
        else if (_wcsicmp(argv[i], L"--geometry-mode") == 0 && i + 1 < argc)
        {
            // [T] runtime toggle, set from CLI for headless / scripted runs.
            //   clusters    - DXR2 cluster-based BLAS (default)
            //   traditional - classic DXR1 per-object monolithic BLAS
            if      (_wcsicmp(argv[i+1], L"clusters")    == 0) m_geometryMode = GeometryMode::Clusters;
            else if (_wcsicmp(argv[i+1], L"traditional") == 0) m_geometryMode = GeometryMode::Traditional;
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--trad-alloc") == 0 && i + 1 < argc)
        {
            // [A] runtime toggle in traditional mode, set from CLI.  Mirrors
            // the cluster path's --clas-alloc but two-mode instead of three.
            if      (_wcsicmp(argv[i+1], L"implicit") == 0) m_traditionalAllocMode = TraditionalAllocMode::Implicit;
            else if (_wcsicmp(argv[i+1], L"compact")  == 0) m_traditionalAllocMode = TraditionalAllocMode::Compact;
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--rebuild-mode") == 0 && i + 1 < argc)
        {
            // Per-frame static-AS rebuild mode = [R] runtime toggle, set from CLI
            // for headless / scripted measurement runs.
            //   none      - off (default)
            //   blas      - re-run BUILD_BLAS_FROM_CLAS every frame
            //   clas-blas - re-run static CLAS + BLAS every frame (CLAS only
            //               actually runs in --clas-alloc implicit)
            if      (_wcsicmp(argv[i+1], L"none")      == 0) m_staticRebuildMode = StaticRebuildMode::None;
            else if (_wcsicmp(argv[i+1], L"blas")      == 0) m_staticRebuildMode = StaticRebuildMode::BlasOnly;
            else if (_wcsicmp(argv[i+1], L"clas-blas") == 0) m_staticRebuildMode = StaticRebuildMode::ClasAndBlas;
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--log-pf-every") == 0 && i + 1 < argc)
        {
            // Every N frames, dump the per-frame timing EMA values (animated
            // INSTANTIATE / animated BLAS / TLAS / static BLAS / static CLAS)
            // to the SampleLog so a wrapper script can scrape them.  0 disables.
            int n = _wtoi(argv[i+1]);
            m_logPfEveryFrames = (UINT)std::max(0, n);
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--log-raw-every") == 0 && i + 1 < argc)
        {
            // Every N frames, dump the RAW (un-smoothed) per-frame timestamp
            // deltas, not the EMA.  Useful for catching transients the EMA
            // smooths away.  0 disables.
            int n = _wtoi(argv[i+1]);
            m_logRawPfEveryFrames = (UINT)std::max(0, n);
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--exit-after-frames") == 0 && i + 1 < argc)
        {
            // Quit cleanly after rendering N frames (post-init).  Headless
            // measurement helper -- pair with --log-pf-every to capture a
            // settled timing run and exit.
            int n = _wtoi(argv[i+1]);
            m_exitAfterFrames = (UINT)std::max(0, n);
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--at") == 0 && i + 1 < argc)
        {
            // Schedule an action to fire at a specific frame.  Format:
            //   --at <frame>:<action>
            // where action is one of:
            //   alloc-implicit / alloc-getsizes / alloc-compact   (=> [A] press)
            //   rebuild-none / rebuild-blas / rebuild-clas-blas  (=> [R] press)
            //   log    -- snapshot all 5 per-frame timing values to SampleLog
            //   exit   -- post WM_QUIT
            // Multiple --at args allowed; executed in order at OnRender time.
            // Frame indices are 0-based and reference m_framesRendered AFTER init.
            std::wstring spec = argv[i+1];
            auto colon = spec.find(L':');
            if (colon != std::wstring::npos)
            {
                ScheduledAction a;
                a.frame  = (UINT)_wtoi(spec.substr(0, colon).c_str());
                a.action = spec.substr(colon + 1);
                m_scheduledActions.push_back(std::move(a));
            }
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
    SampleLog::Write(L">>> BuildTradCidLookup\n");
    BuildTradCidLookup();   // built unconditionally for root-sig binding; cluster path doesn't read it
    SampleLog::Write(L">>> BuildPerInstGeomMaterialTable\n");
    BuildPerInstGeomMaterialTable();
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

    // -------------------------------------------------------------
    // Mixed-material demo: split the smallest sphere (sphere3, was
    // amethyst glass) into chrome upper hemisphere + amethyst glass
    // lower hemisphere.  Per-cluster matRegionIdx assignment drives
    //   - the cluster path's CLAS BaseGeometryIndex stamp (so
    //     GeometryIndex() at hit time returns the region),
    //   - the traditional path's per-region geom-desc layout (one
    //     geom desc per region, NOT per cluster),
    //   - per-(InstIdx, GeomIdx) material lookup
    //     (chrome for region 0, amethyst glass for region 1),
    //   - fixed-function shader-table routing via
    //     MultiplierForGeometryContributionToHitGroupIndex=2 (chrome
    //     hemisphere -> OpaqueHitGroup with no any-hit dispatch;
    //     glass hemisphere -> GlassHitGroup whose any-hit runs for
    //     stochastic translucency).
    // Region split is by cluster centroid Y in object space -- the
    // cluster's tile boundaries already align with parametric latitude
    // rings on the UV sphere, so the equator is a clean cluster seam
    // (no partial-cluster splits).
    // -------------------------------------------------------------
    // -------------------------------------------------------------
    // Mixed-material demo: split the smallest sphere (sphere3, was
    // amethyst glass) into chrome upper hemisphere + amethyst glass
    // lower hemisphere.  Per-cluster matRegionIdx assignment drives:
    //   - the traditional path's per-region geom-desc layout (one
    //     geom desc per region, NOT per cluster) + per-(InstIdx,
    //     GeomIdx) material lookup + fixed-function shader-table
    //     routing via MultiplierForGeometryContributionToHitGroupIndex=2
    //     so chrome hits OpaqueHitGroup (no any-hit dispatch) and
    //     glass hits GlassHitGroup (any-hit runs).
    //   - the cluster path's per-cluster material override via
    //     ClusterMeta::materialSlot (CPU-baked from matRegionIdx +
    //     ClusterObject::perRegionMaterialSlot).
    //     [TODO: when the NVIDIA DXR2 preview driver fixes the
    //     non-zero-BaseGeometryIndex hang on CLAS, the cluster path
    //     can also route via GeometryIndex() and the per-cluster
    //     materialSlot becomes redundant -- both paths converge.]
    // Region split is by cluster centroid Y in object space -- the
    // cluster's tile boundaries already align with parametric latitude
    // rings on the UV sphere, so the equator is a clean cluster seam.
    // -------------------------------------------------------------
    for (auto& obj : m_objects)
    {
        if (obj.instanceID != 3) continue;  // only sphere3 (amethyst -> mixed)

        for (auto& cl : obj.mesh.clusters)
        {
            float centroidY = 0.0f;
            for (const auto& p : cl.positions) centroidY += p.y;
            centroidY /= (float)cl.positions.size();
            cl.matRegionIdx = (centroidY >= 0.0f) ? 0u : 1u;   // 0 = upper (chrome), 1 = lower (glass)
        }
        // Region 0 = chrome (material slot 0, same as sphere0's body).
        // Region 1 = amethyst glass (material slot 3, sphere3's original).
        obj.perRegionMaterialSlot = { 0u, 3u };
        SampleLog::LogF(L"[mixed sphere] obj instanceID=%u split into 2 regions "
                        L"(upper hemisphere -> material slot %u, lower -> %u)\n",
                        obj.instanceID,
                        obj.perRegionMaterialSlot[0],
                        obj.perRegionMaterialSlot[1]);
        break;
    }

    // ---- COMPRESSED1 isolation: erase non-cube objects so only a
    // configurable subset of m_objects is built (to bisect which other
    // clusters need to be in the same batched CLAS build to trigger the
    // NVIDIA COMPRESSED1 corruption).
    //
    // BISECT_KEEP_IID env var (comma-separated list of instanceIDs to
    // keep, besides the cube).  Examples:
    //   (unset) -> only cube
    //   "0" -> cube + sphere0
    //   "1,4" -> cube + sphere1 + torus
    //   "0,1,2,3,4,6,8" -> all non-cube objects
    constexpr bool kFilterObjects = true;
    if (DebugIsolate::kIsolateTarget != DebugIsolate::kIsolateNone && kFilterObjects)
    {
        std::set<UINT> keepSet;
        keepSet.insert(DebugIsolate::kIsolateTarget);
        char envBuf[256] = {};
        DWORD envLen = GetEnvironmentVariableA("BISECT_KEEP_IID", envBuf, sizeof(envBuf));
        if (envLen > 0 && envLen < sizeof(envBuf))
        {
            std::string s(envBuf);
            size_t pos = 0;
            while (pos < s.size())
            {
                size_t comma = s.find(',', pos);
                if (comma == std::string::npos) comma = s.size();
                std::string tok = s.substr(pos, comma - pos);
                try { keepSet.insert((UINT)std::stoi(tok)); } catch (...) {}
                pos = comma + 1;
            }
        }
        m_objects.erase(
            std::remove_if(m_objects.begin(), m_objects.end(),
                [&](const ClusterObject& o) { return keepSet.find(o.instanceID) == keepSet.end(); }),
            m_objects.end());
        SampleLog::LogF(L"[isolation] BISECT_KEEP_IID='%hs' -> kept %zu objects:\n",
                        envBuf, m_objects.size());

        // BISECT_DUP_TARGET=N : append N additional COPIES of the
        // remaining target object (offset on x by 1.5 per copy).  Lets
        // us test "minimum additional clusters needed to trigger" by
        // pairing the cube with copies of itself (6 clusters each).
        // Default 0 = no copies.
        char dupBuf[32] = {};
        DWORD dupLen = GetEnvironmentVariableA("BISECT_DUP_TARGET", dupBuf, sizeof(dupBuf));
        UINT nDup = 0;
        if (dupLen > 0) try { nDup = (UINT)std::stoi(dupBuf); } catch (...) {}
        if (nDup > 0 && !m_objects.empty())
        {
            ClusterObject seed = m_objects.front();
            for (UINT k = 0; k < nDup; ++k)
            {
                ClusterObject copy = seed;
                copy.worldPos.x += 1.5f * float(k + 1);
                copy.instanceID  = 100u + k;    // unique
                m_objects.push_back(std::move(copy));
            }
            SampleLog::LogF(L"[isolation] BISECT_DUP_TARGET=%u: now %zu objects\n",
                            nDup, m_objects.size());
        }

        for (const auto& o : m_objects)
            SampleLog::LogF(L"  instanceID=%u clusters=%zu\n",
                            o.instanceID, o.mesh.clusters.size());
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

    if (m_geometryMode == GeometryMode::Clusters)
    {
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
        if (m_animatedObjectEnabled)
        {
            BuildAnimatedObjectSetup();
            UpdateAnimatedObjectPerFrame();
        }
    }
    else
    {
        // Traditional (DXR1) per-object BLAS path.  Slots 0/1 (CLAS) are
        // stamped with a no-op pair so the ResolveQueryData below has a
        // complete range to resolve; overlay shows them as 0 build time
        // (the truth -- trad mode has no CLAS).  Slot 2/3 bracket the
        // traditional BLAS build for the wall-clock readout.
        commandList->EndQuery(m_buildQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        commandList->EndQuery(m_buildQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        commandList->EndQuery(m_buildQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
        BuildTraditionalStaticAS();
        commandList->EndQuery(m_buildQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);
        // Traditional path's animated ball: build the mesh + per-frame VB +
        // restPos + flat IB + BLAS storage.  Subsequent frames just rebuild/
        // refit tradBlasStorage in UpdateAnimatedTradPerFrame -- this init
        // pass also kicks one rebuild so the BLAS contents are valid before
        // the TLAS build below references it.
        if (m_animatedObjectEnabled)
        {
            BuildAnimatedObjectSetup();
            UpdateAnimatedTradPerFrame();
        }
    }

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
    if (m_geometryMode == GeometryMode::Clusters)
        DumpClusterStatsAsync();

    // Initial snapshot of overlay stats so the first frame of rendering shows
    // the correct numbers (subsequent config changes refresh via the
    // CaptureOverlayStatsSnapshot call at the bottom of
    // RebuildStaticAccelerationStructures).
    CaptureOverlayStatsSnapshot();
}

// ---------------------------------------------------------------------------------
// Hook for Win32Application::Run -- decide between SW_NORMAL and SW_MAXIMIZE
// on the initial ShowWindow.  Two opt-outs:
//   1. Software adapter (WARP / Basic Render) -- maximizing on a CPU
//      rasterizer turns each frame into a seconds-long affair; the window
//      stays at the launched 1280x720 so the app is at least interactive.
//   2. Headless run (any of --screenshot-at, --screenshot-frame,
//      --exit-after-frames is set) -- the back-buffer size is significant
//      because captured screenshots use it; maximizing would silently
//      change the screenshot resolution from 1280x720 to the user's
//      desktop res, breaking any reference-image diff.
// Otherwise (interactive HW run) -- maximize, since the 4K canvas is
// where the cluster geometry actually showcases.
// ---------------------------------------------------------------------------------
bool D3D12RaytracingClusteredGeometry::ShouldMaximizeWindowOnLaunch() const
{
    if (m_screenshotAtSeconds >= 0.0 || m_screenshotFrame >= 0 || m_exitAfterFrames > 0)
        return false;
    const wchar_t* desc = m_deviceResources->GetAdapterDescription();
    const bool isSoftware = (wcsstr(desc, L"WARP") != nullptr) ||
                            (wcsstr(desc, L"Basic Render") != nullptr);
    return !isSoftware;
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

    // Thread-safety note: in async-display mode (WARP), this function MUST
    // be invoked on the worker thread, not the UI thread.  Win32Application's
    // WM_KEYDOWN handler enqueues OnKeyDown into the worker's pending-action
    // queue (DeviceResources::EnqueueAsyncAction); the worker drains the
    // queue at the top of each frame and calls OnKeyDown on its own thread,
    // which then lands here.  Previous attempt: hold a mutex during the
    // teardown so UI thread could safely call here -- caused UI freeze on
    // WARP because the mutex blocked the message pump for the worker's
    // multi-second frame.  See DeviceResources.h m_pendingAsyncActions
    // comment for the full design rationale.

    auto commandList      = m_deviceResources->GetCommandList();
    auto commandAllocator = m_deviceResources->GetCommandAllocator();

    // 1) Flush the GPU so the in-flight frame finishes reading from CLAS/BLAS/TLAS
    //    before we tear them down.  ComPtr<>::Reset() releases the underlying
    //    ID3D12Resource - if the GPU is still touching it the runtime errors out.
    m_deviceResources->WaitForGpu();

    // 2) Stash the PRE-rebuild overlay snapshot as the delta-colour baseline.
    //    Must happen BEFORE the build code below -- specifically before
    //    BuildAnimatedObjectSetup -> MeasureAnimatedClasBytesOneShot writes
    //    s.animatedPerFrameClasActualBytes.  If we stash AFTER, prev gets
    //    the new actual-CLAS value and the per-frame-CLAS-actual field
    //    silently stops lighting up green/red on precision changes.
    StashOverlayStatsAsPrev();

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

    // 3) Drop static-object BLAS storage (BOTH paths' per-object resources --
    //    we may be transitioning between modes and want to release
    //    whichever side is currently holding GPU memory).  The active path
    //    re-allocates whichever subset it needs below.
    for (auto& obj : m_objects)
    {
        obj.blasStorage.Reset();   obj.blasGPUVA      = 0;
        obj.tradVertexBuffer.Reset();
        obj.tradIndexBuffer.Reset();
        obj.tradNormalsBuffer.Reset();
        obj.tradBlasStorage.Reset();
        obj.tradBlasScratch.Reset();
        obj.tradBlasGPUVA  = 0;
        obj.tradVertexCount   = 0;
        obj.tradTriangleCount = 0;
        obj.tradBlasResultBytes  = 0;
        obj.tradBlasScratchBytes = 0;
    }
    m_blasScratchBuffer.Reset();
    m_blasArgsBuffer.Reset();
    m_blasArgsMeta.Reset();
    m_blasResultAddrBuffer.Reset();
    m_traditionalStaticTotalResultBytes  = 0;
    m_traditionalStaticTotalActualBytes  = 0;
    m_traditionalStaticTotalScratchBytes = 0;
    m_traditionalStaticBuildMs           = 0.0;

    // 4) Drop TLAS storage.  Per-frame rebuild path keys off these, so they
    //    MUST exist before the next OnRender; BuildTlasClassic recreates them.
    m_tlasBuffer.Reset();
    m_tlasScratchBuffer.Reset();
    m_tlasInstanceDescs.Reset();

    // 5) Reset the command list/allocator and re-run the static build chain
    //    for the currently-active geometry mode.
    ThrowIfFailed(commandAllocator->Reset());
    ThrowIfFailed(commandList->Reset(commandAllocator, nullptr));

    if (m_geometryMode == GeometryMode::Clusters)
    {
        EncodeCompressedClusters();
        UploadClusterInputs();
        BuildClasIndirect();
        BuildBlasFromClasIndirect();
        if (m_animatedObjectEnabled)
        {
            BuildAnimatedObjectSetup();
            UpdateAnimatedObjectPerFrame();
        }
    }
    else
    {
        // Traditional path: per-object DXR1 BLAS + per-frame animated
        // ball (DXR1 rebuild/refit via [F]).
        BuildTraditionalStaticAS();
        if (m_animatedObjectEnabled)
        {
            BuildAnimatedObjectSetup();
            UpdateAnimatedTradPerFrame();
            // Refresh the tri->cid table now that the animated mesh is
            // part of the trad scene -- BuildTraditionalStaticAS already
            // ran it for static-only, this picks up the animated tris.
            BuildTradCidLookup();
        }
    }
    // Per-(InstIdx, GeomIdx) material table must track the animated-
    // enabled state (which flips between modes).
    BuildPerInstGeomMaterialTable();
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
    // RefreshOverlayStatsCurrent (NOT CaptureOverlayStatsSnapshot) -- the
    // prev-stash already happened at the top of this function before
    // BuildAnimatedObjectSetup wrote the new actual-CLAS value into s.
    RefreshOverlayStatsCurrent();
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
                            float surfTintMul, float refrTintMul, float reflTintMul,
                            UINT defaultMaterialSlot,
                            const std::vector<UINT>* perRegionMaterialSlot = nullptr)
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
#if DXR2_BASEGEOMETRYINDEX_DRIVER_WORKAROUND
            // Per-cluster material slot for the cluster-path fallback
            // (see RaytracingHlslCompat.h driver-workaround gate).  When
            // the gate goes to 0 this whole field disappears and the
            // cluster path joins the trad path on g_perInstGeomMaterial.
            m.materialSlot   = (perRegionMaterialSlot && c.matRegionIdx < perRegionMaterialSlot->size())
                                ? (*perRegionMaterialSlot)[c.matRegionIdx]
                                : defaultMaterialSlot;
#else
            (void)perRegionMaterialSlot;
            (void)defaultMaterialSlot;
#endif

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
                     obj.surfTintMul, obj.refrTintMul, obj.reflTintMul,
                     obj.instanceID,
                     obj.perRegionMaterialSlot.empty() ? nullptr : &obj.perRegionMaterialSlot);
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
                     checker, /*surf*/1.0f, /*refr*/0.75f, /*refl*/1.20f,
                     m_animatedObject.instanceID);
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
            UINT vbGvaLo, vbGvaHi;
        };
        static_assert(sizeof(ClasArgsMeta) == 36, "must match the HLSL load offsets");

        // (Optional) per-cluster dedicated VB resources for the
        // COMPRESSED1 NVIDIA-bug hypothesis test.  Each holds ONLY one
        // cluster's compressed vertex blob; GVA = resource start.
        m_perClusterVbResources.clear();
        const bool kUsePerClusterVbResources =
            DebugIsolate::kPerClusterVbExperiment && !useFloat;
        if (kUsePerClusterVbResources)
        {
            // DEFAULT heap (matches conformance MakeBufferAndInitialize
            // path -- conformance MakeBufferAndInitialize at
            // c:\experimental\src\D3D12Conf\Raytracing\IndirectBuild.cpp:405
            // calls MakeBuffer with D3D12_HEAP_TYPE_DEFAULT then uploads
            // via a staging buffer.  Hypothesis: NVIDIA driver may have
            // a COMPRESSED1 decode bug specifically when the per-cluster
            // VertexBuffer GVA points into an UPLOAD-heap resource (the
            // sample's default).  Earlier same experiment with UPLOAD
            // heap did not fix the corruption -- testing DEFAULT now.
            auto defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            auto uploadHeap  = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
            m_perClusterVbResources.resize(m_totalClusterCount);

            // Per-cluster: create DEFAULT-heap resource (COPY_DEST), then
            // upload via a per-cluster staging UPLOAD resource + CopyBufferRegion.
            // The command list is reset+executed by the caller; we issue
            // copies against the current open list and rely on the existing
            // ExecuteCommandList flow.
            std::vector<ComPtr<ID3D12Resource>> stagings(m_totalClusterCount);
            UINT g2 = 0;
            for (const auto& obj : m_objects)
            for (size_t i = 0; i < obj.mesh.clusters.size(); ++i, ++g2)
            {
                const auto& enc = obj.encoded[i];
                const UINT64 sz = (UINT64)enc.TotalBytes();
                // Default-heap destination
                auto bdDst = CD3DX12_RESOURCE_DESC::Buffer(sz);
                ThrowIfFailed(device->CreateCommittedResource(
                    &defaultHeap, D3D12_HEAP_FLAG_NONE, &bdDst,
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                    IID_PPV_ARGS(&m_perClusterVbResources[g2])));
                m_perClusterVbResources[g2]->SetName(L"Per-cluster COMPRESSED1 VB (DEFAULT heap experiment)");
                // Upload-heap staging
                auto bdSrc = CD3DX12_RESOURCE_DESC::Buffer(sz);
                ThrowIfFailed(device->CreateCommittedResource(
                    &uploadHeap, D3D12_HEAP_FLAG_NONE, &bdSrc,
                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                    IID_PPV_ARGS(&stagings[g2])));
                uint8_t* p = nullptr;
                CD3DX12_RANGE noR(0, 0);
                ThrowIfFailed(stagings[g2]->Map(0, &noR, reinterpret_cast<void**>(&p)));
                memcpy(p, &enc.header, sizeof(enc.header));
                if (!enc.bitstream.empty())
                    memcpy(p + sizeof(enc.header), enc.bitstream.data(), enc.bitstream.size());
                stagings[g2]->Unmap(0, nullptr);
                // Copy + transition to NON_PIXEL_SHADER_RESOURCE.
                auto cl = m_deviceResources->GetCommandList();
                cl->CopyBufferRegion(m_perClusterVbResources[g2].Get(), 0,
                                     stagings[g2].Get(), 0, sz);
                auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
                    m_perClusterVbResources[g2].Get(),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                cl->ResourceBarrier(1, &barrier);
            }
            // Stagings stay alive in the local vector until function end --
            // they get destroyed once the upload's command list has been
            // submitted and waited on (the static-build path's
            // WaitForGpu ensures GPU is done before returning, so this
            // vector goes out of scope safely afterwards).
            SampleLog::LogF(L"[per-cluster-vb experiment] allocated %u dedicated COMPRESSED1 VB resources (DEFAULT heap)\n",
                            m_totalClusterCount);
            // Keep stagings alive at function scope until ExecuteCommandList
            // happens.  Move into the resources vector as a sidecar (we
            // don't strictly need them after this function returns, but
            // pinning until end-of-function is the simplest correct thing).
            m_perClusterVbStagings = std::move(stagings);
        }

        std::vector<ClasArgsMeta> meta(m_totalClusterCount);
        gIdx = 0;
        for (const auto& obj : m_objects)
        for (size_t i = 0; i < obj.mesh.clusters.size(); ++i, ++gIdx)
        {
            const auto& src = obj.mesh.clusters[i];
            const auto& mat = m_materials[obj.instanceID];
            const bool isOpaqueLike = (mat.translucency == 0.0f) && (mat.refractivity == 0.0f);
            meta[gIdx].clusterID    = src.clusterID;
            meta[gIdx].triCount     = (UINT)(src.indices.size() / 3);
            meta[gIdx].vertCount    = (UINT)src.positions.size();
            meta[gIdx].vbOff        = (UINT)slots[gIdx].vbOffset;
            meta[gIdx].ibOff        = (UINT)slots[gIdx].ibOffset;
            meta[gIdx].opaqueFlag   = isOpaqueLike
                ? (UINT)D3D12_RTAS_CLUSTERED_GEOMETRY_FLAG_OPAQUE
                : 0u;
            meta[gIdx].matRegionIdx = src.matRegionIdx;
            // Per-cluster VB GVA: experimental path uses dedicated
            // resource (GVA = resource start), default uses
            // baseSharedBuffer + per-cluster offset.
            const D3D12_GPU_VIRTUAL_ADDRESS vbGva =
                kUsePerClusterVbResources
                    ? m_perClusterVbResources[gIdx]->GetGPUVirtualAddress()
                    : (baseGPUVA + (D3D12_GPU_VIRTUAL_ADDRESS)slots[gIdx].vbOffset);
            meta[gIdx].vbGvaLo = (UINT)(vbGva & 0xFFFFFFFFu);
            meta[gIdx].vbGvaHi = (UINT)(vbGva >> 32);
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
void D3D12RaytracingClusteredGeometry::BuildClasGetSizes()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 185 lines.  Never reached
    // at runtime in cube-only / clusters / Implicit alloc / no-rebuild
    // isolation -- all call sites are gated.
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
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 229 lines.  Never reached
    // at runtime in cube-only / clusters / Implicit alloc / no-rebuild
    // isolation -- all call sites are gated.
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
void D3D12RaytracingClusteredGeometry::BuildTraditionalStaticAS()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 320 lines.  Never reached
    // at runtime in cube-only / clusters / Implicit alloc / no-rebuild
    // isolation -- all call sites are gated.
}

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
void D3D12RaytracingClusteredGeometry::BuildTradCidLookup()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 101 lines.  Never read at runtime in cube-only /
    // clusters / Implicit alloc / no-rebuild isolation, BUT DoRender
    // unconditionally binds m_tradTriToCidBuffer / m_tradGeomTriBaseBuffer
    // GVAs as root SRVs -- so allocate 4-byte dummy upload buffers just
    // to give the bindings a valid (zero-init) GPUVA.
    auto device = m_deviceResources->GetD3DDevice();
    uint32_t zero = 0;
    AllocateUploadBuffer(device, &zero, sizeof(zero), &m_tradTriToCidBuffer,    L"trad tri->cid (stub)");
    AllocateUploadBuffer(device, &zero, sizeof(zero), &m_tradGeomTriBaseBuffer, L"trad geom-tri-base (stub)");
}

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
void D3D12RaytracingClusteredGeometry::BuildPerInstGeomMaterialTable()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Include the animated instance (when enabled) -- InstanceIndex()
    // ranges 0..N_static for the animated case, so the table must be
    // sized to cover it.  Animated has a single region using its
    // own per-instance material slot.
    const UINT animSlots = m_animatedObjectEnabled ? 1u : 0u;
    const UINT N_inst    = (UINT)m_objects.size() + animSlots;
    std::vector<UINT> table(N_inst * kMaxGeomsPerInstance, 0);
    for (size_t oi = 0; oi < m_objects.size(); ++oi)
    {
        const auto& obj = m_objects[oi];
        // Discover how many regions the object actually uses (== max
        // matRegionIdx + 1).  Skipped if perRegionMaterialSlot is set,
        // in which case its size IS the region count.
        UINT regionCount = (UINT)obj.perRegionMaterialSlot.size();
        if (regionCount == 0)
        {
            for (const auto& cl : obj.mesh.clusters)
                regionCount = std::max(regionCount, cl.matRegionIdx + 1);
            if (regionCount == 0) regionCount = 1;
        }
        for (UINT r = 0; r < std::min(regionCount, kMaxGeomsPerInstance); ++r)
        {
            const UINT slot = (r < (UINT)obj.perRegionMaterialSlot.size())
                ? obj.perRegionMaterialSlot[r]
                : obj.instanceID;
            table[oi * kMaxGeomsPerInstance + r] = slot;
        }
    }
    if (m_animatedObjectEnabled)
    {
        // Animated instance: single-region with the animated obj's material slot.
        table[m_objects.size() * kMaxGeomsPerInstance + 0] = m_animatedObject.instanceID;
    }
    AllocateUploadBuffer(device, table.data(),
                         table.size() * sizeof(UINT),
                         &m_perInstGeomMaterialBuffer,
                         L"Per-(InstIdx, GeomIdx) material-slot lookup");
    SampleLog::LogF(L"[per-inst-geom-material] %zu entries (%zu bytes)\n",
                    table.size(), table.size() * sizeof(UINT));
}


void D3D12RaytracingClusteredGeometry::RebuildStaticBlasPerFrame()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 43 lines.  Never reached
    // at runtime in cube-only / clusters / Implicit alloc / no-rebuild
    // isolation -- all call sites are gated.
}

void D3D12RaytracingClusteredGeometry::RebuildStaticClasPerFrame()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 57 lines.  Never reached
    // at runtime in cube-only / clusters / Implicit alloc / no-rebuild
    // isolation -- all call sites are gated.
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
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 573 lines.  Never reached
    // at runtime in cube-only / clusters / Implicit alloc / no-rebuild
    // isolation -- all call sites are gated.
}

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
void D3D12RaytracingClusteredGeometry::BuildAnimatedTraditionalAS()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 80 lines.  Never reached
    // at runtime in cube-only / clusters / Implicit alloc / no-rebuild
    // isolation -- all call sites are gated.
}

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
void D3D12RaytracingClusteredGeometry::UpdateAnimatedTradPerFrame(UINT pfTimestampBase)
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 90 lines.  Never reached
    // at runtime in cube-only / clusters / Implicit alloc / no-rebuild
    // isolation -- all call sites are gated.
}

void D3D12RaytracingClusteredGeometry::MeasureAnimatedClasBytesOneShot()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 116 lines.  Never reached
    // at runtime in cube-only / clusters / Implicit alloc / no-rebuild
    // isolation -- all call sites are gated.
}

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
void D3D12RaytracingClusteredGeometry::StashOverlayStatsAsPrev()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 17 lines.  Not on the
    // critical path under cube-only isolation (overlay/stats/animated
    // PSOs).
}

// =====================================================================================
// Snapshot m_overlayStats from the current live GPU resource state.  Re-reads
// every displayed value EXCEPT animatedPerFrameClasActualBytes (which is set by
// MeasureAnimatedClasBytesOneShot at end-of-build and persists across calls).
// Also arms the 5s delta-colour fade and the "recalculating..." per-frame
// settle flag.  Does NOT stash prev -- that must already have been done by
// either CaptureOverlayStatsSnapshot (one-shot callers) or by an earlier
// StashOverlayStatsAsPrev (rebuild callers).
// =====================================================================================
void D3D12RaytracingClusteredGeometry::RefreshOverlayStatsCurrent()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 112 lines.  Not on the
    // critical path under cube-only isolation (overlay/stats/animated
    // PSOs).
}

// =====================================================================================
// Convenience wrapper: stash prev, then refresh current.  Use this from any
// caller whose code path does NOT mutate m_overlayStats between the prev-stash
// and the refresh.  Rebuild paths (RebuildStaticAccelerationStructures) MUST
// instead bracket the build with StashOverlayStatsAsPrev() at the top and
// RefreshOverlayStatsCurrent() at the bottom -- otherwise the
// MeasureAnimatedClasBytesOneShot write inside the build steals prev.
// =====================================================================================
void D3D12RaytracingClusteredGeometry::CaptureOverlayStatsSnapshot()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 3 lines.  Not on the
    // critical path under cube-only isolation (overlay/stats/animated
    // PSOs).
}

// =====================================================================================
// Per-frame work for the animated object. Called from DoRender BEFORE the TLAS
// rebuild + DispatchRays. Total work: 1 memcpy (positions) + 2 batched
// ExecuteIndirectRTASOperations calls + 2 UAV barriers.
// =====================================================================================
void D3D12RaytracingClusteredGeometry::UpdateAnimatedObjectPerFrame(UINT pfTimestampBase)
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 141 lines.  Never reached
    // at runtime in cube-only / clusters / Implicit alloc / no-rebuild
    // isolation -- all call sites are gated.
}

void D3D12RaytracingClusteredGeometry::BuildTlasClassic()
{
    auto device = m_deviceResources->GetD3DDevice();
    const UINT N_static = (UINT)m_objects.size();
    const UINT N_anim   = m_animatedObjectEnabled ? 1u : 0u;
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
        // DEBUG ISOLATION (DebugIsolate::kIsolateTarget):
        //   kIsolateNone -> every instance visible (normal scene)
        //   otherwise    -> only the target instance (+ optionally the floor)
        //                   has a non-zero mask; everything else is invisible
        //                   to primary rays.
        if (DebugIsolate::kIsolateTarget == DebugIsolate::kIsolateNone ||
            obj.instanceID == DebugIsolate::kIsolateTarget ||
            (DebugIsolate::kKeepFloor && obj.instanceID == DebugIsolate::kIsolateFloor))
            instances[i].InstanceMask= 0xFF;
        else
            instances[i].InstanceMask= 0x00;
        // Hit-group contribution.  For multi-region objects we route
        // by the FIRST region's material kind (geometryIndex 0 lands
        // there with MultiplierForGeometryContributionToHitGroupIndex=2),
        // and subsequent regions add Multiplier*GeomIdx to pick their
        // own hit group automatically -- so the mixed sphere's chrome
        // region (region 0, slot 0) lands at OpaqueHitGroup and its
        // glass region (region 1, slot 3) lands at GlassHitGroup with
        // NO per-pixel shader branch.
        UINT firstRegionMatSlot = obj.perRegionMaterialSlot.empty()
            ? obj.instanceID
            : obj.perRegionMaterialSlot[0];
        const bool firstIsGlass = needsGlass(m_materials[firstRegionMatSlot], obj);
        // If this is a multi-region instance and ANY region is glass,
        // we need any-hit (FORCE_OPAQUE would skip it).  Detect that
        // separately from firstIsGlass for the FORCE_OPAQUE decision.
        bool anyRegionIsGlass = firstIsGlass;
        for (UINT r = 1; r < (UINT)obj.perRegionMaterialSlot.size(); ++r)
        {
            anyRegionIsGlass = anyRegionIsGlass ||
                needsGlass(m_materials[obj.perRegionMaterialSlot[r]], obj);
        }
        instances[i].InstanceContributionToHitGroupIndex =
            firstIsGlass ? kHitGroupContribGlass : kHitGroupContribOpaque;
        instances[i].Flags = anyRegionIsGlass ? D3D12_RAYTRACING_INSTANCE_FLAG_NONE
                                              : D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
        const bool isGlass = anyRegionIsGlass;  // for stats below
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
        // Per-mode BLAS GVA: cluster path uses obj.blasGPUVA (built via
        // BUILD_BLAS_FROM_CLAS); traditional path uses obj.tradBlasGPUVA
        // (built via classic BuildRaytracingAccelerationStructure).
        instances[i].AccelerationStructure =
            (m_geometryMode == GeometryMode::Clusters) ? obj.blasGPUVA : obj.tradBlasGPUVA;
        (isGlass ? nGlass : nOpaque)++;
    }
    // Animated ball is included in both modes -- the cluster path
    // references obj.blasGPUVA (BLAS-from-CLAS) while the traditional
    // path references obj.tradBlasGPUVA (DXR1 BLAS rebuilt/refitted
    // per frame from perFrameVertexBuffer + tradIndexBuffer).
    const bool hasAnimInst = m_animatedObjectEnabled;
    if (hasAnimInst)
    {
        const auto& a = m_animatedObject;
        XMMATRIX m = XMMatrixScaling(a.worldScale, a.worldScale, a.worldScale)
                   * XMMatrixTranslation(a.worldPos.x, a.worldPos.y, a.worldPos.z);
        XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(instances[N_static].Transform), m);
        instances[N_static].InstanceID  = a.instanceID;
        // DEBUG ISOLATION: see static loop above for semantics.
        if (DebugIsolate::kIsolateTarget == DebugIsolate::kIsolateNone ||
            DebugIsolate::kIsolateTarget == DebugIsolate::kIsolateAnimated)
            instances[N_static].InstanceMask= 0xFF;
        else
            instances[N_static].InstanceMask= 0x00;
        // Animated object has no ClusterObject scene config yet; treat by
        // material only (currently refractive glass -> GlassHitGroup).
        const auto& aMat = m_materials[a.instanceID];
        const bool isGlass = (aMat.refractivity > 0.0f) || (aMat.translucency > 0.0f);
        instances[N_static].InstanceContributionToHitGroupIndex =
            isGlass ? kHitGroupContribGlass : kHitGroupContribOpaque;
        instances[N_static].Flags = isGlass ? D3D12_RAYTRACING_INSTANCE_FLAG_NONE
                                            : D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
        // Per-mode BLAS GVA for the animated instance.
        instances[N_static].AccelerationStructure =
            (m_geometryMode == GeometryMode::Clusters) ? a.blasGPUVA : a.tradBlasGPUVA;
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

void D3D12RaytracingClusteredGeometry::DumpClusterStatsAsync()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 39 lines.  Not on the
    // critical path under cube-only isolation (overlay/stats/animated
    // PSOs).
}

// ---------------------------------------------------------------------------------
// Read back the AS build timestamps from m_buildQueryReadback and compute
// per-operation wall-clock times. Called once after init (post-WaitForGpu).
// ---------------------------------------------------------------------------------
void D3D12RaytracingClusteredGeometry::ReadBuildTimestamps()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 78 lines.  Not on the
    // critical path under cube-only isolation (overlay/stats/animated
    // PSOs).
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
        // Tiny per-instance first-cluster-ID lookup (uint per TLAS
        // instance) -- read only by the traditional-BLAS path so it can
        // compute the same cid the cluster path's ClusterID() returns.
        params[GlobalRootSig::TradTriToCidSRVSlot].InitAsShaderResourceView(6);
        params[GlobalRootSig::TradGeomTriBaseSRVSlot].InitAsShaderResourceView(7);
        params[GlobalRootSig::PerInstGeomMaterialSRVSlot].InitAsShaderResourceView(8);
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

    // 1x1 white texture used by the overlay backing-rect pass below
    // (see RenderUI's drawSeg/draw lambdas).  Created via the same
    // ResourceUploadBatch pattern as the spritefont, dropped into the
    // next descriptor heap slot, GPU handle stashed for SpriteBatch::Draw.
    {
        ResourceUploadBatch upload(device);
        upload.Begin();

        D3D12_RESOURCE_DESC texDesc = {};
        texDesc.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDesc.Width              = 1;
        texDesc.Height             = 1;
        texDesc.DepthOrArraySize   = 1;
        texDesc.MipLevels          = 1;
        texDesc.Format             = DXGI_FORMAT_R8G8B8A8_UNORM;
        texDesc.SampleDesc.Count   = 1;
        texDesc.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        texDesc.Flags              = D3D12_RESOURCE_FLAG_NONE;

        auto defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        ThrowIfFailed(device->CreateCommittedResource(
            &defaultHeap, D3D12_HEAP_FLAG_NONE, &texDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_overlayPanelTexture)));
        m_overlayPanelTexture->SetName(L"Overlay 1x1 white (per-line dark backing source)");

        // One white pixel: R=255 G=255 B=255 A=255.  SpriteBatch tints
        // it with the dark+alpha colour we want when drawing the rect.
        const uint8_t whitePixel[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
        D3D12_SUBRESOURCE_DATA sub = {};
        sub.pData      = whitePixel;
        sub.RowPitch   = sizeof(whitePixel);
        sub.SlicePitch = sizeof(whitePixel);

        upload.Upload(m_overlayPanelTexture.Get(), 0, &sub, 1);
        upload.Transition(m_overlayPanelTexture.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        auto done = upload.End(m_deviceResources->GetCommandQueue());
        done.wait();

        // SRV in the next free heap slot; SpriteBatch::Draw consumes a
        // GPU handle.
        D3D12_CPU_DESCRIPTOR_HANDLE cpu;
        UINT slot = AllocateDescriptor(&cpu);
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format                  = DXGI_FORMAT_R8G8B8A8_UNORM;
        srvDesc.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels     = 1;
        device->CreateShaderResourceView(m_overlayPanelTexture.Get(), &srvDesc, cpu);
        m_overlayPanelTextureGpu = CD3DX12_GPU_DESCRIPTOR_HANDLE(
            m_descriptorHeap->GetGPUDescriptorHandleForHeapStart(), slot, m_descriptorSize);
        SampleLog::LogF(L"[ui] overlay-panel 1x1 texture (descriptor slot %u)\n", slot);
    }
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
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 34 lines.  Not on the
    // critical path under cube-only isolation (overlay/stats/animated
    // PSOs).
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
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 35 lines.  Not on the
    // critical path under cube-only isolation (overlay/stats/animated
    // PSOs).
}


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
void D3D12RaytracingClusteredGeometry::CreateFillMoveArgsPipeline()
{
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 12 lines.  Not on the
    // critical path under cube-only isolation (overlay/stats/animated
    // PSOs).
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
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 12 lines.  Not on the
    // critical path under cube-only isolation (overlay/stats/animated
    // PSOs).
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
    // ===== STUB: body removed for COMPRESSED1 minimal repro =====
    // Original body had 944 lines.  Not on the
    // critical path under cube-only isolation (overlay/stats/animated
    // PSOs).
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

    // ---- Headless measurement helpers (--log-pf-every / --exit-after-frames /
    // --at).  Driven off m_framesRendered, executed once per OnRender, all
    // logged via SampleLog::LogF so a wrapper script can scrape them.
    if (m_logPfEveryFrames > 0 && (m_framesRendered % m_logPfEveryFrames) == 0)
    {
        SampleLog::LogF(L"[pf-stats frame=%u alloc=%ls rebuild=%ls] "
                        L"inst=%.3fus animBlas=%.3fus tlas=%.3fus "
                        L"staticBlas=%.3fus staticClas=%.3fus\n",
                        m_framesRendered, ClasAllocModeName(), StaticRebuildModeName(),
                        m_pfInstantiateMs * 1000.0, m_pfBlasRebuildMs * 1000.0,
                        m_pfTlasRebuildMs   * 1000.0,
                        m_pfStaticBlasMs    * 1000.0,
                        m_pfStaticClasMs    * 1000.0);
    }
    for (size_t i = 0; i < m_scheduledActions.size(); )
    {
        const auto& a = m_scheduledActions[i];
        if (a.frame == m_framesRendered)
        {
            const wchar_t* act = a.action.c_str();
            SampleLog::LogF(L"[scheduled frame=%u] action=%ls\n", m_framesRendered, act);
            if      (_wcsicmp(act, L"alloc-implicit")    == 0)
            {   m_clasAllocMode = ClasAllocMode::Implicit;
                RebuildStaticAccelerationStructures(L"scheduled alloc-implicit"); }
            else if (_wcsicmp(act, L"alloc-getsizes")    == 0)
            {   m_clasAllocMode = ClasAllocMode::GetSizes;
                RebuildStaticAccelerationStructures(L"scheduled alloc-getsizes"); }
            else if (_wcsicmp(act, L"alloc-compact")     == 0)
            {   m_clasAllocMode = ClasAllocMode::Compact;
                RebuildStaticAccelerationStructures(L"scheduled alloc-compact"); }
            else if (_wcsicmp(act, L"rebuild-none")      == 0)
            {   m_staticRebuildMode = StaticRebuildMode::None;
                CaptureOverlayStatsSnapshot(); }
            else if (_wcsicmp(act, L"rebuild-blas")      == 0)
            {   m_staticRebuildMode = StaticRebuildMode::BlasOnly;
                CaptureOverlayStatsSnapshot(); }
            else if (_wcsicmp(act, L"rebuild-clas-blas") == 0)
            {   m_staticRebuildMode = StaticRebuildMode::ClasAndBlas;
                CaptureOverlayStatsSnapshot(); }
            else if (_wcsicmp(act, L"geom-clusters")     == 0)
            {   if (m_clustersAndPtlasSupported) {
                    m_geometryMode = GeometryMode::Clusters;
                    RebuildStaticAccelerationStructures(L"scheduled geom-clusters"); } }
            else if (_wcsicmp(act, L"geom-traditional")  == 0)
            {   m_geometryMode = GeometryMode::Traditional;
                RebuildStaticAccelerationStructures(L"scheduled geom-traditional"); }
            else if (_wcsicmp(act, L"trad-implicit")     == 0)
            {   m_traditionalAllocMode = TraditionalAllocMode::Implicit;
                RebuildStaticAccelerationStructures(L"scheduled trad-implicit"); }
            else if (_wcsicmp(act, L"trad-compact")      == 0)
            {   m_traditionalAllocMode = TraditionalAllocMode::Compact;
                RebuildStaticAccelerationStructures(L"scheduled trad-compact"); }
            else if (_wcsicmp(act, L"anim-rebuild")      == 0)
            {   m_traditionalAnimMode = TraditionalAnimMode::Rebuild;
                CaptureOverlayStatsSnapshot(); }
            else if (_wcsicmp(act, L"anim-refit")        == 0)
            {   m_traditionalAnimMode = TraditionalAnimMode::Refit;
                CaptureOverlayStatsSnapshot(); }
            else if (_wcsicmp(act, L"log")               == 0)
            {   SampleLog::LogF(L"[scheduled-snap frame=%u alloc=%ls rebuild=%ls] "
                                L"inst=%.3fus animBlas=%.3fus tlas=%.3fus "
                                L"staticBlas=%.3fus staticClas=%.3fus\n",
                                m_framesRendered, ClasAllocModeName(), StaticRebuildModeName(),
                                m_pfInstantiateMs * 1000.0, m_pfBlasRebuildMs * 1000.0,
                                m_pfTlasRebuildMs   * 1000.0,
                                m_pfStaticBlasMs    * 1000.0,
                                m_pfStaticClasMs    * 1000.0); }
            else if (_wcsicmp(act, L"exit")              == 0)
            {   PostQuitMessage(0); return; }
            // Remove fired action so it doesn't refire.
            m_scheduledActions.erase(m_scheduledActions.begin() + i);
        }
        else
        {
            ++i;
        }
    }
    if (m_exitAfterFrames > 0 && m_framesRendered >= m_exitAfterFrames)
    {
        PostQuitMessage(0);
        return;
    }
    // ----

    bool capture = false;
    if (m_screenshotFrame >= 0 && (UINT)m_screenshotFrame < m_framesRendered && !m_screenshotTaken)
        capture = true;
    // Wait a few frames for swap chain warm-up before time-based capture too.
    // Need at least kPerFrameRingSlots * 2 + 5 frames to also get stable
    // per-frame timestamp EMA reads in the log on shutdown (otherwise the
    // ring buffer hasn't filled yet).  Pump up to m_pfSnapTargetCount + a
    // bit so the rolling per-frame snap window has actually completed when
    // we capture -- so screenshots after a config-change toggle show real
    // post-toggle numbers, not the "recalculating..." placeholder we
    // display during the ~1 s settle.  Old behaviour was a hardcoded
    // frame 15 which exited too early on toggle-tests.
    if (m_screenshotAtSeconds >= 0 && m_framesRendered >= 80 && !m_screenshotTaken)
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
                // Slot pairs: (0,1)=anim INSTANTIATE, (2,3)=anim BLAS,
                // (4,5)=TLAS, (6,7)=static BLAS rebuild ([R]),
                // (8,9)=static CLAS rebuild ([R] mode 2 + Implicit).
                auto deltaMs = [&](UINT a, UINT b) {
                    return (ts[b] >= ts[a]) ? (double)(ts[b] - ts[a]) * 1000.0 / freq : 0.0;
                };
                const double instMs       = deltaMs(0, 1);
                const double blasMs       = deltaMs(2, 3);
                const double tlasMs       = deltaMs(4, 5);
                const double staticBlasMs = deltaMs(6, 7);
                const double staticClasMs = deltaMs(8, 9);
                // EMA, alpha=0.1 for a sub-second smoothing window.
                constexpr double a = 0.1;
                m_pfInstantiateMs = m_pfInstantiateMs * (1.0 - a) + instMs * a;
                m_pfBlasRebuildMs = m_pfBlasRebuildMs * (1.0 - a) + blasMs * a;
                m_pfTlasRebuildMs = m_pfTlasRebuildMs * (1.0 - a) + tlasMs * a;
                m_pfStaticBlasMs  = m_pfStaticBlasMs  * (1.0 - a) + staticBlasMs * a;
                m_pfStaticClasMs  = m_pfStaticClasMs  * (1.0 - a) + staticClasMs * a;

                // Headless raw-per-frame log (--log-raw): dump the unfiltered
                // timestamp deltas for this frame so we can see transients the
                // EMA would smooth away.  Same gating as --log-pf-every but
                // logs the raw values, not the EMA.  Also logs the current
                // EMA + the latest snapshot value the overlay would show, so
                // we can see if the snapshot diverges from the raw timing.
                if (m_logRawPfEveryFrames > 0 &&
                    (m_framesRendered % m_logRawPfEveryFrames) == 0)
                {
                    SampleLog::LogF(L"[pf-raw frame=%u alloc=%ls rebuild=%ls] "
                                    L"raw=%.3fus ema=%.3fus snap=%.3fus  "
                                    L"(anim raw=%.3f ema=%.3f, animBlas raw=%.3f, tlas raw=%.3f)\n",
                                    m_framesRendered, ClasAllocModeName(), StaticRebuildModeName(),
                                    staticBlasMs * 1000.0,
                                    m_pfStaticBlasMs * 1000.0,
                                    m_overlayStats.pfStaticBlasMs * 1000.0,
                                    instMs * 1000.0, m_pfInstantiateMs * 1000.0,
                                    blasMs * 1000.0, tlasMs * 1000.0);
                }

                // Rolling per-frame snapshot.  Accumulate samples into
                // m_pfSnapAccum; when we have m_pfSnapTargetCount, mean
                // them into m_overlayStats and start the next window.
                // m_pfSnapSkipFramesLeft != 0 means a recent toggle and we
                // need to flush stale ring-buffer entries first.  See the
                // header comment on m_pfSnapAccum for the full mechanism.
                if (m_pfSnapSkipFramesLeft > 0)
                {
                    --m_pfSnapSkipFramesLeft;
                }
                else
                {
                    m_pfSnapAccum.instMs       += instMs;
                    m_pfSnapAccum.blasMs       += blasMs;
                    m_pfSnapAccum.tlasMs       += tlasMs;
                    m_pfSnapAccum.staticBlasMs += staticBlasMs;
                    m_pfSnapAccum.staticClasMs += staticClasMs;
                    ++m_pfSnapSamplesCollected;
                    if (m_pfSnapSamplesCollected >= m_pfSnapTargetCount)
                    {
                        const double inv = 1.0 / m_pfSnapTargetCount;
                        m_overlayStats.pfInstantiateMs = m_pfSnapAccum.instMs       * inv;
                        m_overlayStats.pfBlasRebuildMs = m_pfSnapAccum.blasMs       * inv;
                        m_overlayStats.pfTlasRebuildMs = m_pfSnapAccum.tlasMs       * inv;
                        m_overlayStats.pfStaticBlasMs  = m_pfSnapAccum.staticBlasMs * inv;
                        m_overlayStats.pfStaticClasMs  = m_pfSnapAccum.staticClasMs * inv;
                        m_overlayStats.pfTimingValid   = true;
                        // First post-toggle window completed -- live timings
                        // now reflect the new mode, drop the "recalculating..."
                        // placeholder.
                        m_pfTimingSettlingAfterToggle  = false;
                        m_pfSnapAccum            = PfSnapAccum{};
                        m_pfSnapSamplesCollected = 0;
                    }
                }

            }
        }

        // Record this frame's per-frame work.  Timestamp slot layout:
        //   base+0..1  INSTANTIATE_CLUSTER_TEMPLATES (animated)
        //   base+2..3  BUILD_BLAS_FROM_CLAS  (animated)
        //   base+4..5  TLAS rebuild
        //   base+6..7  static BLAS rebuild  ([R] mode 1 or 2; else no-op)
        //   base+8..9  static CLAS rebuild  ([R] mode 2 + Implicit alloc; else no-op)
        // Static CLAS must run BEFORE static BLAS (BLAS reads CLAS); static
        // BLAS must run BEFORE TLAS (TLAS sees BLAS).  Timestamps emitted
        // unconditionally so the resolve range is always valid -- when the
        // mode says no work, the begin/end pair brackets nothing and the
        // delta is ~0 (just GPU query overhead).
        const UINT base = m_pfWriteSlot * kPerFrameTsPerSlot;
        // Per-frame animated update: cluster path uses the
        // INSTANTIATE_CLUSTER_TEMPLATES + BLAS_FROM_CLAS pipeline; trad
        // path uses BuildRaytracingAccelerationStructure (rebuild or
        // refit per [F]).  Both emit the same 4-timestamp layout
        // (base+0..1 = AnimateBall.cs, base+2..3 = AS build) so the
        // overlay split between "anim CS" and "anim build" works for
        // both modes uniformly.
        if (m_geometryMode == GeometryMode::Clusters)
            UpdateAnimatedObjectPerFrame(base);
        else
            UpdateAnimatedTradPerFrame(base);

        cl4->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, base + 8);
        if (m_staticRebuildMode == StaticRebuildMode::ClasAndBlas &&
            m_geometryMode == GeometryMode::Clusters)
            RebuildStaticClasPerFrame();
        cl4->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, base + 9);

        cl4->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, base + 6);
        if (m_staticRebuildMode != StaticRebuildMode::None &&
            m_geometryMode == GeometryMode::Clusters)
            RebuildStaticBlasPerFrame();
        cl4->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, base + 7);

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
    cl4->SetComputeRootShaderResourceView(GlobalRootSig::TradTriToCidSRVSlot,
        m_tradTriToCidBuffer->GetGPUVirtualAddress());
    cl4->SetComputeRootShaderResourceView(GlobalRootSig::TradGeomTriBaseSRVSlot,
        m_tradGeomTriBaseBuffer->GetGPUVirtualAddress());
    cl4->SetComputeRootShaderResourceView(GlobalRootSig::PerInstGeomMaterialSRVSlot,
        m_perInstGeomMaterialBuffer->GetGPUVirtualAddress());
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
    m_overlayPanelTexture.Reset();
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
        // [A] cycles the alloc strategy for whichever path is active:
        //   Clustered BLAS: CLAS alloc mode (Implicit -> GetSizes -> Compact -> ...)
        //   Traditional BLAS: BLAS alloc mode (Compact <-> Implicit)
        // Either triggers a full GPU flush + rebuild of the static AS pipeline.
        if (m_geometryMode == GeometryMode::Clusters)
        {
            switch (m_clasAllocMode)
            {
            case ClasAllocMode::Implicit: m_clasAllocMode = ClasAllocMode::GetSizes; break;
            case ClasAllocMode::GetSizes: m_clasAllocMode = ClasAllocMode::Compact;  break;
            case ClasAllocMode::Compact:  m_clasAllocMode = ClasAllocMode::Implicit; break;
            }
            SampleLog::LogF(L"[input] CLAS alloc mode -> %s\n", ClasAllocModeName());
        }
        else
        {
            m_traditionalAllocMode = (m_traditionalAllocMode == TraditionalAllocMode::Compact)
                                     ? TraditionalAllocMode::Implicit
                                     : TraditionalAllocMode::Compact;
            SampleLog::LogF(L"[input] traditional BLAS alloc mode -> %s\n", TraditionalAllocModeName());
        }
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
    else if (key == 'R' || key == 'r')
    {
        // Cycle [R] static-rebuild mode: None -> BlasOnly -> ClasAndBlas -> ...
        // Per-frame work only; doesn't touch buffers (no GPU flush, no
        // RebuildStaticAccelerationStructures).  The next OnRender picks
        // up the new mode via m_staticRebuildMode and emits the
        // corresponding RTAS ops.  Snap overlay stats so the new mode's
        // per-frame timing values get armed for capture.
        // [R] is cluster-mode-only -- no-op in traditional mode.
        if (m_geometryMode != GeometryMode::Clusters) return;
        switch (m_staticRebuildMode)
        {
        case StaticRebuildMode::None:        m_staticRebuildMode = StaticRebuildMode::BlasOnly;    break;
        case StaticRebuildMode::BlasOnly:    m_staticRebuildMode = StaticRebuildMode::ClasAndBlas; break;
        case StaticRebuildMode::ClasAndBlas: m_staticRebuildMode = StaticRebuildMode::None;        break;
        }
        SampleLog::LogF(L"[input] static rebuild mode -> %s\n", StaticRebuildModeName());
        CaptureOverlayStatsSnapshot();
    }
    else if (key == 'T' || key == 't')
    {
        // Cycle [T] geometry mode: Clusters <-> Traditional.  Triggers a
        // full static-AS rebuild (heavy, like [A]).  Locked off if the
        // adapter doesn't support clusters (we're already pinned to
        // Traditional and stay there).
        if (!m_clustersAndPtlasSupported) return;
        m_geometryMode = (m_geometryMode == GeometryMode::Clusters)
                         ? GeometryMode::Traditional
                         : GeometryMode::Clusters;
        SampleLog::LogF(L"[input] geometry mode -> %s\n", GeometryModeName());
        RebuildStaticAccelerationStructures(L"geometry-mode toggle");
    }
    else if (key == 'F' || key == 'f')
    {
        // [F] toggles animated-BLAS update strategy in traditional mode.
        // No effect in cluster mode (the cluster INSTANTIATE+BLAS-from-CLAS
        // pipeline doesn't have a refit/rebuild dichotomy).  No GPU
        // rebuild needed; the next per-frame animated update picks up the
        // new flag.
        if (m_geometryMode == GeometryMode::Clusters) return;
        m_traditionalAnimMode = (m_traditionalAnimMode == TraditionalAnimMode::Rebuild)
                                ? TraditionalAnimMode::Refit
                                : TraditionalAnimMode::Rebuild;
        SampleLog::LogF(L"[input] traditional animated mode -> %s\n", TraditionalAnimModeName());
        CaptureOverlayStatsSnapshot();
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

