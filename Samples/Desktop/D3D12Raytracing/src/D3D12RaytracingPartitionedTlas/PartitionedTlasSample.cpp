//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// PartitionedTlasSample.cpp  (Milestone 2a)
//
// What's here now:
//   * Device + DXR2 surfaces (queried; logged)
//   * One ball mesh, one BLAS (traditional DXR1 for this milestone -- see
//     BallAssets.h for the rationale; cluster BLAS lands in 2b)
//   * Two ITlasSystem implementations selectable via --tlas-mode
//     (`partitioned` default / `traditional`).  Same scene, same shader,
//     same render.
//   * Raygen + miss + closest-hit shader, output UAV, scene CBV
//   * Headless `--screenshot N path.png` + `--exit-after-frames N`
//
// What's not here yet (see docs/design.md):
//   * Multiple instances, partition grid, flock, displacement,
//     cluster-template animation, GPU-side WRITE/UPDATE/TRANSLATE arg fill
//   * GPU timestamps per pass (the stats hook in ITlasSystem is wired but
//     only CPU-known fields populate today)
//

#include "stdafx.h"
#include "PartitionedTlasSample.h"

#include "TraditionalTlasSystem.h"
#include "PtlasSystem.h"
#include "GpuBuffer.h"
#include "ProceduralGeometry.h"
#include "RaytracingHlslCompat.h"

#include "CompiledShaders/Raytracing.hlsl.h"  // produced by FxCompile -> g_pRaytracing

#include <DirectXMath.h>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

// Aligned helpers (kept local to keep the file self-contained).
static UINT64 AlignUp(UINT64 v, UINT64 a) { return (v + (a - 1)) & ~(a - 1); }

// =============================================================================
// CLI
// =============================================================================

PartitionedTlasSample::PartitionedTlasSample(UINT width, UINT height, std::wstring name)
    : DXSample(width, height, name)
{
}

void PartitionedTlasSample::ParseCommandLineArgs(_In_reads_(argc) WCHAR* argv[], int argc)
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
        else if (_wcsicmp(argv[i], L"--exit-after-frames") == 0 && i + 1 < argc)
        {
            int n = _wtoi(argv[i + 1]);
            m_exitAfterFrames = (UINT)std::max(0, n);
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--tlas-mode") == 0 && i + 1 < argc)
        {
            if      (_wcsicmp(argv[i+1], L"partitioned") == 0) m_tlasMode = TlasMode::Partitioned;
            else if (_wcsicmp(argv[i+1], L"traditional") == 0) m_tlasMode = TlasMode::Traditional;
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--grid") == 0 && i + 1 < argc)
        {
            // --grid WxHxD  e.g. --grid 8x8x8
            uint32_t gx = 0, gy = 0, gz = 0;
            if (swscanf_s(argv[i+1], L"%ux%ux%u", &gx, &gy, &gz) == 3)
            {
                m_scene.gridX = std::max(1u, gx);
                m_scene.gridY = std::max(1u, gy);
                m_scene.gridZ = std::max(1u, gz);
            }
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--balls-per-side") == 0 && i + 1 < argc)
        {
            int n = _wtoi(argv[i + 1]);
            m_scene.ballsPerSide = (uint32_t)std::max(1, n);
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--log-stats-every") == 0 && i + 1 < argc)
        {
            int n = _wtoi(argv[i + 1]);
            m_logStatsEvery = (UINT)std::max(0, n);
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--camera-mode") == 0 && i + 1 < argc)
        {
            if      (_wcsicmp(argv[i+1], L"orbit")        == 0) m_cameraMode = CameraMode::Orbit;
            else if (_wcsicmp(argv[i+1], L"flock-follow") == 0 ||
                     _wcsicmp(argv[i+1], L"flock")        == 0) m_cameraMode = CameraMode::FlockFollow;
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--blas-mode") == 0 && i + 1 < argc)
        {
            if      (_wcsicmp(argv[i+1], L"dxr1")    == 0) m_blasMode = BlasMode::Dxr1;
            else if (_wcsicmp(argv[i+1], L"cluster") == 0) m_blasMode = BlasMode::Cluster;
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--partitions") == 0 && i + 1 < argc)
        {
            int n = _wtoi(argv[i + 1]);
            m_partitionBudget = (uint32_t)std::max(1, n);
            i += 1;
        }
        else if (_wcsicmp(argv[i], L"--resize-at") == 0 && i + 1 < argc)
        {
            // --resize-at FRAME:NEW_BUDGET  (e.g. --resize-at 60:16 --resize-at 120:128)
            UINT64 frame  = 0;
            uint32_t newP = 0;
            wchar_t* endptr = nullptr;
            frame = wcstoull(argv[i+1], &endptr, 10);
            if (endptr && *endptr == L':')
            {
                newP = (uint32_t)wcstoul(endptr + 1, nullptr, 10);
            }
            if (newP > 0)
            {
                m_scheduledResizes.push_back({ frame, newP });
            }
            i += 1;
        }
    }
    SampleLog::LogF(L"OnInit: CLI parsed (tlasMode=%s camera=%s grid=%ux%ux%u "
                    L"ballsPerSide=%u screenshotFrame=%d exitAfterFrames=%u "
                    L"logStatsEvery=%u)\n",
                    m_tlasMode == TlasMode::Partitioned ? L"partitioned" : L"traditional",
                    m_cameraMode == CameraMode::FlockFollow ? L"flock" : L"orbit",
                    m_scene.gridX, m_scene.gridY, m_scene.gridZ, m_scene.ballsPerSide,
                    m_screenshotFrame, m_exitAfterFrames, m_logStatsEvery);
}

// =============================================================================
// OnInit
// =============================================================================

void PartitionedTlasSample::OnInit()
{
    m_startTime = std::chrono::steady_clock::now();

    // DRED for diagnostics on any TDR.
    {
        ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dredSettings;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dredSettings))) && dredSettings)
        {
            dredSettings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dredSettings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            SampleLog::Write(L"OnInit: DRED enabled\n");
        }
    }

    // DXR2 + SM 6.10 experimental opt-in.  MUST be called before any
    // D3D12CreateDevice or ClustersAndPTLAS prebuild queries silently return
    // zero sizes even with caps reporting YES.
    {
        UUID experimentalFeatures[] = { D3D12ExperimentalShaderModels, D3D12RaytracingExperiment };
        HRESULT hrExp = D3D12EnableExperimentalFeatures(_countof(experimentalFeatures),
                                                        experimentalFeatures, nullptr, nullptr);
        SampleLog::LogF(L"OnInit: D3D12EnableExperimentalFeatures hr=0x%08X\n", (unsigned)hrExp);
    }

    m_deviceResources = std::make_unique<DX::DeviceResources>(
        DXGI_FORMAT_R8G8B8A8_UNORM,            // RGBA (UAV-typed; matches our output texture)
        DXGI_FORMAT_UNKNOWN,
        FrameCount,
        D3D_FEATURE_LEVEL_11_0,
        /*options*/0,
        m_adapterIDoverride);
    m_deviceResources->RegisterDeviceNotify(this);
    m_deviceResources->SetWindow(Win32Application::GetHwnd(), m_width, m_height);
    m_deviceResources->InitializeDXGIAdapter();

    m_deviceResources->CreateDeviceResources();
    m_deviceResources->CreateWindowSizeDependentResources();
    CreateDeviceDependentResources();
    CreateWindowSizeDependentResources();
    SampleLog::Write(L"OnInit: complete\n");
}

void PartitionedTlasSample::CreateDeviceDependentResources()
{
    auto device      = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();
    auto cmdQueue    = m_deviceResources->GetCommandQueue();

    // DeviceResources leaves the cmd list CLOSED after CreateDeviceResources
    // (see DeviceResources.cpp).  Reopen against the current frame's
    // allocator so init can record copies / BLAS builds.  alloc is fresh
    // (never used yet for any submission) so we don't need to Reset() it.
    auto alloc = m_deviceResources->GetCommandAllocator();
    ThrowIfFailed(commandList->Reset(alloc, nullptr),
        L"OnInit: failed to reopen DeviceResources cmd list");

    // QI for DXR1 + DXR2 surfaces.
    ThrowIfFailed(device     ->QueryInterface(IID_PPV_ARGS(&m_dxrDevice)),
        L"ID3D12Device5 unavailable\n");
    ThrowIfFailed(commandList->QueryInterface(IID_PPV_ARGS(&m_dxrCommandList)),
        L"ID3D12GraphicsCommandList4 unavailable\n");
    ThrowIfFailed(device     ->QueryInterface(IID_PPV_ARGS(&m_dxr2Device)),
        L"ID3D12DeviceRaytracing2 unavailable -- experimental D3D12Core not loaded?\n");
    ThrowIfFailed(commandList->QueryInterface(IID_PPV_ARGS(&m_dxr2CommandList)),
        L"ID3D12CommandListRaytracing2 unavailable\n");

    // ClustersAndPTLAS cap.
    D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL optsExp = {};
    HRESULT hrCaps = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL,
                                                 &optsExp, sizeof(optsExp));
    if (SUCCEEDED(hrCaps)) m_clustersAndPtlasSupported = optsExp.ClustersAndPTLASSupported != FALSE;
    SampleLog::LogF(L"  Adapter: %s  DXR2/PTLAS=%s\n",
                    m_deviceResources->GetAdapterDescription(),
                    m_clustersAndPtlasSupported ? L"YES" : L"NO");

    // Mirror debug-layer messages into SampleLog so silent breaks become
    // visible.  Also suppress id 1328 (CREATERESOURCE_STATE_IGNORED) which
    // is a benign reminder that buffers on default heaps ignore the
    // initial-state arg.  Pattern lifted from the clustered sample.
    {
        Microsoft::WRL::ComPtr<ID3D12InfoQueue> iq0;
        if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&iq0))))
        {
            D3D12_MESSAGE_ID denied[] = { D3D12_MESSAGE_ID_CREATERESOURCE_STATE_IGNORED };
            D3D12_INFO_QUEUE_FILTER filter = {};
            filter.DenyList.NumIDs  = _countof(denied);
            filter.DenyList.pIDList = denied;
            iq0->AddStorageFilterEntries(&filter);
        }
        Microsoft::WRL::ComPtr<ID3D12InfoQueue1> iq1;
        if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&iq1))))
        {
            DWORD cookie = 0;
            iq1->RegisterMessageCallback(
                [](D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY sev,
                   D3D12_MESSAGE_ID id, LPCSTR desc, void*)
                {
                    if (id == D3D12_MESSAGE_ID_CREATERESOURCE_STATE_IGNORED) return;
                    const wchar_t* sevStr =
                        sev == D3D12_MESSAGE_SEVERITY_CORRUPTION ? L"CORRUPTION" :
                        sev == D3D12_MESSAGE_SEVERITY_ERROR      ? L"ERROR" :
                        sev == D3D12_MESSAGE_SEVERITY_WARNING    ? L"WARNING" :
                        sev == D3D12_MESSAGE_SEVERITY_INFO       ? L"INFO" : L"MSG";
                    wchar_t wbuf[2048] = {};
                    size_t cv = 0;
                    if (desc) mbstowcs_s(&cv, wbuf, desc, _TRUNCATE);
                    SampleLog::LogF(L"[d3d12 %s id=%u] %s\n", sevStr, (unsigned)id, wbuf);
                }, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie);
            SampleLog::Write(L"  InfoQueue1 callback registered\n");
        }
    }

    if (m_tlasMode == TlasMode::Partitioned && !m_clustersAndPtlasSupported)
    {
        SampleLog::Write(L"  WARNING: --tlas-mode partitioned requested but cap is NO; "
                         L"falling back to traditional.\n");
        m_tlasMode = TlasMode::Traditional;
    }

    // Build mesh assets (icosphere ball + torus donut, each with its DXR1
    // BLAS).  Both record into the open cmd list; the caller (us, below)
    // closes / executes / waits at the end of init.
    m_ball.Initialize    (m_dxrDevice.Get(), m_dxrCommandList.Get(),
                          ProceduralGeometry::MakeIcosphere(2), L"BallMeshHi");
    m_ballMid.Initialize (m_dxrDevice.Get(), m_dxrCommandList.Get(),
                          ProceduralGeometry::MakeIcosphere(1), L"BallMeshMid");
    m_ballLow.Initialize (m_dxrDevice.Get(), m_dxrCommandList.Get(),
                          ProceduralGeometry::MakeIcosphere(0), L"BallMeshLo");
    m_donut.Initialize   (m_dxrDevice.Get(), m_dxrCommandList.Get(),
                          ProceduralGeometry::MakeTorus(0.55f, 0.18f, 28, 14), L"DonutMesh");
    // Donut needs accurate per-triangle normals for proper shading +
    // reflection; balls use the unit-sphere shortcut in the shader so
    // they don't.
    m_donut.BuildPerTriVertexNormalsBuffer(m_dxrDevice.Get(), m_dxrCommandList.Get(), L"DonutMesh");

    // Cluster BLAS path (CLAS + BUILD_BLAS_FROM_CLAS).  Built unconditionally
    // -- if --blas-mode dxr1 is selected the cluster BLAS sits unused at the
    // cost of a few KB, but the A/B compare stays trivial.
    {
        ComPtr<ID3D12DeviceRaytracing2> dRT2;
        ThrowIfFailed(m_dxrDevice.As(&dRT2), L"QI(ID3D12DeviceRaytracing2) for cluster BLAS");
        ComPtr<ID3D12CommandListRaytracing2> clRT2;
        ThrowIfFailed(m_dxrCommandList.As(&clRT2), L"QI(ID3D12CommandListRaytracing2) for cluster BLAS");
        m_ball.BuildClusterBlas    (m_dxrDevice.Get(), dRT2.Get(), m_dxrCommandList.Get(), clRT2.Get(), L"BallMeshHi");
        m_ballMid.BuildClusterBlas (m_dxrDevice.Get(), dRT2.Get(), m_dxrCommandList.Get(), clRT2.Get(), L"BallMeshMid");
        m_ballLow.BuildClusterBlas (m_dxrDevice.Get(), dRT2.Get(), m_dxrCommandList.Get(), clRT2.Get(), L"BallMeshLo");
        m_donut.BuildClusterBlas   (m_dxrDevice.Get(), dRT2.Get(), m_dxrCommandList.Get(), clRT2.Get(), L"DonutMesh");
    }

    // Donut flock-member roster.  Members orbit the flock center in a
    // small ring (3D positions baked here; later milestones can animate
    // these offsets to make the flock "shimmer").  Phase 4 keeps it
    // static.
    m_donutMembers.clear();
    m_donutMembers.reserve(kDonutCount);
    for (uint32_t i = 0; i < kDonutCount; ++i)
    {
        const float t = (float)i / (float)kDonutCount;
        const float ring = 0.85f;
        DirectX::XMFLOAT3 off = {
            ring * cosf(t * DirectX::XM_2PI),
            0.05f * sinf(t * DirectX::XM_2PI * 3.0f),    // small vertical wobble
            ring * sinf(t * DirectX::XM_2PI),
        };
        m_donutMembers.push_back({ off, 0.40f });
    }

    // Compute the static scene instance list.  Milestone 2b: every ball is
    // a static instance referencing the single shared BLAS; partition index
    // is the linear index of the partition cell containing the ball.
    BuildSceneInstances();
    SampleLog::LogF(L"[scene] grid=%ux%ux%u  ballsPerSide=%u  total balls=%u "
                    L"cells=%u  partitionBudget=%u\n",
                    m_scene.gridX, m_scene.gridY, m_scene.gridZ,
                    m_scene.ballsPerSide, m_scene.TotalBalls(),
                    m_scene.Partitions(), m_partitionBudget);

    // Set up the rolling-partition manager.  Cell count = grid cell count;
    // partition budget = PTLAS PartitionCount.  Budget < cells means active
    // recycling each frame; budget >= cells means every cell always has a
    // partition (effectively phase 2c behaviour).
    m_rollingParts.Initialize(m_partitionBudget, /*forwardBias*/ 0.30f,
                              m_scene, m_ballWorldPos);

    // Pick the active TLAS system, sized for the partition budget +
    // global-partition flock.
    ITlasSystem::InitDesc tlasInit = {};
    tlasInit.maxInstances                  = m_scene.TotalBalls() + kDonutCount;
    tlasInit.maxPartitions                 = m_partitionBudget;
    tlasInit.maxInstancesPerPartition      = std::max(1u, 2 * m_scene.BallsPerPartition());
    tlasInit.maxInstancesInGlobalPartition = kDonutCount;     // donut flock
    if (m_tlasMode == TlasMode::Partitioned)
        m_tlas = std::make_unique<PtlasSystem>();
    else
        m_tlas = std::make_unique<TraditionalTlasSystem>();
    m_tlas->Initialize(m_dxrDevice.Get(), m_deviceResources.get(), tlasInit);
    SampleLog::LogF(L"  TLAS mode: %s\n", m_tlas->ModeName());

    // Per-resource state: descriptor heap, RT pipeline, shader table, scene CB.
    CreateDescriptorHeap();
    CreateRaytracingPipeline();
    CreateShaderTable();
    CreateSceneConstantBuffer();

    // ---- GPU timestamp query heap + readback ring ----
    {
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type     = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count    = kTimestampsPerFrame * kTimestampSlots;
        ThrowIfFailed(device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&m_timestampHeap)));
        m_timestampHeap->SetName(L"PT/TimestampHeap");

        auto rbHeap  = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto bufDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(UINT64) * qhd.Count);
        ThrowIfFailed(device->CreateCommittedResource(&rbHeap, D3D12_HEAP_FLAG_NONE,
            &bufDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_timestampReadback)));
        m_timestampReadback->SetName(L"PT/TimestampReadback");

        ThrowIfFailed(cmdQueue->GetTimestampFrequency(&m_timestampFreqHz));
        SampleLog::LogF(L"[stats] timestamp frequency = %llu ticks/s "
                        L"(%.3f ns/tick)\n",
                        (unsigned long long)m_timestampFreqHz,
                        1e9 / (double)m_timestampFreqHz);
    }

    // CreateOutputUav is called from CreateWindowSizeDependentResources
    // because the output texture is sized to the back buffer.

    // Close the cmd list -- subsequent OnRender will reopen via Prepare().
    ThrowIfFailed(commandList->Close());
    ID3D12CommandList* lists[] = { commandList };
    cmdQueue->ExecuteCommandLists(_countof(lists), lists);
    m_deviceResources->WaitForGpu();
}

void PartitionedTlasSample::CreateWindowSizeDependentResources()
{
    CreateOutputUav();
}

void PartitionedTlasSample::ReleaseDeviceDependentResources()
{
    m_tlas.reset();
    m_dxr2CommandList.Reset();
    m_dxr2Device.Reset();
    m_dxrCommandList.Reset();
    m_dxrDevice.Reset();
    m_globalRootSig.Reset();
    m_rtStateObject.Reset();
    m_shaderTable.Reset();
    m_sceneCb.Reset();
    m_output.Reset();
    m_descHeap.Reset();
    m_clustersAndPtlasSupported = false;
}

// =============================================================================
// Descriptor heap, output UAV, scene CB
// =============================================================================

void PartitionedTlasSample::CreateDescriptorHeap()
{
    D3D12_DESCRIPTOR_HEAP_DESC d = {};
    d.NumDescriptors = 8;     // skeleton: 1 used, room to grow
    d.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    d.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(m_dxrDevice->CreateDescriptorHeap(&d, IID_PPV_ARGS(&m_descHeap)));
    m_descSize = m_dxrDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_uavHeapIdx = 0;
}

void PartitionedTlasSample::CreateOutputUav()
{
    m_output.Reset();
    CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_DEFAULT);
    auto desc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R8G8B8A8_UNORM, m_width, m_height, 1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ThrowIfFailed(m_dxrDevice->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_output)));
    m_output->SetName(L"RT output UAV");

    D3D12_CPU_DESCRIPTOR_HANDLE cpu = m_descHeap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += SIZE_T(m_uavHeapIdx) * m_descSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
    u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    u.Format        = DXGI_FORMAT_R8G8B8A8_UNORM;
    m_dxrDevice->CreateUnorderedAccessView(m_output.Get(), nullptr, &u, cpu);
}

void PartitionedTlasSample::CreateSceneConstantBuffer()
{
    // 256-byte aligned per-frame slice; ring of FrameCount slices so we
    // never overwrite GPU-in-flight data.
    m_sceneCbStride = AlignUp(sizeof(SceneConstantBuffer), 256);
    const UINT64 bytes = m_sceneCbStride * FrameCount;
    m_sceneCb = PtSample::CreateUploadBuffer(m_dxrDevice.Get(), bytes, L"SceneCBV");
    CD3DX12_RANGE noRead(0, 0);
    ThrowIfFailed(m_sceneCb->Map(0, &noRead, reinterpret_cast<void**>(&m_sceneCbCpu)));
}

// =============================================================================
// RT root signature + state object + shader table
// =============================================================================

void PartitionedTlasSample::CreateRaytracingPipeline()
{
    // ---- Global root signature ----
    //   param 0 : descriptor table -> 1 UAV (u0)   -> output texture
    //   param 1 : root SRV (t0)                    -> TLAS (PTLAS or trad)
    //   param 2 : root CBV (b0)                    -> SceneConstantBuffer
    //   param 3 : root SRV (t1)                    -> donut face-normals
    CD3DX12_DESCRIPTOR_RANGE uavRange = {};
    uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, /*BaseRegister*/0);

    CD3DX12_ROOT_PARAMETER params[4] = {};
    params[PT_GRS_OutputUavSlot].InitAsDescriptorTable(1, &uavRange);
    params[PT_GRS_AccelerationStructureSlot].InitAsShaderResourceView(0);
    params[PT_GRS_SceneCBVSlot].InitAsConstantBufferView(0);
    params[PT_GRS_DonutVertNormalsSrvSlot].InitAsShaderResourceView(1);

    CD3DX12_ROOT_SIGNATURE_DESC rsDesc(_countof(params), params, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_NONE);
    ComPtr<ID3DBlob> blob, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err),
        L"Root sig serialize failed");
    ThrowIfFailed(m_dxrDevice->CreateRootSignature(0, blob->GetBufferPointer(),
        blob->GetBufferSize(), IID_PPV_ARGS(&m_globalRootSig)));
    m_globalRootSig->SetName(L"GlobalRootSig");

    // ---- State object ----
    CD3DX12_STATE_OBJECT_DESC so(D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE);

    auto* lib = so.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    D3D12_SHADER_BYTECODE bc = { g_pRaytracing, ARRAYSIZE(g_pRaytracing) };
    lib->SetDXILLibrary(&bc);
    lib->DefineExport(L"Raygen");
    lib->DefineExport(L"Miss");
    lib->DefineExport(L"ShadowMiss");
    lib->DefineExport(L"ClosestHit_Ball");
    lib->DefineExport(L"ClosestHit_Donut");

    auto* hgBall = so.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    hgBall->SetClosestHitShaderImport(L"ClosestHit_Ball");
    hgBall->SetHitGroupExport(L"HitGroup_Ball");
    hgBall->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

    auto* hgDonut = so.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    hgDonut->SetClosestHitShaderImport(L"ClosestHit_Donut");
    hgDonut->SetHitGroupExport(L"HitGroup_Donut");
    hgDonut->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

    auto* cfg = so.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
    // Payload = float3 colour + uint depth = 16 bytes; ShadowPayload =
    // uint shadowed = 4 bytes.  Pick the larger.  Attr = barycentric
    // float2 = 8 bytes.
    cfg->Config(/*MaxPayload*/16, /*MaxAttr*/8);

    auto* gsig = so.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
    gsig->SetRootSignature(m_globalRootSig.Get());

    auto* pcfg = so.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
    pcfg->Config(/*MaxRecursion*/2);   // primary + 1 reflection bounce

    ThrowIfFailed(m_dxrDevice->CreateStateObject(so, IID_PPV_ARGS(&m_rtStateObject)),
        L"CreateStateObject failed");
}

void PartitionedTlasSample::CreateShaderTable()
{
    ComPtr<ID3D12StateObjectProperties> props;
    ThrowIfFailed(m_rtStateObject.As(&props));
    void* idRaygen     = props->GetShaderIdentifier(L"Raygen");
    void* idMiss       = props->GetShaderIdentifier(L"Miss");
    void* idShadowMiss = props->GetShaderIdentifier(L"ShadowMiss");
    void* idHitBall    = props->GetShaderIdentifier(L"HitGroup_Ball");
    void* idHitDonut   = props->GetShaderIdentifier(L"HitGroup_Donut");

    const UINT64 idBytes  = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    const UINT64 recAlign = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;
    const UINT64 tabAlign = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    const UINT64 recSize  = AlignUp(idBytes, recAlign);

    // Section layout: raygen | miss[2] | hit[2].  Each section starts at
    // a `tabAlign`-aligned offset; within a section, records are spaced
    // `recSize` apart (records contain only the shader identifier in this
    // sample -- no local root-arg data).
    m_rayGenStart       = 0;
    m_rayGenSize        = recSize;
    m_missStart         = AlignUp(m_rayGenStart + recSize, tabAlign);
    m_missRecordSize    = recSize;
    m_missTotalSize     = recSize * 2;       // primary + shadow
    m_hitStart          = AlignUp(m_missStart + m_missTotalSize, tabAlign);
    m_hitRecordSize     = recSize;
    m_hitTotalSize      = recSize * 2;       // HitGroup_Ball + HitGroup_Donut
    const UINT64 total  = m_hitStart + m_hitTotalSize;

    m_shaderTable = PtSample::CreateUploadBuffer(m_dxrDevice.Get(), total, L"ShaderTable");
    UINT8* cpu = nullptr;
    CD3DX12_RANGE noRead(0, 0);
    ThrowIfFailed(m_shaderTable->Map(0, &noRead, reinterpret_cast<void**>(&cpu)));
    memset(cpu, 0, (size_t)total);
    memcpy(cpu + m_rayGenStart,                   idRaygen,     idBytes);
    memcpy(cpu + m_missStart,                     idMiss,       idBytes);
    memcpy(cpu + m_missStart + m_missRecordSize,  idShadowMiss, idBytes);
    memcpy(cpu + m_hitStart,                      idHitBall,    idBytes);
    memcpy(cpu + m_hitStart + m_hitRecordSize,    idHitDonut,   idBytes);
    m_shaderTable->Unmap(0, nullptr);
}

// =============================================================================
// Scene instances (milestone 3b: precompute world transforms + ball positions;
// per-frame partition assignment lives in RollingPartitions)
// =============================================================================
void PartitionedTlasSample::BuildSceneInstances()
{
    using namespace DirectX;
    m_sceneInstances.clear();
    m_sceneInstances.reserve(m_scene.TotalBalls());
    m_ballWorldPos.clear();
    m_ballWorldPos.reserve(m_scene.TotalBalls());

    // Pick the BLAS GPUVA for the active path (`--blas-mode dxr1|cluster`).
    // Cluster mode uses the CLAS+BLAS_FROM_CLAS-built BLAS; DXR1 mode uses
    // the classic BuildRaytracingAccelerationStructure-built BLAS.
    const bool useCluster = (m_blasMode == BlasMode::Cluster);
    const D3D12_GPU_VIRTUAL_ADDRESS blas = useCluster ? m_ball.ClusterBlasGpuVa() : m_ball.BlasGpuVa();
    // PTLAS InstanceIndex layout: donuts occupy [0..kDonutCount), balls
    // start at kDonutCount.  Recorded in inst.instanceIndex so PtlasSystem
    // writes to the correct slot even when we only WRITE_INSTANCE a subset.
    for (uint32_t pk = 0; pk < m_scene.gridZ; ++pk)
    for (uint32_t pj = 0; pj < m_scene.gridY; ++pj)
    for (uint32_t pi = 0; pi < m_scene.gridX; ++pi)
    {
        for (uint32_t bk = 0; bk < m_scene.ballsPerSide; ++bk)
        for (uint32_t bj = 0; bj < m_scene.ballsPerSide; ++bj)
        for (uint32_t bi = 0; bi < m_scene.ballsPerSide; ++bi)
        {
            DirectX::XMFLOAT3 pos = m_scene.BallWorldPos(pi, pj, pk, bi, bj, bk);
            m_ballWorldPos.push_back(pos);

            XMMATRIX m = XMMatrixScaling(m_scene.ballScale, m_scene.ballScale, m_scene.ballScale)
                       * XMMatrixTranslation(pos.x, pos.y, pos.z);
            SceneInstance inst = {};
            XMStoreFloat4x4(&inst.transform, XMMatrixTranspose(m));
            inst.blasGva        = blas;
            const uint32_t ballIdx = m_scene.BallIndex(pi, pj, pk, bi, bj, bk);
            inst.instanceIndex  = kDonutCount + ballIdx;
            inst.instanceID     = ballIdx;
            inst.instanceMask   = 0xFF;
            inst.partitionIndex = 0;   // set per-frame by RollingPartitions
            m_sceneInstances.push_back(inst);
        }
    }
}

// Camera-anchored AS-origin: track the camera every frame so the PTLAS
// sees coords near zero where the rays start.  In flock-follow mode the
// camera trails the flock through the lattice, so as_origin scans across
// the volume of balls -- which in turn means TRANSLATE_PARTITION runs every
// frame on every partition (the partitions effectively move past the
// camera as it flies forward).
void PartitionedTlasSample::RecomputeAsOrigin()
{
    using namespace DirectX;
    const double tsec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - m_startTime).count();

    if (m_cameraMode == CameraMode::FlockFollow)
    {
        m_flock.Update(tsec);
        auto cam = m_flock.CameraPos();
        m_asOrigin = cam;
    }
    else
    {
        // Orbit (phase 2c default).
        auto sceneSize = m_scene.SceneSize();
        const float diag = sqrtf(sceneSize.x*sceneSize.x + sceneSize.y*sceneSize.y + sceneSize.z*sceneSize.z);
        const float r     = diag * 1.2f;
        const float h     = diag * 0.35f;
        const float angle = (float)(tsec * 0.20);
        m_asOrigin = { r * sinf(angle), h, r * cosf(angle) };
    }
}

// Displacement field: a ball at world-rest-position `restPos` is pushed
// radially away from the flock center.  Smooth (1 - d/R)^2 falloff to
// zero at kDisplaceRadius.  Returns the displacement VECTOR (added to
// restPos to get the displaced world position).  Outside the radius
// the result is identically zero.
DirectX::XMFLOAT3 PartitionedTlasSample::ComputeBallDisplacement(const DirectX::XMFLOAT3& restPos) const
{
    DirectX::XMFLOAT3 d = { restPos.x - m_flock.position.x,
                            restPos.y - m_flock.position.y,
                            restPos.z - m_flock.position.z };
    float dist = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
    if (dist >= kDisplaceRadius || dist < 1e-4f) return { 0, 0, 0 };
    float t = 1.0f - dist / kDisplaceRadius;
    float mag = kDisplaceMaxPush * t * t;
    return { d.x / dist * mag, d.y / dist * mag, d.z / dist * mag };
}

void PartitionedTlasSample::UpdateSceneConstantBuffer()
{
    using namespace DirectX;

    const double tsec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - m_startTime).count();

    // Orbit OR flock-follow camera depending on m_cameraMode.  In both cases
    // m_asOrigin (set by RecomputeAsOrigin just before this call) tracks
    // the camera world position; we build the view in AS-space so
    // projectionToWorld unprojects to AS-space points and TraceRay sees
    // the camera-anchored coord frame the PTLAS is laid out in.
    XMVECTOR eyeWorld, targetWorld;
    if (m_cameraMode == CameraMode::FlockFollow)
    {
        auto cam = m_flock.CameraPos();
        auto tgt = m_flock.CameraTarget();
        eyeWorld    = XMVectorSet(cam.x, cam.y, cam.z, 1);
        targetWorld = XMVectorSet(tgt.x, tgt.y, tgt.z, 1);
    }
    else
    {
        // Orbit framing the whole grid.
        auto sceneSize = m_scene.SceneSize();
        const float diag = sqrtf(sceneSize.x*sceneSize.x + sceneSize.y*sceneSize.y + sceneSize.z*sceneSize.z);
        const float r     = diag * 1.2f;
        const float h     = diag * 0.35f;
        const float angle = (float)(tsec * 0.20);
        eyeWorld    = XMVectorSet(r * sinf(angle), h, r * cosf(angle), 1);
        targetWorld = XMVectorSet(0, 0, 0, 1);
    }
    XMVECTOR up = XMVectorSet(0, 1, 0, 0);

    XMVECTOR asOriginVec = XMVectorSet(m_asOrigin.x, m_asOrigin.y, m_asOrigin.z, 0);
    XMVECTOR eyeAs       = XMVectorSubtract(eyeWorld,    asOriginVec);
    XMVECTOR targetAs    = XMVectorSubtract(targetWorld, asOriginVec);

    XMMATRIX viewAs   = XMMatrixLookAtRH(eyeAs, targetAs, up);
    const float aspect = (float)m_width / (float)m_height;
    XMMATRIX proj     = XMMatrixPerspectiveFovRH(XM_PIDIV4, aspect, 0.1f, 1000.0f);
    XMMATRIX viewProj = viewAs * proj;
    XMMATRIX projToAs = XMMatrixInverse(nullptr, viewProj);

    SceneConstantBuffer cb = {};
    // HLSL constant-buffer matrices default to COLUMN-major.  We're a
    // row-major shop, so transpose at upload time and let HLSL's
    // column-major reinterpretation hand us back the row-major matrix the
    // shader wants.
    XMStoreFloat4x4(&cb.projectionToWorld, XMMatrixTranspose(projToAs));
    XMStoreFloat4(&cb.cameraOriginAs, eyeAs);
    cb.asOriginWorld   = { m_asOrigin.x, m_asOrigin.y, m_asOrigin.z, 0 };
    cb.lightDirAndPad  = { 0.55f, 0.65f, 0.52f, 0 };   // sun off to the side + slightly above
    cb.missColorAndTime = { 0.02f, 0.04f, 0.06f, (float)tsec };

    const UINT slot = (UINT)(m_framesRendered % FrameCount);
    memcpy(m_sceneCbCpu + slot * m_sceneCbStride, &cb, sizeof(cb));
}

void PartitionedTlasSample::OnUpdate()
{
    m_timer.Tick();
}

void PartitionedTlasSample::OnRender()
{
    if (!m_deviceResources->IsWindowVisible()) return;
    DoRender();
    ++m_framesRendered;

    // Headless screenshot.
    if (m_screenshotFrame >= 0 && (UINT)m_screenshotFrame < m_framesRendered && !m_screenshotTaken)
    {
        m_screenshotTaken = true;
        SampleLog::LogF(L"[screenshot] capturing frame %llu to %s\n",
                        (unsigned long long)m_framesRendered, m_screenshotPath.c_str());
        CaptureBackBufferToFile(m_screenshotPath);
        if (m_exitAfterFrames == 0) { PostQuitMessage(0); return; }
    }
    if (m_exitAfterFrames > 0 && m_framesRendered >= m_exitAfterFrames)
    {
        PostQuitMessage(0);
    }
}

void PartitionedTlasSample::DoRender()
{
    ApplyResizeIfPending();      // phase 3c: hotkey/scheduled PTLAS resize
    RecomputeAsOrigin();        // updates m_asOrigin BEFORE UpdateSceneConstantBuffer reads it
    UpdateSceneConstantBuffer();

    m_deviceResources->Prepare();   // cmd list reset; back buffer -> RENDER_TARGET
    auto cl  = m_dxrCommandList.Get();
    auto cl2 = m_dxr2CommandList.Get();

    // ---- (P)TLAS build for this frame ----
    //
    // Phase 3b switched from "fixed grid of partitions" to ROLLING WINDOW:
    //   * m_rollingParts picks the P cells closest to the flock as the
    //     active set; partition slot indices recycle as cells enter/leave.
    //   * Each ball reports its current owner partition (or kNoPartition if
    //     its cell isn't in the active set this frame).
    //   * Per-frame: WRITE_INSTANCE for balls whose owner changed
    //     (transfer or transition between active/disabled), and
    //     TRANSLATE_PARTITION for ALL active partitions (their world-space
    //     centroids are fixed but the AS-space offset = home - as_origin
    //     changes every frame as the camera moves).
    //   * Traditional baseline: just rebuilds the flat instance list with
    //     translation = world - as_origin every frame, disabling balls
    //     whose owner is kNoPartition (AccelerationStructure = 0).
    using namespace DirectX;

    m_rollingParts.Update(m_flock.position, m_flock.forward);
    m_cumBallChanges += (UINT64)m_rollingParts.ChangedBalls().size();

    // ---- Compute per-ball displacement state for this frame ----
    //
    // Displaced balls need a per-frame WRITE_INSTANCE (their world
    // translation changes).  Newly-undisplaced balls also need one final
    // WRITE_INSTANCE to settle them back to their rest position.  We
    // track LAST FRAME's displacement state in m_ballDisplaced, recompute
    // THIS FRAME's, and union the two for the "needs write" set.
    const uint32_t ballN = m_rollingParts.BallCount();
    std::vector<uint8_t> displacedNow(ballN, 0);   // 1 if currently displaced
    std::vector<DirectX::XMFLOAT3> displacementVec(ballN);
    std::vector<uint8_t> writeNow(ballN, 0);       // 1 if needs WRITE_INSTANCE this frame
    {
        for (uint32_t b = 0; b < ballN; ++b)
        {
            if (m_rollingParts.Owner(b) == RollingPartitions::kNoPartition) continue;
            auto disp = ComputeBallDisplacement(m_ballWorldPos[b]);
            float mag2 = disp.x*disp.x + disp.y*disp.y + disp.z*disp.z;
            displacementVec[b] = disp;
            if (mag2 > 1e-8f) displacedNow[b] = 1;
            // needs write if currently displaced OR was displaced last frame
            // (the "settling" case -- restore to rest).
            if (displacedNow[b] || (b < m_ballDisplaced.size() && m_ballDisplaced[b]))
                writeNow[b] = 1;
        }
    }

    m_tlas->BeginFrame();
    // Timestamp BEFORE TLAS build.
    const UINT tsSlot     = m_timestampSlotIdx;
    const UINT tsSlotBase = tsSlot * kTimestampsPerFrame;
    cl->EndQuery(m_timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, tsSlotBase + 0);

    if (m_tlasMode == TlasMode::Partitioned)
    {
        // (a) WRITE_INSTANCE for any ball that needs a fresh write this frame.
        //     Three reasons a ball needs writing:
        //       1. Owner changed (rolling-window transfer or active/inactive flip)
        //       2. Currently displaced by the flock (translation moves per frame)
        //       3. Just left the displacement zone (settle back to rest)
        //     #1 comes from RollingPartitions::ChangedBalls(); #2/#3 from
        //     the displacedNow + previous-frame m_ballDisplaced union above.
        //     On the initial pass (m_ptlasInitialWriteDone == false) we
        //     write the full set unconditionally.
        std::vector<SceneInstance> writes;
        // Union ChangedBalls into the writeNow bitmap (it's a tight union).
        for (uint32_t b : m_rollingParts.ChangedBalls())
            if (b < writeNow.size()) writeNow[b] = 1;

        const uint32_t fullCount = ballN;
        const bool initialPass   = !m_ptlasInitialWriteDone;
        const uint32_t writeIterCount = initialPass ? fullCount :
            [&]{ uint32_t c = 0; for (auto v : writeNow) if (v) ++c; return c; }();
        writes.reserve(writeIterCount + (initialPass ? (uint32_t)m_donutMembers.size() : 0));

        // Iterate the union set of (changed | displaced | was_displaced).
        // For the initial pass it's just all balls.
        const uint32_t loopEnd = initialPass ? fullCount : fullCount;
        for (uint32_t b = 0; b < loopEnd; ++b)
        {
            if (!initialPass && !writeNow[b]) continue;

            const uint32_t owner = m_rollingParts.Owner(b);
            SceneInstance inst = m_sceneInstances[b];
            // inst.instanceIndex already = kDonutCount + b from BuildSceneInstances.
            if (owner == RollingPartitions::kNoPartition)
            {
                // Disabled: make the instance DEGENERATE (transform with
                // first-3-elements-of-each-row zero) so per spec it is
                // EXCLUDED ENTIRELY from the PTLAS -- doesn't count toward
                // any partition's instance cap, doesn't get traced.  Just
                // setting AccelerationStructure=0 is "disabled but still in
                // partition" which can exceed MaxInstancePerPartitionCount
                // and TDRs the driver if many instances pile into the same
                // fallback slot.
                inst.blasGva        = 0;
                inst.partitionIndex = 0;
                memset(&inst.transform, 0, sizeof(inst.transform));
            }
            else
            {
                // Active: partition-local transform = ball_world - home.
                const auto& home = m_rollingParts.PartitionHome(owner);
                inst.partitionIndex = owner;
                inst.transform.m[0][3] -= home.x;
                inst.transform.m[1][3] -= home.y;
                inst.transform.m[2][3] -= home.z;
                // Apply per-frame displacement (radial push from flock).
                // Zero for balls outside kDisplaceRadius -- same code path
                // for rest + displaced, no branching needed.
                inst.transform.m[0][3] += displacementVec[b].x;
                inst.transform.m[1][3] += displacementVec[b].y;
                inst.transform.m[2][3] += displacementVec[b].z;
                // LOD-aware BLAS swap: pick by distance from camera (3 bins).
                const float wpx = m_ballWorldPos[b].x + displacementVec[b].x;
                const float wpy = m_ballWorldPos[b].y + displacementVec[b].y;
                const float wpz = m_ballWorldPos[b].z + displacementVec[b].z;
                const float dx = wpx - m_asOrigin.x;
                const float dy = wpy - m_asOrigin.y;
                const float dz = wpz - m_asOrigin.z;
                const float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                const uint8_t lod = SelectLodInitial(dist);
                inst.blasGva = (lod == 0) ? BallBlasFor(0)
                             : (lod == 1) ? BallBlasFor(1)
                                          : BallBlasFor(2);
                // ENABLE_EXPLICIT_AABB: a CONSERVATIVE AABB around the
                // ball's REST partition-local position, padded out by the
                // displacement envelope (kDisplaceAabbPad) so the AABB
                // covers every possible displaced position the ball may
                // take.  This means the AABB is STABLE across frames even
                // as the displaced transform shifts -- avoids the
                // shrink/grow cycle and lets the partition's internal AS
                // stay valid.
                const float restLocalX = m_ballWorldPos[b].x - home.x;
                const float restLocalY = m_ballWorldPos[b].y - home.y;
                const float restLocalZ = m_ballWorldPos[b].z - home.z;
                const float pad        = m_scene.ballScale * 1.05f + kDisplaceAabbPad;
                inst.useExplicitAabb = true;
                inst.aabbMin = { restLocalX - pad, restLocalY - pad, restLocalZ - pad };
                inst.aabbMax = { restLocalX + pad, restLocalY + pad, restLocalZ + pad };
            }
            writes.push_back(inst);
        }
        // (a2) Donut flock in the GLOBAL PARTITION.  Static partition-local
        //      transforms (= flock-local offset + scale).  Written ONCE; the
        //      global partition's translation, updated each frame below,
        //      places them at flock_pos - as_origin in AS-space.
        if (!m_ptlasInitialWriteDone)
        {
            for (uint32_t d = 0; d < (uint32_t)m_donutMembers.size(); ++d)
            {
                const auto& m = m_donutMembers[d];
                XMMATRIX t = XMMatrixScaling(m.scale, m.scale, m.scale)
                           * XMMatrixTranslation(m.localOffset.x, m.localOffset.y, m.localOffset.z);
                SceneInstance inst = {};
                XMStoreFloat4x4(&inst.transform, XMMatrixTranspose(t));
                inst.blasGva        = DonutBlas();
                inst.instanceIndex  = kFlockInstBase + d;
                inst.instanceID     = 0x10000u | d;
                inst.instanceMask   = 0xFF;
                inst.partitionIndex = (UINT)D3D12_RTAS_PARTITIONED_TLAS_PARTITION_INDEX_GLOBAL_PARTITION;
                inst.contributionToHitGroupIndex = 1;   // HitGroup_Donut
                writes.push_back(inst);
            }
        }
        if (!writes.empty())
        {
            m_tlas->WriteInstances(writes.data(), (UINT)writes.size());
        }
        if (!m_ptlasInitialWriteDone)
        {
            m_ptlasInitialWriteDone = true;
            // Per-ball LOD state begins at "hi" (matches the initial write
            // which uses m_ball.BlasGpuVa()).
            m_ballLod.assign(m_rollingParts.BallCount(), 0);
            SampleLog::LogF(L"[ptlas] initial WRITE_INSTANCE pass: %u instances "
                            L"(%u donuts in global partition; partition budget = %u)\n",
                            (unsigned)writes.size(), (unsigned)m_donutMembers.size(),
                            m_partitionBudget);
        }

        // (a3) UPDATE_INSTANCE for balls whose LOD just changed (and that
        //      did NOT get a fresh WRITE_INSTANCE this frame, since those
        //      already carry the latest LOD).  LOD = distance from camera;
        //      bins with hysteresis to limit per-frame flicker.  With
        //      ENABLE_EXPLICIT_AABB set on the prior WRITE, the swap is
        //      cheap (no partition refit).
        std::vector<ITlasSystem::InstanceUpdate> updates;
        if (m_ballLod.size() == ballN)
        {
            for (uint32_t b = 0; b < ballN; ++b)
            {
                if (writeNow[b]) continue;       // already written
                if (m_rollingParts.Owner(b) == RollingPartitions::kNoPartition) continue;
                const auto& pos = m_ballWorldPos[b];
                float dx = pos.x - m_asOrigin.x;
                float dy = pos.y - m_asOrigin.y;
                float dz = pos.z - m_asOrigin.z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                const uint8_t newLod = SelectLod(dist, m_ballLod[b]);
                if (newLod != m_ballLod[b])
                {
                    ITlasSystem::InstanceUpdate u = {};
                    u.instanceIndex = kDonutCount + b;
                    u.newBlas = (newLod == 0) ? BallBlasFor(0)
                              : (newLod == 1) ? BallBlasFor(1)
                                              : BallBlasFor(2);
                    updates.push_back(u);
                    m_ballLod[b] = newLod;
                }
            }
            // KNOWN ISSUE on the current preview NVIDIA driver (2026-05): a
            // PTLAS build call that contains BOTH WRITE_INSTANCE and
            // UPDATE_INSTANCE ops (on disjoint instance sets) TDRs the GPU
            // within ~10 frames.  Verified spec-compliant via the
            // IndirectBuild.cpp conformance test (which exercises exactly
            // this mixed-op pattern with both flags `DisablePartition*Unless
            // Forced` defaulting to false), and verified on WARP (Microsoft
            // Basic Render Driver) which runs the same code with no TDR.
            //
            // Workaround: skip UPDATE when WRITE is non-empty.  LOD swap
            // still works -- the LOD bin is refreshed on the same-frame WRITE,
            // which already picks blasGva from SelectLodInitial(distance).
            // We just lose the cheap UPDATE_INSTANCE optimization on those
            // frames.  Drop the `writes.empty()` half of the gate once the
            // driver fix ships.
            if (!updates.empty() && writes.empty())
            {
                m_tlas->UpdateInstances(updates.data(), (UINT)updates.size());
            }
        }
        // For balls written this frame, also seed m_ballLod from current
        // (possibly-displaced) distance so UPDATE_INSTANCE doesn't fire for
        // them next frame when nothing actually changed.
        for (uint32_t b = 0; b < ballN; ++b)
        {
            if (!writeNow[b] && !initialPass) continue;
            if (b >= m_ballLod.size()) continue;
            if (m_rollingParts.Owner(b) == RollingPartitions::kNoPartition) { m_ballLod[b] = 0; continue; }
            const auto& pos = m_ballWorldPos[b];
            const float wpx = pos.x + displacementVec[b].x;
            const float wpy = pos.y + displacementVec[b].y;
            const float wpz = pos.z + displacementVec[b].z;
            float dx = wpx - m_asOrigin.x;
            float dy = wpy - m_asOrigin.y;
            float dz = wpz - m_asOrigin.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            m_ballLod[b] = SelectLodInitial(dist);
        }
        // Persist displacement state for the next frame's union test.
        m_ballDisplaced = std::move(displacedNow);

        // (b) TRANSLATE_PARTITION for every active partition every frame
        //     (home - as_origin) + ONE for the global partition (flock_pos - as_origin).
        std::vector<ITlasSystem::PartitionTranslate> pts;
        pts.reserve(m_partitionBudget + 1);
        for (uint32_t p = 0; p < m_rollingParts.PartitionCount(); ++p)
        {
            if (!m_rollingParts.PartitionActive(p)) continue;
            const auto& home = m_rollingParts.PartitionHome(p);
            ITlasSystem::PartitionTranslate t = {};
            t.partitionIndex = p;
            t.translation[0] = home.x - m_asOrigin.x;
            t.translation[1] = home.y - m_asOrigin.y;
            t.translation[2] = home.z - m_asOrigin.z;
            pts.push_back(t);
        }
        {
            // Global partition tracks the flock center.
            ITlasSystem::PartitionTranslate t = {};
            t.partitionIndex = (UINT)D3D12_RTAS_PARTITIONED_TLAS_PARTITION_INDEX_GLOBAL_PARTITION;
            t.translation[0] = m_flock.position.x - m_asOrigin.x;
            t.translation[1] = m_flock.position.y - m_asOrigin.y;
            t.translation[2] = m_flock.position.z - m_asOrigin.z;
            pts.push_back(t);
        }
        if (!pts.empty())
        {
            m_tlas->TranslatePartitions(pts.data(), (UINT)pts.size());
        }
    }
    else
    {
        // Traditional: rebuild instance descs with translation baked
        // (world + displacement - as_origin), DISABLING balls outside the
        // active set (AS = 0).  Donuts appended at the end; their world
        // translation = flock_pos + local_offset - as_origin.
        std::vector<SceneInstance> adj;
        adj.reserve(m_sceneInstances.size() + m_donutMembers.size());
        for (size_t i = 0; i < m_sceneInstances.size(); ++i)
        {
            SceneInstance s = m_sceneInstances[i];
            const uint32_t owner = m_rollingParts.Owner((uint32_t)i);
            if (owner == RollingPartitions::kNoPartition)
            {
                s.blasGva = 0;
            }
            // Apply per-frame displacement (zero outside kDisplaceRadius).
            s.transform.m[0][3] += displacementVec[i].x;
            s.transform.m[1][3] += displacementVec[i].y;
            s.transform.m[2][3] += displacementVec[i].z;
            // Then shift into AS-space.
            s.transform.m[0][3] -= m_asOrigin.x;
            s.transform.m[1][3] -= m_asOrigin.y;
            s.transform.m[2][3] -= m_asOrigin.z;
            adj.push_back(s);
        }
        // Persist displacement state for the next-frame's needs-write check
        // (symmetric with the partitioned path; harmless either way).
        m_ballDisplaced = std::move(displacedNow);
        for (uint32_t d = 0; d < (uint32_t)m_donutMembers.size(); ++d)
        {
            const auto& m = m_donutMembers[d];
            XMMATRIX t = XMMatrixScaling(m.scale, m.scale, m.scale)
                       * XMMatrixTranslation(
                            m_flock.position.x + m.localOffset.x - m_asOrigin.x,
                            m_flock.position.y + m.localOffset.y - m_asOrigin.y,
                            m_flock.position.z + m.localOffset.z - m_asOrigin.z);
            SceneInstance inst = {};
            XMStoreFloat4x4(&inst.transform, XMMatrixTranspose(t));
            inst.blasGva        = DonutBlas();
            inst.instanceID     = 0x10000u | d;
            inst.instanceMask   = 0xFF;
            inst.partitionIndex = 0;
            inst.instanceIndex  = (UINT)(m_sceneInstances.size() + d);
            inst.contributionToHitGroupIndex = 1;   // HitGroup_Donut
            adj.push_back(inst);
        }
        m_tlas->WriteInstances(adj.data(), (UINT)adj.size());
    }
    m_tlas->Build(cl, cl2);

    // Timestamp AFTER TLAS build, resolve this slot's pair to readback.
    cl->EndQuery(m_timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, tsSlotBase + 1);
    cl->ResolveQueryData(m_timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
        tsSlotBase, kTimestampsPerFrame,
        m_timestampReadback.Get(), tsSlotBase * sizeof(UINT64));

    // Read N-2 slots back (GPU has finished there).  First few frames have
    // no valid history yet so guard against the underflow.
    if (m_framesRendered >= (kTimestampSlots - 1))
    {
        const UINT readSlot = (tsSlot + 1) % kTimestampSlots;  // oldest slot still in heap
        const UINT64* ts = nullptr;
        D3D12_RANGE readRange = { readSlot * kTimestampsPerFrame * sizeof(UINT64),
                                  (readSlot + 1) * kTimestampsPerFrame * sizeof(UINT64) };
        ThrowIfFailed(m_timestampReadback->Map(0, &readRange, (void**)&ts));
        const UINT64 t0 = ts[readSlot * kTimestampsPerFrame + 0];
        const UINT64 t1 = ts[readSlot * kTimestampsPerFrame + 1];
        m_timestampReadback->Unmap(0, nullptr);
        if (t1 > t0)
        {
            m_tlasBuildMsLast = double(t1 - t0) * 1000.0 / double(m_timestampFreqHz);
            constexpr double kEmaA = 1.0 / 16.0;
            m_tlasBuildMsEma = (m_tlasBuildMsEma == 0.0)
                ? m_tlasBuildMsLast
                : (m_tlasBuildMsEma * (1.0 - kEmaA) + m_tlasBuildMsLast * kEmaA);
        }
    }
    m_timestampSlotIdx = (m_timestampSlotIdx + 1) % kTimestampSlots;

    // Per-frame stats (cheap; once per frame).  GPU timestamps land in a
    // later milestone -- for now we log instance/partition counts and
    // result/scratch budgets straight off the active TLAS system so the
    // headless log shows the workload size.
    if (m_logStatsEvery > 0 && (m_framesRendered % m_logStatsEvery) == 0)
    {
        auto fs = m_tlas->GetLastFrameStats();
        const uint64_t deltaWrites = m_cumBallChanges - m_cumBallChangesLastLog;
        m_cumBallChangesLastLog = m_cumBallChanges;
        SampleLog::LogF(L"[frame %llu] tlas=%s budget=%u "
                        L"writes_delta=%llu cum_writes=%llu  updates_total=%u  part_trans=%u "
                        L"build_ms=%.3f (raw %.3f)  flock=(%.2f, %.2f, %.2f)  "
                        L"fwd=(%.2f, %.2f, %.2f)\n",
                        (unsigned long long)m_framesRendered,
                        m_tlas->ModeName(),
                        m_partitionBudget,
                        (unsigned long long)deltaWrites,
                        (unsigned long long)m_cumBallChanges,
                        fs.instancesSubmitted,
                        fs.partitionsTouched,
                        m_tlasBuildMsEma, m_tlasBuildMsLast,
                        m_flock.position.x, m_flock.position.y, m_flock.position.z,
                        m_flock.forward.x, m_flock.forward.y, m_flock.forward.z);
    }

    // ---- Bind RT pipeline + descriptors ----
    ID3D12DescriptorHeap* heaps[] = { m_descHeap.Get() };
    cl->SetDescriptorHeaps(_countof(heaps), heaps);
    cl->SetComputeRootSignature(m_globalRootSig.Get());
    D3D12_GPU_DESCRIPTOR_HANDLE uavGpu = m_descHeap->GetGPUDescriptorHandleForHeapStart();
    uavGpu.ptr += SIZE_T(m_uavHeapIdx) * m_descSize;
    cl->SetComputeRootDescriptorTable(PT_GRS_OutputUavSlot, uavGpu);
    cl->SetComputeRootShaderResourceView(PT_GRS_AccelerationStructureSlot, m_tlas->Gva());
    cl->SetComputeRootShaderResourceView(PT_GRS_DonutVertNormalsSrvSlot, m_donut.VertNormalsGpuVa());
    const UINT slot = (UINT)(m_framesRendered % FrameCount);
    cl->SetComputeRootConstantBufferView(PT_GRS_SceneCBVSlot,
        m_sceneCb->GetGPUVirtualAddress() + slot * m_sceneCbStride);

    cl->SetPipelineState1(m_rtStateObject.Get());

    // ---- Dispatch ----
    D3D12_DISPATCH_RAYS_DESC drd = {};
    auto stGva = m_shaderTable->GetGPUVirtualAddress();
    drd.RayGenerationShaderRecord.StartAddress = stGva + m_rayGenStart;
    drd.RayGenerationShaderRecord.SizeInBytes  = m_rayGenSize;
    drd.MissShaderTable.StartAddress  = stGva + m_missStart;
    drd.MissShaderTable.SizeInBytes   = m_missTotalSize;
    drd.MissShaderTable.StrideInBytes = m_missRecordSize;
    drd.HitGroupTable.StartAddress    = stGva + m_hitStart;
    drd.HitGroupTable.SizeInBytes     = m_hitTotalSize;
    drd.HitGroupTable.StrideInBytes   = m_hitRecordSize;
    drd.Width  = m_width;
    drd.Height = m_height;
    drd.Depth  = 1;
    cl->DispatchRays(&drd);

    // ---- Output UAV -> back buffer ----
    // Output is in UAV state from creation / from the previous frame's
    // final transition; back buffer is in RENDER_TARGET after Prepare().
    auto bb = m_deviceResources->GetRenderTarget();
    {
        D3D12_RESOURCE_BARRIER pre[2] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_output.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(bb,
                D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST),
        };
        cl->ResourceBarrier(_countof(pre), pre);
    }
    cl->CopyResource(bb, m_output.Get());
    {
        D3D12_RESOURCE_BARRIER post[2] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_output.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            CD3DX12_RESOURCE_BARRIER::Transition(bb,
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET),
        };
        cl->ResourceBarrier(_countof(post), post);
    }

    m_deviceResources->Present();   // RENDER_TARGET -> PRESENT, then Present
}

// =============================================================================
// Notifications
// =============================================================================

void PartitionedTlasSample::OnSizeChanged(UINT width, UINT height, bool minimized)
{
    if (!m_deviceResources->WindowSizeChanged(width, height, minimized)) return;
    UpdateForSizeChange(width, height);
    CreateWindowSizeDependentResources();
}

// =============================================================================
// Hotkeys + PTLAS resize (phase 3c)
// =============================================================================
//
// [  /  ]  : cycle partition budget through a power-of-2 ladder.  Each
//            change schedules a PTLAS tear-down + recreate at the start
//            of the next frame -- the EXPENSIVE update path per spec.
//            (A true SourceAS-based incremental resize is a later
//            milestone; this version recreates from scratch so the spike
//            is unmistakable.)
// SPACE    : pause/resume the flock motion (TODO -- placeholder).
//
void PartitionedTlasSample::OnKeyDown(UINT8 key)
{
    static const uint32_t kBudgetLadder[] = { 8, 16, 32, 64, 128, 216 };
    static const int kBudgetCount = (int)(sizeof(kBudgetLadder) / sizeof(kBudgetLadder[0]));
    auto findCurrentIdx = [&]() -> int {
        for (int i = 0; i < kBudgetCount; ++i)
            if (kBudgetLadder[i] >= m_partitionBudget) return i;
        return kBudgetCount - 1;
    };
    if (key == VK_OEM_4)   // [
    {
        int i = findCurrentIdx();
        if (i > 0)
        {
            m_pendingBudget = kBudgetLadder[i - 1];
            SampleLog::LogF(L"[hotkey] '[' -> queue PTLAS resize to %u partitions (was %u)\n",
                            m_pendingBudget, m_partitionBudget);
        }
    }
    else if (key == VK_OEM_6)   // ]
    {
        int i = findCurrentIdx();
        if (m_partitionBudget < kBudgetLadder[i]) i--;  // bump up if not on a ladder rung
        if (i + 1 < kBudgetCount)
        {
            m_pendingBudget = kBudgetLadder[i + 1];
            SampleLog::LogF(L"[hotkey] ']' -> queue PTLAS resize to %u partitions (was %u)\n",
                            m_pendingBudget, m_partitionBudget);
        }
    }
}

void PartitionedTlasSample::ApplyResizeIfPending()
{
    // (1) Check scheduled resizes (--resize-at).
    for (size_t i = 0; i < m_scheduledResizes.size(); )
    {
        if (m_scheduledResizes[i].frame == m_framesRendered)
        {
            m_pendingBudget = m_scheduledResizes[i].newBudget;
            SampleLog::LogF(L"[scheduled resize at frame %llu] -> %u partitions "
                            L"(was %u)\n",
                            (unsigned long long)m_framesRendered, m_pendingBudget,
                            m_partitionBudget);
            m_scheduledResizes.erase(m_scheduledResizes.begin() + i);
        }
        else
        {
            ++i;
        }
    }
    if (m_pendingBudget == 0 || m_pendingBudget == m_partitionBudget)
    {
        m_pendingBudget = 0;
        return;
    }

    // (2) Tear-down + recreate.  The OLD PtlasSystem owns the OLD PTLAS
    // resource; releasing it via reset() reclaims its memory once the GPU
    // is done.  Wait for GPU to drain in-flight frames first so the
    // resources can be safely freed.
    m_deviceResources->WaitForGpu();
    SampleLog::LogF(L"[ptlas-resize] applying: budget %u -> %u  "
                    L"(tear-down + recreate; spec's expensive path)\n",
                    m_partitionBudget, m_pendingBudget);
    m_partitionBudget = m_pendingBudget;
    m_pendingBudget   = 0;

    // Recreate RollingPartitions with new budget; this resets ownership
    // so the next Update sees all balls as "changed" and the next Build
    // does a full WRITE_INSTANCE pass.
    m_rollingParts.Initialize(m_partitionBudget, /*forwardBias*/ 0.30f,
                              m_scene, m_ballWorldPos);

    // Recreate the TLAS system (PTLAS or Traditional -- both are valid).
    ITlasSystem::InitDesc tlasInit = {};
    tlasInit.maxInstances                  = m_scene.TotalBalls() + kDonutCount;
    tlasInit.maxPartitions                 = m_partitionBudget;
    tlasInit.maxInstancesPerPartition      = std::max(1u, 2 * m_scene.BallsPerPartition());
    tlasInit.maxInstancesInGlobalPartition = kDonutCount;
    m_tlas.reset();
    if (m_tlasMode == TlasMode::Partitioned)
        m_tlas = std::make_unique<PtlasSystem>();
    else
        m_tlas = std::make_unique<TraditionalTlasSystem>();
    m_tlas->Initialize(m_dxrDevice.Get(), m_deviceResources.get(), tlasInit);
    m_ptlasInitialWriteDone = false;
    m_cumBallChanges = 0;
    m_cumBallChangesLastLog = 0;
}

void PartitionedTlasSample::OnDestroy()
{
    m_deviceResources->WaitForGpu();
    OnDeviceLost();
}

void PartitionedTlasSample::OnDeviceLost()
{
    ReleaseDeviceDependentResources();
}

void PartitionedTlasSample::OnDeviceRestored()
{
    CreateDeviceDependentResources();
    CreateWindowSizeDependentResources();
}

// =============================================================================
// Screenshot
// =============================================================================
void PartitionedTlasSample::CaptureBackBufferToFile(const std::wstring& path)
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
    HRESULT hrSave = SaveRGBAToPng(path, width, height,
        (const uint8_t*)mappedRaw + footprint.Offset, footprint.Footprint.RowPitch);
    D3D12_RANGE writeRange = { 0, 0 };
    readback->Unmap(0, &writeRange);

    SampleLog::LogF(L"[screenshot] %s %ux%u -> %s\n",
        SUCCEEDED(hrSave) ? L"wrote" : L"FAILED to write", width, height, path.c_str());
}

HRESULT PartitionedTlasSample::SaveRGBAToPng(const std::wstring& path,
                                             UINT width, UINT height,
                                             const uint8_t* data, UINT rowPitchBytes)
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
    WICPixelFormatGUID pf = GUID_WICPixelFormat32bppRGBA;
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
    if (FAILED(hr = frame->Commit()))   return cleanup(hr);
    if (FAILED(hr = encoder->Commit())) return cleanup(hr);
    return cleanup(S_OK);
}
