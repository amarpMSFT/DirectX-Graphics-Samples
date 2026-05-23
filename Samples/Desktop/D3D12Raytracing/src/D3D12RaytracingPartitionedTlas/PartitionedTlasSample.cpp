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

    // Build ball assets (mesh + DXR1 BLAS).  This records into the
    // already-open cmd list; the caller (us, below) closes / executes /
    // waits at the end of init.
    m_ball.Initialize(m_dxrDevice.Get(), m_dxrCommandList.Get(), /*subdiv*/ 2);

    // Compute the static scene instance list.  Milestone 2b: every ball is
    // a static instance referencing the single shared BLAS; partition index
    // is the linear index of the partition cell containing the ball.
    BuildSceneInstances();
    SampleLog::LogF(L"[scene] grid=%ux%ux%u  ballsPerSide=%u  total balls=%u "
                    L"partitions=%u  perPartMax=%u\n",
                    m_scene.gridX, m_scene.gridY, m_scene.gridZ,
                    m_scene.ballsPerSide, m_scene.TotalBalls(),
                    m_scene.Partitions(), m_scene.BallsPerPartition());

    // Pick the active TLAS system, sized for the scene.
    ITlasSystem::InitDesc tlasInit = {};
    tlasInit.maxInstances                  = m_scene.TotalBalls();
    tlasInit.maxPartitions                 = m_scene.Partitions();
    tlasInit.maxInstancesPerPartition      = m_scene.BallsPerPartition();
    tlasInit.maxInstancesInGlobalPartition = 0;     // reserved for the flock in phase 3
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
    //   param 0 : descriptor table -> 1 UAV (u0)  -> output texture
    //   param 1 : root SRV (t0)                   -> TLAS (PTLAS or trad)
    //   param 2 : root CBV (b0)                   -> SceneConstantBuffer
    CD3DX12_DESCRIPTOR_RANGE uavRange = {};
    uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, /*BaseRegister*/0);

    CD3DX12_ROOT_PARAMETER params[3] = {};
    params[PT_GRS_OutputUavSlot].InitAsDescriptorTable(1, &uavRange);
    params[PT_GRS_AccelerationStructureSlot].InitAsShaderResourceView(0);
    params[PT_GRS_SceneCBVSlot].InitAsConstantBufferView(0);

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
    lib->DefineExport(L"ClosestHit");

    auto* hit = so.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    hit->SetClosestHitShaderImport(L"ClosestHit");
    hit->SetHitGroupExport(L"HitGroup");
    hit->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

    auto* cfg = so.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
    cfg->Config(/*MaxPayload*/16, /*MaxAttr*/8);

    auto* gsig = so.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
    gsig->SetRootSignature(m_globalRootSig.Get());

    auto* pcfg = so.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
    pcfg->Config(/*MaxRecursion*/1);

    ThrowIfFailed(m_dxrDevice->CreateStateObject(so, IID_PPV_ARGS(&m_rtStateObject)),
        L"CreateStateObject failed");
}

void PartitionedTlasSample::CreateShaderTable()
{
    ComPtr<ID3D12StateObjectProperties> props;
    ThrowIfFailed(m_rtStateObject.As(&props));
    void* idRaygen = props->GetShaderIdentifier(L"Raygen");
    void* idMiss   = props->GetShaderIdentifier(L"Miss");
    void* idHit    = props->GetShaderIdentifier(L"HitGroup");

    const UINT64 idBytes  = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    const UINT64 recAlign = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;
    const UINT64 tabAlign = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    const UINT64 recSize  = AlignUp(idBytes, recAlign);
    m_rayGenRecordSize     = recSize;
    m_missRecordStartOffset = AlignUp(recSize,                     tabAlign);
    m_missRecordSize       = recSize;
    m_hitRecordStartOffset = AlignUp(m_missRecordStartOffset + recSize, tabAlign);
    m_hitRecordSize        = recSize;
    const UINT64 total     = m_hitRecordStartOffset + recSize;

    m_shaderTable = PtSample::CreateUploadBuffer(m_dxrDevice.Get(), total, L"ShaderTable");
    UINT8* cpu = nullptr;
    CD3DX12_RANGE noRead(0, 0);
    ThrowIfFailed(m_shaderTable->Map(0, &noRead, reinterpret_cast<void**>(&cpu)));
    memset(cpu, 0, (size_t)total);
    memcpy(cpu + 0,                       idRaygen, idBytes);
    memcpy(cpu + m_missRecordStartOffset, idMiss,   idBytes);
    memcpy(cpu + m_hitRecordStartOffset,  idHit,    idBytes);
    m_shaderTable->Unmap(0, nullptr);
}

// =============================================================================
// Scene instances (milestone 2c: world-space + partition-home table)
// =============================================================================
void PartitionedTlasSample::BuildSceneInstances()
{
    using namespace DirectX;
    m_sceneInstances.clear();
    m_sceneInstances.reserve(m_scene.TotalBalls());
    m_partitionHomes.clear();
    m_partitionHomes.reserve(m_scene.Partitions());

    // Partition home table (world-space center of each partition cell).
    for (uint32_t pk = 0; pk < m_scene.gridZ; ++pk)
    for (uint32_t pj = 0; pj < m_scene.gridY; ++pj)
    for (uint32_t pi = 0; pi < m_scene.gridX; ++pi)
    {
        m_partitionHomes.push_back(m_scene.PartitionHomeWorld(pi, pj, pk));
    }

    const D3D12_GPU_VIRTUAL_ADDRESS blas = m_ball.BlasGpuVa();
    for (uint32_t pk = 0; pk < m_scene.gridZ; ++pk)
    for (uint32_t pj = 0; pj < m_scene.gridY; ++pj)
    for (uint32_t pi = 0; pi < m_scene.gridX; ++pi)
    {
        const uint32_t partIdx = m_scene.PartitionIndex(pi, pj, pk);
        for (uint32_t bk = 0; bk < m_scene.ballsPerSide; ++bk)
        for (uint32_t bj = 0; bj < m_scene.ballsPerSide; ++bj)
        for (uint32_t bi = 0; bi < m_scene.ballsPerSide; ++bi)
        {
            DirectX::XMFLOAT3 pos = m_scene.BallWorldPos(pi, pj, pk, bi, bj, bk);
            // World-space Scale * Translate, then transposed for DXR's
            // column-vector 3x4 instance-transform convention.  Per-frame
            // adjustments (subtracting as_origin for Traditional, subtracting
            // partition_home for Partitioned-local) happen in DoRender by
            // tweaking m[0..2][3] before the TLAS-system call.
            XMMATRIX m = XMMatrixScaling(m_scene.ballScale, m_scene.ballScale, m_scene.ballScale)
                       * XMMatrixTranslation(pos.x, pos.y, pos.z);
            SceneInstance inst = {};
            XMStoreFloat4x4(&inst.transform, XMMatrixTranspose(m));
            inst.blasGva        = blas;
            inst.instanceID     = m_scene.BallIndex(pi, pj, pk, bi, bj, bk);
            inst.instanceMask   = 0xFF;
            inst.partitionIndex = partIdx;
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
    cb.lightDirAndPad  = { 0.5f, 0.8f, 0.3f, 0 };
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
    RecomputeAsOrigin();        // updates m_asOrigin BEFORE UpdateSceneConstantBuffer reads it
    UpdateSceneConstantBuffer();

    m_deviceResources->Prepare();   // cmd list reset; back buffer -> RENDER_TARGET
    auto cl  = m_dxrCommandList.Get();
    auto cl2 = m_dxr2CommandList.Get();

    // ---- (P)TLAS build for this frame ----
    //
    // Both modes must produce hits in the same AS-space coord frame so the
    // shader output is bit-for-bit identical:
    //   * Partitioned (PTLAS):
    //       - WRITE_INSTANCE once at init with PARTITION-LOCAL transforms
    //         (instance translation = ball_world - partition_home).
    //       - TRANSLATE_PARTITION every frame: translation[p] = home[p] - as_origin.
    //       - Per-frame work: ~partitionCount partition translations.  The
    //         partition-local instance writes never repeat.
    //   * Traditional (DXR1 TLAS):
    //       - No partition concept, so we have to bake (world - as_origin)
    //         into every instance transform and re-write all of them each
    //         frame.  This is exactly the cost PTLAS sidesteps -- 5832
    //         instance writes vs 216 partition translations in this scene.
    using namespace DirectX;

    m_tlas->BeginFrame();
    // Timestamp BEFORE TLAS build.
    const UINT tsSlot     = m_timestampSlotIdx;
    const UINT tsSlotBase = tsSlot * kTimestampsPerFrame;
    cl->EndQuery(m_timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, tsSlotBase + 0);

    if (m_tlasMode == TlasMode::Partitioned)
    {
        if (!m_ptlasInitialWriteDone)
        {
            // One-shot: write all instances with partition-local transforms.
            std::vector<SceneInstance> local(m_sceneInstances.size());
            for (size_t i = 0; i < m_sceneInstances.size(); ++i)
            {
                local[i] = m_sceneInstances[i];
                const XMFLOAT3& home = m_partitionHomes[local[i].partitionIndex];
                // SceneInstance.transform is transposed (DXR 3x4 layout):
                // translation lives at m[0][3], m[1][3], m[2][3].
                local[i].transform.m[0][3] -= home.x;
                local[i].transform.m[1][3] -= home.y;
                local[i].transform.m[2][3] -= home.z;
            }
            m_tlas->WriteInstances(local.data(), (UINT)local.size());
            m_ptlasInitialWriteDone = true;
            SampleLog::LogF(L"[ptlas] initial WRITE_INSTANCE pass: %u instances (partition-local transforms)\n",
                            (unsigned)local.size());
        }

        // Per-frame: TRANSLATE_PARTITION for every partition.
        std::vector<ITlasSystem::PartitionTranslate> pts(m_partitionHomes.size());
        for (size_t p = 0; p < m_partitionHomes.size(); ++p)
        {
            pts[p].partitionIndex = (UINT)p;
            pts[p].translation[0] = m_partitionHomes[p].x - m_asOrigin.x;
            pts[p].translation[1] = m_partitionHomes[p].y - m_asOrigin.y;
            pts[p].translation[2] = m_partitionHomes[p].z - m_asOrigin.z;
        }
        m_tlas->TranslatePartitions(pts.data(), (UINT)pts.size());
    }
    else
    {
        // Traditional: rebuild instance descs with translation baked
        // (world - as_origin) so we trace in the same AS-space frame.
        std::vector<SceneInstance> adj(m_sceneInstances.size());
        for (size_t i = 0; i < m_sceneInstances.size(); ++i)
        {
            adj[i] = m_sceneInstances[i];
            adj[i].transform.m[0][3] -= m_asOrigin.x;
            adj[i].transform.m[1][3] -= m_asOrigin.y;
            adj[i].transform.m[2][3] -= m_asOrigin.z;
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
        SampleLog::LogF(L"[frame %llu] tlas=%s instances=%u part_translates=%u "
                        L"result=%llu scratch=%llu  build_ms=%.3f (raw %.3f)  "
                        L"as_origin=(%.2f, %.2f, %.2f)\n",
                        (unsigned long long)m_framesRendered,
                        m_tlas->ModeName(),
                        fs.instancesSubmitted, fs.partitionsTouched,
                        (unsigned long long)fs.resultBytes,
                        (unsigned long long)fs.scratchBytes,
                        m_tlasBuildMsEma, m_tlasBuildMsLast,
                        m_asOrigin.x, m_asOrigin.y, m_asOrigin.z);
    }

    // ---- Bind RT pipeline + descriptors ----
    ID3D12DescriptorHeap* heaps[] = { m_descHeap.Get() };
    cl->SetDescriptorHeaps(_countof(heaps), heaps);
    cl->SetComputeRootSignature(m_globalRootSig.Get());
    D3D12_GPU_DESCRIPTOR_HANDLE uavGpu = m_descHeap->GetGPUDescriptorHandleForHeapStart();
    uavGpu.ptr += SIZE_T(m_uavHeapIdx) * m_descSize;
    cl->SetComputeRootDescriptorTable(PT_GRS_OutputUavSlot, uavGpu);
    cl->SetComputeRootShaderResourceView(PT_GRS_AccelerationStructureSlot, m_tlas->Gva());
    const UINT slot = (UINT)(m_framesRendered % FrameCount);
    cl->SetComputeRootConstantBufferView(PT_GRS_SceneCBVSlot,
        m_sceneCb->GetGPUVirtualAddress() + slot * m_sceneCbStride);

    cl->SetPipelineState1(m_rtStateObject.Get());

    // ---- Dispatch ----
    D3D12_DISPATCH_RAYS_DESC drd = {};
    auto stGva = m_shaderTable->GetGPUVirtualAddress();
    drd.RayGenerationShaderRecord.StartAddress = stGva;
    drd.RayGenerationShaderRecord.SizeInBytes  = m_rayGenRecordSize;
    drd.MissShaderTable.StartAddress  = stGva + m_missRecordStartOffset;
    drd.MissShaderTable.SizeInBytes   = m_missRecordSize;
    drd.MissShaderTable.StrideInBytes = m_missRecordSize;
    drd.HitGroupTable.StartAddress    = stGva + m_hitRecordStartOffset;
    drd.HitGroupTable.SizeInBytes     = m_hitRecordSize;
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
