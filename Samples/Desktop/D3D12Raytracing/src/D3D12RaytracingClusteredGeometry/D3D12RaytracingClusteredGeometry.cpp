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

#include <DirectXMath.h>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace DX;
using namespace DirectX;
using Microsoft::WRL::ComPtr;

// ==== Shader entry-point names (must match Raytracing.hlsl) ====
const wchar_t* D3D12RaytracingClusteredGeometry::c_raygenName     = L"RayGen";
const wchar_t* D3D12RaytracingClusteredGeometry::c_closestHitName = L"Hit";
const wchar_t* D3D12RaytracingClusteredGeometry::c_missName       = L"Miss";
const wchar_t* D3D12RaytracingClusteredGeometry::c_hitGroupName   = L"ClusterHitGroup";

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
    m_deviceResources = std::make_unique<DeviceResources>(
        DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_UNKNOWN,
        FrameCount,
        D3D_FEATURE_LEVEL_11_0,
        DeviceResources::c_RequireTearingSupport,
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
    {
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
    SampleLog::Write(L">>> BuildAccelerationStructures\n");
    BuildAccelerationStructures();
    SampleLog::Write(L">>> CreateRaytracingPipelineAndShaderTables\n");
    CreateRaytracingPipelineAndShaderTables();
    SampleLog::Write(L">>> CreateDescriptorHeapAndRaytracingOutput\n");
    CreateDescriptorHeapAndRaytracingOutput();
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
        SampleLog::Write(L"  WARNING: cluster builds will fail. Try 'd3dconfig device force-warp=true'\n");

    SetCustomWindowText(m_clustersAndPtlasSupported
        ? L"clustered geometry: SUPPORTED"
        : L"clustered geometry: NOT SUPPORTED");
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
    // 12 bits/axis = 36 bits/vertex. Per-axis bit counts in the bitstream
    // (driven by maxDelta per axis) are typically much less for clusters
    // with constant or near-constant components (cube faces have 1 bit on
    // the constant axis).
    constexpr int kCompressedBitsPerComponent = 12;

    auto add = [&](ProceduralGeometry::Mesh&& mesh, const XMFLOAT3& pos, float scale, UINT instanceID)
    {
        ClusterObject obj;
        obj.mesh        = std::move(mesh);
        obj.worldPos    = pos;
        obj.worldScale  = scale;
        obj.instanceID  = instanceID;
        m_objects.push_back(std::move(obj));
    };

    // Arrange the six objects in a roughly-hexagonal cluster around the origin
    // in the XZ plane, with small Y offsets for visual interest. Camera orbits
    // around Y, so every angle frames the whole group evenly (no view ends up
    // with one giant object in front and the rest tiny behind it).
    // Hex coordinates: x = R cos(theta), z = R sin(theta), six positions 60° apart.
    constexpr float R = 1.6f;
    auto hex = [&](int i) {
        float th = (float)i * (2.0f * (float)M_PI / 6.0f);
        return XMFLOAT3(R * std::cos(th), 0.0f, R * std::sin(th));
    };

    auto with_y = [](XMFLOAT3 p, float y) { p.y = y; return p; };

    // Spheres around the hex.
    // TEMP: tiny sphere SHIFTED in X, Y, Z so every vertex is positive.
    auto m_shifted = ProceduralGeometry::GenerateUVSphereSpatialTiles(0.85f, 8, 8, /*tileLat*/4, /*tileLong*/4, 0);
    for (auto& c : m_shifted.clusters)
        for (auto& p : c.positions) { p.x += 1.5f; p.y += 1.5f; p.z += 1.5f; }
    add(std::move(m_shifted),
        XMFLOAT3(-1.5f, -1.5f, -1.5f), 1.0f, 0);

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
    // COMPRESSED1 STATIC PATH - KNOWN ISSUE (open as of 2026-05-15)
    // ----------------------------------------------------------------------------
    // On NVIDIA RTX 4090 with experimental D3D12 (D3D12Core 1.10 preview, agility
    // SDK 722), passing FLOAT32_3 here produces a clean rainbow-tile render of
    // the scene. Passing COMPRESSED1 renders the cube perfectly but mangles the
    // sphere/torus clusters - typically with one cluster appearing as a stretched
    // "tail" extending well beyond the object's bounds, and another cluster
    // missing entirely. The breakage is reproducible with a SINGLE 4-cluster
    // sphere (set kSceneTinyRepro = true to enable this minimal repro scene),
    // and persists across:
    //   * 8-bit, 12-bit, and 16-bit-per-axis encodings
    //   * uniform vs per-axis bit counts
    //   * 16-byte vs 256-byte vertex-buffer alignment
    //   * positive-only anchors (verified by shifting the mesh into +x+y+z)
    //   * MaxCompressedClusterPositionsSize set to exact size vs 4x oversize
    //
    // CPU-side roundtrip via Compressed1::Decode is bit-exact (max error
    // ~0.0002 units, which is sub-quantization-step). Byte-for-byte cluster
    // dumps via DUMP_COMPRESSED1_DIAG match the d3d12conf reference encoder's
    // bit ordering and header layout.
    //
    // Filed as: <TODO: bug-tracker link>. Until resolved, the sample defaults
    // to FLOAT32_3 (see VertexMode::Float32_3 in the header).
    // ============================================================================
    if (m_vertexMode == VertexMode::Compressed1)
    {
        // Pick a single shared compressed1 exponent across the WHOLE scene (all
        // clusters in all objects). Same-grid quantization keeps adjacent
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
            const float minUnit = maxExtent / float((1ull << kCompressedBitsPerComponent) - 1);
            sharedExponent = (int)std::ceil(std::log2(minUnit)) + 127;
            if (sharedExponent < 1)   sharedExponent = 1;
            if (sharedExponent > 232) sharedExponent = 232;
            SampleLog::LogF(L"[compressed1] shared exponent (biased): %d  (unit=%.6f)\n",
                            sharedExponent, std::ldexp(1.0f, sharedExponent - 127));
        }

        size_t totalCompressedBytes = 0, totalUncompressedBytes = 0;
        float  maxRoundtripError = 0.f;
        size_t globalClusterIdx  = 0;
        for (auto& obj : m_objects)
        {
            obj.encoded.reserve(obj.mesh.clusters.size());
            for (const auto& c : obj.mesh.clusters)
            {
                auto enc = Compressed1::Encode(c.positions, kCompressedBitsPerComponent, sharedExponent);
                auto decoded = Compressed1::Decode(enc);
                for (size_t i = 0; i < c.positions.size(); ++i)
                {
                    float dx = decoded[i].x - c.positions[i].x;
                    float dy = decoded[i].y - c.positions[i].y;
                    float dz = decoded[i].z - c.positions[i].z;
                    maxRoundtripError = std::max(maxRoundtripError, std::sqrt(dx*dx + dy*dy + dz*dz));
                }
                // Dump cluster 0 (sphere) and cluster 500 (cube) byte-for-byte
                // for forensic comparison. Cluster 0 is sphere obj #0's first
                // cluster; cluster 500 is the first cube cluster (cube starts
                // at firstClusterID=500).
                // KNOWN-ISSUE FORENSICS (disable for production builds).
                // Set DUMP_COMPRESSED1_DIAG to 1 to print per-cluster header,
                // anchors, bit-counts and a sample of original-vs-decoded
                // positions for the listed cluster IDs. Useful when chasing
                // why the BVH-build's interpretation of a compressed1 cluster
                // diverges from the CPU roundtrip decode.
                #define DUMP_COMPRESSED1_DIAG 0
                #if DUMP_COMPRESSED1_DIAG
                if (c.clusterID == 0 || c.clusterID == 1 || c.clusterID == 10
                    || c.clusterID == 56 || c.clusterID == 500)
                {
                    SampleLog::LogF(L"[compressed1 dump] clusterID=%u "
                                    L"verts=%u  x_bits=%u y_bits=%u z_bits=%u  "
                                    L"anchor=(%d, %d, %d)  exp=%u\n",
                                    c.clusterID, enc.vertexCount,
                                    enc.xBits, enc.yBits, enc.zBits,
                                    (int)DECODE_D3D12_COMPRESSED1_X_ANCHOR(enc.header),
                                    (int)DECODE_D3D12_COMPRESSED1_Y_ANCHOR(enc.header),
                                    (int)DECODE_D3D12_COMPRESSED1_Z_ANCHOR(enc.header),
                                    (unsigned)DECODE_D3D12_COMPRESSED1_EXPONENT(enc.header));
                    SampleLog::LogF(L"  header words: %08X %08X %08X  bitstream-bytes=%zu\n",
                                    enc.header.field0, enc.header.field1, enc.header.field2,
                                    enc.bitstream.size());
                    // First 4 AND last 4 vertices: print original + decoded
                    auto printV = [&](size_t i) {
                        SampleLog::LogF(L"  v[%2zu] orig=(%+.5f,%+.5f,%+.5f) dec=(%+.5f,%+.5f,%+.5f)\n",
                                        i,
                                        c.positions[i].x, c.positions[i].y, c.positions[i].z,
                                        decoded[i].x,     decoded[i].y,     decoded[i].z);
                    };
                    for (size_t i = 0; i < std::min<size_t>(4, c.positions.size()); ++i) printV(i);
                    if (c.positions.size() > 8)
                    {
                        SampleLog::Write(L"  ...\n");
                        for (size_t i = c.positions.size() - 4; i < c.positions.size(); ++i) printV(i);
                    }
                }
                #endif // DUMP_COMPRESSED1_DIAG
                totalCompressedBytes   += enc.TotalBytes();
                totalUncompressedBytes += c.positions.size() * sizeof(ProceduralGeometry::float3);
                obj.encoded.push_back(std::move(enc));
                ++globalClusterIdx;
            }
        }
        SampleLog::LogF(L"[compressed1] %u clusters @ %d bits/comp -> %zu bytes (vs %zu uncompressed = %.2fx)\n",
                        m_totalClusterCount, kCompressedBitsPerComponent,
                        totalCompressedBytes, totalUncompressedBytes,
                        (double)totalUncompressedBytes / (double)totalCompressedBytes);
        SampleLog::LogF(L"[compressed1] worst-case roundtrip position error: %.6f units\n", maxRoundtripError);
    }
    else
    {
        // FLOAT32_3 path: just hold the original positions, no encoding.
        for (auto& obj : m_objects)
        {
            obj.rawPositions.reserve(obj.mesh.clusters.size());
            for (const auto& c : obj.mesh.clusters)
                obj.rawPositions.push_back(c.positions);
        }
        size_t totalBytes = 0;
        for (const auto& obj : m_objects)
            for (const auto& v : obj.rawPositions)
                totalBytes += v.size() * sizeof(ProceduralGeometry::float3);
        SampleLog::LogF(L"[float32_3] %u clusters: %zu bytes total\n",
                        m_totalClusterCount, totalBytes);
    }
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
}

// ---------------------------------------------------------------------------------
// Concatenate per-cluster vertex blob + index buffer + per-cluster build args
// into a single upload buffer. Layout (clusters in object order, all flattened):
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
    cursor = alignTo(cursor, kAlign);
    const size_t argsOffset = cursor;
    const size_t argStride  = sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS);
    const size_t argsSize   = m_totalClusterCount * argStride;
    cursor += argsSize;

    const size_t totalSize = alignTo(cursor, 256);
    SampleLog::LogF(L"[upload] cluster input buffer: %zu bytes (%zu args, %zu data) - %s\n",
                    totalSize, argsSize, totalSize - argsSize,
                    useFloat ? L"FLOAT32_3 path" : L"COMPRESSED1 path");

    auto uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto bufDesc    = CD3DX12_RESOURCE_DESC::Buffer(totalSize);
    ThrowIfFailed(device->CreateCommittedResource(
        &uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_clusterInputBuffer)));
    m_clusterInputBuffer->SetName(L"ClusterInputBuffer");

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

    // Per-cluster CLAS build args.
    auto* args = reinterpret_cast<D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS*>(mapped + argsOffset);
    gIdx = 0;
    for (const auto& obj : m_objects)
    for (size_t i = 0; i < obj.mesh.clusters.size(); ++i, ++gIdx)
    {
        const auto& src = obj.mesh.clusters[i];

        D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS a = {};
        a.ClusterID                         = src.clusterID;
        a.ClusterFlags                      = 0;
        a.TriangleCount                     = (UINT16)(src.indices.size() / 3);
        a.VertexCount                       = (UINT16)src.positions.size();
        a.BaseGeometryIndexAndFlags         = 0;
        a.OpacityMicromapBaseLocation       = 0;
        // For COMPRESSED1 the data is a single blob (stride concept doesn't apply, pass 0);
        // for FLOAT32_3 we have packed float3 per vertex (12 bytes stride).
        a.VertexBufferStride                = useFloat ? (UINT16)(sizeof(ProceduralGeometry::float3)) : 0;
        a.IndexBufferStride                 = 1;     // 1 byte per uint8_t index
        a.OpacityMicromapIndexBufferStride  = 0;
        a.GeometryIndexAndFlagsArrayStride  = 0;
        a.PositionTruncateBitCount          = 0;
        a.ReservedPadding                   = 0;
        a.VertexBuffer                      = baseGPUVA + slots[gIdx].vbOffset;
        a.IndexBuffer                       = baseGPUVA + slots[gIdx].ibOffset;
        a.GeometryIndexAndFlagsArray        = 0;
        a.GeometryIndexAndFlagsIndexBuffer  = 0;
        a.OpacityMicromapArray              = 0;
        a.OpacityMicromapIndexBuffer        = 0;
        args[gIdx] = a;
    }
    m_clusterInputBuffer->Unmap(0, nullptr);

    m_clasArgsArrayGPUVA = baseGPUVA + argsOffset;
    m_clasArgsStride     = (UINT)argStride;
}

// ---------------------------------------------------------------------------------
// Single CLAS build covering ALL clusters from ALL objects in one
// ExecuteIndirectRTASOperations call. Implementation-defined layout in
// IMPLICIT_DESTINATIONS mode; per-cluster addresses written to m_clasAddressArray.
// ---------------------------------------------------------------------------------
void D3D12RaytracingClusteredGeometry::BuildClasIndirect()
{
    auto device = m_deviceResources->GetD3DDevice();
    const UINT N = m_totalClusterCount;
    const bool useFloat = (m_vertexMode == VertexMode::Float32_3);

    // Compute per-cluster maxima for ClusterLimits + (compressed1 path only)
    // the worst-case packed vertex blob size.
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

    D3D12_RTAS_CLUSTER_LIMITS limits = {};
    limits.MaxArgCount                                   = N;
    limits.MaxGeometryIndexValue                         = 0;
    limits.MaxUniqueGeometryIndexAndFlagsCountPerCluster = 1;
    limits.MaxTriangleCountPerCluster                    = maxTris;
    limits.MaxVertexCountPerCluster                      = maxVerts;
    limits.MaxTotalTriangleCount                         = totalTris;
    limits.MaxTotalVertexCount                           = totalVerts;
    limits.MaxOpacityMicromapIndicesPerCluster           = 0;

    D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC clasDesc = {};
    clasDesc.ClusterLimits                       = limits;
    clasDesc.Flags                               = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;
    clasDesc.Mode                                = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
    clasDesc.VertexFormat                        = useFloat ? D3D12_VERTEX_FORMAT_FLOAT32_3
                                                            : D3D12_VERTEX_FORMAT_COMPRESSED1;
    clasDesc.IndexFormat                         = D3D12_INDEX_FORMAT_UINT8;
    clasDesc.GeometryIndexAndFlagsIndexFormat    = D3D12_INDEX_FORMAT_NONE;
    clasDesc.OpacityMicromapIndexFormat          = D3D12_INDEX_FORMAT_NONE;
    // MaxCompressedClusterPositionsSize is only meaningful for COMPRESSED1.
    // TEMP: way oversized to test if undersizing is the bug
    clasDesc.MaxCompressedClusterPositionsSize   = useFloat ? 0u : (maxCompressedSize * 4 + 1024);

    D3D12_RTAS_OPERATION_INPUTS opInputs = {};
    opInputs.Type                  = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
    opInputs.pClusterTrianglesDesc = &clasDesc;

    D3D12_RTAS_OPERATION_PREBUILD_INFO prebuild = {};
    m_dxr2Device->GetRTASOperationPrebuildInfo(&opInputs, &prebuild);
    SampleLog::LogF(L"[CLAS prebuild] %u clusters: result max=%llu bytes, scratch=%llu bytes\n",
                    N,
                    (unsigned long long)prebuild.ResultDataMaxSizeInBytes,
                    (unsigned long long)prebuild.ScratchDataSizeInBytes);

    AllocateUAVBuffer(device, prebuild.ResultDataMaxSizeInBytes,
                      &m_clasResultBuffer, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                      L"All-cluster CLAS result buffer");
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

    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDesc, D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

    D3D12_RESOURCE_BARRIER barriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasResultBuffer.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasAddressArray.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasSizeArray.Get()),
    };
    m_dxrCommandList->ResourceBarrier(_countof(barriers), barriers);
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
    blasDesc.Flags              = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;
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
    // referencing a sub-range of the global CLAS-address array.
    {
        std::vector<D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS> argsVec(N_obj);
        const D3D12_GPU_VIRTUAL_ADDRESS clasArrayGPUVA = m_clasAddressArray->GetGPUVirtualAddress();
        for (UINT i = 0; i < N_obj; ++i)
        {
            const auto& obj = m_objects[i];
            argsVec[i].ClasAddressCount  = obj.clusterCount;
            argsVec[i].ClasAddressStride = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
            argsVec[i].ClasAddressArray  = clasArrayGPUVA
                + (UINT64)obj.globalClusterStart * sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
        }
        AllocateUploadBuffer(device, argsVec.data(),
                             argsVec.size() * sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS),
                             &m_blasArgsBuffer, L"BLAS args (multi-object)");
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
// ClusterIdOffset=10000 in the per-cluster instantiate args. So the shader
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
    // ------------------------------------------------------------------
    constexpr float kRestRadius   = 0.65f;
    constexpr float kEnvelopeScale = 1.18f;       // hint sphere radius = rest * 1.18
    obj.mesh = ProceduralGeometry::GenerateUVSphereSpatialTiles(
        kRestRadius, /*numLat*/24, /*numLong*/48, /*tileLat*/4, /*tileLong*/6, 0);
    obj.clusterCount         = (UINT)obj.mesh.clusters.size();
    obj.worldPos             = XMFLOAT3(0.0f, 1.6f, 0.0f);         // hovers above the hex group
    obj.worldScale           = 1.0f;
    obj.instanceID           = 99;
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
    const size_t totalSize = alignTo(cursor, 256);

    auto uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto bufDesc    = CD3DX12_RESOURCE_DESC::Buffer(totalSize);
    ThrowIfFailed(device->CreateCommittedResource(
        &uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&obj.templateInputBuffer)));
    obj.templateInputBuffer->SetName(L"Animated: cluster-template input buffer");

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
    auto* tArgs = reinterpret_cast<D3D12_RTAS_OPERATION_BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES_ARGS*>(
        mapped + argsOffset);
    for (UINT c = 0; c < obj.clusterCount; ++c)
    {
        const auto& src = obj.mesh.clusters[c];
        D3D12_RTAS_OPERATION_BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES_ARGS a = {};
        // The TrianglesArgs sub-struct is the same shape as for a CLAS build;
        // the driver uses the topology + hint positions to construct a template
        // BVH skeleton that per-frame INSTANTIATE fills in with real positions.
        a.TrianglesArgs.ClusterID           = c;             // 0..N-1; shader sees this + ClusterIdOffset
        a.TrianglesArgs.ClusterFlags        = 0;
        a.TrianglesArgs.TriangleCount       = (UINT16)(src.indices.size() / 3);
        a.TrianglesArgs.VertexCount         = (UINT16)src.positions.size();
        a.TrianglesArgs.VertexBufferStride  = (UINT16)sizeof(XMFLOAT3);
        a.TrianglesArgs.IndexBufferStride   = 1;
        a.TrianglesArgs.VertexBuffer        = templateInputBaseGPUVA + slots[c].vbOffset;
        a.TrianglesArgs.IndexBuffer         = templateInputBaseGPUVA + slots[c].ibOffset;
        // No instantiation-bounds hint - the driver uses the hint vertex AABB.
        a.InstantiationBoundingBoxLimit     = 0;
        tArgs[c] = a;
    }
    obj.templateInputBuffer->Unmap(0, nullptr);
    obj.templateArgsArrayGPUVA = templateInputBaseGPUVA + argsOffset;
    obj.templateArgsStride     = (UINT)argStride;

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
    tplDesc.Flags                     = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;
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
    //    perFrameVertexBuffer  - the new positions for all clusters this frame
    //    perFrameInstArgsBuffer- one INSTANTIATE_CLUSTER_TEMPLATES_ARGS per cluster
    //    perFrameClasResultBuffer + scratch + addresses (IMPLICIT_DESTINATIONS)
    //    blasStorage + scratch + args  (EXPLICIT_DESTINATIONS for fixed GPU VA)
    // ------------------------------------------------------------------
    const size_t vbBytes = (size_t)obj.totalVertexCount * sizeof(XMFLOAT3);
    {
        auto vbDesc = CD3DX12_RESOURCE_DESC::Buffer(alignTo(vbBytes, 256));
        ThrowIfFailed(device->CreateCommittedResource(
            &uploadHeap, D3D12_HEAP_FLAG_NONE, &vbDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&obj.perFrameVertexBuffer)));
        obj.perFrameVertexBuffer->SetName(L"Animated: per-frame vertex buffer");
        ThrowIfFailed(obj.perFrameVertexBuffer->Map(0, &noRead,
            reinterpret_cast<void**>(&obj.perFrameVertexBufferMapped)));
    }
    {
        const size_t instArgsBytes = obj.clusterCount * sizeof(D3D12_RTAS_OPERATION_INSTANTIATE_CLUSTER_TEMPLATES_ARGS);
        auto argsDesc = CD3DX12_RESOURCE_DESC::Buffer(alignTo(instArgsBytes, 256));
        ThrowIfFailed(device->CreateCommittedResource(
            &uploadHeap, D3D12_HEAP_FLAG_NONE, &argsDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&obj.perFrameInstArgsBuffer)));
        obj.perFrameInstArgsBuffer->SetName(L"Animated: per-frame instantiate args");
        ThrowIfFailed(obj.perFrameInstArgsBuffer->Map(0, &noRead,
            reinterpret_cast<void**>(&obj.perFrameInstArgsMapped)));
    }

    // Pre-fill the instantiate args: each cluster's args.ClusterTemplate gets
    // its template GVA, args.VertexBuffer gets a slice of perFrameVertexBuffer.
    // The TEMPLATE GVAs are in obj.templateAddressArray (GPU-only). We can't
    // read them on CPU; the runtime resolves them by-reference at instantiate
    // time via the BatchedOperationData (no - that's wrong, ClusterTemplate
    // in args IS the GVA. Need to read them back, or pre-compute via
    // address-array layout).
    //
    // SOLUTION: we used IMPLICIT_DESTINATIONS for the template build, so the
    // templates are packed contiguously in templateResultBuffer at runtime-
    // determined offsets. But we ALSO got the per-cluster GVAs written into
    // templateAddressArray. The IndirectArgumentArray for INSTANTIATE refers
    // to that ARRAY via the args' ClusterTemplate field... wait, that's a
    // GVA, not an array reference. Hmm.
    //
    // Cleanest path: switch the template build to EXPLICIT_DESTINATIONS, where
    // we pre-allocate per-template buffers and know their GVAs on CPU. But
    // that's clunky (N committed resources). OR: read back the
    // templateAddressArray on a CPU-accessible heap, populate the args.
    //
    // For now, use the approach the test does: have a small compute shader
    // (or just a CPU readback) to populate the ClusterTemplate field in
    // instantiate args. CPU readback is simpler for a once-at-init operation.
    {
        // CPU readback of templateAddressArray to populate ClusterTemplate.
        auto cmdQueue = m_deviceResources->GetCommandQueue();
        auto cmdList  = m_deviceResources->GetCommandList();

        ComPtr<ID3D12Resource> readback;
        auto rbHeap   = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto rbDesc   = CD3DX12_RESOURCE_DESC::Buffer((UINT64)obj.clusterCount * sizeof(D3D12_GPU_VIRTUAL_ADDRESS));
        ThrowIfFailed(device->CreateCommittedResource(&rbHeap, D3D12_HEAP_FLAG_NONE, &rbDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)));

        // Flush the BUILD_CLUSTER_TEMPLATES we recorded above so we can read
        // the addresses array.
        auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(obj.templateAddressArray.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &toCopy);
        cmdList->CopyBufferRegion(readback.Get(), 0, obj.templateAddressArray.Get(), 0,
            (UINT64)obj.clusterCount * sizeof(D3D12_GPU_VIRTUAL_ADDRESS));
        auto fromCopy = CD3DX12_RESOURCE_BARRIER::Transition(obj.templateAddressArray.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(1, &fromCopy);

        m_deviceResources->ExecuteCommandList();
        m_deviceResources->WaitForGpu();

        void* mappedAddrs = nullptr;
        D3D12_RANGE rng = { 0, (SIZE_T)obj.clusterCount * sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
        ThrowIfFailed(readback->Map(0, &rng, &mappedAddrs));
        const auto* templateGVAs = reinterpret_cast<const D3D12_GPU_VIRTUAL_ADDRESS*>(mappedAddrs);

        const D3D12_GPU_VIRTUAL_ADDRESS pfVbGPUVA = obj.perFrameVertexBuffer->GetGPUVirtualAddress();
        size_t vbCursor = 0;
        for (UINT c = 0; c < obj.clusterCount; ++c)
        {
            D3D12_RTAS_OPERATION_INSTANTIATE_CLUSTER_TEMPLATES_ARGS a = {};
            a.GeometryIndexOffset = 0;
            a.ClusterIdOffset     = 10000;                       // unique color range
            a.ClusterTemplate     = templateGVAs[c];
            a.VertexBuffer.StartAddress  = pfVbGPUVA + vbCursor;
            a.VertexBuffer.StrideInBytes = sizeof(XMFLOAT3);
            obj.perFrameInstArgsMapped[c] = a;
            vbCursor += obj.mesh.clusters[c].positions.size() * sizeof(XMFLOAT3);
        }
        D3D12_RANGE noWrite = { 0, 0 };
        readback->Unmap(0, &noWrite);
        SampleLog::LogF(L"[animated] template GVAs read back; first 3 = %llx %llx %llx\n",
                        (unsigned long long)templateGVAs[0],
                        obj.clusterCount > 1 ? (unsigned long long)templateGVAs[1] : 0ull,
                        obj.clusterCount > 2 ? (unsigned long long)templateGVAs[2] : 0ull);

        // Re-open the command list for subsequent setup work in this function.
        auto commandAllocator = m_deviceResources->GetCommandAllocator();
        ThrowIfFailed(commandAllocator->Reset());
        ThrowIfFailed(cmdList->Reset(commandAllocator, nullptr));
    }

    // ------------------------------------------------------------------
    // 5) Per-frame CLAS + BLAS storage. Sized via prebuild info.
    // ------------------------------------------------------------------
    D3D12_RTAS_INSTANTIATE_CLUSTER_TEMPLATE_INPUTS_DESC instDesc = {};
    instDesc.ClusterLimits     = limits;
    instDesc.Flags             = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;
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

    // BLAS-from-CLAS prebuild & alloc (single BLAS, EXPLICIT_DESTINATIONS).
    D3D12_RTAS_CLAS_INPUTS_DESC blasDesc = {};
    blasDesc.Flags              = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;
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

    // Pre-fill BLAS-from-CLAS args (it never changes: 1 arg, fixed CLAS-address array).
    {
        const size_t blasArgsBytes = sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS);
        auto adesc = CD3DX12_RESOURCE_DESC::Buffer(alignTo(blasArgsBytes, 256));
        ThrowIfFailed(device->CreateCommittedResource(
            &uploadHeap, D3D12_HEAP_FLAG_NONE, &adesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&obj.blasArgsBuffer)));
        obj.blasArgsBuffer->SetName(L"Animated: BLAS args");
        ThrowIfFailed(obj.blasArgsBuffer->Map(0, &noRead,
            reinterpret_cast<void**>(&obj.blasArgsMapped)));
        obj.blasArgsMapped->ClasAddressCount  = obj.clusterCount;
        obj.blasArgsMapped->ClasAddressStride = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
        obj.blasArgsMapped->ClasAddressArray  = obj.perFrameClasAddressArray->GetGPUVirtualAddress();
    }
    {
        // BLAS dest-addrs array (1 entry = our fixed BLAS storage GVA).
        AllocateUploadBuffer(device, &obj.blasGPUVA, sizeof(obj.blasGPUVA),
            &obj.blasResultAddrBuffer, L"Animated: BLAS dest addrs");
    }

    m_animatedObjectEnabled = true;
    SampleLog::LogF(L"[animated] setup complete: BLAS at GVA 0x%llx, %u clusters per frame\n",
                    (unsigned long long)obj.blasGPUVA, obj.clusterCount);
}

// =====================================================================================
// Per-frame work for the animated object. Called from DoRender BEFORE the TLAS
// rebuild + DispatchRays. Total work: 1 memcpy (positions) + 2 batched
// ExecuteIndirectRTASOperations calls + 2 UAV barriers.
// =====================================================================================
void D3D12RaytracingClusteredGeometry::UpdateAnimatedObjectPerFrame()
{
    if (!m_animatedObjectEnabled) return;
    auto& obj = m_animatedObject;

    // ------------------------------------------------------------------
    // 1) Compute new positions. Pulsate the sphere (uniform radial scale)
    //    plus a small per-vertex high-frequency wobble so you can see the
    //    individual clusters shimmering.
    // ------------------------------------------------------------------
    const float t          = (float)m_animSeconds;
    const float pulse      = 1.0f + 0.10f * std::sin(t * 2.0f);              // 0.90..1.10
    const float wobbleAmp  = 0.04f;
    const float wobbleFreq = 5.0f;

    XMFLOAT3* dst = obj.perFrameVertexBufferMapped;
    for (UINT c = 0; c < obj.clusterCount; ++c)
    {
        const auto& src = obj.mesh.clusters[c];
        for (size_t v = 0; v < src.positions.size(); ++v)
        {
            const auto& p = src.positions[v];
            // Per-vertex wobble phase from position, so adjacent vertices wobble
            // together (no per-vertex shear that would crack cluster edges).
            float ph = wobbleFreq * (p.x + p.y + p.z) + t * 3.0f;
            float scale = pulse * (1.0f + wobbleAmp * std::sin(ph));
            // Sphere centred at origin -> radial scaling is just multiplicative.
            dst->x = p.x * scale;
            dst->y = p.y * scale;
            dst->z = p.z * scale;
            ++dst;
        }
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
    instDesc.Flags              = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;
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
    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDescInst,
        D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

    D3D12_RESOURCE_BARRIER instBarriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(obj.perFrameClasResultBuffer.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(obj.perFrameClasAddressArray.Get()),
    };
    m_dxrCommandList->ResourceBarrier(_countof(instBarriers), instBarriers);

    // ------------------------------------------------------------------
    // 3) BUILD_BLAS_FROM_CLAS - 1 arg, EXPLICIT_DESTINATIONS so the BLAS
    //    storage VA stays fixed across frames (the BLAS contents get
    //    overwritten in place; the TLAS instance that points at this VA
    //    just needs a rebuild/refit to pick up the new root bounds).
    // ------------------------------------------------------------------
    D3D12_RTAS_CLAS_INPUTS_DESC blasDesc = {};
    blasDesc.Flags              = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE;
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
    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDescBlas,
        D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

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
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances(N_total);
    for (UINT i = 0; i < N_static; ++i)
    {
        const auto& obj = m_objects[i];
        XMMATRIX m = XMMatrixScaling(obj.worldScale, obj.worldScale, obj.worldScale)
                   * XMMatrixTranslation(obj.worldPos.x, obj.worldPos.y, obj.worldPos.z);
        XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(instances[i].Transform), m);
        instances[i].InstanceID                          = obj.instanceID;
        instances[i].InstanceMask                        = 0xFF;
        instances[i].InstanceContributionToHitGroupIndex = 0;
        instances[i].Flags                               = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
        instances[i].AccelerationStructure               = obj.blasGPUVA;
    }
    if (m_animatedObjectEnabled)
    {
        const auto& a = m_animatedObject;
        XMMATRIX m = XMMatrixScaling(a.worldScale, a.worldScale, a.worldScale)
                   * XMMatrixTranslation(a.worldPos.x, a.worldPos.y, a.worldPos.z);
        XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(instances[N_static].Transform), m);
        instances[N_static].InstanceID                          = a.instanceID;
        instances[N_static].InstanceMask                        = 0xFF;
        instances[N_static].InstanceContributionToHitGroupIndex = 0;
        instances[N_static].Flags                               = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
        instances[N_static].AccelerationStructure               = a.blasGPUVA;
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
}

// ---------------------------------------------------------------------------------
// Update the window title with live stats. Throttled to roughly once per second.
// ---------------------------------------------------------------------------------
void D3D12RaytracingClusteredGeometry::UpdateTitleBar()
{
    if (++m_titleUpdateCounter < 30) return;
    m_titleUpdateCounter = 0;

    wchar_t buf[512];
    if (m_animatedObjectEnabled && m_pfFramesCaptured >= kPerFrameRingSlots)
    {
        // Per-frame totals broken out: animated AS rebuild (INSTANTIATE_CLUSTER_TEMPLATES
        // + BUILD_BLAS_FROM_CLAS) + TLAS rebuild.
        swprintf_s(buf,
            L"DXR2 Clusters | %u obj (%u+1) %u CLAS | init build %.2f ms "
            L"(CLAS %.3f, BLAS %.3f, TLAS %.3f) | per-frame AS %.3f ms "
            L"(anim %.3f + TLAS %.3f) | %u FPS",
            (UINT)m_objects.size() + 1, (UINT)m_objects.size(),
            m_totalClusterCount,
            m_totalBuildMs, m_clasBuildMs, m_blasBuildMs, m_tlasBuildMs,
            m_pfAnimRebuildMs + m_pfTlasRebuildMs,
            m_pfAnimRebuildMs, m_pfTlasRebuildMs,
            (unsigned)m_timer.GetFramesPerSecond());
    }
    else
    {
        swprintf_s(buf,
            L"DXR2 Clusters | %u obj | %u CLAS / %.1f KB | BLAS / %.1f KB "
            L"| build %.2f ms (CLAS %.2f, BLAS %.2f, TLAS %.2f) | %u FPS",
            (UINT)m_objects.size(),
            m_totalClusterCount, m_totalClasBytes / 1024.0,
            m_totalBlasBytes / 1024.0,
            m_totalBuildMs, m_clasBuildMs, m_blasBuildMs, m_tlasBuildMs,
            (unsigned)m_timer.GetFramesPerSecond());
    }
    SetCustomWindowText(buf);
}

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
    lib->DefineExport(c_closestHitName);
    lib->DefineExport(c_missName);

    auto hitGroup = pipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    hitGroup->SetClosestHitShaderImport(c_closestHitName);
    hitGroup->SetHitGroupExport(c_hitGroupName);
    hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

    auto shaderConfig = pipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
    shaderConfig->Config(/*payload*/ 4 * sizeof(float), /*attribs*/ 2 * sizeof(float));

    auto globalRS = pipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
    globalRS->SetRootSignature(m_globalRootSignature.Get());

    // ALLOW_CLUSTERED_GEOMETRY is the DXR2 opt-in for the shader to traverse a
    // BLAS built from CLAS. Without it, hits on a Cluster BLAS are undefined.
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
    void* rgID  = props->GetShaderIdentifier(c_raygenName);
    void* missID= props->GetShaderIdentifier(c_missName);
    void* hgID  = props->GetShaderIdentifier(c_hitGroupName);
    const UINT idSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    const UINT recordSize = Align(idSize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
    auto makeTable = [&](void* shaderID, ComPtr<ID3D12Resource>& outTable, const wchar_t* name)
    {
        std::vector<uint8_t> data(recordSize, 0);
        memcpy(data.data(), shaderID, idSize);
        AllocateUploadBuffer(device, data.data(), data.size(), &outTable, name);
    };
    makeTable(rgID,   m_rayGenShaderTable,    L"raygen shader table");
    makeTable(missID, m_missShaderTable,      L"miss shader table");
    makeTable(hgID,   m_hitGroupShaderTable,  L"hit-group shader table");
}

void D3D12RaytracingClusteredGeometry::CreateDescriptorHeapAndRaytracingOutput()
{
    auto device = m_deviceResources->GetD3DDevice();

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = 4;
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

    // Slowly orbit around the scene center. Period = 30s for a full revolution.
    const double t = m_animSeconds;
    const float  angle  = float(t * (2.0 * M_PI / 30.0));
    const float  radius = 6.0f;
    const float  height = 0.8f;
    XMVECTOR eye = XMVectorSet(radius * std::sin(angle), height,
                               -radius * std::cos(angle), 1.0f);
    XMVECTOR at  = XMVectorSet(0.5f, 0.2f, 0.0f, 1.0f);   // scene centre
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
    cb.miscParams.z = 0;
    cb.miscParams.w = 0;
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
                const double animMs = (ts[1] >= ts[0]) ? (double)(ts[1] - ts[0]) * 1000.0 / freq : 0.0;
                const double tlasMs = (ts[3] >= ts[2]) ? (double)(ts[3] - ts[2]) * 1000.0 / freq : 0.0;
                // EMA, alpha=0.1 for a sub-second smoothing window.
                constexpr double a = 0.1;
                m_pfAnimRebuildMs = m_pfAnimRebuildMs * (1.0 - a) + animMs * a;
                m_pfTlasRebuildMs = m_pfTlasRebuildMs * (1.0 - a) + tlasMs * a;
            }
        }

        // Record this frame's per-frame work, straddled with timestamps.
        const UINT base = m_pfWriteSlot * kPerFrameTsPerSlot;
        cl4->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, base + 0);
        UpdateAnimatedObjectPerFrame();   // INSTANTIATE + BLAS rebuild + barriers
        cl4->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, base + 1);

        cl4->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, base + 2);
        RebuildTlasPerFrame();
        cl4->EndQuery(m_pfQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, base + 3);

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
    cl4->SetPipelineState1(m_dxrStateObject.Get());

    auto bbDesc = m_deviceResources->GetRenderTarget()->GetDesc();
    D3D12_DISPATCH_RAYS_DESC drd = {};
    drd.RayGenerationShaderRecord.StartAddress = m_rayGenShaderTable->GetGPUVirtualAddress();
    drd.RayGenerationShaderRecord.SizeInBytes  = m_rayGenShaderTable->GetDesc().Width;
    drd.MissShaderTable.StartAddress           = m_missShaderTable->GetGPUVirtualAddress();
    drd.MissShaderTable.SizeInBytes            = m_missShaderTable->GetDesc().Width;
    drd.MissShaderTable.StrideInBytes          = m_missShaderTable->GetDesc().Width;
    drd.HitGroupTable.StartAddress             = m_hitGroupShaderTable->GetGPUVirtualAddress();
    drd.HitGroupTable.SizeInBytes              = m_hitGroupShaderTable->GetDesc().Width;
    drd.HitGroupTable.StrideInBytes            = m_hitGroupShaderTable->GetDesc().Width;
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
    m_deviceResources->Present();
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
}

void D3D12RaytracingClusteredGeometry::OnDestroy()
{
    SampleLog::Write(L"OnDestroy: starting shutdown\n");
    if (m_animatedObjectEnabled && m_pfFramesCaptured >= kPerFrameRingSlots)
    {
        SampleLog::LogF(L"[per-frame wall-clock, EMA] animated AS rebuild=%.4f ms "
                        L"(INSTANTIATE + BLAS), TLAS rebuild=%.4f ms, total=%.4f ms\n",
                        m_pfAnimRebuildMs, m_pfTlasRebuildMs,
                        m_pfAnimRebuildMs + m_pfTlasRebuildMs);
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

    // Animated object: unmap its persistently-mapped upload buffers before
    // releasing them (same hang-avoidance reason as m_sceneCB above).
    {
        auto& a = m_animatedObject;
        if (a.perFrameVertexBuffer  && a.perFrameVertexBufferMapped)
            { a.perFrameVertexBuffer->Unmap(0, nullptr);  a.perFrameVertexBufferMapped  = nullptr; }
        if (a.perFrameInstArgsBuffer && a.perFrameInstArgsMapped)
            { a.perFrameInstArgsBuffer->Unmap(0, nullptr); a.perFrameInstArgsMapped = nullptr; }
        if (a.blasArgsBuffer && a.blasArgsMapped)
            { a.blasArgsBuffer->Unmap(0, nullptr); a.blasArgsMapped = nullptr; }
        a.templateInputBuffer.Reset();
        a.templateResultBuffer.Reset();
        a.templateScratchBuffer.Reset();
        a.templateAddressArray.Reset();
        a.perFrameVertexBuffer.Reset();
        a.perFrameInstArgsBuffer.Reset();
        a.perFrameClasResultBuffer.Reset();
        a.perFrameClasScratchBuffer.Reset();
        a.perFrameClasAddressArray.Reset();
        a.blasStorage.Reset();
        a.blasScratchBuffer.Reset();
        a.blasArgsBuffer.Reset();
        a.blasResultAddrBuffer.Reset();
        m_animatedObjectEnabled = false;
    }

    // Drop our refs to D3D12 objects in dependency order so destruction is
    // deterministic. (Without this the order is the member-decl order, which
    // would release the device first - DXGI doesn't love that.)
    m_objects.clear();
    m_clusterInputBuffer.Reset();
    m_clasResultBuffer.Reset();
    m_clasScratchBuffer.Reset();
    m_clasAddressArray.Reset();
    m_clasSizeArray.Reset();
    m_blasScratchBuffer.Reset();
    m_blasArgsBuffer.Reset();
    m_blasResultAddrBuffer.Reset();
    m_tlasBuffer.Reset();
    m_tlasScratchBuffer.Reset();
    m_tlasInstanceDescs.Reset();
    m_dxrStateObject.Reset();
    m_globalRootSignature.Reset();
    m_rayGenShaderTable.Reset();
    m_missShaderTable.Reset();
    m_hitGroupShaderTable.Reset();
    m_descriptorHeap.Reset();
    m_raytracingOutput.Reset();
    m_sceneCB.Reset();
    m_dxr2CommandList.Reset();
    m_dxr2Device.Reset();
    m_dxrCommandList.Reset();
    m_dxrDevice.Reset();

    SampleLog::Write(L"OnDestroy: done\n");
}

void D3D12RaytracingClusteredGeometry::OnKeyDown(UINT8 key)
{
    if (key == 'P' || key == 'p')
    {
        m_animPaused = !m_animPaused;
        SampleLog::LogF(L"[input] animation %s\n", m_animPaused ? L"PAUSED" : L"resumed");
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

