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
    // Enable experimental features BEFORE creating any device.
    UUID experimentalFeatures[] = { D3D12ExperimentalShaderModels, D3D12RaytracingExperiment };
    ThrowIfFailed(D3D12EnableExperimentalFeatures(_countof(experimentalFeatures),
                                                   experimentalFeatures, nullptr, nullptr),
        L"D3D12EnableExperimentalFeatures failed.  Is Windows Developer Mode enabled?\n");

    m_deviceResources = std::make_unique<DeviceResources>(
        DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN,
        FrameCount, D3D_FEATURE_LEVEL_11_0, /*options*/0, m_adapterIDoverride);
    m_deviceResources->RegisterDeviceNotify(this);
    m_deviceResources->SetWindow(Win32Application::GetHwnd(), m_width, m_height);
    m_deviceResources->InitializeDXGIAdapter();
    m_deviceResources->CreateDeviceResources();
    m_deviceResources->CreateWindowSizeDependentResources();
    CreateDeviceDependentResources();
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
    if (!m_clustersAndPtlasSupported)
    {
        SampleLog::Write(L"DXR2 clusters not supported on this adapter; sample requires the experimental D3D12Core.\n");
        SetCustomWindowText(L"clusters not supported on this adapter");
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

    // Pick one shared compressed1 exponent across all clusters: same-grid
    // quantization keeps shared-edge vertices bit-identical across cluster
    // boundaries (watertight at the quantization level).
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
    }

    for (auto& obj : m_objects)
    {
        obj.encoded.clear();
        obj.encoded.reserve(obj.mesh.clusters.size());
        for (const auto& c : obj.mesh.clusters)
            obj.encoded.push_back(Compressed1::Encode(c.positions, (int)bitsPerComp, sharedExponent));
    }
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

void D3D12RaytracingClusteredGeometry::UploadClusterInputs()
{
    auto device = m_deviceResources->GetD3DDevice();
    const bool   useFloat = (m_vertexMode == VertexMode::Float32_3);
    // 16-byte alignment is enough for FLOAT32_3.  COMPRESSED1 wants more
    // generous padding (256 B) -- some drivers prefetch past the end of a
    // cluster's compressed blob, and concatenating clusters back-to-back
    // can let that prefetch step into the next cluster's bits.
    const size_t kAlign   = useFloat ? 16 : 256;
    auto alignTo = [](size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); };

    // Pass 1: lay out per-cluster (VB, IB) byte slots in a single buffer.
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
        slots[gIdx].vbSize   = useFloat
            ? src.positions.size() * sizeof(ProceduralGeometry::float3)
            : obj.encoded[i].TotalBytes();
        cursor += slots[gIdx].vbSize;
        cursor = alignTo(cursor, kAlign);
        slots[gIdx].ibOffset = cursor;
        slots[gIdx].ibSize   = src.indices.size() * sizeof(uint16_t);
        cursor += slots[gIdx].ibSize;
    }

    // Pass 2: create one UPLOAD-heap buffer for all VB+IB data and memcpy in.
    auto uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto bufDesc    = CD3DX12_RESOURCE_DESC::Buffer(alignTo(cursor, 256));
    ThrowIfFailed(device->CreateCommittedResource(
        &uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_clusterInputBuffer)));
    m_clusterInputBuffer->SetName(L"Cluster vertex+index buffer");

    uint8_t* mapped = nullptr;
    CD3DX12_RANGE noRead(0, 0);
    ThrowIfFailed(m_clusterInputBuffer->Map(0, &noRead, reinterpret_cast<void**>(&mapped)));
    const D3D12_GPU_VIRTUAL_ADDRESS baseGPUVA = m_clusterInputBuffer->GetGPUVirtualAddress();
    gIdx = 0;
    for (const auto& obj : m_objects)
    for (size_t i = 0; i < obj.mesh.clusters.size(); ++i, ++gIdx)
    {
        const auto& src = obj.mesh.clusters[i];
        if (useFloat)
        {
            memcpy(mapped + slots[gIdx].vbOffset,
                   src.positions.data(),
                   src.positions.size() * sizeof(ProceduralGeometry::float3));
        }
        else
        {
            const auto& enc = obj.encoded[i];
            memcpy(mapped + slots[gIdx].vbOffset, &enc.header, sizeof(enc.header));
            if (!enc.bitstream.empty())
                memcpy(mapped + slots[gIdx].vbOffset + sizeof(enc.header),
                       enc.bitstream.data(), enc.bitstream.size());
        }
        // Source indices are uint8 in the mesh; widen to uint16 (CLAS uses
        // D3D12_INDEX_FORMAT_UINT16 unconditionally to match the conformance
        // test's BUILD_CLAS_FROM_TRIANGLES setup).
        uint16_t* dst16 = reinterpret_cast<uint16_t*>(mapped + slots[gIdx].ibOffset);
        for (size_t k = 0; k < src.indices.size(); ++k)
            dst16[k] = static_cast<uint16_t>(src.indices[k]);
    }
    m_clusterInputBuffer->Unmap(0, nullptr);

    // Pass 3: build the per-cluster BUILD_CLAS_FROM_TRIANGLES_ARGS array
    // (CPU-filled, uploaded as a single buffer; consumed by
    // ExecuteIndirectRTASOperations in BuildClasImplicit).
    std::vector<D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS> args(m_totalClusterCount);
    gIdx = 0;
    for (const auto& obj : m_objects)
    for (size_t i = 0; i < obj.mesh.clusters.size(); ++i, ++gIdx)
    {
        const auto& src = obj.mesh.clusters[i];
        auto& a = args[gIdx];
        a = {};
        a.ClusterID                 = src.clusterID;
        a.TriangleCount             = (UINT16)(src.indices.size() / 3);
        a.VertexCount               = (UINT16)src.positions.size();
        a.BaseGeometryIndexAndFlags = (UINT)D3D12_RTAS_CLUSTERED_GEOMETRY_FLAG_OPAQUE;
        a.VertexBufferStride        = useFloat ? (UINT16)sizeof(ProceduralGeometry::float3) : 0;
        a.IndexBufferStride         = sizeof(uint16_t);
        a.PositionTruncateBitCount  = useFloat ? (UINT16)m_positionTruncateBits : 0;
        a.VertexBuffer              = baseGPUVA + slots[gIdx].vbOffset;
        a.IndexBuffer               = baseGPUVA + slots[gIdx].ibOffset;
    }
    AllocateUploadBuffer(device, args.data(),
                         args.size() * sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS),
                         &m_clasArgsBuffer, L"CLAS-from-triangles args");

    m_clasArgsArrayGPUVA = m_clasArgsBuffer->GetGPUVirtualAddress();
    m_clasArgsStride     = (UINT)sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS);
}

// BuildClasIndirect: thin wrapper -- only Implicit alloc mode supported
// in this minimal repro (Compact / GetSizes paths gutted along with
// the alloc-mode CLI flag).
void D3D12RaytracingClusteredGeometry::BuildClasIndirect() { BuildClasImplicit(); }

// Single IMPLICIT_DESTINATIONS BUILD_CLAS_FROM_TRIANGLES across all
// clusters in m_objects.  Worst-case result-buffer allocation from the
// prebuild query; per-cluster CLAS GVAs land in m_clasAddressArray and
// feed directly into the BUILD_BLAS_FROM_CLAS step that follows.
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

    D3D12_RTAS_CLUSTER_LIMITS limits = {};
    limits.MaxArgCount                                   = N;
    limits.MaxUniqueGeometryIndexAndFlagsCountPerCluster = 256;
    limits.MaxTriangleCountPerCluster                    = maxTris;
    limits.MaxVertexCountPerCluster                      = maxVerts;
    limits.MaxTotalTriangleCount                         = totalTris;
    limits.MaxTotalVertexCount                           = totalVerts;

    D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC clasDesc = {};
    clasDesc.ClusterLimits                    = limits;
    clasDesc.Flags                            = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE | D3D12_RTAS_OPERATION_FLAG_ALLOW_DATA_ACCESS;
    clasDesc.VertexFormat                     = useFloat ? D3D12_VERTEX_FORMAT_FLOAT32_3 : D3D12_VERTEX_FORMAT_COMPRESSED1;
    clasDesc.IndexFormat                      = D3D12_INDEX_FORMAT_UINT16;
    clasDesc.GeometryIndexAndFlagsIndexFormat = D3D12_INDEX_FORMAT_NONE;
    clasDesc.OpacityMicromapIndexFormat       = D3D12_INDEX_FORMAT_NONE;
    if (useFloat)
        clasDesc.MinPositionTruncateBitCount = m_positionTruncateBits;
    else
        clasDesc.MaxCompressedClusterPositionsSize = maxCompressedSize;
    clasDesc.Mode = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;

    D3D12_RTAS_OPERATION_INPUTS opInputs = {};
    opInputs.Type                  = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
    opInputs.pClusterTrianglesDesc = &clasDesc;

    D3D12_RTAS_OPERATION_PREBUILD_INFO prebuild = {};
    m_dxr2Device->GetRTASOperationPrebuildInfo(&opInputs, &prebuild);

    AllocateUAVBuffer(device, prebuild.ResultDataMaxSizeInBytes,
                      &m_clasResultBuffer, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                      L"CLAS result");
    AllocateUAVBuffer(device, std::max<UINT64>(prebuild.ScratchDataSizeInBytes, 256ull),
                      &m_clasScratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS scratch");
    AllocateUAVBuffer(device, (UINT64)N * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                      &m_clasAddressArray, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      L"CLAS address array");

    D3D12_RTAS_BATCHED_OPERATION_DATA batched = {};
    batched.AddressResolutionFlags = D3D12_RTAS_OPERATION_ADDRESS_RESOLUTION_FLAG_NONE;
    batched.BatchResultData        = m_clasResultBuffer->GetGPUVirtualAddress();
    batched.BatchScratchData       = m_clasScratchBuffer->GetGPUVirtualAddress();
    batched.ResultAddressArray     = { m_clasAddressArray->GetGPUVirtualAddress(), sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    batched.IndirectArgumentArray  = { m_clasArgsArrayGPUVA, m_clasArgsStride };

    D3D12_RTAS_OPERATION_DESC opDesc = {};
    opDesc.Inputs                = opInputs;
    opDesc.pBatchedOperationData = &batched;

    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDesc, D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    D3D12_RESOURCE_BARRIER barriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasResultBuffer.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_clasAddressArray.Get()),
    };
    m_dxrCommandList->ResourceBarrier(_countof(barriers), barriers);
}

void D3D12RaytracingClusteredGeometry::BuildBlasFromClasIndirect()
{
    auto device = m_deviceResources->GetD3DDevice();
    const UINT N_obj = (UINT)m_objects.size();

    UINT maxClasPerArg = 0, totalClas = 0;
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

    // One BLAS storage resource per object so each one ends up at its own GVA
    // (which we bake straight into the TLAS instance descs).  EXPLICIT mode.
    for (auto& obj : m_objects)
    {
        AllocateUAVBuffer(device, prebuild.ResultDataMaxSizeInBytes, &obj.blasStorage,
                          D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, L"Cluster BLAS");
        obj.blasGPUVA = obj.blasStorage->GetGPUVirtualAddress();
    }
    AllocateUAVBuffer(device, std::max<UINT64>(prebuild.ScratchDataSizeInBytes, 256ull),
                      &m_blasScratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"BLAS scratch");

    // CPU-fill BUILD_BLAS_FROM_CLAS_ARGS[N_obj] (each entry points at this
    // object's slice of the global CLAS-address array).
    const D3D12_GPU_VIRTUAL_ADDRESS clasArrayGPUVA = m_clasAddressArray->GetGPUVirtualAddress();
    std::vector<D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS> args(N_obj);
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> destAddrs(N_obj);
    for (UINT i = 0; i < N_obj; ++i)
    {
        args[i].ClasAddressCount  = m_objects[i].clusterCount;
        args[i].ClasAddressStride = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
        args[i].ClasAddressArray  = clasArrayGPUVA
            + (UINT64)m_objects[i].globalClusterStart * sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
        destAddrs[i] = m_objects[i].blasGPUVA;
    }
    AllocateUploadBuffer(device, args.data(),
                         args.size() * sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS),
                         &m_blasArgsBuffer, L"BLAS-from-CLAS args");
    AllocateUploadBuffer(device, destAddrs.data(), destAddrs.size() * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                         &m_blasResultAddrBuffer, L"BLAS dest addrs");

    D3D12_RTAS_BATCHED_OPERATION_DATA batched = {};
    batched.BatchScratchData      = m_blasScratchBuffer->GetGPUVirtualAddress();
    batched.ResultAddressArray    = { m_blasResultAddrBuffer->GetGPUVirtualAddress(), sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    batched.IndirectArgumentArray = { m_blasArgsBuffer->GetGPUVirtualAddress(), sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS) };

    D3D12_RTAS_OPERATION_DESC opDesc = {};
    opDesc.Inputs                = opInputs;
    opDesc.pBatchedOperationData = &batched;
    m_dxr2CommandList->ExecuteIndirectRTASOperations(1, &opDesc, D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

    std::vector<D3D12_RESOURCE_BARRIER> uavBarriers;
    uavBarriers.reserve(N_obj);
    for (auto& obj : m_objects)
        uavBarriers.push_back(CD3DX12_RESOURCE_BARRIER::UAV(obj.blasStorage.Get()));
    m_dxrCommandList->ResourceBarrier((UINT)uavBarriers.size(), uavBarriers.data());
}

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

void D3D12RaytracingClusteredGeometry::CreateRaytracingPipelineAndShaderTables()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Global root signature: just the 3 slots the minimal shader binds
    // (Output UAV, AS SRV, Scene CBV).
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
    }

    // Empty local root signature (shaders don't take per-shader records, but
    // the runtime requires one to exist alongside the global RS for any
    // raytracing state object).
    {
        CD3DX12_ROOT_SIGNATURE_DESC localDesc(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);
        ComPtr<ID3DBlob> blob, err;
        ThrowIfFailed(D3D12SerializeRootSignature(&localDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err),
            err ? (wchar_t*)err->GetBufferPointer() : L"D3D12SerializeRootSignature(local) failed\n");
        ThrowIfFailed(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
            IID_PPV_ARGS(&m_localRootSignature)));
    }

    // State object: 1 raygen + 1 miss + 1 opaque hit group, ALLOW_CLUSTERED_GEOMETRY.
    CD3DX12_STATE_OBJECT_DESC pipeline{ D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE };
    pipeline.CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>()->SetRootSignature(m_localRootSignature.Get());

    auto lib = pipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE((void*)g_pRaytracing, ARRAYSIZE(g_pRaytracing));
    lib->SetDXILLibrary(&libdxil);
    lib->DefineExport(c_raygenName);
    lib->DefineExport(c_opaqueClosestHitName);
    lib->DefineExport(c_missName);

    auto opaqueHG = pipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    opaqueHG->SetClosestHitShaderImport(c_opaqueClosestHitName);
    opaqueHG->SetHitGroupExport(c_opaqueHitGroupName);
    opaqueHG->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

    auto shaderConfig = pipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
    shaderConfig->Config(/*payload*/ 4 * sizeof(float) + 2 * sizeof(uint),
                         /*attribs*/ 2 * sizeof(float));
    pipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>()->SetRootSignature(m_globalRootSignature.Get());

    auto pipelineConfig = pipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT>();
    pipelineConfig->Config(/*maxRecursion*/1, D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_CLUSTERED_GEOMETRY);

    ThrowIfFailed(m_dxrDevice->CreateStateObject(pipeline, IID_PPV_ARGS(&m_dxrStateObject)),
        L"CreateStateObject failed\n");

    // Shader tables: one record each (raygen / miss / hit).  Record size =
    // shader identifier size, aligned to D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT.
    ComPtr<ID3D12StateObjectProperties> props;
    ThrowIfFailed(m_dxrStateObject->QueryInterface(IID_PPV_ARGS(&props)));
    const UINT idSize     = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    const UINT recordSize = Align(idSize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
    auto makeTable1 = [&](void* shaderID, ComPtr<ID3D12Resource>& outTable, const wchar_t* name)
    {
        std::vector<uint8_t> data(recordSize, 0);
        memcpy(data.data(), shaderID, idSize);
        AllocateUploadBuffer(device, data.data(), data.size(), &outTable, name);
    };
    makeTable1(props->GetShaderIdentifier(c_raygenName),         m_rayGenShaderTable,   L"raygen ST");
    makeTable1(props->GetShaderIdentifier(c_missName),           m_missShaderTable,     L"miss ST");
    makeTable1(props->GetShaderIdentifier(c_opaqueHitGroupName), m_hitGroupShaderTable, L"hit-group ST");
}

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

    // Simple orbit around the scene origin: yaw 2*PI / 30 s, fixed radius,
    // fixed eye height.  All shaders need is the camera basis + aspect ratio
    // + tan(half-FOV).
    const float angle  = float(m_animSeconds * (2.0 * M_PI / 30.0));
    const float radius = 6.5f;
    const float height = 1.5f;
    XMVECTOR eye = XMVectorSet(radius * std::sin(angle), height,
                              -radius * std::cos(angle), 1.0f);
    XMVECTOR at  = XMVectorSet(0.5f, 0.0f, 0.0f, 1.0f);
    XMVECTOR up  = XMVectorSet(0, 1, 0, 0);
    XMMATRIX view = XMMatrixLookAtLH(eye, at, up);

    SceneConstantBuffer cb = {};
    cb.viewToWorld     = XMMatrixInverse(nullptr, view);
    XMStoreFloat4(&cb.cameraPosition, eye);
    cb.miscParams.x    = (float)m_width / (float)m_height;
    cb.miscParams.y    = std::tan(60.0f * (XM_PI / 180.0f) * 0.5f);   // 60 degree vertical FOV
    memcpy(m_sceneCBMapped, &cb, sizeof(cb));
}

// =====================================================================================
// Per-frame
// =====================================================================================
void D3D12RaytracingClusteredGeometry::OnUpdate()
{
    m_timer.Tick();
    m_animSeconds += m_timer.GetElapsedSeconds();
    UpdateSceneConstantBuffer();
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

    // Bind RT pipeline + global root sig + descriptor heap.
    cl4->SetComputeRootSignature(m_globalRootSignature.Get());
    ID3D12DescriptorHeap* heaps[] = { m_descriptorHeap.Get() };
    cl4->SetDescriptorHeaps(_countof(heaps), heaps);
    cl4->SetComputeRootDescriptorTable(GlobalRootSig::OutputUAVSlot, m_raytracingOutputUAV);
    cl4->SetComputeRootShaderResourceView(GlobalRootSig::AccelerationStructureSlot,
        m_tlasBuffer->GetGPUVirtualAddress());
    cl4->SetComputeRootConstantBufferView(GlobalRootSig::SceneCBVSlot, m_sceneCB->GetGPUVirtualAddress());
    cl4->SetPipelineState1(m_dxrStateObject.Get());

    // DispatchRays at back-buffer resolution.  Shader tables: 1 record each
    // (raygen / miss / hit), stride = identifier size.
    auto bbDesc = m_deviceResources->GetRenderTarget()->GetDesc();
    D3D12_DISPATCH_RAYS_DESC drd = {};
    drd.RayGenerationShaderRecord.StartAddress = m_rayGenShaderTable->GetGPUVirtualAddress();
    drd.RayGenerationShaderRecord.SizeInBytes  = m_rayGenShaderTable->GetDesc().Width;
    drd.MissShaderTable.StartAddress           = m_missShaderTable->GetGPUVirtualAddress();
    drd.MissShaderTable.SizeInBytes            = m_missShaderTable->GetDesc().Width;
    drd.MissShaderTable.StrideInBytes          = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    drd.HitGroupTable.StartAddress             = m_hitGroupShaderTable->GetGPUVirtualAddress();
    drd.HitGroupTable.SizeInBytes              = m_hitGroupShaderTable->GetDesc().Width;
    drd.HitGroupTable.StrideInBytes            = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    drd.Width = (UINT)bbDesc.Width; drd.Height = (UINT)bbDesc.Height; drd.Depth = 1;
    cl4->DispatchRays(&drd);

    // Copy raytracing output (UAV) -> back buffer (RT).
    D3D12_RESOURCE_BARRIER toCopy[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_raytracingOutput.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(m_deviceResources->GetRenderTarget(),
            D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST),
    };
    cl->ResourceBarrier(_countof(toCopy), toCopy);
    cl->CopyResource(m_deviceResources->GetRenderTarget(), m_raytracingOutput.Get());
    D3D12_RESOURCE_BARRIER toRt[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_raytracingOutput.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        CD3DX12_RESOURCE_BARRIER::Transition(m_deviceResources->GetRenderTarget(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET),
    };
    cl->ResourceBarrier(_countof(toRt), toRt);
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
