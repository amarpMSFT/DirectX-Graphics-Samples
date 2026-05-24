//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// MeshAssets.h
//
// Generic mesh -> VB/IB -> DXR1 BLAS bundle.  Used for both balls
// (icosphere) and donuts (torus) in phase 4.  See BallAssets.h
// (deprecated; will be removed once the sample fully migrates) for the
// historical icosphere-specific version.
//
// Milestone 2a uses DXR1 BLAS by design; the cluster path (CLAS +
// BUILD_BLAS_FROM_CLAS) lands in a later milestone behind the same
// BlasGpuVa() accessor so nothing else in the sample changes.
//
#pragma once

#include "stdafx.h"
#include "GpuBuffer.h"
#include "ProceduralGeometry.h"
#include <unordered_map>

class MeshAssets
{
public:
    // Build the supplied mesh + a DXR1 BLAS over its triangles.  Records
    // into the open `cl`; caller is responsible for closing / executing /
    // waiting.  Upload + scratch buffers are kept as members until the
    // owner releases this object so they outlive the GPU submission.
    void Initialize(ID3D12Device5* device,
                    ID3D12GraphicsCommandList4* cl,
                    ProceduralGeometry::Mesh mesh,
                    const wchar_t* nameHint = L"Mesh")
    {
        m_mesh = std::move(mesh);

        const UINT64 vbBytes = m_mesh.positions.size() * sizeof(DirectX::XMFLOAT3);
        const UINT64 ibBytes = m_mesh.indices.size()   * sizeof(uint32_t);
        std::wstring base = nameHint ? nameHint : L"Mesh";
        m_vbUpload = PtSample::CreateUploadBufferWithData(
            device, m_mesh.positions.data(), vbBytes, (base + L"/VB upload").c_str());
        m_ibUpload = PtSample::CreateUploadBufferWithData(
            device, m_mesh.indices.data(),   ibBytes, (base + L"/IB upload").c_str());
        m_vb = PtSample::CreateDefaultBuffer(device, vbBytes, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COMMON, (base + L"/VB").c_str());
        m_ib = PtSample::CreateDefaultBuffer(device, ibBytes, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COMMON, (base + L"/IB").c_str());
        {
            D3D12_RESOURCE_BARRIER toCopy[2] = {
                CD3DX12_RESOURCE_BARRIER::Transition(m_vb.Get(),
                    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
                CD3DX12_RESOURCE_BARRIER::Transition(m_ib.Get(),
                    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
            };
            cl->ResourceBarrier(_countof(toCopy), toCopy);
        }
        cl->CopyBufferRegion(m_vb.Get(), 0, m_vbUpload.Get(), 0, vbBytes);
        cl->CopyBufferRegion(m_ib.Get(), 0, m_ibUpload.Get(), 0, ibBytes);
        {
            D3D12_RESOURCE_BARRIER bars[2] = {
                CD3DX12_RESOURCE_BARRIER::Transition(m_vb.Get(),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                CD3DX12_RESOURCE_BARRIER::Transition(m_ib.Get(),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            };
            cl->ResourceBarrier(_countof(bars), bars);
        }

        D3D12_RAYTRACING_GEOMETRY_DESC geom = {};
        geom.Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geom.Triangles.IndexBuffer            = m_ib->GetGPUVirtualAddress();
        geom.Triangles.IndexCount             = (UINT)m_mesh.indices.size();
        geom.Triangles.IndexFormat            = DXGI_FORMAT_R32_UINT;
        geom.Triangles.Transform3x4           = 0;
        geom.Triangles.VertexFormat           = DXGI_FORMAT_R32G32B32_FLOAT;
        geom.Triangles.VertexCount            = (UINT)m_mesh.positions.size();
        geom.Triangles.VertexBuffer.StartAddress  = m_vb->GetGPUVirtualAddress();
        geom.Triangles.VertexBuffer.StrideInBytes = sizeof(DirectX::XMFLOAT3);

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};
        blasInputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        blasInputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
        blasInputs.NumDescs       = 1;
        blasInputs.pGeometryDescs = &geom;
        blasInputs.Flags          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pre = {};
        device->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputs, &pre);

        m_blas = PtSample::CreateDefaultBuffer(device, pre.ResultDataMaxSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
            D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            (base + L"/BLAS").c_str());
        m_blasScratch = PtSample::CreateDefaultBuffer(device, pre.ScratchDataSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, (base + L"/BLAS scratch").c_str());

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasBuild = {};
        blasBuild.Inputs                          = blasInputs;
        blasBuild.DestAccelerationStructureData    = m_blas->GetGPUVirtualAddress();
        blasBuild.ScratchAccelerationStructureData = m_blasScratch->GetGPUVirtualAddress();
        cl->BuildRaytracingAccelerationStructure(&blasBuild, 0, nullptr);

        auto uav = CD3DX12_RESOURCE_BARRIER::UAV(m_blas.Get());
        cl->ResourceBarrier(1, &uav);

        m_blasResultBytes  = pre.ResultDataMaxSizeInBytes;
        m_blasScratchBytes = pre.ScratchDataSizeInBytes;

        SampleLog::LogF(L"[mesh-assets] %s verts=%u tris=%u  "
                        L"VB=%llu B  IB=%llu B  BLAS=%llu B (scratch=%llu)\n",
                        base.c_str(),
                        (unsigned)m_mesh.positions.size(),
                        (unsigned)(m_mesh.indices.size() / 3),
                        (unsigned long long)vbBytes,
                        (unsigned long long)ibBytes,
                        (unsigned long long)m_blasResultBytes,
                        (unsigned long long)m_blasScratchBytes);
    }

    D3D12_GPU_VIRTUAL_ADDRESS BlasGpuVa() const { return m_blas->GetGPUVirtualAddress(); }
    UINT64                    BlasResultBytes() const { return m_blasResultBytes; }

    // ---- Optional: per-triangle vertex normals SRV ----
    //
    // BuildPerTriVertexNormalsBuffer() packs THREE OBJECT-SPACE vertex
    // normals per triangle (the .normals of the triangle's three vertices,
    // in IB order) into a tightly-packed buffer.  Closest-hit shaders read
    // a triangle's three vertex normals by PrimitiveIndex() * 36, then
    // interpolate them via the barycentric attributes for smooth shading
    // (the "vertex-normal interpolation" pattern used by the clustered
    // sample at line 550 of its Raytracing.hlsl).  Buffer layout:
    //
    //   bytes [primIdx * 36     .. primIdx * 36 + 12) : vertex 0 normal
    //   bytes [primIdx * 36 + 12.. primIdx * 36 + 24) : vertex 1 normal
    //   bytes [primIdx * 36 + 24.. primIdx * 36 + 36) : vertex 2 normal
    //
    // Records into the same cmd list as Initialize(); the upload staging
    // buffer is kept as a member until the owner releases this object.
    //
    // The "barycentric weight 0 = 1 - x - y" convention used in the shader
    // assumes the same vertex order as IB[primIdx*3 + 0/1/2] which is what
    // we pack here.
    void BuildPerTriVertexNormalsBuffer(ID3D12Device5* device,
                                        ID3D12GraphicsCommandList4* cl,
                                        const wchar_t* nameHint = L"Mesh")
    {
        if (m_mesh.normals.size() != m_mesh.positions.size())
        {
            // Caller forgot to populate per-vertex normals for this mesh.
            // (ProceduralGeometry::MakeIcosphere / MakeTorus both fill them;
            // a foreign mesh might not.)
            return;
        }
        const uint32_t triCount = (uint32_t)(m_mesh.indices.size() / 3);
        std::vector<DirectX::XMFLOAT3> packed(triCount * 3);
        for (uint32_t t = 0; t < triCount; ++t)
        {
            packed[t*3 + 0] = m_mesh.normals[m_mesh.indices[t*3 + 0]];
            packed[t*3 + 1] = m_mesh.normals[m_mesh.indices[t*3 + 1]];
            packed[t*3 + 2] = m_mesh.normals[m_mesh.indices[t*3 + 2]];
        }
        const UINT64 bytes = (UINT64)packed.size() * sizeof(DirectX::XMFLOAT3);
        std::wstring base = nameHint ? nameHint : L"Mesh";
        m_vertNormalsUpload = PtSample::CreateUploadBufferWithData(
            device, packed.data(), bytes, (base + L"/VertN upload").c_str());
        m_vertNormals = PtSample::CreateDefaultBuffer(device, bytes, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COMMON, (base + L"/VertN").c_str());
        auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(m_vertNormals.Get(),
            D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        cl->ResourceBarrier(1, &toCopy);
        cl->CopyBufferRegion(m_vertNormals.Get(), 0, m_vertNormalsUpload.Get(), 0, bytes);
        auto toSrv = CD3DX12_RESOURCE_BARRIER::Transition(m_vertNormals.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        cl->ResourceBarrier(1, &toSrv);
        SampleLog::LogF(L"[mesh-assets] %s per-tri vertex normals: %u entries (%llu B)\n",
                        base.c_str(), triCount * 3, (unsigned long long)bytes);
    }
    D3D12_GPU_VIRTUAL_ADDRESS VertNormalsGpuVa() const
    {
        return m_vertNormals ? m_vertNormals->GetGPUVirtualAddress() : 0;
    }

    // ---- Optional: Cluster BLAS path (CLAS + BUILD_BLAS_FROM_CLAS) ----
    //
    // Splits the mesh into clusters of at most kTrianglesPerCluster
    // triangles each.  Builds one CLAS per cluster via
    // BUILD_CLAS_FROM_TRIANGLES (BATCHED operation, IMPLICIT destinations,
    // result-address-array populated by the driver), then one Cluster BLAS
    // from the list of CLAS GPUVAs via BUILD_BLAS_FROM_CLAS.  The Cluster
    // BLAS is a drop-in replacement for the DXR1 BLAS the Initialize()
    // path builds: same GPUVA semantics in the PTLAS instance
    // AccelerationStructure field.
    //
    // For the sample's tiny meshes (up to 320 tris), kTrianglesPerCluster=64
    // gives at most 5 clusters per mesh; well under the 256-vert limit.
    //
    // Spec: D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES and
    //       D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS.
    void BuildClusterBlas(ID3D12Device5* device,
                          ID3D12DeviceRaytracing2* deviceRT2,
                          ID3D12GraphicsCommandList4* cl,
                          ID3D12CommandListRaytracing2* cl2,
                          const wchar_t* nameHint = L"Mesh")
    {
        constexpr UINT16 kTrianglesPerCluster = 64;
        std::wstring base = nameHint ? nameHint : L"Mesh";

        const UINT triTotal = (UINT)(m_mesh.indices.size() / 3);
        if (triTotal == 0) return;

        // ---- Split triangles into clusters --------------------------------
        struct ClusterRange { UINT firstTri; UINT triCount; UINT vertCount; };
        std::vector<ClusterRange> clusters;
        for (UINT t = 0; t < triTotal; t += kTrianglesPerCluster)
        {
            UINT triCount = std::min<UINT>(kTrianglesPerCluster, triTotal - t);
            // Count unique vertex indices in this cluster's triangle range.
            std::unordered_map<uint32_t, bool> seen;
            for (UINT i = 0; i < triCount * 3; ++i)
                seen.emplace(m_mesh.indices[(t + 0) * 3 + i], true);
            clusters.push_back({ t, triCount, (UINT)seen.size() });
        }
        const UINT clasCount = (UINT)clusters.size();

        // ---- Per-cluster build args (CPU side, uploaded once) -------------
        std::vector<D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS> args(clasCount);
        for (UINT c = 0; c < clasCount; ++c)
        {
            const ClusterRange& cl_ = clusters[c];
            auto& a = args[c];
            a.ClusterID                = c;
            a.ClusterFlags             = D3D12_RTAS_CLUSTER_OPERATION_CLAS_FLAG_NONE;
            a.TriangleCount            = (UINT16)cl_.triCount;
            a.VertexCount              = (UINT16)cl_.vertCount;
            a.BaseGeometryIndexAndFlags = 0;
            a.OpacityMicromapBaseLocation = 0;
            a.VertexBufferStride       = (UINT16)sizeof(DirectX::XMFLOAT3);
            a.IndexBufferStride        = (UINT16)sizeof(uint32_t);
            a.OpacityMicromapIndexBufferStride = 0;
            a.GeometryIndexAndFlagsArrayStride = 0;
            a.PositionTruncateBitCount = 0;
            a.ReservedPadding          = 0;
            a.VertexBuffer             = m_vb->GetGPUVirtualAddress();
            a.IndexBuffer              = m_ib->GetGPUVirtualAddress() + cl_.firstTri * 3 * sizeof(uint32_t);
            a.GeometryIndexAndFlagsArray       = 0;
            a.GeometryIndexAndFlagsIndexBuffer = 0;
            a.OpacityMicromapArray             = 0;
            a.OpacityMicromapIndexBuffer       = 0;
        }
        const UINT64 argsBytes = clasCount * sizeof(args[0]);
        m_clasArgsUpload = PtSample::CreateUploadBufferWithData(
            device, args.data(), argsBytes, (base + L"/CLAS args").c_str());

        // ---- Sizing -------------------------------------------------------
        // CLAS prebuild: maxes for ClusterLimits.
        D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC ctd = {};
        ctd.ClusterLimits.MaxArgCount               = clasCount;
        ctd.ClusterLimits.MaxGeometryIndexValue     = 0;
        ctd.ClusterLimits.MaxUniqueGeometryIndexAndFlagsCountPerCluster = 1;
        ctd.ClusterLimits.MaxTriangleCountPerCluster = kTrianglesPerCluster;
        ctd.ClusterLimits.MaxVertexCountPerCluster  = kTrianglesPerCluster * 3;
        ctd.ClusterLimits.MaxTotalTriangleCount     = triTotal;
        ctd.ClusterLimits.MaxTotalVertexCount       = triTotal * 3;
        ctd.ClusterLimits.MaxOpacityMicromapIndicesPerCluster = 0;
        ctd.Flags                                   = D3D12_RTAS_OPERATION_FLAG_NONE;
        ctd.Mode                                    = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
        ctd.VertexFormat                            = D3D12_VERTEX_FORMAT_FLOAT32_3;
        ctd.IndexFormat                             = D3D12_INDEX_FORMAT_UINT32;
        ctd.GeometryIndexAndFlagsIndexFormat        = D3D12_INDEX_FORMAT_NONE;
        ctd.OpacityMicromapIndexFormat              = D3D12_INDEX_FORMAT_NONE;
        ctd.MinPositionTruncateBitCount             = 0;

        D3D12_RTAS_OPERATION_INPUTS clasOpIn = {};
        clasOpIn.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
        clasOpIn.pClusterTrianglesDesc = &ctd;
        D3D12_RTAS_OPERATION_PREBUILD_INFO clasPre = {};
        deviceRT2->GetRTASOperationPrebuildInfo(&clasOpIn, &clasPre);

        // Cluster BLAS prebuild.
        D3D12_RTAS_CLAS_INPUTS_DESC blasInputs = {};
        blasInputs.Flags              = D3D12_RTAS_OPERATION_FLAG_NONE;
        blasInputs.MaxArgCount        = 1;       // one BLAS-from-CLAS arg
        blasInputs.Mode               = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
        blasInputs.MaxTotalClasCount  = clasCount;
        blasInputs.MaxClasCountPerArg = clasCount;
        D3D12_RTAS_OPERATION_INPUTS blasOpIn = {};
        blasOpIn.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
        blasOpIn.pClasDesc = &blasInputs;
        D3D12_RTAS_OPERATION_PREBUILD_INFO blasPre = {};
        deviceRT2->GetRTASOperationPrebuildInfo(&blasOpIn, &blasPre);

        // ---- Allocate result + scratch buffers ----------------------------
        m_clasResult = PtSample::CreateDefaultBuffer(device, clasPre.ResultDataMaxSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
            D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            (base + L"/CLAS result").c_str());
        m_clasScratch = PtSample::CreateDefaultBuffer(device, clasPre.ScratchDataSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, (base + L"/CLAS scratch").c_str());
        m_clasAddresses = PtSample::CreateDefaultBuffer(device,
            clasCount * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, (base + L"/CLAS addresses").c_str());
        m_clusterBlas = PtSample::CreateDefaultBuffer(device, blasPre.ResultDataMaxSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
            D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            (base + L"/ClusterBLAS").c_str());
        m_clusterBlasScratch = PtSample::CreateDefaultBuffer(device, blasPre.ScratchDataSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, (base + L"/ClusterBLAS scratch").c_str());

        // ---- 1) CLAS build (BATCHED, IMPLICIT destinations) ---------------
        D3D12_RTAS_BATCHED_OPERATION_DATA clasBatch = {};
        clasBatch.AddressResolutionFlags = D3D12_RTAS_OPERATION_ADDRESS_RESOLUTION_FLAG_NONE;
        clasBatch.BatchResultData        = m_clasResult->GetGPUVirtualAddress();
        clasBatch.BatchScratchData       = m_clasScratch->GetGPUVirtualAddress();
        clasBatch.ResultAddressArray.StartAddress  = m_clasAddresses->GetGPUVirtualAddress();
        clasBatch.ResultAddressArray.StrideInBytes = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
        clasBatch.ResultSizeArray = {};
        clasBatch.IndirectArgumentArray.StartAddress  = m_clasArgsUpload->GetGPUVirtualAddress();
        clasBatch.IndirectArgumentArray.StrideInBytes = sizeof(args[0]);
        clasBatch.IndirectArgumentArraySize = 0;     // 0 = use MaxArgCount
        clasBatch.ToolsInfo = 0;
        D3D12_RTAS_OPERATION_DESC clasDesc = {};
        clasDesc.Inputs                = clasOpIn;
        clasDesc.pBatchedOperationData = &clasBatch;
        cl2->ExecuteIndirectRTASOperations(1, &clasDesc,
            D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
        {
            // CLAS results read by the next op; UAV barrier on the result
            // buffer (the acceleration structure) and addresses buffer.
            D3D12_RESOURCE_BARRIER bars[2] = {};
            bars[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            bars[0].UAV.pResource = m_clasResult.Get();
            bars[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            bars[1].UAV.pResource = m_clasAddresses.Get();
            cl->ResourceBarrier(2, bars);
        }

        // ---- 2) BLAS-from-CLAS (BATCHED, IMPLICIT destinations) -----------
        // Build args: one D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS
        // pointing at the CLAS addresses buffer.
        D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS blasArgs = {};
        blasArgs.ClasAddressCount  = clasCount;
        blasArgs.ClasAddressStride = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
        blasArgs.ClasAddressArray  = m_clasAddresses->GetGPUVirtualAddress();
        m_clusterBlasArgsUpload = PtSample::CreateUploadBufferWithData(
            device, &blasArgs, sizeof(blasArgs), (base + L"/ClusterBLAS args").c_str());

        D3D12_RTAS_BATCHED_OPERATION_DATA blasBatch = {};
        blasBatch.AddressResolutionFlags = D3D12_RTAS_OPERATION_ADDRESS_RESOLUTION_FLAG_NONE;
        blasBatch.BatchResultData        = m_clusterBlas->GetGPUVirtualAddress();
        blasBatch.BatchScratchData       = m_clusterBlasScratch->GetGPUVirtualAddress();
        blasBatch.IndirectArgumentArray.StartAddress  = m_clusterBlasArgsUpload->GetGPUVirtualAddress();
        blasBatch.IndirectArgumentArray.StrideInBytes = sizeof(blasArgs);
        blasBatch.IndirectArgumentArraySize = 0;
        blasBatch.ToolsInfo = 0;
        D3D12_RTAS_OPERATION_DESC blasDesc = {};
        blasDesc.Inputs                = blasOpIn;
        blasDesc.pBatchedOperationData = &blasBatch;
        cl2->ExecuteIndirectRTASOperations(1, &blasDesc,
            D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
        {
            D3D12_RESOURCE_BARRIER uav = {};
            uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource = m_clusterBlas.Get();
            cl->ResourceBarrier(1, &uav);
        }

        m_clasCount = clasCount;
        SampleLog::LogF(L"[mesh-assets] %s ClusterBLAS: %u CLAS (max %u tris each), "
                        L"result=%llu B  CLAS_result=%llu B  scratch=%llu+%llu B\n",
                        base.c_str(), clasCount, (UINT)kTrianglesPerCluster,
                        (unsigned long long)blasPre.ResultDataMaxSizeInBytes,
                        (unsigned long long)clasPre.ResultDataMaxSizeInBytes,
                        (unsigned long long)clasPre.ScratchDataSizeInBytes,
                        (unsigned long long)blasPre.ScratchDataSizeInBytes);
    }
    D3D12_GPU_VIRTUAL_ADDRESS ClusterBlasGpuVa() const
    {
        return m_clusterBlas ? m_clusterBlas->GetGPUVirtualAddress() : 0;
    }
    UINT ClasCount() const { return m_clasCount; }

private:
    ProceduralGeometry::Mesh m_mesh;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vb;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_ib;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vbUpload;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_ibUpload;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blas;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vertNormals;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vertNormalsUpload;
    // Cluster BLAS path (CLAS + BUILD_BLAS_FROM_CLAS).  Allocated +
    // populated by BuildClusterBlas() if the sample uses the cluster path.
    Microsoft::WRL::ComPtr<ID3D12Resource> m_clasArgsUpload;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_clasResult;          // CLAS data (BatchResultData)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_clasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_clasAddresses;       // GPUVAs of each built CLAS (driver-populated)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_clusterBlas;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_clusterBlasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_clusterBlasArgsUpload;
    UINT m_clasCount = 0;
    UINT64 m_blasResultBytes  = 0;
    UINT64 m_blasScratchBytes = 0;
};
