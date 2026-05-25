//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// AnimatedDonutMesh.h
//
// Per-frame cluster-BLAS rebuild for an animated (pulsating) torus.  Exercises
// the canonical use case the spec calls out for the DXR2 cluster pipeline:
// vertex positions deform per frame while the underlying topology is constant,
// and the freshly-built BLAS gets swapped into the PTLAS via WRITE_INSTANCE
// (which carries the new AccelerationStructure pointer alongside the rest of
// the instance args).
//
// Implementation:
//   * IB is static (topology of a 28x14 torus, 784 triangles).
//   * VB lives in an UPLOAD heap, written each frame from the CPU with
//     pulsated vertex positions.
//   * Per-frame: BUILD_CLAS_FROM_TRIANGLES (one CLAS per cluster of up to
//     64 triangles, BATCHED + IMPLICIT_DESTINATIONS) followed by
//     BUILD_BLAS_FROM_CLAS (BATCHED + IMPLICIT_DESTINATIONS).  UAV barriers
//     between.  All result + scratch buffers allocated ONCE at init and
//     reused per frame -- safe because we wait fence between frames.
//   * Build args (CPU-filled UPLOAD buffer) point at the animated VB by
//     GPUVA, which is stable across frames; only the VB CONTENTS change.
//
// What's NOT cluster TEMPLATES yet: BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES +
// INSTANTIATE_CLUSTER_TEMPLATES is the optimized path where topology is
// baked into a template once, then per-frame instantiation only needs the
// new vertex data.  This file does the unoptimized per-frame CLAS-from-
// triangles rebuild; switching to templates is a future polish item.
//
#pragma once

#include "stdafx.h"
#include "GpuBuffer.h"
#include "ProceduralGeometry.h"
#include <unordered_map>
#include <vector>

class AnimatedDonutMesh
{
public:
    // Build the static infrastructure (resources, args, initial CLAS+BLAS
    // build) and run one initial Tick so the BLAS GPUVA is valid before
    // the first DispatchRays.
    void Initialize(ID3D12Device5* device,
                    ID3D12DeviceRaytracing2* deviceRT2,
                    ID3D12GraphicsCommandList4* cl,
                    ID3D12CommandListRaytracing2* cl2,
                    float majorR, float minorR, uint32_t segMajor, uint32_t segMinor,
                    const wchar_t* nameHint = L"AnimatedDonut")
    {
        constexpr UINT16 kTrianglesPerCluster = 64;
        std::wstring base = nameHint ? nameHint : L"AnimatedDonut";

        // ---- Topology + initial vertex generation -----------------------
        m_majorR   = majorR;
        m_minorR   = minorR;
        m_segMajor = segMajor;
        m_segMinor = segMinor;
        m_vertCount = segMajor * segMinor;
        m_indices.reserve(segMajor * segMinor * 6);
        for (uint32_t u = 0; u < segMajor; ++u)
        {
            uint32_t un = (u + 1) % segMajor;
            for (uint32_t v = 0; v < segMinor; ++v)
            {
                uint32_t vn = (v + 1) % segMinor;
                uint32_t i00 = u  * segMinor + v;
                uint32_t i10 = un * segMinor + v;
                uint32_t i01 = u  * segMinor + vn;
                uint32_t i11 = un * segMinor + vn;
                // Same winding as ProceduralGeometry::MakeTorus.
                m_indices.insert(m_indices.end(), { i00, i11, i10, i00, i01, i11 });
            }
        }
        const UINT triCount = (UINT)(m_indices.size() / 3);

        // ---- Allocate IB (static, DEFAULT heap) + IB upload --------------
        const UINT64 ibBytes = m_indices.size() * sizeof(uint32_t);
        m_ibUpload = PtSample::CreateUploadBufferWithData(device, m_indices.data(), ibBytes,
                                                          (base + L"/IB upload").c_str());
        m_ib = PtSample::CreateDefaultBuffer(device, ibBytes, D3D12_RESOURCE_FLAG_NONE,
                                             D3D12_RESOURCE_STATE_COMMON, (base + L"/IB").c_str());
        {
            auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(m_ib.Get(),
                D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            cl->ResourceBarrier(1, &toCopy);
            cl->CopyBufferRegion(m_ib.Get(), 0, m_ibUpload.Get(), 0, ibBytes);
            auto toRead = CD3DX12_RESOURCE_BARRIER::Transition(m_ib.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
            cl->ResourceBarrier(1, &toRead);
        }

        // ---- Allocate VB (animated, UPLOAD heap, mapped persistently) ----
        const UINT64 vbBytes = (UINT64)m_vertCount * sizeof(DirectX::XMFLOAT3);
        m_vb = PtSample::CreateUploadBuffer(device, vbBytes, (base + L"/VB animated").c_str());
        D3D12_RANGE noRead = {0, 0};
        ThrowIfFailed(m_vb->Map(0, &noRead, &m_vbPtr), L"map animated donut VB");

        // ---- Cluster split ----------------------------------------------
        struct ClusterRange { UINT firstTri; UINT triCount; UINT vertCount; };
        std::vector<ClusterRange> clusters;
        for (UINT t = 0; t < triCount; t += kTrianglesPerCluster)
        {
            UINT triN = std::min<UINT>(kTrianglesPerCluster, triCount - t);
            std::unordered_map<uint32_t, bool> seen;
            for (UINT i = 0; i < triN * 3; ++i)
                seen.emplace(m_indices[(t * 3) + i], true);
            clusters.push_back({ t, triN, (UINT)seen.size() });
        }
        m_clasCount = (UINT)clusters.size();

        // ---- CLAS build args (CPU-filled UPLOAD, static after init) -----
        std::vector<D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS> args(m_clasCount);
        for (UINT c = 0; c < m_clasCount; ++c)
        {
            const auto& cl_ = clusters[c];
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
            a.VertexBuffer             = m_vb->GetGPUVirtualAddress();    // stable across frames
            a.IndexBuffer              = m_ib->GetGPUVirtualAddress() + cl_.firstTri * 3 * sizeof(uint32_t);
            a.GeometryIndexAndFlagsArray       = 0;
            a.GeometryIndexAndFlagsIndexBuffer = 0;
            a.OpacityMicromapArray             = 0;
            a.OpacityMicromapIndexBuffer       = 0;
        }
        m_clasArgs = PtSample::CreateUploadBufferWithData(device, args.data(),
            m_clasCount * sizeof(args[0]), (base + L"/CLAS args").c_str());

        // ---- Sizing -----------------------------------------------------
        D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC ctd = {};
        ctd.ClusterLimits.MaxArgCount               = m_clasCount;
        ctd.ClusterLimits.MaxUniqueGeometryIndexAndFlagsCountPerCluster = 1;
        ctd.ClusterLimits.MaxTriangleCountPerCluster = kTrianglesPerCluster;
        ctd.ClusterLimits.MaxVertexCountPerCluster  = kTrianglesPerCluster * 3;
        ctd.ClusterLimits.MaxTotalTriangleCount     = triCount;
        ctd.ClusterLimits.MaxTotalVertexCount       = triCount * 3;
        ctd.Flags                                   = D3D12_RTAS_OPERATION_FLAG_NONE;
        ctd.Mode                                    = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
        ctd.VertexFormat                            = D3D12_VERTEX_FORMAT_FLOAT32_3;
        ctd.IndexFormat                             = D3D12_INDEX_FORMAT_UINT32;
        ctd.GeometryIndexAndFlagsIndexFormat        = D3D12_INDEX_FORMAT_NONE;
        ctd.OpacityMicromapIndexFormat              = D3D12_INDEX_FORMAT_NONE;
        m_clasInputs = ctd;
        D3D12_RTAS_OPERATION_INPUTS clasOpIn = {};
        clasOpIn.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
        clasOpIn.pClusterTrianglesDesc = &m_clasInputs;
        D3D12_RTAS_OPERATION_PREBUILD_INFO clasPre = {};
        deviceRT2->GetRTASOperationPrebuildInfo(&clasOpIn, &clasPre);

        D3D12_RTAS_CLAS_INPUTS_DESC blasInputs = {};
        blasInputs.Flags              = D3D12_RTAS_OPERATION_FLAG_NONE;
        blasInputs.MaxArgCount        = 1;
        blasInputs.Mode               = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
        blasInputs.MaxTotalClasCount  = m_clasCount;
        blasInputs.MaxClasCountPerArg = m_clasCount;
        m_blasInputs = blasInputs;
        D3D12_RTAS_OPERATION_INPUTS blasOpIn = {};
        blasOpIn.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
        blasOpIn.pClasDesc = &m_blasInputs;
        D3D12_RTAS_OPERATION_PREBUILD_INFO blasPre = {};
        deviceRT2->GetRTASOperationPrebuildInfo(&blasOpIn, &blasPre);

        // ---- Allocate result + scratch (DEFAULT, reused per frame) ------
        m_clasResult = PtSample::CreateDefaultBuffer(device, clasPre.ResultDataMaxSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
            D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            (base + L"/CLAS result").c_str());
        m_clasScratch = PtSample::CreateDefaultBuffer(device, clasPre.ScratchDataSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, (base + L"/CLAS scratch").c_str());
        m_clasAddresses = PtSample::CreateDefaultBuffer(device,
            m_clasCount * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, (base + L"/CLAS addresses").c_str());
        m_blas = PtSample::CreateDefaultBuffer(device, blasPre.ResultDataMaxSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
            D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            (base + L"/ClusterBLAS").c_str());
        m_blasScratch = PtSample::CreateDefaultBuffer(device, blasPre.ScratchDataSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, (base + L"/ClusterBLAS scratch").c_str());

        // ---- BLAS-from-CLAS args (UPLOAD, static -- points at CLAS addresses) ----
        D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS blasArgs = {};
        blasArgs.ClasAddressCount  = m_clasCount;
        blasArgs.ClasAddressStride = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
        blasArgs.ClasAddressArray  = m_clasAddresses->GetGPUVirtualAddress();
        m_blasArgs = PtSample::CreateUploadBufferWithData(device, &blasArgs, sizeof(blasArgs),
            (base + L"/BLAS-from-CLAS args").c_str());

        SampleLog::LogF(L"[anim-donut] %s init: %u CLAS, vbBytes=%llu  CLAS_result=%llu B  BLAS=%llu B\n",
                        base.c_str(), m_clasCount, (unsigned long long)vbBytes,
                        (unsigned long long)clasPre.ResultDataMaxSizeInBytes,
                        (unsigned long long)blasPre.ResultDataMaxSizeInBytes);

        // ---- Initial Tick so BlasGpuVa() is valid for the first frame --
        Tick(cl, cl2, 0.0);
    }

    // Per-frame: pulsate vertex positions, rebuild CLAS + Cluster BLAS.
    // The Cluster BLAS GPUVA returned by BlasGpuVa() stays valid because
    // we reuse the same dest buffer; the contents are overwritten.
    void Tick(ID3D12GraphicsCommandList4* cl,
              ID3D12CommandListRaytracing2* cl2,
              double timeSeconds)
    {
        // ---- 1) CPU: compute pulsated vertex positions, write to mapped VB.
        //         minorR oscillates 70% .. 130% of base; majorR steady.
        const float kTwoPi = 6.28318530718f;
        const float pulse  = 1.0f + 0.30f * std::sin((float)timeSeconds * 2.4f);
        const float minor  = m_minorR * pulse;
        DirectX::XMFLOAT3* dst = static_cast<DirectX::XMFLOAT3*>(m_vbPtr);
        for (uint32_t u = 0; u < m_segMajor; ++u)
        {
            const float a = (float)u * kTwoPi / (float)m_segMajor;
            const float ca = std::cos(a), sa = std::sin(a);
            for (uint32_t v = 0; v < m_segMinor; ++v)
            {
                const float b = (float)v * kTwoPi / (float)m_segMinor;
                const float cb = std::cos(b), sb = std::sin(b);
                const float r = m_majorR + minor * cb;
                dst[u * m_segMinor + v] = { r * ca, minor * sb, r * sa };
            }
        }

        // ---- 2) GPU: BUILD_CLAS_FROM_TRIANGLES (batched, implicit dest) -
        D3D12_RTAS_BATCHED_OPERATION_DATA clasBatch = {};
        clasBatch.AddressResolutionFlags = D3D12_RTAS_OPERATION_ADDRESS_RESOLUTION_FLAG_NONE;
        clasBatch.BatchResultData        = m_clasResult->GetGPUVirtualAddress();
        clasBatch.BatchScratchData       = m_clasScratch->GetGPUVirtualAddress();
        clasBatch.ResultAddressArray.StartAddress  = m_clasAddresses->GetGPUVirtualAddress();
        clasBatch.ResultAddressArray.StrideInBytes = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
        clasBatch.IndirectArgumentArray.StartAddress  = m_clasArgs->GetGPUVirtualAddress();
        clasBatch.IndirectArgumentArray.StrideInBytes = sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS);
        clasBatch.IndirectArgumentArraySize = 0;
        D3D12_RTAS_OPERATION_INPUTS clasOpIn = {};
        clasOpIn.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
        clasOpIn.pClusterTrianglesDesc = &m_clasInputs;
        D3D12_RTAS_OPERATION_DESC clasDesc = {};
        clasDesc.Inputs                = clasOpIn;
        clasDesc.pBatchedOperationData = &clasBatch;
        cl2->ExecuteIndirectRTASOperations(1, &clasDesc,
            D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
        {
            D3D12_RESOURCE_BARRIER bars[2] = {};
            bars[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            bars[0].UAV.pResource = m_clasResult.Get();
            bars[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            bars[1].UAV.pResource = m_clasAddresses.Get();
            cl->ResourceBarrier(2, bars);
        }

        // ---- 3) GPU: BUILD_BLAS_FROM_CLAS (batched, implicit dest) -----
        D3D12_RTAS_BATCHED_OPERATION_DATA blasBatch = {};
        blasBatch.AddressResolutionFlags = D3D12_RTAS_OPERATION_ADDRESS_RESOLUTION_FLAG_NONE;
        blasBatch.BatchResultData        = m_blas->GetGPUVirtualAddress();
        blasBatch.BatchScratchData       = m_blasScratch->GetGPUVirtualAddress();
        blasBatch.IndirectArgumentArray.StartAddress  = m_blasArgs->GetGPUVirtualAddress();
        blasBatch.IndirectArgumentArray.StrideInBytes = sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS);
        blasBatch.IndirectArgumentArraySize = 0;
        D3D12_RTAS_OPERATION_INPUTS blasOpIn = {};
        blasOpIn.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
        blasOpIn.pClasDesc = &m_blasInputs;
        D3D12_RTAS_OPERATION_DESC blasDesc = {};
        blasDesc.Inputs                = blasOpIn;
        blasDesc.pBatchedOperationData = &blasBatch;
        cl2->ExecuteIndirectRTASOperations(1, &blasDesc,
            D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
        {
            D3D12_RESOURCE_BARRIER uav = {};
            uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource = m_blas.Get();
            cl->ResourceBarrier(1, &uav);
        }
    }

    D3D12_GPU_VIRTUAL_ADDRESS BlasGpuVa() const
    {
        return m_blas ? m_blas->GetGPUVirtualAddress() : 0;
    }
    UINT ClasCount() const { return m_clasCount; }

private:
    // Topology + parameters.
    float                m_majorR = 0, m_minorR = 0;
    uint32_t             m_segMajor = 0, m_segMinor = 0;
    uint32_t             m_vertCount = 0;
    std::vector<uint32_t> m_indices;

    // Buffers.
    Microsoft::WRL::ComPtr<ID3D12Resource> m_ib;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_ibUpload;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vb;
    void*                                  m_vbPtr = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_clasArgs;          // static after init
    Microsoft::WRL::ComPtr<ID3D12Resource> m_clasResult;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_clasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_clasAddresses;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blas;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blasScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blasArgs;          // static after init
    UINT                                   m_clasCount = 0;

    // Cached input descs so prebuild + per-frame call match exactly.
    D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC m_clasInputs = {};
    D3D12_RTAS_CLAS_INPUTS_DESC               m_blasInputs = {};
};
