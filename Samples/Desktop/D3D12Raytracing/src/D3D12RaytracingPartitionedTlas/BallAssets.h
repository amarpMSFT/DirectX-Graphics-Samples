//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// BallAssets.h  (milestone 2a -- traditional DXR1 BLAS path)
//
// Owns the one ball mesh, uploads its vertex + index data to the GPU, and
// builds a single traditional BLAS for it via
// `BuildRaytracingAccelerationStructure()`.  The BLAS GVA is exposed so a
// (P)TLAS can instance the ball any number of times.
//
// Milestone 2a uses DXR1 here on purpose: the PTLAS is the centerpiece of
// this sample, and a known-good BLAS path lets us debug PTLAS without also
// debugging cluster-BLAS-from-CLAS.  Phase 2b/3 swaps in the DXR2 cluster
// path (CLAS + BUILD_BLAS_FROM_CLAS) behind the same `BlasGpuVa()`
// accessor so nothing else in the sample changes.
//
#pragma once

#include "stdafx.h"
#include "GpuBuffer.h"
#include "ProceduralGeometry.h"

class BallAssets
{
public:
    // Build the ball mesh + BLAS.  This call ONLY RECORDS commands into
    // `cl`; the caller is responsible for closing / executing / waiting.
    // We keep the upload + scratch buffers as members so they outlive the
    // GPU submission.
    void Initialize(ID3D12Device5* device,
                    ID3D12GraphicsCommandList4* cl,
                    uint32_t icosphereSubdiv = 2)
    {
        m_mesh = ProceduralGeometry::MakeIcosphere(icosphereSubdiv);

        // ---- Upload VB + IB to default-heap buffers ----
        const UINT64 vbBytes = m_mesh.positions.size() * sizeof(DirectX::XMFLOAT3);
        const UINT64 ibBytes = m_mesh.indices.size()   * sizeof(uint32_t);
        m_vbUpload = PtSample::CreateUploadBufferWithData(
            device, m_mesh.positions.data(), vbBytes, L"BallAssets/VB upload");
        m_ibUpload = PtSample::CreateUploadBufferWithData(
            device, m_mesh.indices.data(),   ibBytes, L"BallAssets/IB upload");
        m_vb = PtSample::CreateDefaultBuffer(device, vbBytes, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COMMON, L"BallAssets/VB");
        m_ib = PtSample::CreateDefaultBuffer(device, ibBytes, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COMMON, L"BallAssets/IB");
        // Transition COMMON -> COPY_DEST before the copies.  (Buffers on the
        // default heap always start in COMMON; the initial-state arg above
        // is essentially ignored, see D3D12 debug-layer id 1328.)
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

        // ---- Build a traditional DXR1 BLAS over those triangles ----
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
            L"BallAssets/BLAS");
        m_blasScratch = PtSample::CreateDefaultBuffer(device, pre.ScratchDataSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, L"BallAssets/BLAS scratch");

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasBuild = {};
        blasBuild.Inputs                          = blasInputs;
        blasBuild.DestAccelerationStructureData    = m_blas->GetGPUVirtualAddress();
        blasBuild.ScratchAccelerationStructureData = m_blasScratch->GetGPUVirtualAddress();
        cl->BuildRaytracingAccelerationStructure(&blasBuild, 0, nullptr);

        // UAV barrier so any TLAS build later in the same cmd list sees
        // the completed BLAS.
        auto uav = CD3DX12_RESOURCE_BARRIER::UAV(m_blas.Get());
        cl->ResourceBarrier(1, &uav);

        m_blasResultBytes  = pre.ResultDataMaxSizeInBytes;
        m_blasScratchBytes = pre.ScratchDataSizeInBytes;

        SampleLog::LogF(L"[ball-assets] icosphere subdiv=%u verts=%u tris=%u "
                        L"VB=%llu B  IB=%llu B  BLAS=%llu B (scratch=%llu)\n",
                        icosphereSubdiv,
                        (unsigned)m_mesh.positions.size(),
                        (unsigned)(m_mesh.indices.size() / 3),
                        (unsigned long long)vbBytes,
                        (unsigned long long)ibBytes,
                        (unsigned long long)m_blasResultBytes,
                        (unsigned long long)m_blasScratchBytes);
    }

    // Address that goes into a TLAS instance desc (D3D12_RAYTRACING_INSTANCE_DESC
    // ::AccelerationStructure) or a PTLAS WRITE_INSTANCE_ARGS::AccelerationStructure.
    D3D12_GPU_VIRTUAL_ADDRESS BlasGpuVa() const { return m_blas->GetGPUVirtualAddress(); }
    UINT64                    BlasResultBytes() const { return m_blasResultBytes; }

private:
    ProceduralGeometry::Mesh m_mesh;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vb;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_ib;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vbUpload;     // kept alive past Initialize
    Microsoft::WRL::ComPtr<ID3D12Resource> m_ibUpload;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blas;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blasScratch;  // kept alive past Initialize
    UINT64 m_blasResultBytes  = 0;
    UINT64 m_blasScratchBytes = 0;
};
