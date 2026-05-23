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

private:
    ProceduralGeometry::Mesh m_mesh;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vb;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_ib;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_vbUpload;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_ibUpload;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blas;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blasScratch;
    UINT64 m_blasResultBytes  = 0;
    UINT64 m_blasScratchBytes = 0;
};
