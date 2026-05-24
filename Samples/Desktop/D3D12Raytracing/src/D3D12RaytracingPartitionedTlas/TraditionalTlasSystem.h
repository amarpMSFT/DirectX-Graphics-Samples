//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// TraditionalTlasSystem.h
//
// The DXR1 baseline: every frame, write a flat array of
// D3D12_RAYTRACING_INSTANCE_DESC into an upload buffer and call
// `BuildRaytracingAccelerationStructure()`.  Exists as the A/B comparison
// against `PtlasSystem` and as a sanity reference: if the same scene
// renders identically under both modes then the PTLAS plumbing is correct.
//
// No partitions, no per-partition translations, no LOD-via-update -- this
// is literally the DXR1 reference, modernised only by living behind the
// `ITlasSystem` interface.
//
#pragma once

#include "ITlasSystem.h"
#include "GpuBuffer.h"

class TraditionalTlasSystem final : public ITlasSystem
{
public:
    void Initialize(ID3D12Device5* device,
                    DX::DeviceResources* /*dr*/,
                    const InitDesc& desc) override
    {
        m_device = device;
        m_maxInstances = desc.maxInstances;

        // Upload buffer big enough for the max instance count, persistently
        // mapped (upload heaps are CPU-coherent on x64).
        const UINT64 instBytes = (UINT64)m_maxInstances * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
        m_instUpload = PtSample::CreateUploadBuffer(device, instBytes, L"Trad/InstanceDescUpload");
        CD3DX12_RANGE noRead(0, 0);
        ThrowIfFailed(m_instUpload->Map(0, &noRead, reinterpret_cast<void**>(&m_instUploadCpu)));

        // Worst-case prebuild now so we know the result/scratch sizes up
        // front and don't have to re-query (and possibly re-allocate) when
        // the per-frame instance count changes.
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
        tlasInputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlasInputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
        tlasInputs.NumDescs       = m_maxInstances;
        tlasInputs.InstanceDescs  = m_instUpload->GetGPUVirtualAddress();
        tlasInputs.Flags          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pre = {};
        device->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &pre);

        m_resultBytes  = pre.ResultDataMaxSizeInBytes;
        m_scratchBytes = pre.ScratchDataSizeInBytes;
        m_tlas    = PtSample::CreateDefaultBuffer(device, m_resultBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
            D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            L"Trad/TLAS");
        m_scratch = PtSample::CreateDefaultBuffer(device, m_scratchBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, L"Trad/Scratch");

        SampleLog::LogF(L"[trad-tlas] init max=%u result=%llu scratch=%llu\n",
                        m_maxInstances,
                        (unsigned long long)m_resultBytes,
                        (unsigned long long)m_scratchBytes);
    }

    void BeginFrame() override
    {
        m_frameInstCount = 0;
    }

    void WriteInstances(const SceneInstance* instances, UINT count) override
    {
        for (UINT i = 0; i < count; ++i)
        {
            ThrowIfFalse(m_frameInstCount < m_maxInstances,
                         L"TraditionalTlasSystem: too many instances this frame");
            const SceneInstance& s = instances[i];
            D3D12_RAYTRACING_INSTANCE_DESC d = {};
            // Row-major 3x4 from our 4x4 (SceneInstance.transform is
            // already transposed for the DXR 3x4 convention; see
            // PartitionedTlasSample::BuildSceneInstances).
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    d.Transform[r][c] = s.transform.m[r][c];
            d.InstanceID                          = s.instanceID;
            d.InstanceMask                        = s.instanceMask;
            d.InstanceContributionToHitGroupIndex = s.contributionToHitGroupIndex;
            d.Flags                               = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
            d.AccelerationStructure               = s.blasGva;
            m_instUploadCpu[m_frameInstCount++] = d;
        }
    }

    void TranslatePartitions(const PartitionTranslate* /*args*/, UINT /*count*/) override
    {
        // No-op: traditional TLAS has no partition translation concept.
        // The sample bakes any per-frame world-shift into the per-instance
        // transforms it hands to WriteInstances above.
    }

    void UpdateInstances(const InstanceUpdate* /*args*/, UINT /*count*/) override
    {
        // No-op: the traditional path rewrites the full instance list
        // every frame, so any LOD/BLAS swap is already reflected in the
        // SceneInstance.blasGva values the sample feeds into WriteInstances.
    }

    void Build(ID3D12GraphicsCommandList4* cl,
               ID3D12CommandListRaytracing2* /*cl2*/) override
    {
        // Build a TLAS whose NumDescs == the actual count this frame, but
        // pointing into the same upload buffer.  (The prebuild query was
        // for the worst case, so the scratch/result we allocated is enough.)
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
        tlasInputs.Type          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlasInputs.DescsLayout   = D3D12_ELEMENTS_LAYOUT_ARRAY;
        tlasInputs.NumDescs      = m_frameInstCount;
        tlasInputs.InstanceDescs = m_instUpload->GetGPUVirtualAddress();
        tlasInputs.Flags         = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build = {};
        build.Inputs                              = tlasInputs;
        build.DestAccelerationStructureData       = m_tlas->GetGPUVirtualAddress();
        build.ScratchAccelerationStructureData    = m_scratch->GetGPUVirtualAddress();
        cl->BuildRaytracingAccelerationStructure(&build, 0, nullptr);

        auto uav = CD3DX12_RESOURCE_BARRIER::UAV(m_tlas.Get());
        cl->ResourceBarrier(1, &uav);
    }

    D3D12_GPU_VIRTUAL_ADDRESS Gva() const override { return m_tlas->GetGPUVirtualAddress(); }
    const wchar_t* ModeName() const override { return L"traditional"; }

    FrameStats GetLastFrameStats() const override
    {
        FrameStats s = {};
        s.resultBytes        = m_resultBytes;
        s.scratchBytes       = m_scratchBytes;
        s.instancesSubmitted = m_frameInstCount;
        s.partitionsTouched  = 0;
        return s;
    }

private:
    ID3D12Device5* m_device = nullptr;
    UINT m_maxInstances     = 0;
    UINT m_frameInstCount   = 0;
    UINT64 m_resultBytes    = 0;
    UINT64 m_scratchBytes   = 0;

    Microsoft::WRL::ComPtr<ID3D12Resource>      m_tlas;
    Microsoft::WRL::ComPtr<ID3D12Resource>      m_scratch;
    Microsoft::WRL::ComPtr<ID3D12Resource>      m_instUpload;
    D3D12_RAYTRACING_INSTANCE_DESC*             m_instUploadCpu = nullptr;
};
