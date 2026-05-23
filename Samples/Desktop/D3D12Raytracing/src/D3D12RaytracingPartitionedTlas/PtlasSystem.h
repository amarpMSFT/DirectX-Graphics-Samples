//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// PtlasSystem.h
//
// Self-contained Partitioned-TLAS (DXR2) manager.  Designed to be the
// single source of truth a reader can study to learn how a PTLAS is built
// and updated in D3D12.  See PtlasSystem.cpp for the entire lifecycle in
// ~150 lines.
//
// Quick API tour:
//
//   PtlasSystem ptlas;
//   ptlas.Initialize(device, dr, { instanceCount, partitionCount, ... });
//   // ... per frame:
//   ptlas.BeginFrame();
//   ptlas.WriteInstances(scene_instances, n);   // partition-aware
//   // (optional)  ptlas.UpdateInstances(...)
//   // (optional)  ptlas.TranslatePartitions(...)
//   ptlas.Build(cl, cl2);                       // ExecuteIndirectRTASOperations
//   // ptlas.Gva() is now bindable as a RaytracingAccelerationStructure SRV.
//
#pragma once

#include "ITlasSystem.h"

class PtlasSystem final : public ITlasSystem
{
public:
    void Initialize(ID3D12Device5* device,
                    DX::DeviceResources* dr,
                    const InitDesc& desc) override;
    void BeginFrame() override;
    void WriteInstances(const SceneInstance* instances, UINT count) override;
    void Build(ID3D12GraphicsCommandList4* cl,
               ID3D12CommandListRaytracing2* cl2) override;
    D3D12_GPU_VIRTUAL_ADDRESS Gva() const override { return m_ptlas->GetGPUVirtualAddress(); }
    const wchar_t* ModeName() const override { return L"partitioned"; }
    FrameStats GetLastFrameStats() const override;

private:
    // Persistent (created in Initialize, valid for the lifetime of the
    // system).
    ID3D12Device5* m_device = nullptr;
    D3D12_RTAS_PARTITIONED_TLAS_INPUTS_DESC m_inputs = {};

    Microsoft::WRL::ComPtr<ID3D12Resource> m_ptlas;       // result buffer
    Microsoft::WRL::ComPtr<ID3D12Resource> m_scratch;     // build scratch
    UINT64 m_resultBytes  = 0;
    UINT64 m_scratchBytes = 0;

    bool m_haveBuiltOnce = false;   // first build vs incremental?  See Build().

    // Per-frame upload-heap arenas.  We grow a fresh chunk each frame so we
    // never overwrite GPU work in flight.  Three is overkill for our small
    // arg buffers; one shared 1MB arena per frame slot is plenty.
    static constexpr UINT kFrameSlots = 3;
    struct FrameSlot
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> arena;
        UINT8*                                 cpu = nullptr;
        UINT64                                 used = 0;
        UINT64                                 capacity = 0;
        D3D12_GPU_VIRTUAL_ADDRESS              gva = 0;
    };
    FrameSlot m_slots[kFrameSlots];
    UINT      m_currentSlot = 0;

    // What we've staged this frame.  Each entry lands in one
    // D3D12_RTAS_PARTITIONED_TLAS_OPERATION.  Built ops MUST have unique
    // Type values per the spec, so this vector has at most 3 entries
    // (WRITE / UPDATE / TRANSLATE).
    struct PendingOp
    {
        D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE type;
        D3D12_GPU_VIRTUAL_ADDRESS argsGva;
        UINT                       argCount;
        UINT                       stride;
    };
    std::vector<PendingOp> m_pendingOps;

    // Stats for the overlay / log.
    UINT m_lastWriteCount     = 0;
    UINT m_lastUpdateCount    = 0;
    UINT m_lastTranslateCount = 0;

    // Helpers (defined in .cpp).
    D3D12_GPU_VIRTUAL_ADDRESS WriteToFrameArena(const void* src, UINT64 bytes, UINT64 align);
    void AppendPendingOp(D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE t,
                         D3D12_GPU_VIRTUAL_ADDRESS gva, UINT count, UINT stride);
};
