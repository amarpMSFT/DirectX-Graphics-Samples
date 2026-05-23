//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// PtlasSystem.cpp
//
// Complete lifecycle of a DXR2 Partitioned Top-Level Acceleration Structure:
//   * Initialize -- size + allocate the PTLAS storage and scratch
//   * BeginFrame -- rotate the per-frame upload arena
//   * WriteInstances / UpdateInstances / TranslatePartitions --
//                  stage GPU-side op-arg arrays
//   * Build      -- assemble the operation header array + call
//                  ExecuteIndirectRTASOperations
//
// The whole file is ~150 lines and meant to be read top-to-bottom.  Every
// PTLAS call boils down to that one ExecuteIndirectRTASOperations
// invocation at the bottom of Build().
//
//   --------------------------------------------------------------------
//   Why "indirect"?
//   --------------------------------------------------------------------
//   All op-arg arrays live on the GPU.  The CPU only ships pointers + a
//   compact "what types of ops to run" header (max 3 entries: WRITE,
//   UPDATE, TRANSLATE).  This lets future versions of the sample have a
//   compute shader emit the op-args directly (e.g. instance transfer
//   detection) without ever copying them to/from the CPU.  Milestone 2a
//   uses CPU staging via an upload heap to keep this file legible; the
//   GPU-arg-fill path arrives in a later milestone but doesn't change
//   PtlasSystem itself -- the caller just hands in a different GVA.
//
// Spec references (Raytracing2.md):
//   * `Partitioned TLAS overview`                       -- big picture
//   * `D3D12_RTAS_PARTITIONED_TLAS_INPUTS_DESC`         -- sizing
//   * `D3D12_RTAS_PARTITIONED_TLAS_OPERATION_DATA`      -- build call inputs
//   * `D3D12_RTAS_PARTITIONED_TLAS_OPERATION_*_ARGS`    -- per-op-type args
//

#include "stdafx.h"
#include "PtlasSystem.h"
#include "GpuBuffer.h"

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Initialize -- query prebuild sizes from the driver, allocate the PTLAS
// + scratch buffers, and allocate the per-frame upload-arena ring used for
// staging op headers and (in milestone 2a) the per-op-type arg arrays.
// ---------------------------------------------------------------------------
void PtlasSystem::Initialize(ID3D12Device5* device, DX::DeviceResources* /*dr*/,
                             const InitDesc& desc)
{
    m_device = device;

    // PTLAS prebuild/build calls live on ID3D12DeviceRaytracing2 (DXR2),
    // not ID3D12Device5.  QI here so the caller doesn't have to thread an
    // extra interface pointer through ITlasSystem.
    Microsoft::WRL::ComPtr<ID3D12DeviceRaytracing2> dxr2;
    ThrowIfFailed(device->QueryInterface(IID_PPV_ARGS(&dxr2)),
        L"PtlasSystem: ID3D12DeviceRaytracing2 unavailable (experimental D3D12Core not loaded?)");

    // (1) Describe the worst-case shape of the PTLAS.  See
    //     `D3D12_RTAS_PARTITIONED_TLAS_INPUTS_DESC` in Raytracing2.md.
    //     These values are the upper bounds: actual per-frame ops can
    //     reference any subset of [0..InstanceCount) and [0..PartitionCount).
    m_inputs                                    = {};
    m_inputs.Flags                              = D3D12_RTAS_PARTITIONED_TLAS_FLAG_FAST_TRACE
                                                | D3D12_RTAS_PARTITIONED_TLAS_FLAG_ENABLE_PARTITION_TRANSLATION;
    m_inputs.InstanceCount                      = desc.maxInstances;
    m_inputs.PartitionCount                     = desc.maxPartitions;
    m_inputs.MaxInstancePerPartitionCount       = desc.maxInstancesPerPartition;
    m_inputs.MaxInstanceInGlobalPartitionCount  = desc.maxInstancesInGlobalPartition;
    // ENABLE_PARTITION_TRANSLATION enabled unconditionally: this sample's
    // milestone 2c uses per-frame TRANSLATE_PARTITION ops to re-center
    // every partition on the camera each frame.  Per spec the flag adds a
    // small memory cost (stores the un-translated instance transforms) but
    // unlocks the cheap-many-partitions update path.

    // (2) Ask the driver for result + scratch byte budgets.
    D3D12_RTAS_OPERATION_INPUTS opInputs   = {};
    opInputs.Type                           = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
    opInputs.pPartitionedTLASInputsDesc     = &m_inputs;

    D3D12_RTAS_OPERATION_PREBUILD_INFO pre = {};
    dxr2->GetRTASOperationPrebuildInfo(&opInputs, &pre);
    m_resultBytes  = pre.ResultDataMaxSizeInBytes;
    m_scratchBytes = pre.ScratchDataSizeInBytes;

    // (3) Allocate the PTLAS storage and scratch.
    m_ptlas = PtSample::CreateDefaultBuffer(device, m_resultBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
        D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        L"PTLAS/Result");
    m_scratch = PtSample::CreateDefaultBuffer(device, m_scratchBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
        L"PTLAS/Scratch");

    // (4) Pre-allocate the per-frame upload arenas.  Each arena is small
    // (1MB) but rotates so we never write into a slot the GPU is still
    // reading from a previous frame.
    constexpr UINT64 kArenaBytes = 1u << 20;
    for (UINT i = 0; i < kFrameSlots; ++i)
    {
        auto& s = m_slots[i];
        wchar_t name[64];
        swprintf_s(name, L"PTLAS/Arena[%u]", i);
        s.arena    = PtSample::CreateUploadBuffer(device, kArenaBytes, name);
        s.capacity = kArenaBytes;
        s.used     = 0;
        s.gva      = s.arena->GetGPUVirtualAddress();
        CD3DX12_RANGE noRead(0, 0);
        ThrowIfFailed(s.arena->Map(0, &noRead, reinterpret_cast<void**>(&s.cpu)));
    }

    SampleLog::LogF(L"[ptlas] init: instances=%u partitions=%u per-part-max=%u "
                    L"global-max=%u  result=%llu scratch=%llu\n",
                    desc.maxInstances, desc.maxPartitions,
                    desc.maxInstancesPerPartition,
                    desc.maxInstancesInGlobalPartition,
                    (unsigned long long)m_resultBytes,
                    (unsigned long long)m_scratchBytes);
}

// ---------------------------------------------------------------------------
// BeginFrame -- rotate to the next arena slot and clear staged ops.
// ---------------------------------------------------------------------------
void PtlasSystem::BeginFrame()
{
    m_currentSlot = (m_currentSlot + 1) % kFrameSlots;
    m_slots[m_currentSlot].used = 0;
    m_pendingOps.clear();
    m_lastWriteCount = m_lastUpdateCount = m_lastTranslateCount = 0;
}

// ---------------------------------------------------------------------------
// WriteToFrameArena -- bump-allocate `bytes` aligned to `align` out of the
// current frame's upload arena.  Returns the GPU virtual address.  Writes
// must fit; sizing of the arena is conservative (1MB / frame today).
// ---------------------------------------------------------------------------
D3D12_GPU_VIRTUAL_ADDRESS
PtlasSystem::WriteToFrameArena(const void* src, UINT64 bytes, UINT64 align)
{
    FrameSlot& s = m_slots[m_currentSlot];
    UINT64 aligned = (s.used + (align - 1)) & ~(align - 1);
    ThrowIfFalse(aligned + bytes <= s.capacity,
                 L"PtlasSystem: per-frame arena exhausted -- grow kArenaBytes");
    std::memcpy(s.cpu + aligned, src, (size_t)bytes);
    s.used = aligned + bytes;
    return s.gva + aligned;
}

void PtlasSystem::AppendPendingOp(D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE t,
                                  D3D12_GPU_VIRTUAL_ADDRESS gva, UINT count, UINT stride)
{
    if (count == 0) return;  // empty op-arg arrays are pointless; skip
    PendingOp op = { t, gva, count, stride };
    m_pendingOps.push_back(op);
}

// ---------------------------------------------------------------------------
// WriteInstances -- stage a WRITE_INSTANCE op for `count` scene instances.
// One WRITE_INSTANCE arg = full instance descriptor (transform, mask,
// flags, BLAS pointer, instance index slot, partition index).
//
// Spec: D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS
// ---------------------------------------------------------------------------
void PtlasSystem::WriteInstances(const SceneInstance* instances, UINT count)
{
    if (count == 0) return;

    // Build the args contiguously in a temporary buffer, then copy in one
    // shot into the frame arena.  Keeps the upload-heap write linear.
    std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS> args(count);
    for (UINT i = 0; i < count; ++i)
    {
        const SceneInstance& s = instances[i];
        auto& a = args[i];
        // Row-major 3x4 from our row-major 4x4: drop the last (0,0,0,1) row.
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                a.Transform[r][c] = s.transform.m[r][c];
        a.InstanceID                          = s.instanceID;
        a.InstanceMask                        = s.instanceMask;
        a.InstanceContributionToHitGroupIndex = 0;
        a.InstanceFlags                       = D3D12_RTAS_PARTITIONED_TLAS_INSTANCE_FLAG_NONE;
        a.AccelerationStructure               = s.blasGva;
        a.InstanceIndex                       = i;
        a.PartitionIndex                      = s.partitionIndex;
        a.ExplicitAABB                        = {};   // not used (flag not set)
    }

    constexpr UINT kStride = (UINT)sizeof(D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS);
    D3D12_GPU_VIRTUAL_ADDRESS gva =
        WriteToFrameArena(args.data(), (UINT64)args.size() * kStride, /*align*/8);
    AppendPendingOp(D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE_WRITE_INSTANCE,
                    gva, count, kStride);
    m_lastWriteCount += count;
}

// ---------------------------------------------------------------------------
// TranslatePartitions -- stage a TRANSLATE_PARTITION op for `count` partitions.
// Each arg is { partitionIndex, translation[3] }.  partitionIndex can be
// any value in [0..PartitionCount) for regular partitions or
// D3D12_RTAS_PARTITIONED_TLAS_PARTITION_INDEX_GLOBAL_PARTITION (0xFFFFFFFF)
// for the global partition.  Translation values REPLACE (not accumulate) the
// previous translation for that partition.
//
// Use case in this sample: every frame, compute
//   translation[p] = partition_home_world - as_origin_world
// and submit one TRANSLATE_PARTITION op for every partition.  Combined with
// PARTITION-LOCAL instance transforms (translation_world(b) - home_world(p)),
// the PTLAS sees acceleration-structure-space coords centered on the camera
// every frame -- best float precision exactly where rays start.
//
// Spec: D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TRANSLATE_PARTITION_ARGS
// ---------------------------------------------------------------------------
void PtlasSystem::TranslatePartitions(const PartitionTranslate* args, UINT count)
{
    if (count == 0) return;

    // Translate args are 16 bytes each (UINT + 3 floats, naturally
    // 4-byte-aligned).  Build them contiguously in a local vector then
    // copy in one shot into the frame arena.
    std::vector<D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TRANSLATE_PARTITION_ARGS> a(count);
    for (UINT i = 0; i < count; ++i)
    {
        a[i].PartitionIndex          = args[i].partitionIndex;
        a[i].PartitionTranslation[0] = args[i].translation[0];
        a[i].PartitionTranslation[1] = args[i].translation[1];
        a[i].PartitionTranslation[2] = args[i].translation[2];
    }

    constexpr UINT kStride = (UINT)sizeof(D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TRANSLATE_PARTITION_ARGS);
    D3D12_GPU_VIRTUAL_ADDRESS gva =
        WriteToFrameArena(a.data(), (UINT64)a.size() * kStride, /*align*/4);
    AppendPendingOp(D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE_TRANSLATE_PARTITION,
                    gva, count, kStride);
    m_lastTranslateCount += count;
}

// ---------------------------------------------------------------------------
// Build -- assemble the per-frame op header array + invoke
// ExecuteIndirectRTASOperations with a PARTITIONED_TLAS desc.  This is the
// entire PTLAS-build call.
//
// On the first call (m_haveBuiltOnce == false) SourceAccelerationStructure
// is null which tells the driver "treat the PTLAS as zero-initialised, then
// apply these ops".  On every subsequent call we point Source = Dest so
// the build is incremental on top of the previous frame's state.
//
// Spec: D3D12_RTAS_PARTITIONED_TLAS_OPERATION_DATA
//       (`ScratchAccelerationStructureData` is missing from the spec snippet
//        for the struct definition but present in the conformance test +
//        d3d12.h; confirmed via IndirectBuild.cpp line ~8025.)
// ---------------------------------------------------------------------------
void PtlasSystem::Build(ID3D12GraphicsCommandList4* cl,
                        ID3D12CommandListRaytracing2* cl2)
{
    // (a) Stage the operation header array (max 3 entries).
    D3D12_RTAS_PARTITIONED_TLAS_OPERATION ops[3] = {};
    for (size_t i = 0; i < m_pendingOps.size(); ++i)
    {
        const PendingOp& p = m_pendingOps[i];
        ops[i].Type                  = p.type;
        ops[i].ArgCount              = p.argCount;
        ops[i].ArgData.StartAddress  = p.argsGva;
        ops[i].ArgData.StrideInBytes = p.stride;
    }
    const UINT numOps = (UINT)m_pendingOps.size();
    if (numOps == 0)
    {
        // Nothing to do.  Skip the build call entirely -- a zero-op rebuild
        // would still cost a driver round-trip with no effect.
        return;
    }
    const UINT64 opsBytes = numOps * sizeof(D3D12_RTAS_PARTITIONED_TLAS_OPERATION);
    D3D12_GPU_VIRTUAL_ADDRESS opsGva = WriteToFrameArena(ops, opsBytes, /*align*/8);

    // (b) Stage the operation count (one UINT32 in upload memory).
    UINT32 numOps32 = numOps;
    D3D12_GPU_VIRTUAL_ADDRESS countGva =
        WriteToFrameArena(&numOps32, sizeof(numOps32), /*align*/4);

    // (c) Fill in the operation data + inputs.
    D3D12_RTAS_OPERATION_INPUTS opInputs   = {};
    opInputs.Type                           = D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS;
    opInputs.pPartitionedTLASInputsDesc     = &m_inputs;

    D3D12_RTAS_PARTITIONED_TLAS_OPERATION_DATA opData = {};
    opData.SourceAccelerationStructureData    = m_haveBuiltOnce ? m_ptlas->GetGPUVirtualAddress() : 0;
    opData.DestAccelerationStructureData       = m_ptlas->GetGPUVirtualAddress();
    opData.ScratchAccelerationStructureData    = m_scratch->GetGPUVirtualAddress();
    opData.IndirectPartitionedTlasOpCount      = countGva;
    opData.IndirectPartitionedTlasOps          = opsGva;

    D3D12_RTAS_OPERATION_DESC desc = {};
    desc.Inputs                     = opInputs;
    desc.pPartitionedTlasOperationData = &opData;

    // (d) The whole PTLAS build is this one driver call.
    cl2->ExecuteIndirectRTASOperations(1, &desc,
        D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);

    // ⚠ GOTCHA: insert a UAV barrier on the PTLAS dest buffer before any
    // downstream TraceRay / RayQuery / DispatchRays.  The PTLAS resource
    // is in D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE which
    // is "sticky" (no transition required between build and trace), BUT
    // memory visibility between the build's UAV writes and the trace's
    // RTAS reads is NOT implicit.  Without this barrier the GPU may TDR
    // on the first frame: the trace reads zero-initialised PTLAS contents
    // and loops/faults.  Same rule as a BLAS build feeding a TLAS build.
    {
        auto uav = CD3DX12_RESOURCE_BARRIER::UAV(m_ptlas.Get());
        cl->ResourceBarrier(1, &uav);
    }

    m_haveBuiltOnce = true;
}

ITlasSystem::FrameStats PtlasSystem::GetLastFrameStats() const
{
    FrameStats s = {};
    s.resultBytes        = m_resultBytes;
    s.scratchBytes       = m_scratchBytes;
    s.instancesSubmitted = m_lastWriteCount + m_lastUpdateCount;
    s.partitionsTouched  = m_lastTranslateCount;
    return s;
}
