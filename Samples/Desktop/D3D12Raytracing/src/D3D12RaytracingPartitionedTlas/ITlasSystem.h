//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT License (MIT).
//
//*********************************************************
//
// ITlasSystem.h
//
// Abstract top-level-acceleration-structure interface.  The sample owns one
// of these and swaps between concrete implementations:
//   * TraditionalTlasSystem -- DXR1 BuildRaytracingAccelerationStructure
//                              over a flat D3D12_RAYTRACING_INSTANCE_DESC[]
//   * PtlasSystem           -- DXR2 ExecuteIndirectRTASOperations with
//                              D3D12_RTAS_OPERATION_TYPE_PARTITIONED_TLAS
//
// The same shader binding (g_tlas : register(t0)) works for both because
// PTLAS is shader-transparent: the GPU virtual address goes into a normal
// RaytracingAccelerationStructure SRV.
//
// Why an interface instead of code-paths-with-an-enum?  Two reasons:
//   1. Toggling at runtime cleanly tears down resources for the OTHER mode
//      so memory comparisons are honest.
//   2. The PtlasSystem implementation can be lifted into a blog post / other
//      engine without dragging in TLAS-toggle plumbing.
//
#pragma once

#include "stdafx.h"
#include <DirectXMath.h>

// One scene instance.  Mode-agnostic input the sample feeds to whichever
// TLAS system is active.  Milestone 2a uses a single instance per ball; later
// milestones add PartitionIndex (PTLAS-only -- traditional TLAS just folds
// every instance into one flat list) and ExplicitAABB (PTLAS-only).
struct SceneInstance
{
    DirectX::XMFLOAT4X4 transform;     // row-major, last row is (0,0,0,1)
    D3D12_GPU_VIRTUAL_ADDRESS blasGva; // 256-byte aligned BLAS
    UINT instanceIndex;                // PTLAS slot to write into.  Distinct from
                                       // instanceID -- this is the address-in-PTLAS,
                                       // not the user-visible value.  Traditional
                                       // TLAS ignores this and uses array position.
    UINT instanceID;                   // accessible via InstanceID() in HLSL
    UINT instanceMask;                 // 0xFF by default
    UINT partitionIndex;               // 0xFFFFFFFF = global (PTLAS); ignored by traditional
};

class ITlasSystem
{
public:
    virtual ~ITlasSystem() = default;

    // One-time init.  `maxInstances` and `maxPartitions` (and the global-
    // partition cap) size the worst-case allocations.  Implementations are
    // free to round up.
    struct InitDesc
    {
        UINT maxInstances;
        UINT maxPartitions;                  // ignored by Traditional
        UINT maxInstancesPerPartition;       // ignored by Traditional
        UINT maxInstancesInGlobalPartition;  // ignored by Traditional
    };
    virtual void Initialize(ID3D12Device5* device,
                            DX::DeviceResources* dr,
                            const InitDesc& desc) = 0;

    // Begin a frame's instance updates.  Cheap reset of any per-frame state
    // (operation list, upload offsets).  Called once per frame.
    virtual void BeginFrame() = 0;

    // Submit one or more instance writes / updates.  This is the boundary
    // where the abstraction unifies the two backing models:
    //   * Traditional batches a flat array of D3D12_RAYTRACING_INSTANCE_DESC.
    //   * PTLAS batches WRITE_INSTANCE / UPDATE_INSTANCE arg structs and
    //     remembers which partitions are touched.
    virtual void WriteInstances(const SceneInstance* instances, UINT count) = 0;

    // GPU-side arg structure for a single TRANSLATE_PARTITION operation.
    // Mirrors D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TRANSLATE_PARTITION_ARGS
    // but we redefine it CPU-side so callers don't have to include the full
    // PTLAS header just to fill these.  (Traditional impls also accept the
    // struct -- they just ignore the data.)
    struct PartitionTranslate
    {
        UINT  partitionIndex;
        FLOAT translation[3];
    };

    // Optional per-frame call.  PTLAS impl emits a TRANSLATE_PARTITION op
    // with `args[0..count)` baked into the current frame's arg buffer; the
    // Traditional impl is a no-op (no partition concept).  Counts can be
    // up to InitDesc::maxPartitions + 1 (the +1 covers the global partition;
    // identified via D3D12_RTAS_PARTITIONED_TLAS_PARTITION_INDEX_GLOBAL_PARTITION).
    virtual void TranslatePartitions(const PartitionTranslate* args, UINT count) = 0;

    // Build/finalize the top-level structure.  Records GPU work into `cl`.
    virtual void Build(ID3D12GraphicsCommandList4* cl,
                       ID3D12CommandListRaytracing2* cl2) = 0;

    // GPU virtual address bindable as a RaytracingAccelerationStructure
    // SRV (i.e. what goes into the SRV at register t0).
    virtual D3D12_GPU_VIRTUAL_ADDRESS Gva() const = 0;

    // Short human-readable name used by logs and the overlay
    // ("traditional" / "partitioned").
    virtual const wchar_t* ModeName() const = 0;

    // Most-recent build stats for the overlay / log.  Implementations fill
    // what they have; zeros for everything else.
    struct FrameStats
    {
        UINT64 resultBytes;          // size of the top-level structure
        UINT64 scratchBytes;
        UINT   instancesSubmitted;
        UINT   partitionsTouched;    // 0 for Traditional
        // Future: GPU timestamp delta for build, instance-update fill CS, ...
    };
    virtual FrameStats GetLastFrameStats() const = 0;
};
