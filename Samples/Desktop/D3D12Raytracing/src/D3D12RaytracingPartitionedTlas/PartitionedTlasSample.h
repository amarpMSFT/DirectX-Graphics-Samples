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

#pragma once

#include "DXSample.h"
#include "StepTimer.h"
#include "ITlasSystem.h"
#include "MeshAssets.h"
#include "SceneLayout.h"
#include "FlockMotion.h"
#include "RollingPartitions.h"

#include <string>
#include <memory>
#include <vector>

// PartitionedTlasSample
// ---------------------------------------------------------------------------
// Milestone 2a (this commit): one ball, one TLAS instance, two TLAS modes
// (`traditional` baseline / `partitioned`) toggleable at startup via
// `--tlas-mode`.  Both produce identical pixels so the toggle is a
// before/after sanity check on the PtlasSystem implementation.
//
// Milestones 2b+ scale up to a grid of partitions and the moving flock; see
// docs/design.md.
class PartitionedTlasSample : public DXSample
{
public:
    PartitionedTlasSample(UINT width, UINT height, std::wstring name);

    // IDeviceNotify
    virtual void OnDeviceLost() override;
    virtual void OnDeviceRestored() override;

    // DXSample overrides
    virtual void OnInit() override;
    virtual void OnUpdate() override;
    virtual void OnRender() override;
    virtual void OnSizeChanged(UINT width, UINT height, bool minimized) override;
    virtual void OnDestroy() override;
    virtual void OnKeyDown(UINT8 key) override;
    virtual IDXGISwapChain* GetSwapchain() override { return m_deviceResources->GetSwapChain(); }
    virtual void ParseCommandLineArgs(_In_reads_(argc) WCHAR* argv[], int argc) override;

private:
    static const UINT FrameCount = 3;

    // ---- DXR1 + DXR2 device / cmdlist surfaces ----
    Microsoft::WRL::ComPtr<ID3D12Device5>                m_dxrDevice;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4>   m_dxrCommandList;
    Microsoft::WRL::ComPtr<ID3D12DeviceRaytracing2>      m_dxr2Device;
    Microsoft::WRL::ComPtr<ID3D12CommandListRaytracing2> m_dxr2CommandList;
    bool   m_clustersAndPtlasSupported = false;
    UINT64 m_framesRendered            = 0;
    UINT64 m_cumBallChanges            = 0;   // accumulates since startup
    UINT64 m_cumBallChangesLastLog     = 0;   // value at previous log frame
    StepTimer m_timer;
    std::chrono::steady_clock::time_point m_startTime;

    // ---- CLI / headless state ----
    int          m_screenshotFrame    = -1;
    std::wstring m_screenshotPath;
    bool         m_screenshotTaken    = false;
    UINT         m_exitAfterFrames    = 0;
    UINT         m_logStatsEvery      = 60;

    // ---- TLAS mode (selectable; instrumentation hook) ----
    enum class TlasMode { Partitioned, Traditional };
    TlasMode m_tlasMode = TlasMode::Partitioned;
    std::unique_ptr<ITlasSystem> m_tlas;

    // ---- Static assets ----
    MeshAssets m_ball;       // hi-LOD icosphere   (subdiv=2, 320 tris)
    MeshAssets m_ballMid;    // mid-LOD icosphere  (subdiv=1, 80 tris)  -- phase 5
    MeshAssets m_ballLow;    // lo-LOD icosphere   (subdiv=0, 20 tris)  -- phase 5 LOD swap
    MeshAssets m_donut;
    // The donut flock-members live in the PTLAS GLOBAL PARTITION (phase 4).
    // Their transforms are FLOCK-LOCAL (small offsets from flock center)
    // and the global partition's translation is updated each frame to
    // place the flock cluster at flock_pos - as_origin in AS-space.
    struct FlockMember {
        DirectX::XMFLOAT3 localOffset;   // relative to flock position
        float             scale;
    };
    std::vector<FlockMember> m_donutMembers;
    static constexpr UINT kDonutCount     = 9;
    static constexpr UINT kFlockInstBase  = 0; // donut instance indices: [kFlockInstBase .. kFlockInstBase+kDonutCount)
    // Ball instances live ABOVE donuts in the PTLAS InstanceIndex range.

    // ---- Scene layout (grid of partitions; each partition holds a sub-grid
    // of balls; balls are equally spaced across the whole lattice). ----
    SceneLayout m_scene;
    std::vector<SceneInstance>      m_sceneInstances;          // world-space rest transforms (built once)
    std::vector<DirectX::XMFLOAT3>  m_ballWorldPos;            // ball REST positions (fed to RollingPartitions)
    std::vector<uint8_t>            m_ballLod;                 // per-ball current LOD (0=hi, 1=mid, 2=lo)
    std::vector<uint8_t>            m_ballDisplaced;           // per-ball: displaced THIS frame? (set in DoRender)
    bool                            m_ptlasInitialWriteDone = false;
    DirectX::XMFLOAT3               m_asOrigin = { 0, 0, 0 };  // current frame's AS-space origin
    void BuildSceneInstances();
    void RecomputeAsOrigin();
    // Compute the radial-push displacement applied at a ball's rest position
    // given the flock center.  Returns zero outside kDisplaceRadius.
    DirectX::XMFLOAT3 ComputeBallDisplacement(const DirectX::XMFLOAT3& restPos) const;

    // Displacement parameters.  Smooth radial falloff: max push at
    // distance 0, zero push beyond kDisplaceRadius.
    static constexpr float kDisplaceRadius   = 3.0f;
    static constexpr float kDisplaceMaxPush  = 0.85f;
    // Conservative AABB pad applied during WRITE_INSTANCE for a ball that
    // CAN be displaced -- the AABB encompasses the full displacement
    // envelope so subsequent UPDATE_INSTANCE/UPDATE_INSTANCE remains cheap.
    static constexpr float kDisplaceAabbPad  = kDisplaceMaxPush + 0.05f;

    // LOD thresholds.  Three bins (hi/mid/lo) selected by distance from
    // camera, with hysteresis to prevent per-frame flicker as a ball sits
    // near a threshold.  Margin = the band you have to clearly cross
    // before the LOD changes; bigger margin = fewer transitions, more
    // stable visual.
    static constexpr float kLodHiDist  = 2.5f;
    static constexpr float kLodMidDist = 5.0f;
    static constexpr float kLodMargin  = 0.5f;
    static uint8_t SelectLod(float dist, uint8_t cur)
    {
        // Hysteresis: only step DOWN (higher number = less detail) once
        // we're past the threshold by +margin; only step UP once we're
        // below by -margin.  In between, keep current.
        if (cur == 0)
        {
            if (dist > kLodHiDist  + kLodMargin) return 1;
        }
        else if (cur == 1)
        {
            if (dist < kLodHiDist  - kLodMargin) return 0;
            if (dist > kLodMidDist + kLodMargin) return 2;
        }
        else // cur == 2
        {
            if (dist < kLodMidDist - kLodMargin) return 1;
        }
        return cur;
    }
    // Initial-LOD selector ignores hysteresis (no "current" LOD yet).
    static uint8_t SelectLodInitial(float dist)
    {
        if (dist < kLodHiDist)  return 0;
        if (dist < kLodMidDist) return 1;
        return 2;
    }

    // ---- Camera / flock-follow rig.  --camera-mode orbit|flock-follow
    // selects between the static orbit camera (phase 2c) and the
    // camera-trails-flock rig (phase 3+).  In both cases m_asOrigin
    // tracks the camera so partition translations and the AS-space camera
    // CB stay consistent. ----
    enum class CameraMode { Orbit, FlockFollow };
    CameraMode  m_cameraMode = CameraMode::FlockFollow;
    FlockMotion m_flock;

    // ---- Rolling-partition manager (phase 3b+).  Budget is set via
    // --partitions; defaults to a smaller value than the cell count so
    // recycling is visible.  Phase 3c turns the budget into a runtime
    // knob (hotkeys [ / ]) that triggers a full PTLAS tear-down + rebuild
    // -- the more expensive of the two PTLAS update paths per spec. ----
    RollingPartitions m_rollingParts;
    uint32_t          m_partitionBudget = 64;
    uint32_t          m_pendingBudget   = 0;   // 0 = no pending resize
    // Phase 3c headless: --resize-at FRAME:N triggers a resize to N at
    // frame FRAME.  Multiple --resize-at flags queue multiple resizes.
    struct ScheduledResize { UINT64 frame; uint32_t newBudget; };
    std::vector<ScheduledResize> m_scheduledResizes;
    void ApplyResizeIfPending();

    // ---- RT pipeline + shader table + bindings ----
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_globalRootSig;
    Microsoft::WRL::ComPtr<ID3D12StateObject>   m_rtStateObject;
    Microsoft::WRL::ComPtr<ID3D12Resource>      m_shaderTable;
    // Shader table layout (record size = D3D12_SHADER_RECORD_BYTE_ALIGNMENT
    // aligned).  Table sections (each contiguous):
    //   * 1 raygen record   at offset m_rayGenStart
    //   * 2 miss records    at offset m_missStart    (primary, shadow)
    //   * 2 hit-group recs  at offset m_hitStart     (HitGroup_Ball, HitGroup_Donut)
    UINT64                                      m_rayGenStart            = 0;
    UINT64                                      m_rayGenSize             = 0;
    UINT64                                      m_missStart              = 0;
    UINT64                                      m_missRecordSize         = 0;
    UINT64                                      m_missTotalSize          = 0;
    UINT64                                      m_hitStart               = 0;
    UINT64                                      m_hitRecordSize          = 0;
    UINT64                                      m_hitTotalSize           = 0;

    // ---- Per-frame scene constants (3-deep upload-heap ring) ----
    Microsoft::WRL::ComPtr<ID3D12Resource>      m_sceneCb;
    UINT8*                                      m_sceneCbCpu     = nullptr;
    UINT64                                      m_sceneCbStride  = 0;

    // ---- Output UAV + descriptor heap ----
    Microsoft::WRL::ComPtr<ID3D12Resource>       m_output;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descHeap;     // CBV_SRV_UAV, shader-visible
    UINT                                          m_descSize    = 0;
    UINT                                          m_uavHeapIdx  = 0;  // index of output UAV in m_descHeap

    // ---- GPU timestamps (per-frame, ring-buffered) ----
    // 2 timestamps per slot, 1 slot per frame in flight + headroom.  CPU
    // reads from slot ~(FrameCount + 1) behind the writer so the GPU has
    // definitely completed the writes.  Resolved into the readback buffer
    // by ResolveQueryData() at end of cmd list each frame.
    static const UINT kTimestampsPerFrame = 2;          // [tlas_build_begin, tlas_build_end]
    static const UINT kTimestampSlots     = FrameCount + 2;
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> m_timestampHeap;
    Microsoft::WRL::ComPtr<ID3D12Resource>  m_timestampReadback;
    UINT64                                  m_timestampFreqHz   = 0;
    double                                  m_tlasBuildMsEma    = 0.0;
    double                                  m_tlasBuildMsLast   = 0.0;
    UINT                                    m_timestampSlotIdx  = 0;   // current write slot
    void CreateDeviceDependentResources();
    void ReleaseDeviceDependentResources();
    void CreateWindowSizeDependentResources();
    void CreateRaytracingPipeline();
    void CreateShaderTable();
    void CreateOutputUav();
    void CreateDescriptorHeap();
    void CreateSceneConstantBuffer();

    void UpdateSceneConstantBuffer();
    void DoRender();

    void CaptureBackBufferToFile(const std::wstring& path);
    HRESULT SaveRGBAToPng(const std::wstring& path, UINT width, UINT height,
                          const uint8_t* data, UINT rowPitchBytes);
};
