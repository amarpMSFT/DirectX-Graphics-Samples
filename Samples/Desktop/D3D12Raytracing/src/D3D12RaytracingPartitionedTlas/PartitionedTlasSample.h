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
#include "BallAssets.h"
#include "SceneLayout.h"
#include "FlockMotion.h"

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
    BallAssets m_ball;

    // ---- Scene layout (grid of partitions; each partition holds a sub-grid
    // of balls; balls are equally spaced across the whole lattice). ----
    SceneLayout m_scene;
    std::vector<SceneInstance>      m_sceneInstances;          // world-space (built once)
    std::vector<DirectX::XMFLOAT3>  m_partitionHomes;          // world-space per partition
    bool                            m_ptlasInitialWriteDone = false;
    DirectX::XMFLOAT3               m_asOrigin = { 0, 0, 0 };  // current frame's AS-space origin
    void BuildSceneInstances();
    void RecomputeAsOrigin();

    // ---- Camera / flock-follow rig.  --camera-mode orbit|flock-follow
    // selects between the static orbit camera (phase 2c) and the
    // camera-trails-flock rig (phase 3+).  In both cases m_asOrigin
    // tracks the camera so partition translations and the AS-space camera
    // CB stay consistent. ----
    enum class CameraMode { Orbit, FlockFollow };
    CameraMode  m_cameraMode = CameraMode::FlockFollow;
    FlockMotion m_flock;

    // ---- RT pipeline + shader table + bindings ----
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_globalRootSig;
    Microsoft::WRL::ComPtr<ID3D12StateObject>   m_rtStateObject;
    Microsoft::WRL::ComPtr<ID3D12Resource>      m_shaderTable;
    UINT64                                      m_rayGenRecordSize       = 0;
    UINT64                                      m_missRecordStartOffset  = 0;
    UINT64                                      m_missRecordSize         = 0;
    UINT64                                      m_hitRecordStartOffset   = 0;
    UINT64                                      m_hitRecordSize          = 0;

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
