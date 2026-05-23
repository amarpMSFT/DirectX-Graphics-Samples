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

#include <string>

// PartitionedTlasSample (Milestone 1 SKELETON)
// ---------------------------------------------------------------------------
// At this stage the sample only:
//   * Initialises a D3D12 device + swap chain (via DeviceResources)
//   * Queries DXR2 / Clusters+PTLAS support (logs result, does NOT fail if
//     unsupported -- so we can verify build/run on machines without DXR2)
//   * Clears the back buffer to a unique teal so screenshots are
//     unambiguously THIS sample
//   * Honours --screenshot N path and --exit-after-frames N for unattended
//     headless verification (mirrors the clustered-geometry sample's CLI)
//
// Future milestones layer in PtlasSystem / ClusterSystem / SceneState /
// GpuMemory and the actual PTLAS work. See docs/design.md for the plan.
class PartitionedTlasSample : public DXSample
{
public:
    PartitionedTlasSample(UINT width, UINT height, std::wstring name);

    // IDeviceNotify (from DXSample base)
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

    // Device + DXR2 surfaces (DXR2 is queried but its features are not yet
    // exercised; the dxr2 pointers stay null until later milestones).
    Microsoft::WRL::ComPtr<ID3D12Device5>                m_dxrDevice;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4>   m_dxrCommandList;
    Microsoft::WRL::ComPtr<ID3D12DeviceRaytracing2>      m_dxr2Device;
    Microsoft::WRL::ComPtr<ID3D12CommandListRaytracing2> m_dxr2CommandList;
    bool   m_clustersAndPtlasSupported = false;
    UINT64 m_framesRendered            = 0;

    StepTimer m_timer;

    // ---- Headless / screenshot CLI state -------------------------------
    int          m_screenshotFrame    = -1;   // --screenshot N path
    std::wstring m_screenshotPath;
    bool         m_screenshotTaken    = false;
    UINT         m_exitAfterFrames    = 0;    // --exit-after-frames N (0 = never)

    // ---- Helpers (defined in .cpp) ------------------------------------
    void CreateDeviceDependentResources();
    void ReleaseDeviceDependentResources();
    void CaptureBackBufferToFile(const std::wstring& path);
    HRESULT SaveBGRAToPng(const std::wstring& path, UINT width, UINT height,
                          const uint8_t* data, UINT rowPitchBytes);
};
