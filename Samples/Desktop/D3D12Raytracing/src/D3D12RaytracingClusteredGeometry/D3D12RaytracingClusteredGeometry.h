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
#include "ProceduralGeometry.h"
#include "Compressed1.h"
#include "RaytracingHlslCompat.h"
#include <DirectXMath.h>

namespace GlobalRootSig {
    enum {
        OutputUAVSlot = 0,
        AccelerationStructureSlot,
        SceneCBVSlot,
        Count
    };
}

// One object in the scene. Each object becomes one Cluster BLAS in the TLAS.
struct ClusterObject
{
    ProceduralGeometry::Mesh                 mesh;            // CPU-side mesh + cluster decomp
    // Per-cluster vertex payload. For COMPRESSED1 we keep the encoded blob;
    // for FLOAT32_3 we just keep the original positions array. The two paths
    // upload different bytes and use a different VertexFormat in the CLAS build
    // args. Only one of these is populated, picked at scene-build time by
    // m_vertexMode.
    std::vector<Compressed1::EncodedCluster>             encoded;
    std::vector<std::vector<ProceduralGeometry::float3>> rawPositions;
    UINT                                     globalClusterStart = 0;
    UINT                                     clusterCount       = 0;

    // Set after BuildBlasFromClasIndirect (we use EXPLICIT_DESTINATIONS so the
    // BLAS GPU VA is known on CPU and can be baked into the TLAS instance).
    Microsoft::WRL::ComPtr<ID3D12Resource>   blasStorage;
    D3D12_GPU_VIRTUAL_ADDRESS                blasGPUVA = 0;

    // Per-object world placement (used by TLAS instance desc).
    DirectX::XMFLOAT3                        worldPos    = { 0, 0, 0 };
    float                                    worldScale  = 1.0f;
    UINT                                     instanceID  = 0;
};

class D3D12RaytracingClusteredGeometry : public DXSample
{
public:
    D3D12RaytracingClusteredGeometry(UINT width, UINT height, std::wstring name);

    // IDeviceNotify
    virtual void OnDeviceLost() override;
    virtual void OnDeviceRestored() override;

    // DXSample messages.
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

    // ---------- DXR1 + DXR2 device / cmdlist interfaces ----------
    ComPtr<ID3D12Device5>                m_dxrDevice;
    ComPtr<ID3D12GraphicsCommandList4>   m_dxrCommandList;
    ComPtr<ID3D12DeviceRaytracing2>      m_dxr2Device;
    ComPtr<ID3D12CommandListRaytracing2> m_dxr2CommandList;
    bool m_clustersAndPtlasSupported = false;

    // Vertex format for cluster builds. Toggle via --vertex-format float|compressed.
    // Default is FLOAT32_3 — currently looks cleanest. The COMPRESSED1 path
    // exists for the demo but has a bug in my encoder (visible as shifted
    // geometry / cracks at cluster boundaries); see TODO in Compressed1.h.
    enum class VertexMode { Compressed1, Float32_3 };
    VertexMode                           m_vertexMode = VertexMode::Float32_3;

    // ---------- Scene = N procedurally-generated objects ----------
    std::vector<ClusterObject>           m_objects;
    UINT                                 m_totalClusterCount = 0;

    // ---------- One big upload buffer holding all per-cluster vert+idx + the
    // concatenated BUILD_CLAS_FROM_TRIANGLES_ARGS array. Built once at init. ----------
    ComPtr<ID3D12Resource>               m_clusterInputBuffer;
    D3D12_GPU_VIRTUAL_ADDRESS            m_clasArgsArrayGPUVA = 0;
    UINT                                 m_clasArgsStride     = 0;

    // ---------- AS results / scratch / address arrays ----------
    // Single CLAS build covers ALL clusters from ALL objects (one batched op).
    ComPtr<ID3D12Resource>               m_clasResultBuffer;
    ComPtr<ID3D12Resource>               m_clasScratchBuffer;
    ComPtr<ID3D12Resource>               m_clasAddressArray;       // N_total_clusters x GVA
    ComPtr<ID3D12Resource>               m_clasSizeArray;          // N_total_clusters x UINT64

    // Single BLAS-from-CLAS build covers all BLASes (one per object).
    ComPtr<ID3D12Resource>               m_blasScratchBuffer;
    ComPtr<ID3D12Resource>               m_blasArgsBuffer;         // N_objects x BUILD_BLAS_FROM_CLAS_ARGS
    ComPtr<ID3D12Resource>               m_blasResultAddrBuffer;   // N_objects x GVA (explicit dests)

    ComPtr<ID3D12Resource>               m_tlasBuffer;
    ComPtr<ID3D12Resource>               m_tlasScratchBuffer;
    ComPtr<ID3D12Resource>               m_tlasInstanceDescs;      // upload heap, N_objects descs

    // ---------- Raytracing pipeline + shader tables ----------
    ComPtr<ID3D12StateObject>            m_dxrStateObject;
    ComPtr<ID3D12RootSignature>          m_globalRootSignature;
    ComPtr<ID3D12RootSignature>          m_localRootSignature;       // empty; required by WARP
    ComPtr<ID3D12Resource>               m_rayGenShaderTable;
    ComPtr<ID3D12Resource>               m_missShaderTable;
    ComPtr<ID3D12Resource>               m_hitGroupShaderTable;

    // ---------- Output texture + descriptor heap ----------
    ComPtr<ID3D12DescriptorHeap>         m_descriptorHeap;
    UINT                                 m_descriptorSize       = 0;
    UINT                                 m_descriptorsAllocated = 0;
    ComPtr<ID3D12Resource>               m_raytracingOutput;
    D3D12_GPU_DESCRIPTOR_HANDLE          m_raytracingOutputUAV  = {};

    // ---------- Scene constant buffer ----------
    ComPtr<ID3D12Resource>               m_sceneCB;
    SceneConstantBuffer*                 m_sceneCBMapped = nullptr;

    // ---------- Timestamp queries for AS build wall-clocks ----------
    ComPtr<ID3D12QueryHeap>              m_buildQueryHeap;
    ComPtr<ID3D12Resource>               m_buildQueryReadback;
    static const UINT                    kBuildTimestampCount = 8;
    UINT64                               m_timestampFrequency = 0;
    double                               m_clasBuildMs        = 0.0;
    double                               m_blasBuildMs        = 0.0;
    double                               m_tlasBuildMs        = 0.0;
    double                               m_totalBuildMs       = 0.0;
    UINT64                               m_totalClasBytes     = 0;
    UINT64                               m_totalBlasBytes     = 0;
    UINT                                 m_titleUpdateCounter = 0;

    // ---------- App state ----------
    StepTimer m_timer;
    double    m_animSeconds       = 0.0;          // wall-clock pan time (frame-rate independent)
    bool      m_animPaused        = false;

    // Screenshot capture (--screenshot N path.png).
    int          m_screenshotFrame     = -1;
    double       m_screenshotAtSeconds = -1.0;
    std::wstring m_screenshotPath;
    UINT         m_framesRendered = 0;
    bool         m_screenshotTaken = false;

    // ---------- Initialization helpers ----------
    void CreateDeviceDependentResources();
    void QueryDXR2Support();
    void BuildScene();                              // populates m_objects (CPU-side meshes + encoding)
    void BuildAccelerationStructures();
    void UploadClusterInputs();
    void BuildClasIndirect();
    void BuildBlasFromClasIndirect();
    void BuildTlasClassic();
    void DumpClusterStatsAsync();
    void ReadBuildTimestamps();
    void UpdateTitleBar();
    void CreateRaytracingPipelineAndShaderTables();
    void CreateDescriptorHeapAndRaytracingOutput();
    void UpdateSceneConstantBuffer();

    UINT AllocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE* outCpu);
    void DoRender();
    void CaptureBackBufferToFile(const std::wstring& path);
    static HRESULT SaveBGRAToPng(const std::wstring& path, UINT width, UINT height,
                                 const uint8_t* data, UINT rowPitchBytes);

    // Shader entry-point names (must match Raytracing.hlsl).
    static const wchar_t* c_raygenName;
    static const wchar_t* c_closestHitName;
    static const wchar_t* c_missName;
    static const wchar_t* c_hitGroupName;
};

