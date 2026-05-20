//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
//
//*********************************************************

// Minimal COMPRESSED1 driver-bug repro.  One cube object (6 face clusters
// of 4 verts / 2 tris each) packed into a single BUILD_BLAS_FROM_CLAS
// BLAS, one TLAS instance, primary-ray-only DXR pipeline.  See the .cpp
// header comment for the full bug description and command-line.

#pragma once

#include "DXSample.h"
#include "StepTimer.h"
#include "ProceduralGeometry.h"
#include "Compressed1.h"
#include "RaytracingHlslCompat.h"
#include <DirectXMath.h>
#include <vector>

namespace GlobalRootSig {
    enum {
        OutputUAVSlot = 0,
        AccelerationStructureSlot,
        SceneCBVSlot,
        Count
    };
}

// One object in the scene.  Each object becomes one BLAS in the TLAS.
struct ClusterObject
{
    ProceduralGeometry::Mesh                            mesh;
    // Per-cluster compressed-vertex payload.  Populated unconditionally at
    // scene-build time so the runtime --vertex-format toggle can flip
    // between FLOAT32_3 and COMPRESSED1 without re-running BuildScene.
    // The FLOAT32_3 path reads `mesh.clusters[i].positions` directly.
    std::vector<Compressed1::EncodedCluster>            encoded;

    UINT                                                globalClusterStart = 0;
    UINT                                                clusterCount       = 0;

    // EXPLICIT_DESTINATIONS BLAS storage + GVA, set by BuildBlasFromClasIndirect.
    Microsoft::WRL::ComPtr<ID3D12Resource>              blasStorage;
    D3D12_GPU_VIRTUAL_ADDRESS                           blasGPUVA = 0;

    DirectX::XMFLOAT3                                   worldPos      = { 0, 0, 0 };
    float                                               worldScale    = 1.0f;
    DirectX::XMFLOAT3                                   worldRotEuler = { 0, 0, 0 };
    UINT                                                instanceID    = 0;
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
    virtual bool ShouldMaximizeWindowOnLaunch() const override;
    virtual void OnUpdate() override;
    virtual void OnRender() override;
    virtual void OnSizeChanged(UINT width, UINT height, bool minimized) override;
    virtual void OnDestroy() override;
    virtual void OnKeyDown(UINT8 key) override;
    virtual IDXGISwapChain* GetSwapchain() override { return m_deviceResources->GetSwapChain(); }
    virtual void ParseCommandLineArgs(_In_reads_(argc) WCHAR* argv[], int argc) override;

private:
    static const UINT FrameCount = 3;

    // ---- DXR1 + DXR2 device / cmdlist interfaces ----
    ComPtr<ID3D12Device5>                m_dxrDevice;
    ComPtr<ID3D12GraphicsCommandList4>   m_dxrCommandList;
    ComPtr<ID3D12DeviceRaytracing2>      m_dxr2Device;
    ComPtr<ID3D12CommandListRaytracing2> m_dxr2CommandList;
    bool                                 m_clustersAndPtlasSupported = false;

    // ---- Vertex format toggle (--vertex-format float|compressed) ----
    enum class VertexMode { Float32_3, Compressed1 };
    VertexMode                           m_vertexMode = VertexMode::Compressed1;  // bug-repro default
    UINT                                 m_compressedBitsPerComponent = 12;       // COMPRESSED1 quantizer precision
    UINT                                 m_positionTruncateBits       = 0;        // FLOAT32_3 mantissa-truncate floor

    // ---- Scene ----
    std::vector<ClusterObject>           m_objects;
    UINT                                 m_totalClusterCount  = 0;
    UINT                                 m_totalTriangleCount = 0;

    // ---- CLAS+BLAS (all clusters batched into one BUILD_CLAS_FROM_TRIANGLES,
    //                 followed by one BUILD_BLAS_FROM_CLAS per object) ----
    ComPtr<ID3D12Resource>               m_clusterInputBuffer;     // concatenated per-cluster vert+idx
    ComPtr<ID3D12Resource>               m_clasArgsBuffer;         // BUILD_CLAS_FROM_TRIANGLES_ARGS[N], CPU-filled
    D3D12_GPU_VIRTUAL_ADDRESS            m_clasArgsArrayGPUVA = 0;
    UINT                                 m_clasArgsStride     = 0;
    ComPtr<ID3D12Resource>               m_clasResultBuffer;       // worst-case CLAS storage (implicit alloc)
    ComPtr<ID3D12Resource>               m_clasScratchBuffer;
    ComPtr<ID3D12Resource>               m_clasAddressArray;       // per-cluster CLAS GVA, filled by the build

    ComPtr<ID3D12Resource>               m_blasArgsBuffer;         // BUILD_BLAS_FROM_CLAS_ARGS[N_obj], CPU-filled
    ComPtr<ID3D12Resource>               m_blasResultAddrBuffer;   // per-object dest addresses (EXPLICIT_DESTINATIONS)
    ComPtr<ID3D12Resource>               m_blasScratchBuffer;

    // ---- TLAS ----
    ComPtr<ID3D12Resource>               m_tlasBuffer;
    ComPtr<ID3D12Resource>               m_tlasScratchBuffer;
    ComPtr<ID3D12Resource>               m_tlasInstanceDescs;

    // ---- DXR pipeline + shader tables ----
    ComPtr<ID3D12StateObject>            m_dxrStateObject;
    ComPtr<ID3D12RootSignature>          m_globalRootSignature;
    ComPtr<ID3D12RootSignature>          m_localRootSignature;     // empty (no per-shader records)
    ComPtr<ID3D12Resource>               m_rayGenShaderTable;
    ComPtr<ID3D12Resource>               m_missShaderTable;
    ComPtr<ID3D12Resource>               m_hitGroupShaderTable;
    static const wchar_t* c_raygenName;
    static const wchar_t* c_opaqueClosestHitName;
    static const wchar_t* c_missName;
    static const wchar_t* c_opaqueHitGroupName;

    // ---- DXR output texture + descriptor heap (1 UAV) ----
    ComPtr<ID3D12DescriptorHeap>         m_descriptorHeap;
    UINT                                 m_descriptorSize       = 0;
    UINT                                 m_descriptorsAllocated = 0;
    ComPtr<ID3D12Resource>               m_raytracingOutput;
    D3D12_GPU_DESCRIPTOR_HANDLE          m_raytracingOutputUAV  = {};

    // ---- Scene CB (camera basis, aspect, fov) ----
    ComPtr<ID3D12Resource>               m_sceneCB;
    SceneConstantBuffer*                 m_sceneCBMapped = nullptr;

    // ---- Animation clock (drives the orbit camera) ----
    StepTimer                            m_timer;
    double                               m_animSeconds  = 0.0;
    UINT                                 m_framesRendered = 0;

    // ---- Setup / per-frame ----
    void CreateDeviceDependentResources();
    void QueryDXR2Support();
    void BuildScene();
    void EncodeCompressedClusters();
    void BuildAccelerationStructures();
    void UploadClusterInputs();
    void BuildClasIndirect();
    void BuildClasImplicit();
    void BuildBlasFromClasIndirect();
    void BuildTlasClassic();
    void CreateRaytracingPipelineAndShaderTables();
    void CreateDescriptorHeapAndRaytracingOutput();
    void UpdateSceneConstantBuffer();
    void DoRender();
    UINT AllocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE* outCpu);
};
