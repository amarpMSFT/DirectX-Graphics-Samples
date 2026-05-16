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
#include <array>

namespace GlobalRootSig {
    enum {
        OutputUAVSlot = 0,
        AccelerationStructureSlot,
        SceneCBVSlot,
        MaterialsSRVSlot,           // Stage C: per-instance MaterialDesc[] indexed by InstanceID()
        ClusterNormalsSRVSlot,      // Stage F: StructuredBuffer<float3> g_clusterNormals  (per-vertex)
        ClusterIndicesSRVSlot,      // Stage F: StructuredBuffer<uint>   g_clusterIndices  (uint32 per index)
        ClusterOffsetsSRVSlot,      // Stage F: StructuredBuffer<uint2>  g_clusterOffsets  (per-cluster
                                    //                                                      vertOff, idxOff)
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
    DirectX::XMFLOAT3                        worldPos      = { 0, 0, 0 };
    float                                    worldScale    = 1.0f;
    // Euler rotation in radians, applied as Rx * Ry * Rz before scale +
    // translation. Used so the torus can present its donut hole to the
    // camera and the Klein bottle's figure-8 lobe stands proud instead of
    // being viewed straight-on. Defaults to identity for objects that
    // don't care about orientation.
    DirectX::XMFLOAT3                        worldRotEuler = { 0, 0, 0 };
    UINT                                     instanceID    = 0;
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

    // Read-only accessor consumed by the BuildSharedClusterTrianglesInputs
    // free function in the .cpp - keeps the helper's signature short.
    UINT PositionTruncateBits() const { return m_positionTruncateBits; }
private:
    static const UINT FrameCount = 3;

    // ---------- DXR1 + DXR2 device / cmdlist interfaces ----------
    ComPtr<ID3D12Device5>                m_dxrDevice;
    ComPtr<ID3D12GraphicsCommandList4>   m_dxrCommandList;
    ComPtr<ID3D12DeviceRaytracing2>      m_dxr2Device;
    ComPtr<ID3D12CommandListRaytracing2> m_dxr2CommandList;
    bool m_clustersAndPtlasSupported = false;

    // Vertex format for cluster builds. Toggle via --vertex-format float|compressed.
    // Default is FLOAT32_3. The COMPRESSED1 path produces correct bytes (WARP
    // renders them cleanly) but hits a current NVIDIA driver bug; see the
    // banner in BuildScene() in the .cpp.
    enum class VertexMode { Compressed1, Float32_3 };
    VertexMode                           m_vertexMode = VertexMode::Float32_3;

    // ---------- CLAS memory-allocation strategy (selectable via --clas-alloc) ----------
    // Implicit  - one IMPLICIT_DESTINATIONS build into a worst-case-sized buffer.
    //             Simplest, lowest CPU overhead, highest steady memory.
    // GetSizes  - 2-pass: first runs MODE_GET_SIZES to learn per-cluster bytes,
    //             then EXPLICIT_DESTINATIONS build into an exact-sized buffer.
    //             No over-allocation; CPU stalls on size-array readback.
    // Compact   - 1-pass IMPLICIT build (worst-case alloc) PLUS a per-cluster
    //             size readback (free side effect of the build); then
    //             MOVE_CLUSTER_OBJECTS in IMPLICIT mode compacts the live CLAS
    //             into a tightly-sized buffer. Peak GPU memory = worst-case +
    //             compacted; final GPU memory = compacted.
    enum class ClasAllocMode { Implicit, GetSizes, Compact };
    ClasAllocMode                        m_clasAllocMode = ClasAllocMode::Implicit;

    // ---------- Position-truncate bits (FLOAT32_3 mode only) ----------
    // Per-vertex float positions can have their LOW N mantissa bits zeroed
    // before the CLAS build sees them - the bits-needed savings are passed
    // to the driver via D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC's
    // MinPositionTruncateBitCount + the per-cluster
    // D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS::PositionTruncateBitCount.
    // The driver may store positions more compactly when both fields agree
    // that low bits are zero. Range 0 (no truncation) to ~22 (kills all
    // mantissa, useful only as a stress test).
    //
    // DEFAULT: 12 bits. On this scene's ~1m bounding box that leaves
    // ~244 micrometer per-vertex resolution - sub-pixel-equivalent at any
    // sane camera distance, visually identical to 0-bit, ~27% smaller CLAS
    // bytes. Pass --position-truncate 0 for the no-truncation baseline,
    // or --position-truncate 16/22 to see the artifacts kick in.
    //
    // For larger scenes you'd pick a smaller value (or normalize positions
    // first); the right knob is "what per-vertex precision do you need in
    // world units" not "how many bits".
    //
    // Ignored in COMPRESSED1 mode (the union slot is taken by
    // MaxCompressedClusterPositionsSize there).
    UINT                                 m_positionTruncateBits = 12;

    // ANTIALIASING / SUPERSAMPLING.  How many primary rays per output pixel,
    // each at a different sub-pixel jitter offset.  Set via --aa-samples N
    // (1, 2, or 4).  Default 1 keeps the sample stable on Debug builds (4x
    // jitter + recursive reflection/refraction is heavy on the CPU-side
    // shader validation; on Release a 4090 handles 4 samples comfortably).
    // Pass-through to the raygen shader is via SceneConstantBuffer.miscParams.z.
    UINT                                 m_aaSamplesPerPixel = 4;

    // CLUSTER-RAINBOW VISUALISATION KNOB.  miscParams.w in the scene CB.
    // Multiplies the cosine-palette per-cluster tint into the material
    // baseColor in the closesthit:
    //   tint = lerp(white, ClusterColor(cid), m_clusterTint)
    // Default 0.5 - clusters are clearly visible (this IS a sample about
    // clustered geometry after all) without completely overwriting the
    // material colours.  Set to 0 via --cluster-tint 0 for pure material
    // rendering, 1 for the original "cluster rainbow dominates everything"
    // look.
    float                                m_clusterTint = 0.85f;
    const wchar_t*                       ClasAllocModeName() const
    {
        switch (m_clasAllocMode)
        {
        case ClasAllocMode::Implicit: return L"implicit";
        case ClasAllocMode::GetSizes: return L"get-sizes";
        case ClasAllocMode::Compact:  return L"compact";
        }
        return L"?";
    }
    // Stats captured by BuildClasIndirect, reported via UpdateTitleBar + log.
    struct ClasMemStats
    {
        UINT64 resultPrebuildMax  = 0;  // prebuild.ResultDataMaxSizeInBytes (worst case)
        UINT64 resultInitialBytes = 0;  // what we actually allocated initially
        UINT64 resultFinalBytes   = 0;  // final live result-buffer size after this function returns
        UINT64 peakResidentBytes  = 0;  // max concurrent live result-buffer memory
        UINT64 scratchBytesPhase1 = 0;
        UINT64 scratchBytesPhase2 = 0;
        UINT64 sumActualBytes     = 0;  // sum of per-cluster sizes (= the irreducible CLAS storage)
        double cpuWallMsPhase1    = 0.0;
        double cpuWallMsPhase2    = 0.0;
    };
    ClasMemStats                         m_clasMemStats;

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
    ComPtr<ID3D12Resource>               m_clasMoveArgsBuffer;     // ClasAllocMode::Compact only - kept alive across function returns

    // Single BLAS-from-CLAS build covers all BLASes (one per object).
    ComPtr<ID3D12Resource>               m_blasScratchBuffer;
    ComPtr<ID3D12Resource>               m_blasArgsBuffer;         // N_objects x BUILD_BLAS_FROM_CLAS_ARGS
    ComPtr<ID3D12Resource>               m_blasResultAddrBuffer;   // N_objects x GVA (explicit dests)

    ComPtr<ID3D12Resource>               m_tlasBuffer;
    ComPtr<ID3D12Resource>               m_tlasScratchBuffer;
    ComPtr<ID3D12Resource>               m_tlasInstanceDescs;      // upload heap, N_objects descs

    // ============ ANIMATED OBJECT (cluster templates + per-frame instantiation) ============
    // One additional object whose Cluster BLAS is rebuilt every frame from
    // pre-built cluster templates. CPU computes new positions per frame from a
    // pulsating-sphere formula; INSTANTIATE_CLUSTER_TEMPLATES turns the
    // templates + fresh positions into CLAS; BUILD_BLAS_FROM_CLAS rebuilds the
    // BLAS storage in-place (the BLAS storage GPU VA stays fixed - the
    // animated TLAS instance always points at the same VA, only the BVH
    // contents change). TLAS is rebuilt every frame to pick up the new BLAS
    // root bounds.
    struct AnimatedObject
    {
        ProceduralGeometry::Mesh             mesh;                   // topology + rest positions
        std::vector<std::vector<DirectX::XMFLOAT3>> hintPositions;   // per-cluster, max-deformation envelope
        UINT                                 clusterCount       = 0;
        UINT                                 maxTrisPerCluster  = 0;
        UINT                                 maxVertsPerCluster = 0;
        UINT                                 totalVertexCount   = 0;
        UINT                                 vertexBufferStride = 0; // bytes per vertex (sizeof(float3))

        DirectX::XMFLOAT3                    worldPos     = {0,0,0};
        float                                worldScale   = 1.0f;
        UINT                                 instanceID   = 0;

        // --- Built once at init ---
        // Hint vertex blob + per-cluster CLUSTER_TEMPLATES_FROM_TRIANGLES args.
        Microsoft::WRL::ComPtr<ID3D12Resource> templateInputBuffer;
        D3D12_GPU_VIRTUAL_ADDRESS              templateArgsArrayGPUVA = 0;
        UINT                                   templateArgsStride     = 0;
        // Template BVH storage (one templated CLAS per source cluster).
        Microsoft::WRL::ComPtr<ID3D12Resource> templateResultBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> templateScratchBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> templateAddressArray;  // per-cluster GVA into templateResultBuffer

        // --- Reused every frame ---
        // Per-frame new positions (CPU-side computed + memcpy'd into mapped upload buffer).
        Microsoft::WRL::ComPtr<ID3D12Resource> perFrameVertexBuffer;        // upload heap, persistently mapped
        DirectX::XMFLOAT3*                     perFrameVertexBufferMapped = nullptr;
        // Per-frame INSTANTIATE_CLUSTER_TEMPLATES_ARGS array (one per cluster).
        // Each arg points to its cluster template + the slice of perFrameVertexBuffer.
        Microsoft::WRL::ComPtr<ID3D12Resource> perFrameInstArgsBuffer;      // upload heap, persistently mapped
        D3D12_RTAS_OPERATION_INSTANTIATE_CLUSTER_TEMPLATES_ARGS* perFrameInstArgsMapped = nullptr;
        // Instantiated CLAS results (rewritten each frame in IMPLICIT_DESTINATIONS mode).
        Microsoft::WRL::ComPtr<ID3D12Resource> perFrameClasResultBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> perFrameClasScratchBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> perFrameClasAddressArray;    // N_clusters x GVA
        // BLAS storage (fixed GPU VA across frames, rebuilt in place every frame).
        Microsoft::WRL::ComPtr<ID3D12Resource> blasStorage;
        Microsoft::WRL::ComPtr<ID3D12Resource> blasScratchBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> blasArgsBuffer;              // upload, mapped, 1 entry
        D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS* blasArgsMapped = nullptr;
        Microsoft::WRL::ComPtr<ID3D12Resource> blasResultAddrBuffer;        // upload, 1 entry = blasStorage VA
        D3D12_GPU_VIRTUAL_ADDRESS              blasGPUVA = 0;
    };
    AnimatedObject                          m_animatedObject;
    bool                                    m_animatedObjectEnabled = false;

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

    // ---------- Per-instance materials ----------
    // m_materials[InstanceID] = MaterialDesc.  Indexed by InstanceID() in
    // HLSL via a structured-buffer SRV at root parameter MaterialsSRVSlot.
    // Populated in BuildMaterials() and uploaded once at init time.
    std::array<MaterialDesc, NUM_MATERIAL_SLOTS> m_materials = {};
    ComPtr<ID3D12Resource>               m_materialsBuffer;
    void BuildMaterials();

    // ---------- Per-vertex normal side channel (smooth shading) ----------
    // DXR2 cluster geometry only carries positions in the CLAS vertex buffer,
    // so per-vertex normals and the cluster index buffer have to travel
    // through a SEPARATE channel for the closesthit shader to interpolate
    // them by barycentrics.  Three upload-heap StructuredBuffers + an
    // offset table:
    //
    //   m_clusterNormalsBuffer : float3[]   - all per-cluster normals
    //                                          concatenated.  Vertex offset
    //                                          per cluster comes from the
    //                                          offsets table below.
    //   m_clusterIndicesBuffer : uint[]     - all per-cluster index buffers
    //                                          concatenated.  Stored as
    //                                          uint32 per index for clean
    //                                          StructuredBuffer access; the
    //                                          source uint8 indices are
    //                                          widened on upload.
    //   m_clusterOffsetsBuffer : uint2[]    - indexed by ClusterID().  .x =
    //                                          start in normals buffer; .y
    //                                          = start in indices buffer.
    //                                          Sized to max(ClusterID)+1
    //                                          (sparse - unused slots
    //                                          contain (~0u,~0u)).
    //
    // The closesthit shader then does:
    //   uint cid = ClusterID();
    //   uint2 off = g_clusterOffsets[cid];
    //   uint i0 = g_clusterIndices[off.y + PrimitiveIndex()*3 + 0];
    //   ... interpolate g_clusterNormals[off.x + i_k] via barycentrics.
    //
    // For the animated sphere we reuse its BASE normals (computed once at
    // gen time) - the per-frame morph deformation is small enough that the
    // static normals stay visually plausible without per-frame re-upload.
    ComPtr<ID3D12Resource>               m_clusterNormalsBuffer;
    ComPtr<ID3D12Resource>               m_clusterIndicesBuffer;
    ComPtr<ID3D12Resource>               m_clusterOffsetsBuffer;
    UINT                                 m_clusterNormalsCount = 0;
    UINT                                 m_clusterIndicesCount = 0;
    UINT                                 m_clusterOffsetsCount = 0;
    void BuildClusterShaderSideBuffers();

    // ---------- Timestamp queries for AS build wall-clocks ----------
    // Init-time slots 0..5 straddle the static CLAS / BLAS / TLAS builds (3 pairs).
    // Per-frame uses a separate heap + readback with 3 ring-buffer slots so
    // the CPU reads timestamps written ~3 frames ago (safely past GPU
    // completion) without stalling.
    ComPtr<ID3D12QueryHeap>              m_buildQueryHeap;
    ComPtr<ID3D12Resource>               m_buildQueryReadback;
    ComPtr<ID3D12QueryHeap>              m_pfQueryHeap;
    ComPtr<ID3D12Resource>               m_pfQueryReadback;
    static const UINT                    kBuildTimestampCount    = 8;
    static const UINT                    kPerFrameTsPerSlot      = 4;   // 2 op pairs
    static const UINT                    kPerFrameRingSlots      = 3;
    UINT64                               m_timestampFrequency = 0;
    double                               m_clasBuildMs        = 0.0;
    double                               m_blasBuildMs        = 0.0;
    double                               m_tlasBuildMs        = 0.0;
    double                               m_totalBuildMs       = 0.0;
    // Per-frame (EMA-smoothed).
    double                               m_pfAnimRebuildMs    = 0.0;  // INSTANTIATE + BLAS rebuild
    double                               m_pfTlasRebuildMs    = 0.0;  // TLAS rebuild
    UINT                                 m_pfWriteSlot        = 0;
    UINT                                 m_pfFramesCaptured   = 0;
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
    void BuildClasIndirect();              // dispatches to one of the three below
    void BuildClasImplicit();              // ClasAllocMode::Implicit
    void BuildClasGetSizes();              // ClasAllocMode::GetSizes
    void BuildClasCompact();               // ClasAllocMode::Compact
    void BuildBlasFromClasIndirect();
    void BuildTlasClassic();
    void RebuildTlasPerFrame();
    void BuildAnimatedObjectSetup();          // generates mesh, builds templates ONCE
    void UpdateAnimatedObjectPerFrame();      // CPU positions -> INSTANTIATE -> BLAS rebuild (per frame)
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
    // Shader entry-point names (must match Raytracing.hlsl). Two ray indices:
    //   0 = primary ray  (Miss + ClusterHitGroup)
    //   1 = shadow ray   (ShadowMiss + ShadowHitGroup -- the shadow hit group's
    //                     closesthit is never invoked because shadow rays use
    //                     SKIP_CLOSEST_HIT_SHADER, but DXR still requires a
    //                     hit-group record per ray-contribution index)
    static const wchar_t* c_raygenName;
    static const wchar_t* c_closestHitName;
    static const wchar_t* c_anyHitName;
    static const wchar_t* c_missName;
    static const wchar_t* c_shadowMissName;
    static const wchar_t* c_hitGroupName;
    static const wchar_t* c_shadowHitGroupName;
};

