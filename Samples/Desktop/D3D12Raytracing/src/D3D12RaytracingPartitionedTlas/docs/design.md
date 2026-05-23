# D3D12 Raytracing Partitioned TLAS sample — design

Living design document. **The PTLAS-management code is the centerpiece**; everything
else exists to give it a meaningful workload to exercise. Layered architecture so a
reader can lift `PtlasSystem` out for their own engine without dragging in the scene
demo.

Companion read: `d:\docs\d3d\Raytracing2.md` (PTLAS spec) and
`d:\experimental\src\d3d12conf\raytracing\IndirectBuild.cpp` (conformance-test
patterns).

## Goal recap

Exercise **all** PTLAS operations on a non-trivial scene:

- WRITE_INSTANCE   — initial fill + partition transfers
- UPDATE_INSTANCE  — per-frame BLAS pointer swap (LOD changes, cluster-template
                     re-instantiation for animated balls; explicit AABB enabled)
- TRANSLATE_PARTITION — at least one partition gets a per-frame translation
                     to demonstrate the operation (and to motivate the
                     ENABLE_PARTITION_TRANSLATION flag in inputs)

Done **GPU-side**. CPU does setup once, then per frame just kicks compute +
ExecuteIndirectRTASOperations.

## Scene

- **Volumetric grid of partitions.** Start at ~12×8×12 regions (small enough for
  WARP-friendly first run; tunable). Each region is a non-global PTLAS partition.
- **Balls in regions.** NxNxN balls per region (N=2 initial) with equal spacing
  across region boundaries so the global lattice looks continuous. Each ball is
  cluster BLAS with 3 LODs.
- **Flock of donuts** in the **global partition**. Camera trails behind. Donuts
  also cluster BLAS.
- **Displacement field.** Invisible sphere of influence around flock center
  pushes nearby balls aside (smooth radial falloff, clipped at radius_max).
  Displaced balls *pulsate* their geometry in proportion to displacement
  magnitude (driven by cluster-template instantiation per frame, so each frame
  produces a fresh CLAS+CBLAS for that ball). Balls displaced enough to cross
  a region boundary get **WRITE_INSTANCE**'d into the new partition.
- **Camera follows flock** automatically; a debug free-cam toggle later.

## Camera-anchored coordinate system (drives TRANSLATE_PARTITION usage)

The sample uses partition translation as the **primary** mechanism for keeping
acceleration-structure-side world coordinates near origin while the camera flies
through the grid. This both motivates `ENABLE_PARTITION_TRANSLATION` honestly
(matches the spec's stated use case in `Partition translation`) and naturally
exercises `TRANSLATE_PARTITION` on every partition every frame.

Convention:

- Every partition has a constant **`home_world_pos`** (grid cell center for
  spatial partitions; flock center for the global partition).
- Each instance's stored `Transform` is **partition-local** — i.e. ball offset
  from its partition's `home_world_pos`. These local coordinates stay within a
  single cell's extents, so the floats used at WRITE_INSTANCE time are small.
- Per frame we pick an **`as_origin`** = (optionally quantized) camera position.
- Per frame we set `partition_translation = home_world_pos − as_origin` for
  every partition, via a single TRANSLATE_PARTITION op containing
  `partition_count + 1` arg entries (one per spatial partition + the global one).
  Args are produced by a tiny CS reading `home_world_pos` and `as_origin`.
- Ray generation shader subtracts `as_origin` from the camera ray origin so
  traversal happens in AS-origin-centered coordinates. World positions seen by
  the closest-hit are `as_origin + hit_pos_in_as`.

Consequences this design wants the reader to see:

- TRANSLATE_PARTITION is exercised across the **entire** partition set every
  frame — the demo doesn't have to invent a contrived single-partition wobble.
- WRITE_INSTANCE transforms are always small. When a ball transfers across a
  partition boundary, its new stored transform = (old_world_pos − new_partition_home),
  computed in the partition-transfer CS without ever needing large world coords.
- Displacement field acts in world space conceptually but the stored result is
  always converted to the local frame of whichever partition the ball ends up in.
- The global partition gets the same treatment — its translation tracks the flock
  center relative to `as_origin`, keeping flock-instance transforms small even as
  the flock crosses the world.

Quantization knob (later refinement, not initial milestone): snap `as_origin` to
a grid (e.g. partition step) so TRANSLATE_PARTITION only fires when the camera
crosses a snap boundary. Demo first uses per-frame updates because the
demonstration value of seeing the op fire every frame outweighs the
micro-efficiency of throttling.

## Subsystems & factoring

```
Sample/App
  PartitionedTlasSample (DXSample)        — wiring, render loop, command-list driver
  SceneState                              — camera, time, flock path, ball roster
  InputHandling                           — CLI + hotkeys
  OverlayUI                               — text overlay (debug stats, partition counts)

Cluster layer
  ClusterAssets                           — procedural ball/donut + LOD generation
  ClusterSystem                           — CLAS / cluster templates / BLAS management
                                            (static balls one-shot,
                                             animated displaced balls + flock per-frame)

PTLAS layer
  PtlasSystem                             — the centerpiece; resource + ops + build
                                            scheduling; SCENE-AGNOSTIC API surface

Memory layer
  GpuMemory                               — shared persistent / per-frame / scratch arenas
                                            cluster + ptlas + scene share heaps here
                                            (GPUVA-into-big-resource style, like the
                                             clustered sample's trad-VB/IB pool)

Infrastructure (straight copy from clustered sample)
  Main.cpp           Agility-SDK exports
  DeviceResources    swap chain + async-display for WARP
  DXSample           base class
  Win32Application   message pump, async-display dispatch
  DXSampleHelper     ComPtr helpers, throwifFailed, GpuUploadBuffer
  DirectXRaytracingHelper  ShaderTable
  SampleLog          log-file mirror for headless runs
  StepTimer          frame timing
  HlslCompat         shared CPU/HLSL types
  stdafx.h/cpp       precompiled header
  ExperimentalD3D12.props   experimental D3D12 + DXC build hookup
```

## PtlasSystem API (target shape — refined as I write it)

```cpp
class PtlasSystem
{
public:
    struct InitDesc {
        UINT  instanceCount;
        UINT  partitionCount;
        UINT  maxInstancesPerPartition;
        UINT  maxInstancesInGlobalPartition;
        D3D12_RTAS_PARTITIONED_TLAS_FLAGS flags;   // FAST_TRACE / FAST_BUILD / NONE + ENABLE_PARTITION_TRANSLATION
    };

    // CPU-side init: queries prebuild sizes, allocates PTLAS storage + scratch
    // (memory comes from GpuMemory; PtlasSystem does NOT call CreateCommittedResource directly).
    void Initialize(ID3D12Device14* device, GpuMemory& mem, const InitDesc&);

    // Per-frame: caller submits GPU-side op intents.  All args are GPU-driven —
    // each operation is a (count GVA, args GVA, stride) tuple.  The actual op-header
    // array is assembled by the system and lives in transient GPU memory.

    void BeginFrame(GpuMemory& mem);

    void SubmitWriteInstances    (D3D12_GPU_VIRTUAL_ADDRESS countGva,
                                  D3D12_GPU_VIRTUAL_ADDRESS argsGva,
                                  UINT maxCount);  // upper bound for prebuild compatibility

    void SubmitUpdateInstances   (D3D12_GPU_VIRTUAL_ADDRESS countGva,
                                  D3D12_GPU_VIRTUAL_ADDRESS argsGva,
                                  UINT maxCount);

    void SubmitTranslatePartitions(D3D12_GPU_VIRTUAL_ADDRESS countGva,
                                   D3D12_GPU_VIRTUAL_ADDRESS argsGva,
                                   UINT maxCount);

    // Emits the ExecuteIndirectRTASOperations call.  Decides incremental-vs-first
    // build via internal "have built once" flag.
    void Build(ID3D12CommandListRaytracing2* cmdList);

    D3D12_GPU_VIRTUAL_ADDRESS GpuVA() const;   // for ShaderResource bind on TLAS slot
};
```

Notes:
- Initial WRITE_INSTANCE for every ball happens once at OnInit (filled by a
  one-shot compute pass that reads the scene's static roster).
- Per-frame: an "ownership analysis" CS scans the ball roster, classifies each
  ball as static / displaced / transferred / flock, and bins it into the
  appropriate args buffer with an InterlockedAdd count. Output: 3 typed args
  buffers + 3 counts. Those go straight into the PTLAS ops.
- TRANSLATE_PARTITION exercised by translating one chosen partition by a small
  oscillating offset each frame, OR by re-centering all partitions periodically
  (the spec's intended use). Either way, exercises the op.

## ClusterSystem responsibilities

- **Static balls (3 LODs)**: build cluster templates once. Each ball instance's
  BLAS is a BUILD_BLAS_FROM_CLAS over the right LOD's instantiated CLAS. LOD
  swap = UPDATE_INSTANCE on the PTLAS pointing at the alternate-LOD BLAS.
  Cluster templates pre-instantiated per LOD; BLAS allocated per-LOD in a pool;
  each ball's per-LOD BLAS GVA is known up-front.

- **Animated balls (displaced)**: per-frame INSTANTIATE_CLUSTER_TEMPLATES with
  vertex offsets (driven by displacement-aware AnimateBall.hlsl-style CS) →
  BUILD_BLAS_FROM_CLAS into a per-ball-per-frame BLAS slot. PTLAS instance
  gets ENABLE_EXPLICIT_AABB set big enough to cover all pulsation amplitudes,
  so per-frame UPDATE_INSTANCE keeps it cheap.

- **Flock donuts (global partition)**: same template-instantiate + BLAS pattern
  as displaced balls; written into the global partition. Once flock stabilizes
  spatially (NOT in the demo — they fly continuously), the spec hint is to
  promote them to a spatial partition. We won't bother; they stay global.

## GpuMemory responsibilities

- Big persistent heap pool: cluster templates, static-LOD CLAS, static-LOD BLAS,
  ball/partition records.
- Per-frame transient arena (ring of 3, one per frame in flight): LOD choices,
  partition transfer lists, PTLAS op headers, all the indirect-arg buffers.
- Scratch arena: PTLAS build scratch, BLAS-from-CLAS scratch, INSTANTIATE
  scratch, compute scratch.

Allocator handed to both ClusterSystem and PtlasSystem so they share heaps but
don't see each other's slot internals.

## CLI surface (mirrors clustered sample's style)

- `--screenshot N path.png` — render N frames, capture, exit (headless workflow)
- `--screenshot-at SECONDS path.png` — jump animation clock to time, settle, capture
- `--exit-after-frames N` — clean exit (headless)
- `-forceAdapter ID`
- `--scene-grid WxHxD` — partition grid dims
- `--balls-per-region N` — N^3 balls per partition
- `--log-pf-every N` — per-frame timing dumps
- Hotkeys: `[P]` pause, `[C]` toggle free-cam, `[B]` toggle debug partition viz,
  `[L]` toggle LOD viz, `[T]` cycle translate-partition demo mode, …

## Phasing (re-affirming)

1. **Skeleton**: builds, queries DXR2 caps, clears teal, exits via --screenshot.
2. **Static scene**: balls in grid + initial PTLAS WRITE_INSTANCE pass. Trace.
3. **GPU LOD**: distance-based LOD with PTLAS UPDATE_INSTANCE.
4. **Flock + global partition**.
5. **Displacement field + ball partition transfers + animated cluster templates**.
6. **TRANSLATE_PARTITION exercise**.
7. **Polish + debug viz**.

## WARP / hardware notes

- d3dconfig present at `c:\windows\system32\d3dconfig.exe`. Switch via
  `d3dconfig device force-warp=true|false`. Currently false (NVIDIA driver).
- WARP needs `d3d10warp.dll` copied to the EXE dir (clustered sample's
  ExperimentalD3D12.props handles this; mirror the rule).
- WARP path: reduce res + scene complexity. Stay at default 1280×720,
  small grid (4×4×4) for first WARP runs.

## Open questions / things to revisit

- PTLAS resize: do we attempt growable PTLAS or pre-allocate worst case? Start
  pre-allocated worst-case; resizing is a later milestone per user.
- TRANSLATE_PARTITION semantics in the demo: best motivation would be re-centering
  to flock position to avoid float-precision loss far from origin. Defer until
  scene is moving.
- Materials/shading: start with normal-derived color so visual debugging works
  without much shading code. PBR/reflections later if useful.

## TLAS comparison toggle (Vanilla vs Partitioned)

Per user request: support a runtime/CLI toggle that switches the entire
scene's top-level acceleration structure from PTLAS to a **traditional DXR1
TLAS** built each frame via `BuildRaytracingAccelerationStructure()`. The
scene, BLAS set, materials, and shaders all stay the same — only the
top-level structure swaps.

- `--tlas-mode partitioned|traditional` CLI flag (initial value).
- `[M]` runtime hotkey to cycle modes; triggers a clean tear-down + rebuild
  of the active top-level structure (the OTHER mode's resources are released
  to keep memory honest in comparison).
- The shader's `g_tlas : register(t0)` slot binding is unchanged across
  modes; the SRV just points at whichever top-level structure is active.
- A `TlasSystem` interface wraps both:
  ```cpp
  class ITlasSystem {
  public:
      virtual void Initialize(...)             = 0;
      virtual void BeginFrame(...)             = 0;
      virtual void UpdateInstances(...)        = 0;  // per-frame instance changes
      virtual void Build(cmdList)              = 0;
      virtual D3D12_GPU_VIRTUAL_ADDRESS Gva()  const = 0;
      virtual const wchar_t* ModeName()        const = 0;  // for overlay
  };
  ```
  `PtlasSystem` and `TraditionalTlasSystem` both implement this. The sample
  holds a `std::unique_ptr<ITlasSystem>` that swaps on mode-toggle.

Why this matters: it lets the same demo answer the "why PTLAS?" question
visually — the user can see build-time and instance-update behavior side-by-
side. Also acts as a sanity check on my BLAS setup (the per-frame
traditional TLAS rebuild stresses the BLAS GVAs the same way PTLAS will, so
if one path works the other should too).

## Instrumentation

Borrowing the clustered-geometry sample's design wholesale, because the
patterns are already field-proven for these workloads.

GPU timestamps (via `ID3D12QueryHeap`/`D3D12_QUERY_TYPE_TIMESTAMP`),
captured around named "passes":

- `clas_build`           (when cluster path active)
- `cluster_template_inst` (per-frame animated balls + flock)
- `blas_from_clas`       (per-frame for animated)
- `tlas_build`           (traditional path)
- `ptlas_build`          (partitioned path)
- `ptlas_write_args_cs`  (per-frame WRITE_INSTANCE arg-fill CS)
- `ptlas_update_args_cs` (per-frame UPDATE_INSTANCE arg-fill CS)
- `ptlas_translate_args_cs` (per-frame TRANSLATE_PARTITION arg-fill CS)
- `dispatch_rays`

EMA-smoothed `m_pf*Ms` doubles for the overlay; raw last-frame values for
`--log-raw-every`. CPU wall-clock side-by-side via `std::chrono::steady_clock`
for the frame-time / FPS overlay line.

CPU-side counts logged per frame (or per mode-change):
- partition counts: total, balls displaced this frame, balls transferred this
  frame across partition boundaries, flock instance count
- LOD distribution: count per LOD level
- AS-mode active (`partitioned` / `traditional`)
- PTLAS sizing: `InstanceCount`, `PartitionCount`, `MaxInstancePerPartitionCount`,
  `MaxInstanceInGlobalPartitionCount`, result size, scratch size
- BLAS pool occupancy: bytes used vs allocated

CLI / scheduled-action surface (mirrors clustered sample):
- `--log-pf-every N`           dump EMA per-frame stats every N frames
- `--log-raw-every N`          dump raw last-frame timings every N frames
- `--bench-seconds N`          run for N seconds, then write `--bench-out` and exit
- `--bench-out PATH.json`      JSON dump (one record per measurement)
- `--at FRAME:ACTION`          scheduled-action queue (one fires once at FRAME)
  - `tlas-partitioned`, `tlas-traditional` — switch AS mode
  - `lod-static`, `lod-camera`               — freeze LOD vs camera-distance
  - `flock-pause`, `flock-resume`            — freeze the flock motion
  - `log`                                    — emit a snapshot line
  - `exit`                                   — clean quit
- `--exit-after-frames N`      already wired in milestone 1

On-screen overlay (when DirectXTK12 is added back in, milestone 4+):
- FPS / ms-per-frame
- AS mode + scene size knobs
- Per-pass timings table
- Partition activity meter (transfers/frame, displaced/frame)
- Memory totals
- Active hotkey hints

Keyboard hotkeys (interactive runs):
- `[P]` pause animation
- `[M]` cycle TLAS mode (partitioned / traditional)
- `[C]` free-cam vs flock-follow camera
- `[L]` LOD viz overlay
- `[B]` partition-color viz overlay
- `[T]` cycle TRANSLATE_PARTITION demo: off / per-frame-track-camera /
        snap-on-grid-step
- `,` / `.` adjust scene density (balls per region)
- `[` / `]` adjust partition grid dimensions

Phase wisely: not all of this in milestone 2a. The instrumentation belt
fleshes out as features land. What ships per milestone:
- 2a: SampleLog wiring of every state change + a per-frame timing
  scaffold; bench/sweep CLI flags reserved (no-op if unsupported).
- 2b: GPU timestamp pipeline for the few passes that exist
  (`tlas_build`/`ptlas_build`, `dispatch_rays`). TLAS toggle wired.
- 3+: per-pass timings as each pass lands; scheduled actions; bench mode;
  overlay.
