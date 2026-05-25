# D3D12 Raytracing Partitioned TLAS sample

Demonstrates **DXR2 Partitioned Top-Level Acceleration Structures (PTLAS)**
in D3D12 with a "flock-of-donuts flying through a lattice of balls" scene
that exercises every PTLAS operation type, the cluster-BLAS pipeline
(CLAS + BUILD_BLAS_FROM_CLAS), shadow + reflection rays, and a runtime
PTLAS resize via UI controls.

> Status: **work in progress**.  Branch: `user/amarp/ptlas-sample` on the
> author's fork.

## What's in the scene

- **Static ball lattice**: 18×18×18 = 5832 ball instances (icosphere) in
  6×6×6 = 216 cells of 27 balls each.  Balls fixed in world space.  Three
  LOD bins (subdiv 2 / 1 / 0 = 320 / 80 / 20 triangles), selected per-ball
  by distance from the camera with hysteresis to avoid per-frame flicker.
- **Camera trails a flock** along a Lissajous-style arcing path through
  the lattice.  Flock is a 9-donut ring (torus, 28×14 segments) orbiting
  the flock center.  Each donut spins around its local Y axis at a
  slightly different rate per donut.  Donut **geometry pulsates each
  frame** — minor radius oscillates ±30% — via a per-frame
  `BUILD_CLAS_FROM_TRIANGLES` → `BUILD_BLAS_FROM_CLAS` rebuild against
  freshly-uploaded vertex positions; the resulting Cluster BLAS GPUVA
  gets WRITE_INSTANCE'd into the PTLAS donut instances each frame.
  Donuts live in the PTLAS **global partition** -- the per-frame
  `TRANSLATE_PARTITION(GLOBAL_PARTITION)` moves the whole ring with the
  flock while per-frame `WRITE_INSTANCE` on each donut rotates it in
  place and swaps in the new animated BLAS.
- **Rolling partition window**: configurable budget `P` (default 64,
  hotkey `[ ]` cycles 8/16/32/64/128/216) of PTLAS partition slots cover
  the `P` cells closest to the flock.  Cells outside the active set have
  their balls excluded (degenerate transform).  As the flock flies,
  partition slots are recycled forward.
- **Displacement field**: an invisible repulsion field around the flock
  pushes nearby balls aside as it passes; balls return to rest after the
  flock moves on.
- **Shadows + reflections**:
    - Hard shadow ray from each surface toward an angled sun (skipped on
      bounce hits to stay under MaxRecursionDepth = 2).
    - Donuts cast reflection rays (Fresnel-weighted, ~45-90% reflective
      depending on grazing angle) showing the surrounding lattice.
- **Cluster-BLAS path**: CLAS (one per cluster, max 64 tris) + Cluster
  BLAS via `BUILD_BLAS_FROM_CLAS`.  Drop-in replacement for the DXR1 BLAS
  path; `--blas-mode` selects between them.
- **A/B against traditional DXR1 TLAS**: every screenshot reproduces
  under `--tlas-mode traditional`.

## DXR2 features exercised

| Feature                                    | Where in the sample                                       |
| ------------------------------------------ | --------------------------------------------------------- |
| `WRITE_INSTANCE`                           | Initial fill (all 5841 instances at frame 0); per-frame for rolling-window cell transitions + displaced balls + settling + LOD bin transitions + animated-donut BLAS swap + donut Y-axis spin; full rewrite on PTLAS resize. |
| `UPDATE_INSTANCE`                          | Currently NOT used.  Was used for LOD bin swap; promoted to WRITE_INSTANCE (which carries the BLAS GPUVA among the other args, so the same effect with no need for a separate UPDATE op) so the sample sidesteps a preview NVIDIA driver bug -- see "Known issues" below.  When the driver fix lands, LOD swap can move back to UPDATE_INSTANCE for the no-partition-refit fast path. |
| `TRANSLATE_PARTITION`                      | Every active partition every frame (`home - as_origin`); global partition tracks the flock center (`flock_pos - as_origin`). |
| `TRANSLATE_PARTITION(GLOBAL_PARTITION)`    | Donut flock translation each frame.                       |
| Global partition                           | 9-donut flock instances.                                  |
| PTLAS resize (`InitDesc.PartitionCount` change) | Hotkey `[ ]` + `--resize-at FRAME:N` CLI.            |
| `ENABLE_EXPLICIT_AABB`                     | On every ball WRITE; AABB padded for full displacement envelope so re-WRITE on LOD swap or displacement doesn't grow/shrink the partition AABB. |
| `BUILD_CLAS_FROM_TRIANGLES`                | Two callers: (1) one-shot at init per mesh for the static Cluster BLAS path; (2) per-frame for animated donut geometry (13 CLAS, ≤64 tris each). |
| `BUILD_BLAS_FROM_CLAS`                     | Same two callers: per-mesh static Cluster BLAS + per-frame animated donut Cluster BLAS. |

## Source layout

The PTLAS-specific code is deliberately small and self-contained so it
can be lifted into a blog post or another engine.

| File                              | Purpose                                                                                |
| --------------------------------- | -------------------------------------------------------------------------------------- |
| **`PtlasSystem.{h,cpp}`**         | **PTLAS centerpiece.** ~300 lines.  Full lifecycle: `Initialize`, `BeginFrame`, `WriteInstances` / `UpdateInstances` / `TranslatePartitions`, `Build`.  Builds via `ExecuteIndirectRTASOperations`. |
| **`AnimatedDonutMesh.h`**         | **Cluster-pipeline animation example.** ~340 lines.  Per-frame `BUILD_CLAS_FROM_TRIANGLES` + `BUILD_BLAS_FROM_CLAS` against an UPLOAD-heap VB whose contents are CPU-written each frame.  All result/scratch buffers reused. |
| `TraditionalTlasSystem.h`         | DXR1 baseline implementing the same `ITlasSystem` interface for A/B comparison.       |
| `ITlasSystem.h`                   | Abstract interface so the sample doesn't know which AS mode is active.                 |
| `RollingPartitions.h`             | Top-P cells follow the flock; partition slots recycle as the flock flies forward.  Pure CPU, no D3D12 dependency. |
| `FlockMotion.h`                   | Lissajous-style flock motion + camera-follow rig.                                      |
| `SceneLayout.h`                   | Grid dimensions + index math.                                                          |
| `MeshAssets.h`                    | Generic VB/IB + DXR1-BLAS + optional CLAS+Cluster-BLAS builders + optional per-vertex normal SRV. |
| `ProceduralGeometry.h`            | Icosphere + torus mesh generators with per-vertex normals (header-only).               |
| `Raytracing.hlsl`                 | Raygen + miss + ball/donut closest-hits + shadow miss.  Ball hit uses unit-sphere normal shortcut; donut hit uses per-vertex normal interpolation (cluster-demo pattern). |
| `RaytracingHlslCompat.h`          | Shared CPU/HLSL types (scene CB).                                                      |
| `GpuBuffer.h`                     | CreateUploadBuffer / CreateDefaultBuffer helpers.                                      |
| `PartitionedTlasSample.{h,cpp}`   | Sample glue: scene state, per-frame orchestration, screenshot capture.                 |
| Infrastructure                    | `DeviceResources`, `DXSample`, `Win32Application`, `SampleLog`, `StepTimer`, `HlslCompat`, `stdafx`, `ExperimentalD3D12.props` (copied from clustered sample). |

## Build

Same recipe as the sibling D3D12RaytracingClusteredGeometry sample.
Requires an experimental D3D12 install (`$DxrExperimentalRoot`, default
`d:\experimental\`).

```
msbuild D3D12RaytracingPartitionedTlas.sln /p:Configuration=Debug /p:Platform=x64
```

`ExperimentalD3D12.props` copies `D3D12Core.dll` + `D3D12SDKLayers.dll`
into `bin\<cfg>\D3D12\` and `d3d10warp.dll` + `dxcompiler.dll` + `dxil.dll`
next to the EXE.

## Run

```
D3D12RaytracingPartitionedTlas.exe [options]
```

| Flag                          | Default        | Notes                                                                  |
| ----------------------------- | -------------- | ---------------------------------------------------------------------- |
| `--tlas-mode <m>`             | `partitioned`  | `partitioned` (DXR2) or `traditional` (DXR1).                          |
| `--blas-mode <m>`             | `cluster`      | `cluster` (CLAS + BUILD_BLAS_FROM_CLAS) or `dxr1` (classic BLAS).      |
| `--animate-donuts on\|off`    | `on`           | When `on`, donut geometry pulsates via per-frame CLAS+Cluster-BLAS rebuild. |
| `--camera-mode <m>`           | `flock-follow` | `flock-follow` or `orbit`.                                             |
| `--grid WxHxD`                | `6x6x6`        | Partition grid.                                                        |
| `--balls-per-side N`          | `3`            | N×N×N balls per cell.                                                  |
| `--partitions N`              | `64`           | PTLAS partition budget.  Smaller = more visible recycling.             |
| `--resize-at FRAME:N`         | (none)         | Schedule a PTLAS resize at the given frame.  Repeatable.               |
| `--log-stats-every N`         | `60`           | Per-frame stats line cadence; 0 disables.                              |
| `--screenshot N path.png`     | (off)          | Render frame N, capture, then exit.                                    |
| `--exit-after-frames N`       | (off)          | Clean quit after N frames.                                             |

Hotkeys:

| Key   | Action                                                             |
| ----- | ------------------------------------------------------------------ |
| `[`   | Decrement PTLAS partition budget along the 8/16/32/64/128/216 ladder.  Triggers PTLAS resize. |
| `]`   | Increment PTLAS partition budget.  Triggers PTLAS resize.          |

### Headless

`_buildrun.py` builds the .sln, runs the EXE with sensible defaults,
dumps the SampleLog tail, and exits.  Honors the `TLAS_MODE` env var so
wrapper scripts can drive A/B sweeps:

```
TLAS_MODE=partitioned python _buildrun.py
TLAS_MODE=traditional python _buildrun.py
```

### Testing on WARP

```
d3dconfig device force-warp=true     # add d:\samples2\ to apps via d3dconfig apps add ...
d3dconfig device force-warp=false    # restore hardware
```

WARP correctness was verified end-to-end via the **stripped repro**
(see `../PtlasWriteUpdateRepro/`) -- the full sample at 1280×720 runs but
is too slow for an interactive smoke test (~30 sec/frame on WARP due to
shadow + reflection rays).  WARP-friendly mode (reduced resolution +
disabled shadows/reflections) is a future polish item.

## Measured numbers (NVIDIA RTX 4090, partitioned mode, 5832 balls)

| Configuration                                  | `build_ms` (EMA) |
| ---------------------------------------------- | ---------------- |
| Cluster BLAS + ENABLE_EXPLICIT_AABB, rolling 64-of-216, steady state | ~0.12 ms |
| Cluster BLAS + EXPLICIT_AABB, rolling 32-of-216                       | ~0.09 ms |
| Cluster BLAS + EXPLICIT_AABB, rolling 128-of-216                      | ~0.11–0.30 ms |
| Traditional DXR1 TLAS                                                  | ~0.155 ms (no rolling) |
| PTLAS resize                                                           | one-frame spike (~ tens of ms) |

## Known issues

**WRITE_INSTANCE + UPDATE_INSTANCE same call TDRs the GPU on the preview
NVIDIA driver (2026-05).**  Verified spec-compliant via the
`IndirectBuild.cpp` conformance test (which exercises mixed W+U on
disjoint instances by default) and via WARP (Microsoft Basic Render
Driver) which runs the same code with no TDR.  The minimal isolated
repro is in the sibling branch `user/amarp/ptlas-write-update-repro` --
demonstrates that `TDR iff WRITE.InstanceIndex != 0 AND UPDATE.InstanceIndex
!= 0` in the same `ExecuteIndirectRTASOperations` call.  A two-call
split workaround (W+T in one ExecuteIndirect, U in the next) also
TDR'd on the same driver, suggesting the issue is broader than
single-call op-type mixing.

**Workaround in this sample:** stop emitting `UPDATE_INSTANCE` entirely.
LOD bin swap (the only existing UPDATE_INSTANCE user) was promoted to
`WRITE_INSTANCE`, which carries the BLAS GPUVA among the other args, so
the same effect with no need for a separate UPDATE op.  Animated donut
BLAS swap (new in milestone 11) similarly uses per-frame
`WRITE_INSTANCE` to swap the freshly-built Cluster BLAS into the donut
PTLAS instances.

Cost vs UPDATE_INSTANCE: `WRITE_INSTANCE` triggers a partition refit
(rebuild of the partition's internal AS tree) while UPDATE_INSTANCE
with `ENABLE_EXPLICIT_AABB` does not.  For LOD swap the cost is
negligible (rare per frame; partitions are small).  For the animated
donut every frame, the global partition's refit is part of the per-frame
PTLAS build; build_ms goes from ~0.12ms (no animation) to ~0.20ms
(animation on, 13 CLAS rebuilt per frame).

When the driver fix ships, both code paths can move back to
`UPDATE_INSTANCE` for the cheap-update fast path.

## Roadmap

Remaining work:

1. **Cluster TEMPLATES** (`BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES` +
   `INSTANTIATE_CLUSTER_TEMPLATES`) -- this sample currently does the
   simpler per-frame CLAS-from-triangles rebuild for the animated donut.
   Templates would let the per-frame work skip topology setup, only
   touching the new vertex data.
2. **GPU-driven `WRITE_INSTANCE_ARGS` / `TRANSLATE_PARTITION_ARGS`
   generation** -- compute shaders fill the indirect-arg buffers so the
   CPU just kicks off `ExecuteIndirectRTASOperations`.
3. **On-screen overlay** (DirectXTK12) -- FPS, build_ms EMA, partition
   activity, LOD distribution, hotkey hints.
4. **WARP-friendly mode** -- reduced resolution + disabled shadows/
   reflections for an interactive WARP smoke test.
