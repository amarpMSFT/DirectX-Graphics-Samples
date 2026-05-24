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
  the flock center.  Donuts live in the PTLAS **global partition**.
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
| `WRITE_INSTANCE`                           | Initial fill (all 5841 instances at frame 0); per-frame for rolling-window cell transitions + displaced balls + settling; full rewrite on PTLAS resize. |
| `UPDATE_INSTANCE`                          | LOD swap (BLAS pointer change with `ENABLE_EXPLICIT_AABB` -> cheap, no partition refit).  Currently gated off on frames with any WRITE work due to a preview NVIDIA driver bug -- see "Known issues" below. |
| `TRANSLATE_PARTITION`                      | Every active partition every frame (`home - as_origin`); global partition tracks the flock center (`flock_pos - as_origin`). |
| `TRANSLATE_PARTITION(GLOBAL_PARTITION)`    | Donut flock translation each frame.                       |
| Global partition                           | 9-donut flock instances.                                  |
| PTLAS resize (`InitDesc.PartitionCount` change) | Hotkey `[ ]` + `--resize-at FRAME:N` CLI.            |
| `ENABLE_EXPLICIT_AABB`                     | On every ball WRITE; AABB padded for full displacement envelope so UPDATE_INSTANCE stays cheap. |
| `BUILD_CLAS_FROM_TRIANGLES`                | One CLAS per cluster (≤ 64 tris); batched, implicit destinations. |
| `BUILD_BLAS_FROM_CLAS`                     | One Cluster BLAS per mesh, takes the CLAS-addresses buffer from the previous step. |

## Source layout

The PTLAS-specific code is deliberately small and self-contained so it
can be lifted into a blog post or another engine.

| File                              | Purpose                                                                                |
| --------------------------------- | -------------------------------------------------------------------------------------- |
| **`PtlasSystem.{h,cpp}`**         | **PTLAS centerpiece.** ~300 lines.  Full lifecycle: `Initialize`, `BeginFrame`, `WriteInstances` / `UpdateInstances` / `TranslatePartitions`, `Build`.  Builds via `ExecuteIndirectRTASOperations`. |
| `TraditionalTlasSystem.h`         | DXR1 baseline implementing the same `ITlasSystem` interface for A/B comparison.       |
| `ITlasSystem.h`                   | Abstract interface so the sample doesn't know which AS mode is active.                 |
| `RollingPartitions.h`             | Top-P cells follow the flock; partition slots recycle as the flock flies forward.  Pure CPU, no D3D12 dependency. |
| `FlockMotion.h`                   | Lissajous-style flock motion + camera-follow rig.                                      |
| `SceneLayout.h`                   | Grid dimensions + index math.                                                          |
| `MeshAssets.h`                    | Generic VB/IB + DXR1-BLAS bundle.  Optional per-vertex normal SRV and CLAS+Cluster-BLAS builders. |
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
!= 0` in the same `ExecuteIndirectRTASOperations` call.

Attempted workaround #1: gate UPDATE_INSTANCE off on frames with any
WRITE_INSTANCE.  Sample is in this state; LOD bin is still kept current
via the per-frame WRITE picking the LOD-appropriate `blasGva` at write
time (just loses the cheap UPDATE_INSTANCE optimization on those frames).

Attempted workaround #2: split each frame's PTLAS work into two
`ExecuteIndirectRTASOperations` calls (W+T in one, U in the other).
**Also TDRs at the same ~frame-8 point.**  Suggests the issue is broader
than single-call op-type mixing -- back-to-back PTLAS builds on the
same resource within one frame's CL also misbehave on this driver.
Code path stayed in place (PtlasSystem::Build clears pending ops so it
can be re-called within a frame) but the second call is currently
commented out; drop the gate and re-enable when the driver fix lands.

## Roadmap

Remaining work:

1. **Cluster template animation** — `BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES`
   once at init, per-frame `INSTANTIATE_CLUSTER_TEMPLATES` + `BUILD_BLAS_FROM_CLAS`
   for animated/deforming donut geometry, `UPDATE_INSTANCE` to swap the
   per-frame BLAS into the PTLAS.  This is the spec's canonical
   "UPDATE_INSTANCE with stable AABB" use case but currently blocked
   end-to-end by the driver bug.
2. **GPU-driven `WRITE_INSTANCE_ARGS` / `UPDATE_INSTANCE_ARGS` /
   `TRANSLATE_PARTITION_ARGS` generation** — compute shaders fill the
   indirect-arg buffers so the CPU just kicks off `ExecuteIndirectRTASOperations`.
3. **On-screen overlay** (DirectXTK12) — FPS, build_ms EMA, partition
   activity, LOD distribution, hotkey hints.
4. **WARP-friendly mode** — reduced resolution + disabled shadows/reflections
   for an interactive WARP smoke test.
