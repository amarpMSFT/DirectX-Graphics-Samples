# D3D12 Raytracing Partitioned TLAS sample

Demonstrates **DXR2 Partitioned Top-Level Acceleration Structures (PTLAS)**
in D3D12, with a "flock flying through a lattice of balls" scene that
exercises the realistic combinations of PTLAS operations.

> Status: **work in progress** (phases 1–4 done, see `docs/design.md`).
> Branch: `user/amarp/ptlas-sample`.

## What's rendered today (phase 4)

- **Static lattice**: 18×18×18 = 5832 ball instances (icosphere subdiv=2),
  grouped into 6×6×6 = 216 cells of 27 balls each.  Balls fixed in world space.
- **Flock-follow camera**: trails a Lissajous-style flock center as it
  gradually arcs through the lattice.  Camera position is the AS-space
  origin — partition translations track it every frame for best float
  precision where rays start.
- **9-donut flock** (torus, 28×14 segments) orbiting the flock center,
  living in the PTLAS **global partition**.
- **Rolling partition window**: configurable budget `P` (default 64) of
  PTLAS partition slots covers the `P` cells closest to the flock.  As the
  flock flies, partition slots are recycled: a slot bound to a cell that
  drops out of top-P gets reassigned to a fresh cell that just entered.
  Balls outside the active set are excluded (degenerate transform).
- **A/B against traditional DXR1 TLAS**: every screenshot reproduces under
  `--tlas-mode traditional`.

## Which PTLAS ops are exercised

| Op                          | When                                                                                    |
| --------------------------- | --------------------------------------------------------------------------------------- |
| `WRITE_INSTANCE`            | Initial full write at frame 0; per-frame for balls that cross active/inactive boundary or change partition slot; PTLAS resize triggers a full rewrite |
| `TRANSLATE_PARTITION`       | Every active partition every frame (home − as_origin); global partition gets its own translation (flock_pos − as_origin) |
| `UPDATE_INSTANCE`           | NOT YET — phase 5 will use it for LOD swap with `ENABLE_EXPLICIT_AABB`                  |
| **PTLAS resize**            | Hotkey `[ ]` cycles partition budget through 8/16/32/64/128/216, triggering a tear-down+recreate of the PTLAS.  `--resize-at FRAME:N` for headless captures. |

## Source organisation

The PTLAS-management code is deliberately tiny and self-contained so it
can be lifted into a blog post / other engine.

| File                              | Purpose                                                                 |
| --------------------------------- | ----------------------------------------------------------------------- |
| **`PtlasSystem.{h,cpp}`**         | **The centerpiece.** ~250 lines.  Full PTLAS lifecycle: `Initialize`, `BeginFrame`, `WriteInstances` / `UpdateInstances` (TODO) / `TranslatePartitions`, `Build`. |
| `TraditionalTlasSystem.h`         | DXR1 baseline implementing the same `ITlasSystem` interface; A/B comparison.                                       |
| `ITlasSystem.h`                   | Abstract interface so the sample doesn't know which AS mode it's running. |
| `RollingPartitions.h`             | "Top-P cells follow the flock; slots recycle as flock flies forward."  Pure CPU; no D3D12 dependency. |
| `FlockMotion.h`                   | Lissajous-style flock-center motion + camera-follow rig.                |
| `SceneLayout.h`                   | Grid dimensions + index math.                                           |
| `MeshAssets.h`                    | Generic VB+IB+DXR1-BLAS bundle.                                         |
| `ProceduralGeometry.h`            | Icosphere + torus generators (header-only).                             |
| `RaytracingHlslCompat.h`          | Shared CPU/HLSL types (scene CB).                                       |
| `Raytracing.hlsl`                 | Raygen + miss + closest-hit (unit-sphere normal shortcut; phase 5 will switch to per-vertex normal SRV). |
| `GpuBuffer.h`                     | CreateUploadBuffer / CreateDefaultBuffer helpers.                       |
| `PartitionedTlasSample.{h,cpp}`   | Sample glue: scene state, per-frame orchestration, screenshot capture. |
| Infrastructure                    | `DeviceResources`, `DXSample`, `Win32Application`, `SampleLog`, `StepTimer`, `HlslCompat`, `stdafx`, `ExperimentalD3D12.props` (copied from clustered sample). |

## Build

Same recipe as the sibling D3D12RaytracingClusteredGeometry sample.
Requires an experimental D3D12 install (`$DxrExperimentalRoot`, default
`d:\experimental`).

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

Options:

| Flag                          | Default          | Notes                                                |
| ----------------------------- | ---------------- | ---------------------------------------------------- |
| `--tlas-mode <m>`             | `partitioned`    | `partitioned` (DXR2) or `traditional` (DXR1).        |
| `--camera-mode <m>`           | `flock-follow`   | `flock-follow` or `orbit`.                           |
| `--grid WxHxD`                | `6x6x6`          | Partition grid.                                       |
| `--balls-per-side N`          | `3`              | N×N×N balls per cell.                                |
| `--partitions N`              | `64`             | PTLAS partition budget.  Smaller = more visible recycling. |
| `--resize-at FRAME:N`         | (none)           | Schedule a PTLAS resize at the given frame.  Repeatable. |
| `--log-stats-every N`         | `60`             | Per-frame stats line cadence; 0 disables.            |
| `--screenshot N path.png`     | (off)            | Render frame N, capture, then exit.                  |
| `--exit-after-frames N`       | (off)            | Clean quit after N frames.                           |

Hotkeys:

| Key   | Action                                                             |
| ----- | ------------------------------------------------------------------ |
| `[`   | Decrement PTLAS partition budget (8/16/32/64/128/216 ladder).  Triggers PTLAS resize. |
| `]`   | Increment PTLAS partition budget.  Triggers PTLAS resize.          |

### Headless

`_buildrun.py` builds the sln, runs the exe with sensible defaults, dumps
the SampleLog tail, and exits.  Honors the `TLAS_MODE` env var so wrapper
scripts can drive A/B sweeps:

```
TLAS_MODE=partitioned python _buildrun.py
TLAS_MODE=traditional python _buildrun.py
```

### Testing on WARP

```
d3dconfig device force-warp=true      # use WARP (CPU rasteriser, supports DXR2)
d3dconfig device force-warp=false     # use the hardware adapter
```

The sample doesn't auto-pick the adapter; toggle via `d3dconfig`.  Drop
`--grid` and `--balls-per-side` for WARP-friendly scenes.

## Measured numbers (NVIDIA RTX 4090, 5832 balls, 216 cells)

| Configuration                        | Per-frame `build_ms` (EMA) | Notes                                       |
| ------------------------------------ | -------------------------- | ------------------------------------------- |
| Partitioned, static (phase 2c, all 216 cells active) | ~0.123 ms        | Pure TRANSLATE_PARTITION, no instance churn |
| Traditional, static                  | ~0.155 ms                  | Full TLAS rebuild every frame               |
| Partitioned, rolling 32-of-216       | ~0.09 ms                   | Few cells/frame transit the active set      |
| Partitioned, rolling 128-of-216      | ~0.11–0.30 ms              | Larger active set; more partition work       |
| PTLAS resize                         | one-frame spike            | Full PTLAS tear-down + rewrite              |

## Roadmap

See `docs/design.md` for the full architectural plan.  Remaining major
items:

1. **`UPDATE_INSTANCE` for LOD swap** with `ENABLE_EXPLICIT_AABB` — exercises
   the third PTLAS op type.
2. **Per-vertex normal SRV** in closest-hit so the donut tori shade correctly
   (currently use the unit-sphere shortcut).
3. **Displacement field** — flock physically pushes balls aside as it flies through;
   balls return to rest behind the flock.  Naturally generates `WRITE_INSTANCE`
   transfers as displaced balls cross cell boundaries.
4. **Cluster BLAS** (CLAS + `BUILD_BLAS_FROM_CLAS`) replacing the DXR1 BLAS path.
5. **Cluster template instancing** per frame for displaced balls (pulsation).
6. **On-screen overlay** (DirectXTK12) — stats, hotkey hints, partition viz.
7. **GPU-driven build-arg generation** — compute shaders fill the
   `WRITE_INSTANCE_ARGS` / `UPDATE_INSTANCE_ARGS` / `TRANSLATE_PARTITION_ARGS`
   buffers, so the CPU just kicks off `ExecuteIndirectRTASOperations`.
