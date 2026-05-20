# D3D12 Raytracing Clustered Geometry sample

This sample demonstrates **DXR2 clustered geometry** in D3D12: building Cluster
Level Acceleration Structures (CLAS) and Cluster BLAS via the new GPU-driven
indirect acceleration-structure operations API, with both the static path
(`BUILD_CLAS_FROM_TRIANGLES` + compressed1 vertex format) and the animated path
(`BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES` + per-frame
`INSTANTIATE_CLUSTER_TEMPLATES`).

> Status: **work in progress**. Milestone 1 currently builds, queries DXR2
> support, and clears the back buffer to teal. Cluster build + render lands in
> milestone 2.

## Build

The sample requires an **experimental D3D12 build** (no public Agility SDK
NuGet supports DXR2 yet). The expected layout is:

```
<root>\install\windows\x64\debug\
    d3d12.lib, D3D12Core.dll, D3D12SDKLayers.dll
    include\d3d12.h, d3dx12.h, ...
<root>\packages\OSS\dxcbin_preview\bin\sdk\amd64\
    dxc.exe, dxcompiler.dll, dxil.dll      (must support lib_6_10)
```

Defaults assume `<root>=d:\experimental`. Override:

```
set DxrExperimentalRoot=d:\my\d3d12
set DxrExperimentalDxcRoot=d:\my\dxc
msbuild D3D12RaytracingClusteredGeometry.sln /p:Configuration=Debug /p:Platform=x64
```

The build copies `D3D12Core.dll` + `D3D12SDKLayers.dll` into
`bin\<cfg>\D3D12\` and `dxcompiler.dll` + `dxil.dll` next to the EXE. The
system `d3d12.dll` loads the experimental `D3D12Core.dll` via the standard
Agility SDK loader (`D3D12SDKVersion` + `D3D12SDKPath` exports in `Main.cpp`).

When a public Agility SDK NuGet shipping DXR2 is available, replace
`ExperimentalD3D12.props` with the standard NuGet-based pattern (see e.g.
`D3D12RaytracingOpacityMicromaps.vcxproj`).

## Run

```
D3D12RaytracingClusteredGeometry.exe [-forceAdapter <ID>] [--screenshot N path.png]
```

`--screenshot N path.png` renders `N` frames, dumps the back buffer to PNG,
and exits. Useful for unattended visual verification.

### Testing on WARP / NVIDIA / etc.

This sample doesn't pick the adapter. Use `d3dconfig` to switch:

```
d3dconfig device force-warp=true     # use WARP (CPU rasteriser, supports DXR2)
d3dconfig device force-warp=false    # use the system's hardware adapter
```

Currently exercised against:
- **WARP**: works.
### Controls (placeholder for milestone 2+)

### Runtime controls

On-screen overlay (right column) lists the active value for each toggle.

| Key      | Action                                                                                  |
| -------- | --------------------------------------------------------------------------------------- |
| `[T]`    | Geometry path: **clustered** ↔ **traditional** (DXR1 BLAS). Triggers static AS rebuild. |
| `[A]`    | Cluster mode: cycle CLAS alloc strategy (`implicit-dest` → `get-sizes` → `compact`).    |
|          | Trad mode: cycle BLAS alloc strategy (`implicit-dest` ↔ `compact`).                     |
| `[V]`    | Vertex format: `FLOAT32_3` ↔ `COMPRESSED1` (shared-exponent quantized).                 |
| `[F]`    | Trad mode: animated BLAS update strategy (`rebuild` ↔ `refit`).                         |
| `[R]`    | Cluster mode: per-frame static-AS rebuild (`off` → `BLAS only` → `CLAS+BLAS`).          |
| `[P]`    | Pause / resume animation.                                                               |
| `[N]`    | Workload scaling — cycle extra clones (`0` → `100` → `1,000` → `10,000` → `0`).         |
|          | Each clone gets its OWN CLAS array + BLAS (1:1 instance:BLAS), so toggling progressively |
|          | stresses the CLAS/BLAS/template paths.  Clones spiral out from the floor in a           |
|          | sunflower pattern; distance-based LOD swaps the source mesh for a lower-tessellation    |
|          | variant as radius grows, so the back rings stay cheap.                                  |
| `,` `.`  | Bounce slider — reflection + refraction depth (held +2 apart, range 0..5).              |
| `[` `]`  | Vertex precision slider — fewer / more position bits (`FLOAT32_3` truncate-bits or      |
|          | `COMPRESSED1` bits/component). Triggers full static rebuild on change.                  |

### Headless / scripted runs

`--screenshot-at <seconds> path.png --exit-after-frames <N>` renders headless,
takes a screenshot, then quits. Useful for unattended visual diffs.

`--at <seconds>:<action>` queues a runtime action that fires at the given
wall-clock time. Combine multiple `--at` for scripted state sweeps. Actions:

| Action               | Equivalent hotkey                          |
| -------------------- | ------------------------------------------ |
| `geom-clusters`      | `[T]` → clustered                          |
| `geom-traditional`   | `[T]` → traditional                        |
| `alloc-implicit`     | `[A]` in cluster mode → implicit-dest      |
| `alloc-getsizes`     | `[A]` in cluster mode → get-sizes          |
| `alloc-compact`      | `[A]` in cluster mode → compact            |
| `trad-implicit`      | `[A]` in trad mode → implicit-dest         |
| `trad-compact`       | `[A]` in trad mode → compact               |
| `anim-rebuild`       | `[F]` → rebuild                            |
| `anim-refit`         | `[F]` → refit                              |
| `rebuild-none`       | `[R]` → off                                |
| `rebuild-blas`       | `[R]` → BLAS only                          |
| `rebuild-clas-blas`  | `[R]` → CLAS+BLAS                          |
| `extra-none`         | `[N]` → 0                                  |
| `extra-100`          | `[N]` → 100                                |
| `extra-1k`           | `[N]` → 1,000                              |
| `extra-10k`          | `[N]` → 10,000                             |

### Window behavior

On HW adapters the window launches maximized; on WARP / Basic Render it stays
at the default 1280×720 since maximizing a CPU rasterizer to 4K would make
each frame take seconds. Headless runs (any of the above CLI flags) also stay
at 1280×720 so screenshot resolution is deterministic.
| B   | Pause animation                                            |
| S   | Toggle stats overlay (cluster sizes, build wallclocks)     |

## Things to note

- The static region of the scene uses the new **compressed1** vertex format
  (DXR2's recommended encoding for static cluster geometry). The animated
  region uses float32 positions because per-frame requantization would obscure
  the templates story.
- Cluster size statistics are read back via the
  `D3D12_RTAS_OPERATION_MODE_GET_SIZES` mode and via `ResultSizeArray` on
  implicit-destination builds.
- Per-frame cluster-template instantiate args are populated by a tiny compute
  shader (`IndirectArgs.hlsl`) - illustrating the spec's GPU-driven intent
  even though the demo's argument count is small enough that CPU upload would
  also work.
