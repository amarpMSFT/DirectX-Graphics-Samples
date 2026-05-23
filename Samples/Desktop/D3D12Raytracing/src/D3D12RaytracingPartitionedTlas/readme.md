# D3D12 Raytracing Partitioned TLAS sample

Demonstrates **DXR2 Partitioned Top-Level Acceleration Structures (PTLAS)**
in D3D12.  See `docs/design.md` for the full layered design (subsystems,
phasing, factoring rationale).

> Status: **work in progress**.  Milestone 1 (this commit) builds, queries
> DXR2 / Clusters+PTLAS support, clears the back buffer to teal, and supports
> `--screenshot N path.png` for unattended verification.  PTLAS work lands in
> milestone 2+.

## Build

The sample requires an **experimental D3D12 build** (no public Agility SDK
NuGet supports DXR2 yet).  See `ExperimentalD3D12.props` for the expected
layout; defaults assume `<root>=d:\experimental`.

```
msbuild D3D12RaytracingPartitionedTlas.sln /p:Configuration=Debug /p:Platform=x64
```

The build copies `D3D12Core.dll` + `D3D12SDKLayers.dll` into
`bin\<cfg>\D3D12\` and `dxcompiler.dll` + `dxil.dll` + `d3d10warp.dll` next
to the EXE.

## Run

```
D3D12RaytracingPartitionedTlas.exe [-forceAdapter <ID>] [--screenshot N path.png] [--exit-after-frames N]
```

### Testing on WARP / NVIDIA

This sample doesn't pick the adapter.  Use `d3dconfig`:

```
d3dconfig device force-warp=true     # use WARP (CPU rasteriser, supports DXR2)
d3dconfig device force-warp=false    # use the hardware adapter
```

## Headless / scripted runs

`_buildrun.py` builds the sln, runs the exe with `--screenshot 5 <path>`,
prints the SampleLog tail, and exits.  `_run_interactive.py` launches with an
auto-kill timeout so a forgotten window doesn't lock the EXE for the next
rebuild.

## File map (this milestone)

| File                                | Role                                            |
| ----------------------------------- | ----------------------------------------------- |
| `Main.cpp`                          | Agility SDK exports + WinMain                   |
| `PartitionedTlasSample.{h,cpp}`     | DXSample subclass (skeleton)                    |
| `DeviceResources.{cpp,h}`           | Swap chain + WARP async-display                 |
| `DXSample.{cpp,h}`                  | Base class                                      |
| `DXSampleHelper.h`                  | ComPtr helpers, ThrowIfFailed, GpuUploadBuffer  |
| `DirectXRaytracingHelper.h`         | ShaderTable etc.                                |
| `Win32Application.{cpp,h}`          | Message pump + async-display dispatch           |
| `SampleLog.h`                       | Log-file mirror for headless runs               |
| `StepTimer.h`                       | Frame timing                                    |
| `HlslCompat.h`                      | Shared CPU/HLSL types                           |
| `stdafx.{h,cpp}`                    | Precompiled header                              |
| `ExperimentalD3D12.props`           | Build hookup for experimental D3D12 + DXC       |
| `docs/design.md`                    | Architectural plan                              |

Future milestones add (per `docs/design.md`):
* `PtlasSystem.{cpp,h}` (the centerpiece)
* `ClusterSystem.{cpp,h}` (cluster BLAS / templates)
* `SceneState.{cpp,h}`   (camera, flock, ball roster)
* `GpuMemory.{cpp,h}`    (shared frame/scratch arenas)
* `shaders/` (RT pipeline + Fill*Args CS for indirect args)
