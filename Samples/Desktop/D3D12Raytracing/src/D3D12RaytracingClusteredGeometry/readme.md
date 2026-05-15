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
- **NVIDIA preview driver**: works.

### Controls (placeholder for milestone 2+)

| Key | Action                                                     |
| --- | ---------------------------------------------------------- |
| T   | Toggle templates+instantiate vs direct-CLAS rebuild        |
| C   | Toggle ClusterID-color shading vs lit                      |
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
