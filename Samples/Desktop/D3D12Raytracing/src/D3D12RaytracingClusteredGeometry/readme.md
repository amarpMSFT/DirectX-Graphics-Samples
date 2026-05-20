# D3D12 DXR2 COMPRESSED1 cluster-geometry min repro

Single-file repro for the bug where `D3D12_VERTEX_FORMAT_COMPRESSED1` on
the DXR2 cluster build path silently drops 5 of 6 cube faces from BVH
traversal.

## Build

Requires an experimental D3D12 build that supports DXR2 (clustered
geometry + `lib_6_10`). `ExperimentalD3D12.props` hooks it up; the
defaults expect:

```
<root>\install\windows\x64\debug\         (d3d12.lib, D3D12Core.dll, SDKLayers, headers)
<root>\packages\OSS\dxcbin_preview\bin\sdk\amd64\
                                          (dxc.exe, dxcompiler.dll, dxil.dll)
```

where `<root>` defaults to `d:\experimental`. Override with
`/p:DxrExperimentalRoot=<path>` or the env var of the same name.

```
msbuild D3D12RaytracingClusteredGeometry.sln /p:Configuration=Debug /p:Platform=x64
```

## Run

```
D3D12RaytracingClusteredGeometry.exe                       (default - BUG: 1 face)
D3D12RaytracingClusteredGeometry.exe --vertex-format float  (correct - full cube)
```

Press `Esc` to quit.

The only difference between the two runs is the `VertexFormat` field on
the `BUILD_CLAS_FROM_TRIANGLES` op (`D3D12_VERTEX_FORMAT_COMPRESSED1` vs
`D3D12_VERTEX_FORMAT_FLOAT32_3`). Same scene, same camera, same TLAS, same
6 clusters / 12 triangles. In COMPRESSED1 mode only one face survives
traversal.

## Files

- `Main.cpp`                     - everything (window, device, scene, COMPRESSED1 encoder, DXR2 pipeline)
- `ExperimentalD3D12.props`      - hookup to the local experimental D3D12 build
- `D3D12RaytracingClusteredGeometry.{vcxproj,sln}`
