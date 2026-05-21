# D3D12 DXR2 COMPRESSED1 cluster-geometry min repro

Single-file repro for a bug in the NVIDIA preview driver where
`BUILD_CLAS_FROM_TRIANGLES` with `D3D12_VERTEX_FORMAT_COMPRESSED1` silently
drops all but the first cluster's geometry when
`ClusterLimits.MaxVertexCountPerCluster < 32`.

## TL;DR

```
default                          -> NVIDIA HW: 1 of 6 cube faces renders   (BUG)
default                          -> WARP:      full cube renders            (correct)
--vertex-format float            -> NVIDIA HW: full cube renders            (correct)
--fix                            -> NVIDIA HW: full cube renders            (correct)
                                    (sets MaxVertexCountPerCluster = 32)
```

## The one-line fix

```cpp
D3D12_RTAS_CLUSTER_LIMITS limits = {};
limits.MaxArgCount                = N;
limits.MaxTriangleCountPerCluster = 2;        // tight (actual)
limits.MaxVertexCountPerCluster   = 32;       // <- MUST be >= 32 on NVIDIA preview driver
limits.MaxTotalTriangleCount      = 2 * N;    // tight (actual)
limits.MaxTotalVertexCount        = 4 * N;    // tight (actual)
```

The spec allows any value `>= actual` (up to the spec max of 256). FLOAT32_3
works at any value; COMPRESSED1 on the NVIDIA preview driver requires `>= 32`.

`MaxTriangleCountPerCluster`, `MaxTotalTriangleCount`, `MaxTotalVertexCount`
are all irrelevant - only `MaxVertexCountPerCluster` matters.

## Threshold bisection (NVIDIA RTX 4090, preview driver, COMPRESSED1)

| `MaxVertexCountPerCluster` | Render |
|---:|---|
| 4 (= actual) ... 10 | broken (only cluster 0 renders) |
| 11 ... 16 | partial + **geometrically distorted** (extra triangles spilling beyond cube silhouette) |
| 17 ... 20 | broken again (1 face) |
| 24 ... 31 | partial + distorted |
| **32** | **✓ full cube** |
| 33 | partial (regresses!) |
| 64 / 128 / 256 | ✓ full cube |

The non-monotonic, 32-aligned pattern suggests the driver computes a per-
cluster vertex-storage stride as roughly
`roundUpToMultipleOf(MaxVertexCountPerCluster, 32)` - 32 being NVIDIA's
warp size. When the value doesn't round to a 32-multiple, adjacent
clusters' storage overlaps or is read from wrong slots, leaving only the
first cluster's CLAS with traversable geometry.

## Why the conformance test doesn't catch it

`d3d12conf/raytracing/IndirectBuild.cpp:4858` sets
`MaxVertexCountPerCluster = 256` unconditionally, regardless of actual
cluster sizes. The test never exercises a tight value with multiple
COMPRESSED1 clusters, so the buggy code path is never reached. If the test
were modified to set tight `Max*` limits (which the spec allows), it would
fail on this driver.

## Build

Requires an experimental D3D12 build that supports DXR2 (clustered geometry
+ `lib_6_10`). `ExperimentalD3D12.props` hooks it up; defaults expect:

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
D3D12RaytracingClusteredGeometry.exe                       (default - BUG on NVIDIA HW)
D3D12RaytracingClusteredGeometry.exe --vertex-format float (correct - FLOAT reference)
D3D12RaytracingClusteredGeometry.exe --fix                 (correct - workaround applied)
D3D12RaytracingClusteredGeometry.exe --adapter warp        (correct - WARP)
```

Press `Esc` to quit. Diagnostic info via `OutputDebugString` (visible in
DebugView).

## Files

- `Main.cpp`                     - everything (window, device, scene, COMPRESSED1 encoder, DXR2 pipeline)
- `ExperimentalD3D12.props`      - hookup to the local experimental D3D12 build
- `D3D12RaytracingClusteredGeometry.{vcxproj,sln}`
