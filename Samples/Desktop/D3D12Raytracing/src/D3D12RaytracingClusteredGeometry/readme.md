# D3D12 DXR2 COMPRESSED1 cluster-geometry min repro

Single-file repro for a bug where `D3D12_VERTEX_FORMAT_COMPRESSED1` on the
DXR2 cluster build path silently drops 5 of 6 cube faces from BVH traversal
**on the NVIDIA preview driver**, but renders correctly on WARP.

## TL;DR

```
COMPRESSED1 + NVIDIA preview driver -> only 1 cube face renders   (BUG)
COMPRESSED1 + WARP                  -> full cube renders           (correct)
FLOAT32_3   + NVIDIA preview driver -> full cube renders           (correct)
```

### THE ONE-LINE FIX

Setting **`ClusterLimits.MaxVertexCountPerCluster = 32`** (or anywhere up to
the spec max of 256) makes the NVIDIA driver render the full cube. All other
fields can stay tight to actual values. This is the same minimal change that
brings the repro to working.

```cpp
D3D12_RTAS_CLUSTER_LIMITS limits = {};
limits.MaxArgCount                = N;
limits.MaxTriangleCountPerCluster = 2;        // tight (actual)
limits.MaxVertexCountPerCluster   = 32;       // <- MUST be >= 32 on NVIDIA preview driver
limits.MaxTotalTriangleCount      = 2 * N;    // tight (actual)
limits.MaxTotalVertexCount        = 4 * N;    // tight (actual)
```

### Threshold pattern (NVIDIA RTX 4090, preview driver)

| `MaxVertexCountPerCluster` | Result |
|---:|---|
| 4 (= actual) ... 10 | broken (1 face) |
| 11 ... 16 | **partial + geometrically distorted** (extra triangles stretching beyond cube) |
| 17 ... 20 | broken again (1 face) |
| 24 ... 31 | partial + distorted |
| **32** | **✓ full cube** |
| 33 | partial (regresses!) |
| 64 / 128 / 256 | ✓ full cube |

The non-monotonic pattern suggests the driver rounds per-cluster vertex
storage to a wave-aligned stride (warp size = 32 on NVIDIA), and uses
`MaxVertexCountPerCluster` to compute it. At values that don't round
correctly to a 32-multiple, adjacent clusters' vertex storage overlaps or
reads from wrong slots.

### Why the conformance test doesn't hit this

The d3d12conf `ClusterCompressed1VertexFormat` test sets
`MaxVertexCountPerCluster = 256` unconditionally (IndirectBuild.cpp:4858),
regardless of actual cluster sizes. That hides the bug. The test never
exercises a tight `MaxVertexCountPerCluster` with multiple COMPRESSED1
clusters per build.

The visible face was always whichever cluster has index 0 in the args array
(proven by `--rotate-faces N`).  All 6 CLAS addresses come back valid and
distinct from `BUILD_CLAS_FROM_TRIANGLES`; only cluster 0's CLAS contains
traversable geometry.

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
D3D12RaytracingClusteredGeometry.exe                                     (default - BUG on HW)
D3D12RaytracingClusteredGeometry.exe --vertex-format float               (correct - FLOAT reference)
D3D12RaytracingClusteredGeometry.exe --adapter warp                      (correct - WARP)
D3D12RaytracingClusteredGeometry.exe --max-vert-per-cluster 32           (FIX - the minimal workaround)
D3D12RaytracingClusteredGeometry.exe --fix-loose-limits                  (FIX - 256 on all the Max*)
```

Press `Esc` to quit. Log mirrored to `%TEMP%\compressed1_repro.log`.

## Diagnostic / experimental flags

For investigating the bug:

| Flag | Effect |
|---|---|
| `--diag` | Dump per-cluster COMPRESSED1 header + decode roundtrip + post-build CLAS address readback |
| `--adapter warp` / `--adapter hw` | Force WARP or first HW adapter |
| `--rotate-faces N` | Rotate cube-face emit order by N (test which cluster is "winning") |
| `--reverse-faces` | Reverse face emit order |
| `--fix-split` | CPU sync between CLAS+BLAS builds (mirrors d3d12conf test). No effect. |
| `--fix-reupload` | Readback CLAS addresses + re-upload via fresh CPU buffer. No effect. |
| `--fix-explicit` | EXPLICIT_DESTINATIONS for CLAS build. No effect (and broken in this repro). |
| `--fix-separate-vb` | Per-cluster VB+IB resources (vs concatenated). No effect. |
| `--fix-blas-implicit` | IMPLICIT_DESTINATIONS for BUILD_BLAS_FROM_CLAS. No effect. |
| `--fix-geo-flag-array` | Supply per-tri GeometryIndexAndFlagsArray (no effect; rules out the known NVIDIA flags bug as the cause) |
| **`--fix-loose-limits`** | **THE FIX** — set ClusterLimits.Max{Triangle,Vertex}CountPerCluster = 256 |
| **`--max-vert-per-cluster N`** | Override `MaxVertexCountPerCluster`. **N=32 is the MINIMAL fix; N<32 broken, N=33 partial.** |
| `--max-tri-per-cluster N` / `--max-total-tri N` / `--max-total-vert N` | Individually override one ClusterLimits field; bisecting showed only `--max-vert-per-cluster` matters |
| `--wa-1blas-per-clas` | Build 6 separate BLASes (1 CLAS each), 6 TLAS instances. No effect (cluster 0 still wins). |
| `--wa-1clas-per-build` | Build 6 single-arg BUILD_CLAS_FROM_TRIANGLES calls (currently broken - device-removes) |

## Investigation summary

**Final root cause** (NVIDIA RTX 4090, preview driver): the COMPRESSED1
multi-cluster CLAS build mis-strides per-cluster vertex storage when
`MaxVertexCountPerCluster < 32`. With tight limits (e.g. 4 for a 4-vert
cluster), adjacent clusters' vertex slots overlap or get read from the wrong
cluster, leaving only the first cluster's CLAS with traversable geometry.

Tested fixes that do NOT help (all produce identical 1-face render except
where noted):

| Workaround | Result |
|---|---|
| CPU sync between CLAS and BLAS builds (`--fix-split`) | broken (same 1 face) |
| CPU readback + re-upload of CLAS address array (`--fix-reupload`) | broken |
| EXPLICIT_DESTINATIONS for CLAS build (`--fix-explicit`) | broken |
| Per-cluster separate VB+IB resources (`--fix-separate-vb`) | broken |
| IMPLICIT vs EXPLICIT for BLAS-from-CLAS (`--fix-blas-implicit`) | broken |
| 6 separate BLAS + 6 TLAS instances (`--wa-1blas-per-clas`) | broken (cluster 0 still wins) |
| Per-tri GeometryIndexAndFlagsArray (`--fix-geo-flag-array`) | broken |
| **`MaxVertexCountPerCluster = 32` (`--max-vert-per-cluster 32`)** | **✓ FULL CUBE** |

Verified correctness of inputs / non-driver-side code:
- CPU-side COMPRESSED1 encoder is valid (decode roundtrip max-err 0.000049
  vs quantization step 0.000244, i.e. quantization-grid perfect)
- WARP renders all 6 faces in COMPRESSED1 mode -> the encoded bytes and API
  usage are spec-valid
- FLOAT32_3 mode renders all 6 faces on NVIDIA + `--wa-1blas-per-clas` ->
  the app-side per-cluster-BLAS plumbing is correct

Strongest evidence for "first cluster always wins": `--rotate-faces N` makes
the visible face track cluster index 0, not any specific geometry:

| N | cluster 0 = | Result |
|---|---|---|
| 0 | +X face | +X face shows (centroid (695,458), 15549 px) |
| 1 | -X face | -X face edge visible (back-face culled, centroid (647,506), 6120 px) |
| 2 | +Y face | top face shows (centroid (647,404), 11488 px) |
| 5 | -Z face | -Z face shows (centroid (598,458), 15485 px) |

Threshold bisection on `MaxVertexCountPerCluster` (other fields tight):

| Value | Result |
|---:|---|
| 4 (actual) ... 10 | broken (1 face, 9212 px) |
| 11 ... 16 | partial + **geometrically distorted** (extra triangles spilling beyond cube silhouette) |
| 17 ... 20 | broken again (1 face) |
| 24 ... 31 | partial + distorted |
| **32** | **✓ FULL CUBE (23645 px)** |
| 33 | partial (regresses!) |
| 64 / 128 / 256 | ✓ full cube |

The non-monotonic + 32-aligned pattern suggests the driver computes
per-cluster vertex-storage stride as `roundUpToWarpSize(MaxVertexCountPerCluster)`
where warp size = 32 on NVIDIA. Values that don't round to 32-multiples
produce wrong strides -> cluster storage overlaps or is read from wrong slots.

## Files

- `Main.cpp`                     - everything (window, device, scene, COMPRESSED1 encoder, DXR2 pipeline, diagnostic + experimental toggles)
- `ExperimentalD3D12.props`      - hookup to the local experimental D3D12 build
- `D3D12RaytracingClusteredGeometry.{vcxproj,sln}`
