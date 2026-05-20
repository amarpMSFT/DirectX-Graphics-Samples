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

The visible face is always **cluster index 0** in the `BUILD_CLAS_FROM_TRIANGLES`
arg array — proven by `--rotate-faces N` (which makes a different face be
cluster 0 and shows that face wins instead). All 6 CLAS addresses come back
non-zero and distinct, but only cluster 0's CLAS contains traversable
geometry.

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
D3D12RaytracingClusteredGeometry.exe                       (default - BUG on HW)
D3D12RaytracingClusteredGeometry.exe --vertex-format float (correct - reference)
D3D12RaytracingClusteredGeometry.exe --adapter warp        (correct - WARP)
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
| `--fix-split` | CPU sync between CLAS+BLAS builds (mirrors d3d12conf test) |
| `--fix-reupload` | Readback CLAS addresses + re-upload via fresh CPU buffer |
| `--fix-explicit` | EXPLICIT_DESTINATIONS for CLAS build |
| `--fix-separate-vb` | Per-cluster VB+IB resources (vs concatenated) |
| `--fix-blas-implicit` | IMPLICIT_DESTINATIONS for BUILD_BLAS_FROM_CLAS |
| `--wa-1blas-per-clas` | Build 6 separate BLASes (1 CLAS each), 6 TLAS instances |
| `--wa-1clas-per-build` | Build 6 single-arg BUILD_CLAS_FROM_TRIANGLES calls (currently buggy - device-removes) |

## Investigation summary

Tested fixes that do NOT help on NVIDIA HW (all produce same 1-face render):
- CPU sync between CLAS and BLAS builds (`--fix-split`)
- CPU readback + re-upload of CLAS address array (`--fix-reupload`)
- EXPLICIT_DESTINATIONS for CLAS build (`--fix-explicit`)
- Per-cluster separate VB+IB resources (`--fix-separate-vb`)
- IMPLICIT vs EXPLICIT for BLAS-from-CLAS (`--fix-blas-implicit`)
- 6 separate BLAS (1 CLAS each) + 6 TLAS instances (`--wa-1blas-per-clas`)

Verified correctness of inputs / non-driver-side code:
- CPU-side COMPRESSED1 encoder produces valid output (decode roundtrip
  max-error 0.000049 vs quantization step 0.000244, i.e. quantization-grid
  perfect)
- WARP renders all 6 faces in COMPRESSED1 mode -> the encoded bytes are
  spec-valid + the API usage is spec-valid
- FLOAT32_3 mode renders all 6 faces on NVIDIA + `--wa-1blas-per-clas` -> the
  app-side per-cluster-BLAS plumbing is correct

Strongest evidence for the bug shape: `--rotate-faces N` makes the visible
face track cluster index 0 (not any specific geometry). With N rotations:

| N | cluster 0 = | Result |
|---|---|---|
| 0 | +X face | +X face shows (centroid (695,458), 15549 px) |
| 1 | -X face | -X face edge visible (back-face culled, centroid (647,506), 6120 px) |
| 2 | +Y face | top face shows (centroid (647,404), 11488 px) |
| 5 | -Z face | -Z face shows (centroid (598,458), 15485 px) |

## Files

- `Main.cpp`                     - everything (window, device, scene, COMPRESSED1 encoder, DXR2 pipeline, diagnostic + experimental toggles)
- `ExperimentalD3D12.props`      - hookup to the local experimental D3D12 build
- `D3D12RaytracingClusteredGeometry.{vcxproj,sln}`
