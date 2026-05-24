# PtlasWriteUpdateRepro

Minimal D3D12 DXR2 repro: a PTLAS build call that contains both a
`WRITE_INSTANCE` op and an `UPDATE_INSTANCE` op (on **disjoint** instance
indices) TDRs the GPU on the preview NVIDIA driver.  Repros on **frame 1**
with as little as **4 instances, 1 partition, no dispatch, no
translate_partition op, no `ENABLE_EXPLICIT_AABB` flag**.

## TL;DR

```
# REPROS (TDR within 1-2 frames):
PtlasWriteUpdateRepro.exe
PtlasWriteUpdateRepro.exe --no-translate --no-dispatch --instances 4 --partitions 1

# CLEAN baselines (no TDR for 20 frames):
PtlasWriteUpdateRepro.exe --no-update             # WRITE_INSTANCE only
PtlasWriteUpdateRepro.exe --no-write              # UPDATE_INSTANCE only
PtlasWriteUpdateRepro.exe --use-warp              # same code on WARP, clean
```

## Repro environment

- NVIDIA GeForce RTX 4090
- Driver: preview build, 2026-05
- Windows 11
- D3D12 SDK = 721 (experimental D3D12Core.dll)
- DXR2 cluster + PTLAS experimental features enabled
- Verified clean on WARP (Microsoft Basic Render Driver)
- Verified spec-compliant: the public DXR2 conformance test
  `IndirectBuild.cpp` exercises the same WRITE+UPDATE-on-disjoint-instances
  pattern by default (both `DisablePartitionUpdateInstanceUnlessForced` and
  `DisablePartitionWriteInstanceUnlessForced` default to `false`).

## What it does

Builds 2 trivial triangle BLASes, creates a PTLAS sized for N instances and
M partitions, does one full initial `WRITE_INSTANCE` pass to populate it,
then per frame:

  1. **WRITE_INSTANCE** for one instance (transform tweak + alternate BLAS).
  2. **UPDATE_INSTANCE** for a *different* instance (BLAS pointer swap).
  3. (optionally) **TRANSLATE_PARTITION** for all partitions.
  4. (optionally) **DispatchRays** (64×64) so the PTLAS is consumed by a trace.
  5. Flush + wait fence.

Each `ExecuteIndirectRTASOperations` call contains **at most one op of each
type** (WRITE × 1, UPDATE × 1, TRANSLATE × 1) and **the WRITE arg references
a different `InstanceIndex` than the UPDATE arg**.  Per spec, this is legal.

## Bisect grid (--frames 20)

| Config                                                | Result            |
| ----------------------------------------------------- | ----------------- |
| default (WRITE + UPDATE + TRANSLATE + dispatch + AABB) | TDR after frame 1 |
| `--no-update` (WRITE only)                            | **OK 20 frames**  |
| `--no-write` (UPDATE only)                            | **OK 20 frames**  |
| `--no-translate` (drop TRANSLATE_PARTITION)           | TDR after frame 1 |
| `--no-dispatch` (no DispatchRays anywhere)            | TDR after frame 1 |
| `--no-aabb` (no ENABLE_EXPLICIT_AABB flag)            | TDR after frame 2 |
| `--partitions 1` (one partition)                      | TDR after frame 1 |
| `--instances 4 --partitions 2`                        | TDR after frame 1 |
| `--no-translate --no-dispatch` (W+U only, no trace)   | TDR after frame 1 |
| `--no-translate --no-aabb`                            | TDR after frame 2 |
| `--use-warp` (same code on WARP)                      | **OK 20 frames**  |

Conclusion:
- TDR requires **both** a `WRITE_INSTANCE` op and an `UPDATE_INSTANCE` op
  in the same `ExecuteIndirectRTASOperations` call.
- TDR happens **with or without** TRANSLATE_PARTITION, with or without
  DispatchRays, with or without `ENABLE_EXPLICIT_AABB`.
- TDR happens with **1 partition + 4 instances**, the smallest config tested.
- Running the **exact same EXE** with `--use-warp` is clean for 20 frames.

## Build

Requires:
- Visual Studio 2022 / MSBuild
- Experimental D3D12 install at `D:\experimental\` (override with
  `$(DxrExperimentalRoot)` env var or msbuild property -- see
  `ExperimentalD3D12.props` header comment).  Provides:
    - `install\windows\x64\debug\d3d12.lib`
    - `install\windows\x64\debug\D3D12Core.dll`
    - `install\windows\x64\debug\D3D12SDKLayers.dll`
    - `install\windows\x64\debug\d3d10warp.dll`
    - `install\windows\x64\debug\include\d3d12.h`
    - `packages\OSS\dxcbin_preview\bin\sdk\amd64\dxcompiler.dll`
    - `packages\OSS\dxcbin_preview\bin\sdk\amd64\dxil.dll`

```
msbuild PtlasWriteUpdateRepro.vcxproj /p:Configuration=Debug /p:Platform=x64
```

Output: `bin\x64\Debug\PtlasWriteUpdateRepro.exe`
(`D3D12Core.dll`, `D3D12SDKLayers.dll` copied into `bin\x64\Debug\D3D12\`;
`dxcompiler.dll`, `dxil.dll`, `d3d10warp.dll` copied next to the exe.)

## Run / CLI

```
PtlasWriteUpdateRepro.exe [options]

  --frames N           : run N frames (default 30)
  --no-write           : skip the WRITE_INSTANCE arg (baseline: no TDR)
  --no-update          : skip the UPDATE_INSTANCE arg (baseline: no TDR)
  --no-translate       : skip TRANSLATE_PARTITION op
  --no-dispatch        : skip DispatchRays (build alone TDRs too)
  --no-aabb            : don't set ENABLE_EXPLICIT_AABB flag on writes
  --partitions N       : partition count (default 4)
  --instances N        : total instance count (default 16)
  --use-warp           : force the WARP adapter (sanity comparison; no TDR)
```

Exit code: 0 = no TDR, 1 = TDR detected, 2 = setup error (no DXR2, build
failure, etc.).

## Source files

- `Main.cpp` (~700 LOC) -- the entire repro: D3D12 init, BLAS build, PTLAS
  setup, inline HLSL (compiled at runtime via DXC), render loop, fence wait,
  TDR-via-`InfoQueue1`-callback detection.  Inline HLSL is ~30 lines (raygen +
  miss + closesthit) for the optional trace path.
- `PtlasWriteUpdateRepro.vcxproj` -- minimal MSBuild project.
- `ExperimentalD3D12.props` -- experimental D3D12 SDK + DXC hookup.
