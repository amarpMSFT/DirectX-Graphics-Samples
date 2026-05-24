# PtlasWriteUpdateRepro

Minimal DXR2 driver-bug repro: a PTLAS `ExecuteIndirectRTASOperations` call
containing BOTH a `WRITE_INSTANCE` op AND an `UPDATE_INSTANCE` op TDRs the
GPU on the preview NVIDIA driver -- **specifically when both args target a
non-zero `InstanceIndex`**.

> **`TDR iff WRITE.InstanceIndex != 0 AND UPDATE.InstanceIndex != 0`.**

If either op references `InstanceIndex = 0`, the build is clean.

## TL;DR

```
# REPROS (TDR on frame 1):
PtlasWriteUpdateRepro.exe                                  # W=1, U=2 (default)
PtlasWriteUpdateRepro.exe --write 2 --update 3 --instances 4

# CLEAN baselines (5 frames no TDR):
PtlasWriteUpdateRepro.exe --write 0 --update 1             # one arg references 0
PtlasWriteUpdateRepro.exe --write 1 --update 0
PtlasWriteUpdateRepro.exe --no-update                      # WRITE_INSTANCE only
PtlasWriteUpdateRepro.exe --no-write                       # UPDATE_INSTANCE only
PtlasWriteUpdateRepro.exe --use-warp                       # same EXE on WARP
```

## The (W, U) bisect grid

`--frames 15`, 4 instances, 1 partition, single shared BLAS:

```
       U=0   U=1   U=2   U=3
W=0     -    OK    OK    OK
W=1    OK     -    TDR   TDR
W=2    OK    TDR    -    TDR
W=3    OK    TDR   TDR    -
```

`-` = same instance (spec-forbidden). Everything off the diagonal where
both indices are >= 1 TDRs; the entire row/column for index 0 is clean.

## What the program does

Single 353-line source file.  No shaders, no pipeline state, no descriptor
heaps, no DispatchRays, no TRANSLATE_PARTITION, no `ENABLE_EXPLICIT_AABB`.

1. Build one 1-triangle BLAS.
2. Create a PTLAS: `InstanceCount = 3`, `PartitionCount = 1`, `Flags = NONE`.
3. Initial pass: one `WRITE_INSTANCE` op populating all 3 instances.
4. Per frame, one `ExecuteIndirectRTASOperations` call with two ops:
     - `WRITE_INSTANCE`  (`InstanceIndex = g_writeInst`, default 1)
     - `UPDATE_INSTANCE` (`InstanceIndex = g_updateInst`, default 2)
   Both args reference the same BLAS GPUVA.  Fence wait.

Per spec (`Raytracing2.md`): "Multiple operations of different types can be
used in the same call."  Per-op-type uniqueness is satisfied (one arg
each, on different instances).

## What does NOT affect the bug

Confirmed via bisect: removing each of these in isolation still TDRs:

- `TRANSLATE_PARTITION` op  (present or absent)
- `DispatchRays` after the build  (build alone TDRs)
- `ENABLE_EXPLICIT_AABB` flag on writes
- `ENABLE_PARTITION_TRANSLATION` flag in inputs
- `InstanceFlags` value  (NONE / FORCE_OPAQUE)
- Partition count  (1, 2, 4)
- Instance count  (3 minimum so W=1 U=2 fits)
- Same vs different BLAS pointer in W vs U
- Same vs varying transform per frame

## What DOES affect the bug

Only this:
- **Both ops in the same `ExecuteIndirectRTASOperations` call.**  Submitting them in two separate calls (each with one op) is clean.
- **Both args' `InstanceIndex` non-zero.**  Either op touching index 0 makes the build clean.

## Related observation (full sample only)

Splitting the per-frame PTLAS work into TWO `ExecuteIndirectRTASOperations`
calls back-to-back on the same PTLAS resource within one command list
(first call = `WRITE_INSTANCE` + `TRANSLATE_PARTITION`, second call =
`UPDATE_INSTANCE` only, with a UAV barrier on the PTLAS resource between
them) also TDRs at the same ~frame-8 timing in the full sample
(`user/amarp/ptlas-sample` branch, `PartitionedTlasSample::DoRender`).
This is NOT reproduced in the minimal repro here (which does only one
build per frame), but the observation is captured for the IHV's
investigation: the bug surface may be broader than just single-call op-type
mixing, and back-to-back PTLAS builds on the same resource within one CL
may share whatever code path is misbehaving.

## Repro environment

- NVIDIA GeForce RTX 4090
- Driver: preview, 2026-05
- Windows 11
- D3D12 SDK = 721 (experimental D3D12Core.dll)
- DXR2 cluster + PTLAS experimental features enabled
- Verified clean on WARP (Microsoft Basic Render Driver) with the same EXE
- Verified spec-compliant: `IndirectBuild.cpp` conformance test exercises
  mixed WRITE+UPDATE-on-disjoint-instances by default; passes on the same
  driver because its random index split frequently lands on index 0.

## Build

Requires:
- Visual Studio 2022 / MSBuild
- Experimental D3D12 install at `D:\experimental\` (override with
  `$(DxrExperimentalRoot)` env var or msbuild property -- see
  `ExperimentalD3D12.props` for the layout it expects).

```
msbuild PtlasWriteUpdateRepro.vcxproj /p:Configuration=Debug /p:Platform=x64
```

Output: `bin\x64\Debug\PtlasWriteUpdateRepro.exe`.  Props file copies
`D3D12Core.dll` + `D3D12SDKLayers.dll` into `bin\x64\Debug\D3D12\`, and
`d3d10warp.dll` next to the EXE (for `--use-warp`).

## CLI

```
PtlasWriteUpdateRepro.exe [options]

  --frames N      : run N frames (default 5)
  --write IDX     : InstanceIndex for the WRITE_INSTANCE arg (default 1)
  --update IDX    : InstanceIndex for the UPDATE_INSTANCE arg (default 2)
  --no-write      : skip WRITE_INSTANCE  -> baseline (no TDR)
  --no-update     : skip UPDATE_INSTANCE -> baseline (no TDR)
  --use-warp      : force WARP adapter   -> baseline (no TDR)
```

Exit: 0 = no TDR, 1 = TDR detected, 2 = setup error.

## Files

- `Main.cpp` -- the entire repro (~350 LOC).
- `PtlasWriteUpdateRepro.vcxproj` + `ExperimentalD3D12.props` -- build.
