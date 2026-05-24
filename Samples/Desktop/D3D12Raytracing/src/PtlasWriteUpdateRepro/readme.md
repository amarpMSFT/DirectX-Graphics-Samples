# PtlasWriteUpdateRepro

Minimal D3D12 DXR2 repro: a PTLAS `ExecuteIndirectRTASOperations` call that
contains BOTH a `WRITE_INSTANCE` op AND an `UPDATE_INSTANCE` op TDRs the GPU
on the preview NVIDIA driver -- **specifically when the WRITE's
`InstanceIndex` is non-zero AND the UPDATE's `InstanceIndex` is non-zero**.

If either op targets `InstanceIndex = 0` the build is clean.

## TL;DR

```
# REPROS (TDR on frame 1):
PtlasWriteUpdateRepro.exe                                  # W=1 U=2 (default)
PtlasWriteUpdateRepro.exe --write 1 --update 3 --instances 4
PtlasWriteUpdateRepro.exe --write 2 --update 3 --instances 4

# CLEAN baselines (no TDR for 30 frames):
PtlasWriteUpdateRepro.exe --write 0 --update 1             # W targets InstanceIndex 0
PtlasWriteUpdateRepro.exe --write 1 --update 0             # U targets InstanceIndex 0
PtlasWriteUpdateRepro.exe --no-update                      # WRITE_INSTANCE only
PtlasWriteUpdateRepro.exe --no-write                       # UPDATE_INSTANCE only
PtlasWriteUpdateRepro.exe --use-warp                       # same EXE on WARP
```

## The (W, U) bisect grid

`--frames 15`, 4 instances, 1 partition, single BLAS shared by WRITE+UPDATE,
no DispatchRays, no TRANSLATE, no `ENABLE_EXPLICIT_AABB`:

```
       U=0   U=1   U=2   U=3
W=0     -    OK    OK    OK
W=1    OK     -    TDR   TDR
W=2    OK    TDR    -    TDR
W=3    OK    TDR   TDR    -
```

**Rule: TDR iff `WRITE.InstanceIndex != 0` AND `UPDATE.InstanceIndex != 0`.**
The driver's PTLAS update path appears to behave differently when either op
references `InstanceIndex = 0`.  The very narrow `(W>0, U>0)` region is the
entire bug.

## Other knobs that DON'T affect the bug (all confirmed via bisect)

| Knob                                         | Notes |
| -------------------------------------------- | ----- |
| `TRANSLATE_PARTITION` op present / not       | TDR either way |
| `DispatchRays` consumer present / not        | TDR either way (build alone TDRs) |
| `ENABLE_EXPLICIT_AABB` on the WRITE arg     | TDR either way |
| `ENABLE_PARTITION_TRANSLATION` in inputs flags | TDR either way |
| Partition count = 1, 2, 4                   | TDR either way |
| Instance count = 3, 4, 16                   | TDR (min is 3, so W=1 U=2 fits) |
| `D3D12_RAYTRACING_INSTANCE_FLAG_*` on writes | TDR either way (tested NONE + FORCE_OPAQUE) |
| Same vs different BLAS pointer in W vs U    | TDR either way |
| Same vs varying transform per frame         | TDR either way |

## Repro environment

- NVIDIA GeForce RTX 4090
- Driver: preview build, 2026-05
- Windows 11
- D3D12 SDK = 721 (experimental D3D12Core.dll)
- DXR2 cluster + PTLAS experimental features enabled
- Verified clean on WARP (Microsoft Basic Render Driver) with the same EXE
- Verified spec-compliant: `IndirectBuild.cpp` conformance test exercises
  mixed WRITE+UPDATE-on-disjoint-instances by default and passes on the
  same driver (just happens to often touch `InstanceIndex = 0` by chance,
  not deterministically hitting the bug).

## What the program does

The single source file (~440 LOC, no shaders, no pipeline state, no
DispatchRays) does:

1. Build one trivial 1-triangle BLAS.
2. Create a PTLAS with `InstanceCount = 3`, `PartitionCount = 1`,
   `Flags = NONE`.
3. Initial pass: a `WRITE_INSTANCE` op that populates all 3 instances.
4. Per frame, call `ExecuteIndirectRTASOperations` once with TWO ops:
     - `WRITE_INSTANCE`  (InstanceIndex = 1, by default)
     - `UPDATE_INSTANCE` (InstanceIndex = 2, by default)
   Both args reference the same BLAS pointer.
5. Flush + wait fence.

Per spec, mixing op types in the same call is legal
(`Raytracing2.md`: "Multiple operations of different types can be used in
the same call.").  Per-op-type uniqueness is satisfied (one arg each, on
different instances; the `or` clause prohibiting same-instance W+U is also
satisfied).

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

```
msbuild PtlasWriteUpdateRepro.vcxproj /p:Configuration=Debug /p:Platform=x64
```

Output: `bin\x64\Debug\PtlasWriteUpdateRepro.exe`
(`D3D12Core.dll`, `D3D12SDKLayers.dll` copied into `bin\x64\Debug\D3D12\`;
`d3d10warp.dll` copied next to the EXE for `--use-warp`.)

There is no `.hlsl` file and no shader compile step -- the program contains
zero shaders.

## CLI

```
PtlasWriteUpdateRepro.exe [options]

  --frames N         : run N frames (default 30)
  --instances N      : PTLAS instance count (default 3; min 3 to trigger)
  --write IDX        : WRITE_INSTANCE arg's InstanceIndex (default 1; -1 = vary)
  --update IDX       : UPDATE_INSTANCE arg's InstanceIndex (default 2; -1 = vary)
  --no-write         : skip the WRITE_INSTANCE arg  (baseline: no TDR)
  --no-update        : skip the UPDATE_INSTANCE arg (baseline: no TDR)
  --use-warp         : force the WARP adapter        (baseline: no TDR)
```

Exit code: 0 = no TDR, 1 = TDR detected, 2 = setup error (no DXR2, etc.).

## Source

- `Main.cpp` -- everything (D3D12 init, BLAS, PTLAS, per-frame W+U build,
  fence wait, TDR detection via `ID3D12InfoQueue1::RegisterMessageCallback`).
- `PtlasWriteUpdateRepro.vcxproj` + `ExperimentalD3D12.props` -- the build.
