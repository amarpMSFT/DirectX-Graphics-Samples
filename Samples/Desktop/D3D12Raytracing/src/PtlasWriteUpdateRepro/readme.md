# PtlasWriteUpdateRepro

Minimal D3D12 DXR2 repro: a PTLAS build call that contains both a
`WRITE_INSTANCE` op and an `UPDATE_INSTANCE` op (on **disjoint** instance
indices) TDRs the GPU on the preview NVIDIA driver.  Repros on **frame 1-2**
with **4 instances, 1 partition** as the only PTLAS work in the call --
no DispatchRays, no TRANSLATE_PARTITION, no `ENABLE_EXPLICIT_AABB`, no
shaders or pipeline state in the program at all.

## TL;DR

```
# REPROS (TDR within 1-2 frames):
PtlasWriteUpdateRepro.exe

# CLEAN baselines (no TDR for 30 frames):
PtlasWriteUpdateRepro.exe --no-update             # WRITE_INSTANCE only
PtlasWriteUpdateRepro.exe --no-write              # UPDATE_INSTANCE only
PtlasWriteUpdateRepro.exe --use-warp              # same EXE on WARP
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

The single source file (~440 LOC, no shaders, no pipeline state, no
DispatchRays) does:

  1. Build two trivial 1-triangle BLASes (so we have two GPUVAs to alternate).
  2. Create a PTLAS with `InstanceCount = 4`, `PartitionCount = 1`,
     `Flags = NONE` (no global partition, no partition translation).
  3. Initial `WRITE_INSTANCE` pass: populate all 4 instances.
  4. Per frame, call `ExecuteIndirectRTASOperations` once with TWO ops in
     the operation array:
       - `WRITE_INSTANCE` × 1 -- args reference instance #i
       - `UPDATE_INSTANCE` × 1 -- args reference instance #j  (i ≠ j)
     Each op writes one distinct `InstanceIndex`.  The op order is
     `WRITE_INSTANCE` first then `UPDATE_INSTANCE`.
  5. Flush + wait fence.

Per spec, mixing different op types in the same call is legal
(`Raytracing2.md`: "Multiple operations of different types can be used in
the same call.").  Per-op-type uniqueness is satisfied (one arg each, on
different instances).

## Bisect that landed on this trim

Earlier versions of this repro had a TRANSLATE_PARTITION op, a 64×64
DispatchRays, and `ENABLE_EXPLICIT_AABB` on the writes.  Removing each
in turn confirmed each is irrelevant to the bug; the bisect grid:

| Config (--frames 20)                                  | Result            |
| ----------------------------------------------------- | ----------------- |
| WRITE + UPDATE (+ everything else removed)            | TDR after frame 2 |
| `--no-update` (WRITE only)                            | OK 30 frames      |
| `--no-write` (UPDATE only)                            | OK 30 frames      |
| (previous, with translate+dispatch+AABB) default      | TDR after frame 1 |
| `--use-warp` (same EXE on WARP)                       | OK 20+ frames     |

Conclusion: TDR is triggered exclusively by the **coexistence of WRITE +
UPDATE op types in a single `ExecuteIndirectRTASOperations` call**.  All
other PTLAS feature toggles tested are irrelevant; the bug does not depend
on a consumer of the PTLAS (no DispatchRays needed).

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
`d3d10warp.dll` copied next to the exe for `--use-warp`.)

There is no `.hlsl` file and no shader compile step -- the program contains
zero shaders.

## Run / CLI

```
PtlasWriteUpdateRepro.exe [options]

  --frames N           : run N frames (default 30)
  --no-write           : skip the WRITE_INSTANCE arg (baseline: no TDR)
  --no-update          : skip the UPDATE_INSTANCE arg (baseline: no TDR)
  --partitions N       : partition count (default 1)
  --instances N        : total instance count (default 4)
  --use-warp           : force the WARP adapter (sanity comparison; no TDR)
```

Exit code: 0 = no TDR, 1 = TDR detected, 2 = setup error (no DXR2, build
failure, etc.).

## Source files

- `Main.cpp` (~440 LOC) -- the entire repro: D3D12 init, BLAS build, PTLAS
  setup, per-frame `WRITE_INSTANCE` + `UPDATE_INSTANCE`, fence wait, TDR
  detection via `ID3D12InfoQueue1::RegisterMessageCallback`.  No shaders,
  no pipeline state, no DispatchRays.
- `PtlasWriteUpdateRepro.vcxproj` -- minimal MSBuild project.
- `ExperimentalD3D12.props` -- experimental D3D12 SDK hookup.
