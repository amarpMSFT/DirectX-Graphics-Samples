# DXR2 Clustered-Geometry Benchmark Suite

End-to-end parameter sweep + HTML reporting for the
`D3D12RaytracingClusteredGeometry` sample. Runs the same exe many times with
different command-line configurations, gathers per-run performance and memory
data via the built-in JSON-snapshot mode, and produces a self-contained HTML
report with grouped charts.

## What you need

* The `D3D12RaytracingClusteredGeometry.exe` built (any config — Debug works
  fine). Default location: `..\bin\x64\Debug\`.
* Python 3.10+ (anything modern; only stdlib used). Tested on 3.11/3.14.
* `chart.umd.min.js` next to the scripts (already vendored — 200 KB). The
  report can also fall back to a CDN if missing, but the offline copy is
  recommended for build-box use.

## Quick start

```powershell
# Full sweep on this machine (~3 minutes, 26 runs, skips 10K trad to save time):
python sweep.py --no-10k-trad

# 30-second smoke test (12 runs, no precision sweeps):
python sweep.py --quick --no-10k --bench-seconds 3

# Custom exe + custom output path:
python sweep.py --exe C:\my\app.exe --out my_report.html

# Reuse most-recent JSONs and just rebuild the HTML (useful while iterating
# on the chart definitions; no exe re-runs):
python sweep.py --no-run
```

Output lands in `results\<adapter_safe_name>\<timestamp>\`:
* `report.html` — the self-contained report
* one `*.json` per run — the raw measurement data

## Cross-machine comparison

To compare two or more machines, run `sweep.py` on each, copy the result
directories somewhere accessible, then:

```powershell
python compare.py path\to\machine_A\<sweep_id> path\to\machine_B\<sweep_id> --out compare.html
```

Each input directory contributes one series per chart. Headline numbers
panel at the top shows absolute values plus % delta vs the first machine.

You can run `compare.py` with more than two machines — each additional
machine becomes another curve on the line charts / additional bar on the
bar charts.

## How the data flows

```
sweep.py  →  spawns D3D12RaytracingClusteredGeometry.exe with --bench-seconds N --bench-out X.json
                  (the exe is already configured for headless / scripted use
                   via the existing --geometry-mode, --vertex-format, etc.
                   args this script wraps)
              ↓
              exe runs for N seconds wall-clock, snapshots state to X.json, exits
              ↓
sweep.py  →  parses each JSON, builds Chart objects, renders self-contained HTML
              with vendored Chart.js for offline interactivity
```

The exe's `--bench-seconds` + `--bench-out` flags are part of the sample
itself — added alongside this benchmark suite. The snapshot captures:

* **Adapter + driver** identity (description, PCI VID/PID, dedicated VRAM,
  Windows User-Mode Driver version)
* **Config** — every CLI flag that was set, normalized to round-trippable
  tokens (`"clusters"` not `"Clusters"`)
* **Build times** — static CLAS / BLAS / TLAS / total, in milliseconds
* **Frame perf** — FPS, ms/frame, per-frame INSTANTIATE / BLAS / TLAS
  microseconds (EMA-smoothed plus 60-sample snap means)
* **Memory** — every AS bucket the overlay tracks (CLAS, BLAS, TLAS,
  cluster-input, anim templates, anim BLAS, clone pools, etc.)
* **Scene** — cluster count, triangle count, anim-clones-pooled

JSON schema is in the WriteBenchmarkSnapshot function in the sample's
D3D12RaytracingClusteredGeometry.cpp. `schema_version: 1`.

## What the sweep covers

26 unique configurations by default (the matrix is in `build_sweep_matrix`
inside `sweep.py`):

| Axis | Values | What it shows |
|---|---|---|
| Scene scale (cluster mode) | 0 / 100 / 1k / [10k] | Cluster path memory + perf scaling |
| Scene scale (trad mode) | 0 / 100 / 1k / [10k]* | DXR1 path scaling (the comparison) |
| Vertex format | float / compressed | COMPRESSED1 vs FLOAT32_3 |
| CLAS alloc | implicit / get-sizes / compact | Memory vs build-time trade-off |
| BVH build flags | none / fast-build / fast-trace | Build-time vs traversal-perf |
| Position truncate (float) | 0 / 4 / 8 / 12 / 16 / 20 bits | Precision sweep |
| COMPRESSED1 bits/comp | 2 / 4 / 6 / 8 / 10 / 12 / 14 / 16 | Precision sweep |
| Trad alloc | implicit / compact | Trad-mode compaction effect |

`*` 10K trad mode is skipped by default (~90 s init) — pass `--include-10k-trad`
to enable. The user-side experience is that 1k already shows the trend.

## Charts produced

16 charts across 6 categories:

* **Scaling**: 7 charts — total AS memory, build time, FPS, per-frame TLAS,
  per-frame animated BLAS, per-frame INSTANTIATE, cluster+tri count.
* **Vertex format**: 1 bar chart contrasting FLOAT32_3 vs COMPRESSED1 across
  6 key metrics.
* **CLAS alloc**: 1 bar chart, 3 strategies × 3 metrics.
* **Build flags**: 1 bar chart, 3 flag modes × 3 metrics.
* **Precision (FLOAT32_3)**: 3 charts — CLAS bytes, INSTANTIATE time, FPS
  vs truncate bits.
* **Precision (COMPRESSED1)**: 3 charts — same axes vs bits/component.

Each chart describes itself in the rendered HTML; hover any point/bar for
the raw value + units.

## Notes & observations panel

The report auto-flags physically interesting (but non-buggy) numbers:

* **COMPRESSED1 cb=2 FPS spike** — at 4 quantization levels per axis,
  triangles collapse to degenerate strips and many rays miss. Not a perf
  win; the geometry is wrong.
* **FLOAT32_3 trunc=20 FPS drop** — extreme truncation distorts glass
  surfaces enough that refraction trace paths get longer.
* **Cluster vs trad memory ratio** at each scale — the headline DXR2
  value proposition number.
* **pf_timing_valid=false** runs — the 60-sample snap window didn't fill
  within the bench window (slow scenes need longer `--bench-seconds`); the
  EMA-smoothed values are reported instead.

## Two real reporting bugs were caught + fixed while writing this

The sample is in active development; the sweep ran across enough
configurations that it surfaced two genuine sample-side bugs:

1. **`animated_pf_clas_actual` always 0.** `MeasureAnimatedClasBytesOneShot()`
   early-returned on `!m_animatedObjectEnabled`, but that flag was set
   *after* it was called from `BuildAnimatedObjectSetup`. Fixed in the
   build sequence — flag is now set before the measure runs.
2. **`pf_*_us = 0` at high clone counts.** The overlay's per-frame snap
   window needs 60 samples (~1 s at 60 fps, but ~4 s at 16 fps), longer
   than typical 3-5 s bench windows on heavy scenes. The benchmark JSON
   now reports the EMA (always current) for headline values and includes
   a `*_snap` companion field for the exact 60-sample mean when available.

If you're modifying the sample, the sweep is a useful regression detector
— rerun `sweep.py` after any AS-pipeline change and compare to the prior
report with `compare.py`.

## Files

| File | What |
|---|---|
| `sweep.py` | Parameter-sweep driver + single-machine report renderer |
| `compare.py` | Cross-machine overlay-comparison report |
| `chart.umd.min.js` | Vendored Chart.js 4.4.1 (for offline reports) |
| `results/` | Per-machine output directory (gitignored — these are local artifacts) |
