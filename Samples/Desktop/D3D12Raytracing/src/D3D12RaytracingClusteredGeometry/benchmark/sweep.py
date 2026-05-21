#!/usr/bin/env python3
"""
sweep.py - parameter-space benchmark sweep for D3D12RaytracingClusteredGeometry.

Drives the sample exe headlessly, gathers per-config performance + memory data
via --bench-seconds + --bench-out, then builds a self-contained HTML report
with grouped charts (memory scaling, build-time scaling, FPS scaling, per-frame
timings, orthogonal effect studies).

Design intent:
  * Every run gets the same warmup + measurement window (--bench-seconds).
    Default 5 s -- the first ~1 s settles the per-frame timing snap window
    and the FPS rolling average, the remainder smooths jitter.
  * Each unique config is run ONCE.  Charts are consumers of the run set --
    adding new charts doesn't require re-running.
  * Output is a SINGLE .html file embedding Chart.js (vendored under the
    same dir) so the report works fully offline.  JSON results are also
    written to results/<adapter_safe>/<sweep_id>/ for cross-machine
    comparison via compare.py.
  * Adapter + driver version are captured into the report header so a
    report from one machine is unambiguous.

Usage:
    python sweep.py                          # full sweep, default exe path
    python sweep.py --quick                  # short matrix for smoke test
    python sweep.py --bench-seconds 8        # longer measurement per run
    python sweep.py --exe path\\to\\app.exe   # custom exe
    python sweep.py --out report.html        # custom output path
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

THIS_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
DEFAULT_EXE = PROJECT_DIR / "bin" / "x64" / "Debug" / "D3D12RaytracingClusteredGeometry.exe"
DEFAULT_RESULTS_ROOT = THIS_DIR / "results"


# =====================================================================================
# Run definition
# =====================================================================================

@dataclass(frozen=True)
class RunConfig:
    """One unique exe invocation. Field names mirror the CLI flag names so the
    config dict (and resulting JSON file) round-trip cleanly. None means 'use
    sample default' (i.e. don't pass that flag).
    """
    geometry_mode: str        = "clusters"        # --geometry-mode
    vertex_format: str        = "float"           # --vertex-format
    clas_alloc: str           = "compact"         # --clas-alloc
    trad_alloc: str           = "compact"         # --trad-alloc
    static_rebuild: str       = "none"            # --rebuild-mode
    build_flags: str          = "fast-trace"      # --build-flags
    extra_instances: int      = 0                 # --at <f>:extra-<n>  (we use CLI-equivalent below)
    position_truncate_bits: int | None = None     # --position-truncate
    compressed_bits: int | None        = None     # --compressed-bits
    aa_samples: int | None             = None     # --aa-samples

    def to_cli(self) -> list[str]:
        args: list[str] = [
            "--geometry-mode", self.geometry_mode,
            "--vertex-format", self.vertex_format,
            "--clas-alloc",    self.clas_alloc,
            "--trad-alloc",    self.trad_alloc,
            "--rebuild-mode",  self.static_rebuild,
            "--build-flags",   self.build_flags,
            "--extra-instances", str(self.extra_instances),
        ]
        if self.position_truncate_bits is not None:
            args += ["--position-truncate", str(self.position_truncate_bits)]
        if self.compressed_bits is not None:
            args += ["--compressed-bits", str(self.compressed_bits)]
        if self.aa_samples is not None:
            args += ["--aa-samples", str(self.aa_samples)]
        return args

    def slug(self) -> str:
        """Stable filename slug for the JSON output.  Includes only the fields
        that differ from the dataclass default to keep the slug readable."""
        parts: list[str] = []
        for f in self.__dataclass_fields__.values():
            v = getattr(self, f.name)
            if v == f.default:
                continue
            # Compact names
            short = {
                "geometry_mode": "geom",
                "vertex_format": "vtx",
                "clas_alloc":    "clas",
                "trad_alloc":    "trad",
                "static_rebuild": "reb",
                "build_flags":   "bf",
                "extra_instances": "n",
                "position_truncate_bits": "trunc",
                "compressed_bits": "cb",
                "aa_samples":    "aa",
            }.get(f.name, f.name)
            parts.append(f"{short}-{v}")
        if not parts:
            return "default"
        return "_".join(parts)


@dataclass
class RunResult:
    config: RunConfig
    json_path: pathlib.Path
    data: dict[str, Any]
    wall_seconds: float


# =====================================================================================
# Sweep matrix
# =====================================================================================

def build_sweep_matrix(quick: bool, include_10k_trad: bool) -> list[RunConfig]:
    """Returns the unique list of configs to run.  Charts later consume
    subsets of these by filter.  Keep this LOGICAL -- duplicates are removed
    automatically by the dedupe step in main().

    The matrix is layered:
      1. Scene-size scaling: extra_instances ∈ {0, 100, 1k, [10k]} × geometry_mode
         -- the headline DXR2-vs-DXR1 comparison.
      2. Orthogonal effects at the DEFAULT scene size (extra=0, clusters):
         a. vertex_format: float vs compressed1
         b. clas_alloc:    implicit / get-sizes / compact
         c. build_flags:   none / fast-build / fast-trace
         d. position_truncate_bits: precision sweep (FLOAT32_3 only)
         e. compressed_bits: precision sweep (COMPRESSED1 only)
      3. Trad-mode orthogonal at extra=0:
         a. trad_alloc: implicit / compact

    Quick mode drops the 10K scaling run (60s init in trad mode) and the
    precision sweeps (~6 extra runs).
    """
    runs: list[RunConfig] = []

    # --- 1. Scene scaling --------------------------------------------------
    # Cluster mode at 10K is fast (~5s init); trad mode at 10K is slow
    # (~60-90s init) and dominates wall-clock.  Include cluster 10K by default;
    # gate trad 10K on include_10k_trad.  --quick skips 10K entirely.
    cluster_scales = [0, 100, 1000] + ([10000] if not quick else [])
    trad_scales    = [0, 100, 1000] + ([10000] if (include_10k_trad and not quick) else [])
    for n in cluster_scales:
        runs.append(RunConfig(geometry_mode="clusters",    extra_instances=n))
    for n in trad_scales:
        runs.append(RunConfig(geometry_mode="traditional", extra_instances=n))

    # --- 2. Orthogonal at default scene size (clusters, extra=0) ----------
    runs.append(RunConfig(vertex_format="compressed"))

    for ca in ("implicit", "get-sizes"):  # compact is the default
        runs.append(RunConfig(clas_alloc=ca))

    for bf in ("none", "fast-build"):  # fast-trace is the default
        runs.append(RunConfig(build_flags=bf))
        # Trad-mode variant so the build-flag analysis can also chart
        # traditional BLAS size + build time per flag.  Each one is a quick
        # ~6s run; adds 2 to the matrix.
        runs.append(RunConfig(geometry_mode="traditional", build_flags=bf))

    if not quick:
        # Precision sweep -- FLOAT32_3.  Default position_truncate_bits=0.
        for bits in (0, 4, 8, 12, 16, 20):
            runs.append(RunConfig(position_truncate_bits=bits))
        # Precision sweep -- COMPRESSED1.  Default compressed_bits=11.
        # Use multiples of 32 / 8 increments across the valid 1..16 range.
        for cb in (2, 4, 6, 8, 10, 12, 14, 16):
            runs.append(RunConfig(vertex_format="compressed", compressed_bits=cb))

    # --- 3. Trad alloc orthogonal -----------------------------------------
    runs.append(RunConfig(geometry_mode="traditional", trad_alloc="implicit"))

    # Deduplicate while preserving order
    seen: set[tuple] = set()
    deduped: list[RunConfig] = []
    for c in runs:
        key = tuple(sorted(c.__dict__.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


# =====================================================================================
# Run executor
# =====================================================================================

def run_one(exe: pathlib.Path, cfg: RunConfig, out_dir: pathlib.Path,
            bench_seconds: float, init_timeout: float, verbose: bool) -> RunResult:
    """Spawn the exe once for `cfg`, wait for it to exit with the JSON written,
    return the parsed result.  Init timeout accommodates trad-mode 10K which
    can take 60+ seconds before the bench window even begins.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{cfg.slug()}.json"
    if out_json.exists():
        out_json.unlink()
    cmd = [str(exe)] + cfg.to_cli() + [
        "--bench-seconds", str(bench_seconds),
        "--bench-out", str(out_json),
    ]
    if verbose:
        print(f"  exec: {' '.join(cmd[1:])}")

    timeout = init_timeout + bench_seconds + 15.0  # generous tail for AS rebuild + JSON write
    t0 = time.time()
    try:
        cp = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return RunResult(config=cfg, json_path=out_json, data={"_error": f"timeout after {timeout:.1f}s"},
                         wall_seconds=time.time() - t0)
    elapsed = time.time() - t0
    if cp.returncode != 0:
        return RunResult(config=cfg, json_path=out_json,
                         data={"_error": f"exit={cp.returncode}", "_stderr": cp.stderr[:200].decode(errors='replace')},
                         wall_seconds=elapsed)
    if not out_json.exists():
        return RunResult(config=cfg, json_path=out_json,
                         data={"_error": "no JSON file produced"}, wall_seconds=elapsed)
    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as e:
        return RunResult(config=cfg, json_path=out_json, data={"_error": f"parse: {e}"},
                         wall_seconds=elapsed)
    return RunResult(config=cfg, json_path=out_json, data=data, wall_seconds=elapsed)


# =====================================================================================
# Report rendering
# =====================================================================================

# Convert bytes to a friendly unit string.  Matches the sample's overlay
# convention so charts feel familiar.
def fmt_bytes(n: float) -> str:
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024**3):.2f} GB"
    if n >= 1024 * 1024:
        return f"{n / (1024**2):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n:.0f} B"

def fmt_us(n: float) -> str:
    if n >= 1000:
        return f"{n / 1000:.2f} ms"
    return f"{n:.1f} µs"


# Each Chart describes one ECharts/Chart.js panel.  We emit Chart.js datasets.
@dataclass
class Chart:
    title: str
    category: str
    description: str
    chart_type: str            # "line" or "bar"
    x_label: str
    y_label: str
    # x_scale: 'linear' (default, evenly spaced numeric ticks), 'log' (for
    # exponentially-distributed numeric data; rejects x<=0), or 'category'
    # (for set-of-discrete-labels where positions should be evenly spaced
    # regardless of numeric distance -- 0/100/1k/10k is exponential, so on
    # a linear axis the 0 and 100 would cluster against the left edge).
    x_scale: str = "linear"
    x_is_log: bool = False     # kept for back-compat; equivalent to x_scale='log'
    y_is_log: bool = False
    # series: dict from series_label -> list of (x_value, y_value, hover_text)
    series: dict[str, list[tuple[Any, float, str]]] = field(default_factory=dict)


def _g(d: dict, *path, default=None):
    """Safe nested dict get."""
    x = d
    for p in path:
        if not isinstance(x, dict) or p not in x:
            return default
        x = x[p]
    return x


def find_anomalies(runs: list[RunResult]) -> list[tuple[str, str]]:
    """Auto-detect numbers that may surprise readers of the report.  Not bugs
    in the SAMPLE -- bugs are fixed at source -- but real physical / numerical
    quirks worth highlighting so the reader doesn't misinterpret a chart.

    Returns list of (severity, message), severity = 'info' | 'warn'.
    """
    notes: list[tuple[str, str]] = []
    by_slug: dict[str, RunResult] = {r.config.slug(): r for r in runs if "_error" not in r.data}

    # Helper
    def fps(slug: str) -> float | None:
        r = by_slug.get(slug)
        if not r: return None
        return _g(r.data, "frame_perf", "fps", default=None)

    def mem(slug: str, field: str) -> int | None:
        r = by_slug.get(slug)
        if not r: return None
        return _g(r.data, "memory_bytes", field, default=None)

    # ----- 1. Extreme-precision FPS spikes (geometry collapse) -------------
    # If COMPRESSED1 at the lowest cb shows >20% FPS gain vs the next step,
    # flag it: that's almost certainly degenerate triangles letting rays miss.
    f_cb2 = fps("vtx-compressed_cb-2")
    f_cb4 = fps("vtx-compressed_cb-4")
    if f_cb2 and f_cb4 and f_cb2 > f_cb4 * 1.20:
        notes.append(("warn",
            f"COMPRESSED1 cb=2 reports {f_cb2:.1f} fps vs cb=4 at {f_cb4:.1f} fps (+{(f_cb2/f_cb4-1)*100:.0f}%).  "
            f"Almost certainly geometry collapse -- 2 bits/component = 4 quantization levels per axis = "
            f"degenerate triangles letting many rays miss.  Do NOT read this as a perf win."))

    # FLOAT32_3 extreme truncate sometimes shows a FPS drop -- different
    # failure mode (degeneracy + glass refraction going wrong) -> rays
    # bounce wrong, may hit MORE surfaces.
    f_t16 = fps("trunc-16")
    f_t20 = fps("trunc-20")
    if f_t16 and f_t20 and f_t20 < f_t16 * 0.92:
        notes.append(("info",
            f"FLOAT32_3 trunc=20 reports {f_t20:.1f} fps vs trunc=16 at {f_t16:.1f} fps "
            f"({(f_t20/f_t16-1)*100:+.0f}%).  Extreme truncation collapses positions; glass refraction "
            f"on degenerate geometry can cost more, not less.  Useful as a 'precision floor' marker."))

    # ----- 2. Cluster vs trad memory ratio at scale ------------------------
    # Show the headline number explicitly so readers see the DXR2 value prop.
    pairs = [(0,"default","geom-traditional"),
             (100,"n-100","geom-traditional_n-100"),
             (1000,"n-1000","geom-traditional_n-1000"),
             (10000,"n-10000","geom-traditional_n-10000")]
    for n, sc, st in pairs:
        mc = mem(sc, "total_as"); mt = mem(st, "total_as")
        if mc and mt and mc > 0:
            ratio = mt / mc
            notes.append(("info",
                f"At scene size +{n} clones: traditional uses {fmt_bytes(mt)} vs cluster {fmt_bytes(mc)} "
                f"(traditional is {ratio:.1f}x larger)."))

    # ----- 3. animated_pf_clas_actual sanity --------------------------------
    # Pre-fix bug: was always 0.  If we see 0 with non-zero alloc, the bug
    # came back.
    for r in runs:
        if "_error" in r.data: continue
        alloc  = _g(r.data, "memory_bytes", "animated_pf_clas_alloc",  default=0)
        actual = _g(r.data, "memory_bytes", "animated_pf_clas_actual", default=0)
        if alloc > 0 and actual == 0 and r.config.geometry_mode == "clusters":
            notes.append(("warn",
                f"{r.config.slug()}: animated_pf_clas_actual=0 with alloc={fmt_bytes(alloc)}.  "
                f"The one-shot CLAS-size readback didn't fire (regression of the m_animatedObjectEnabled-gate bug?)"))

    # ----- 4. pf_timing_valid never true (=> snap window never filled) -----
    # Not necessarily a bug -- short bench + slow scene -- but worth flagging
    # so the reader knows the displayed pf_*_us values are EMA, not exact mean.
    n_invalid = sum(1 for r in runs if "_error" not in r.data
                                    and not _g(r.data, "frame_perf", "pf_timing_valid", default=False))
    n_total   = sum(1 for r in runs if "_error" not in r.data)
    if n_invalid > 0:
        notes.append(("info",
            f"{n_invalid}/{n_total} runs report pf_timing_valid=false (the per-frame snap window of "
            f"60 samples didn't complete within the bench window).  Reported pf_*_us values for those "
            f"runs are EMA-smoothed (alpha=0.1) rather than exact 60-sample means.  Increase "
            f"--bench-seconds to make them all 'valid' (snap window ~= 60 / steady-state FPS seconds)."))

    return notes


def build_charts(runs: list[RunResult]) -> list[Chart]:
    """Consume the run set and produce a list of Chart objects."""
    # Filter out errored runs but report them later
    ok = [r for r in runs if "_error" not in r.data]

    def match(filt: dict) -> list[RunResult]:
        out = []
        for r in ok:
            cfg = r.config
            if all(getattr(cfg, k) == v for k, v in filt.items()):
                out.append(r)
        return out

    charts: list[Chart] = []

    # =========================================================================
    # CATEGORY: Scaling -- memory, build time, FPS, per-frame ops vs scene size
    # =========================================================================
    scale_filter = dict(vertex_format="float", clas_alloc="compact",
                        build_flags="fast-trace", static_rebuild="none",
                        position_truncate_bits=None, compressed_bits=None,
                        aa_samples=None)

    # --- Total AS memory vs scene size, by geometry mode -----------------
    c = Chart(
        title="Total acceleration-structure memory vs scene size",
        category="Scaling",
        description=("Sum of all BLAS/CLAS/TLAS bytes the sample reports for "
                     "the active geometry mode.  Lower is better.  The cluster "
                     "(DXR2) path amortises geometry via cluster templates + "
                     "BLAS-from-CLAS; the traditional (DXR1) path stores a full "
                     "monolithic BLAS per unique object."),
        chart_type="line", x_label="Extra instances (clones)", y_label="Total AS memory (bytes)",
        x_scale="category", y_is_log=True,
    )
    for mode in ("clusters", "traditional"):
        pts = []
        for r in match({**scale_filter, "geometry_mode": mode, "trad_alloc": "compact"}):
            x = r.config.extra_instances
            y = _g(r.data, "memory_bytes", "total_as", default=0)
            pts.append((x, y, f"{fmt_bytes(y)}  ({r.config.extra_instances} clones)"))
        pts.sort()
        c.series[mode] = pts
    if any(c.series.values()):
        charts.append(c)

    # --- Static AS build time vs scene size -------------------------------
    c = Chart(
        title="Static acceleration-structure build time vs scene size",
        category="Scaling",
        description=("Wall-clock ms to build the STATIC AS pipeline (CLAS + "
                     "BLAS + TLAS for cluster mode, BLAS + TLAS for traditional).  "
                     "Measured once at startup; per-frame anim work excluded.  "
                     "Lower is better."),
        chart_type="line", x_label="Extra instances (clones)", y_label="Build time (ms)",
        x_scale="category", y_is_log=True,
    )
    for mode in ("clusters", "traditional"):
        pts = []
        for r in match({**scale_filter, "geometry_mode": mode, "trad_alloc": "compact"}):
            x = r.config.extra_instances
            y = _g(r.data, "build_times_ms", "total", default=0)
            pts.append((x, y, f"{y:.1f} ms"))
        pts.sort()
        c.series[mode] = pts
    if any(c.series.values()):
        charts.append(c)

    # --- FPS vs scene size ------------------------------------------------
    c = Chart(
        title="Steady-state FPS vs scene size",
        category="Scaling",
        description=("Rolling-average FPS over the bench window.  Higher is "
                     "better.  Reflects total per-frame cost including ray "
                     "traversal, shading, AS rebuilds (anim BLAS), and present."),
        chart_type="line", x_label="Extra instances (clones)", y_label="FPS",
        x_scale="category",
    )
    for mode in ("clusters", "traditional"):
        pts = []
        for r in match({**scale_filter, "geometry_mode": mode, "trad_alloc": "compact"}):
            x = r.config.extra_instances
            y = _g(r.data, "frame_perf", "fps", default=0)
            pts.append((x, y, f"{y:.1f} fps  ({_g(r.data,'frame_perf','ms_per_frame',default=0):.1f} ms/frame)"))
        pts.sort()
        c.series[mode] = pts
    if any(c.series.values()):
        charts.append(c)

    # --- Per-frame TLAS rebuild time vs scene size ------------------------
    c = Chart(
        title="Per-frame TLAS rebuild time vs scene size",
        category="Scaling",
        description=("GPU time spent rebuilding the TLAS each frame, smoothed "
                     "by the per-frame snap window.  Includes all clones since "
                     "TLAS NumDescs must cover every instance.  Lower is better."),
        chart_type="line", x_label="Extra instances (clones)", y_label="TLAS rebuild (µs)",
        x_scale="category", y_is_log=True,
    )
    for mode in ("clusters", "traditional"):
        pts = []
        for r in match({**scale_filter, "geometry_mode": mode, "trad_alloc": "compact"}):
            x = r.config.extra_instances
            y = _g(r.data, "frame_perf", "pf_tlas_us", default=0)
            pts.append((x, y, fmt_us(y)))
        pts.sort()
        c.series[mode] = pts
    if any(c.series.values()):
        charts.append(c)

    # --- Per-frame animated BLAS-from-CLAS / animated trad BLAS rebuild ---
    c = Chart(
        title="Per-frame animated BLAS rebuild time vs scene size",
        category="Scaling",
        description=("Cluster mode: BUILD_BLAS_FROM_CLAS time for the animated "
                     "ball + clones.  Traditional mode: per-frame DXR1 BLAS "
                     "rebuild for the animated objects.  Both modes use the "
                     "same field (pf_blas_us); the meaning depends on the path. "
                     "Lower is better."),
        chart_type="line", x_label="Extra instances (clones)", y_label="Anim BLAS (µs)",
        x_scale="category", y_is_log=True,
    )
    for mode in ("clusters", "traditional"):
        pts = []
        for r in match({**scale_filter, "geometry_mode": mode, "trad_alloc": "compact"}):
            x = r.config.extra_instances
            y = _g(r.data, "frame_perf", "pf_blas_us", default=0)
            pts.append((x, y, fmt_us(y)))
        pts.sort()
        c.series[mode] = pts
    if any(c.series.values()):
        charts.append(c)

    # --- Per-frame INSTANTIATE: INTENTIONALLY OMITTED from the scaling group.
    # Data is constant ~12µs across scene scale -- the cluster path only runs
    # INSTANTIATE for the source animated sphere (768 clusters / 16384 tris);
    # animated clones share the source's per-frame CLAS via TLAS instancing,
    # so scale doesn't add INSTANTIATE work.  Putting it on a scaling chart
    # produces a flat-line non-signal.  Per-frame INSTANTIATE is still charted
    # in the Precision categories where it shows real precision sensitivity.

    # --- Scene cluster count vs scene size --------------------------------
    # Cluster + trad modes use IDENTICAL geometry so cluster/triangle counts
    # match across modes.  Earlier version emitted both lines; the second
    # was hidden behind the first (same y values).  Just plot one series.
    c = Chart(
        title="Scene cluster count vs scene size",
        category="Scaling",
        description=("Total cluster count in the active scene.  Counts are "
                     "identical between cluster mode and traditional mode "
                     "(both modes consume the same source geometry; trad mode "
                     "happens to also report cluster counts even though they "
                     "don't drive the BVH layout there).  Y axis is logarithmic. "
                     "If clones use lower LOD than the base scene, you'd see "
                     "sublinear growth -- in this sample they don't, so the "
                     "curve scales near-linearly with clone count."),
        chart_type="line", x_label="Extra instances (clones)", y_label="Clusters",
        x_scale="category", y_is_log=True,
    )
    pts = []
    for r in match({**scale_filter, "geometry_mode": "clusters", "trad_alloc": "compact"}):
        x = r.config.extra_instances
        y = _g(r.data, "scene", "total_clusters", default=0)
        pts.append((x, y, f"{y:,} clusters"))
    pts.sort()
    if pts:
        c.series["clusters"] = pts
        charts.append(c)

    # --- Scene triangle count vs scene size -------------------------------
    c = Chart(
        title="Scene triangle count vs scene size",
        category="Scaling",
        description=("Total triangle count.  Same caveat as clusters: identical "
                     "across cluster + trad modes.  Y axis logarithmic.  Compare "
                     "this curve's slope to the cluster-count curve above to see "
                     "if clones use higher or lower triangles-per-cluster than "
                     "the base scene."),
        chart_type="line", x_label="Extra instances (clones)", y_label="Triangles",
        x_scale="category", y_is_log=True,
    )
    pts = []
    for r in match({**scale_filter, "geometry_mode": "clusters", "trad_alloc": "compact"}):
        x = r.config.extra_instances
        y = _g(r.data, "scene", "total_triangles", default=0)
        pts.append((x, y, f"{y:,} triangles"))
    pts.sort()
    if pts:
        c.series["triangles"] = pts
        charts.append(c)

    # =========================================================================
    # CATEGORY: Vertex format effect at default scene size
    # =========================================================================
    base_default = dict(geometry_mode="clusters", clas_alloc="compact",
                        build_flags="fast-trace", static_rebuild="none",
                        extra_instances=0, position_truncate_bits=None,
                        compressed_bits=None, aa_samples=None, trad_alloc="compact")

    # Vertex format effect at default scene size: bar chart of static CLAS bytes + INSTANTIATE us.
    cm_runs = match({**base_default, "vertex_format": "float"})
    c1_runs = match({**base_default, "vertex_format": "compressed"})
    if cm_runs and c1_runs:
        cm = cm_runs[0]
        c1 = c1_runs[0]
        c = Chart(
            title="Vertex format: FLOAT32_3 vs COMPRESSED1 (default scene)",
            category="Vertex format",
            description=("Bar chart contrasting key metrics between the two "
                         "cluster vertex formats at the default scene (no clones).  "
                         "COMPRESSED1 typically saves ~3x on CLAS bytes and "
                         "cluster-input bytes at the cost of small INSTANTIATE "
                         "overhead and slight precision loss.  Lower bars = better "
                         "memory; higher bars on the FPS panel = better perf."),
            chart_type="bar", x_label="Metric", y_label="Value",
        )
        metrics = [
            ("CLAS bytes",        "memory_bytes.static_clas_actual"),
            ("Cluster input B",   "memory_bytes.static_cluster_input"),
            ("BLAS bytes",        "memory_bytes.static_blas_total"),
            ("Total AS bytes",    "memory_bytes.total_as"),
            ("INSTANTIATE µs",    "frame_perf.pf_instantiate_us"),
            ("FPS",               "frame_perf.fps"),
        ]
        for label, run in (("FLOAT32_3", cm), ("COMPRESSED1", c1)):
            pts = []
            for name, path in metrics:
                v = _g(run.data, *path.split("."), default=0)
                pts.append((name, v, f"{name}: {v:,.2f}"))
            c.series[label] = pts
        charts.append(c)

    # =========================================================================
    # CATEGORY: CLAS alloc strategy
    # =========================================================================
    rows = []
    for ca in ("implicit", "get-sizes", "compact"):
        m = match({**base_default, "vertex_format": "float", "clas_alloc": ca})
        if m:
            rows.append((ca, m[0]))
    if len(rows) >= 2:
        c = Chart(
            title="CLAS alloc strategy: memory + build time (cluster mode, default scene)",
            category="CLAS alloc",
            description=("How CLAS allocation strategy affects bytes resident "
                         "and build wall-clock at the default scene.  Implicit "
                         "lets the driver pick; GetSizes does an explicit prebuild "
                         "for exact-fit allocation (2 passes, more time, less "
                         "memory); Compact does implicit then a post-build compact "
                         "(extra pass, smallest result)."),
            chart_type="bar", x_label="CLAS alloc mode", y_label="Value",
        )
        c.series["CLAS alloc bytes"]   = [(ca, _g(r.data,"memory_bytes","static_clas_alloc",default=0),
                                           fmt_bytes(_g(r.data,"memory_bytes","static_clas_alloc",default=0)))
                                          for ca, r in rows]
        c.series["CLAS actual bytes"]  = [(ca, _g(r.data,"memory_bytes","static_clas_actual",default=0),
                                           fmt_bytes(_g(r.data,"memory_bytes","static_clas_actual",default=0)))
                                          for ca, r in rows]
        c.series["Static CLAS build (ms)"] = [(ca, _g(r.data,"build_times_ms","static_clas",default=0),
                                                f"{_g(r.data,'build_times_ms','static_clas',default=0):.1f} ms")
                                              for ca, r in rows]
        charts.append(c)

    # =========================================================================
    # CATEGORY: BVH build flags -- one chart per metric (separated y axes)
    # =========================================================================
    # Previously: one bar chart with FPS / build-time-ms / BLAS bytes on ONE
    # y-axis -- unreadable (values spanned 6 orders of magnitude).  Now: one
    # chart per metric.  Each chart contrasts the 3 build-flag values (NONE /
    # FAST_BUILD / FAST_TRACE) for BOTH cluster mode (driven by --build-flags)
    # and trad mode (same flag applied via DXR1 PREFER_FAST_BUILD / PREFER_FAST_TRACE).
    flag_to_runs = {}  # flag -> {"clusters": run, "traditional": run}
    # NOTE: drop build_flags from the base filter -- otherwise the filter pins
    # to "fast-trace" (the default) and excludes the bf-none / bf-fast-build
    # variants we're explicitly trying to chart.
    bf_filter_c = {k: v for k, v in base_default.items() if k != "build_flags"}
    bf_filter_c["vertex_format"] = "float"
    bf_filter_c["geometry_mode"] = "clusters"
    bf_filter_t = dict(bf_filter_c)
    bf_filter_t["geometry_mode"] = "traditional"
    flag_runs_cluster = match(bf_filter_c)
    flag_runs_trad    = match(bf_filter_t)
    for r in flag_runs_cluster:
        flag_to_runs.setdefault(r.config.build_flags, {})["clusters"]    = r
    for r in flag_runs_trad:
        flag_to_runs.setdefault(r.config.build_flags, {})["traditional"] = r

    # Stable category order matching the CLI tokens
    flag_order = [bf for bf in ("none", "fast-build", "fast-trace") if bf in flag_to_runs]

    if len(flag_order) >= 2:
        # Helper: returns (xs, fmt-fn) for the per-mode bar series
        def _flag_chart(title, desc, path, fmt_fn, y_label):
            c = Chart(
                title=title, category="Build flags",
                description=desc, chart_type="bar",
                x_label="Build flag", y_label=y_label,
            )
            for mode_key in ("clusters", "traditional"):
                pts = []
                for bf in flag_order:
                    r = flag_to_runs.get(bf, {}).get(mode_key)
                    if not r:
                        continue
                    v = _g(r.data, *path, default=None)
                    if v is None:
                        continue
                    pts.append((bf, v, f"{mode_key} {bf}: {fmt_fn(v)}"))
                if pts:
                    c.series[mode_key] = pts
            return c

        charts.append(_flag_chart(
            "Build flags: steady-state FPS",
            "Higher is better.  FAST_TRACE typically wins for traversal perf; FAST_BUILD trades it for build time.",
            ("frame_perf", "fps"),
            lambda v: f"{v:.1f} fps", "FPS",
        ))
        charts.append(_flag_chart(
            "Build flags: static-BLAS build time (ms)",
            "Wall-clock ms to build the static-scene BLAS.  FAST_BUILD should be fastest, FAST_TRACE slowest.",
            ("build_times_ms", "static_blas"),
            lambda v: f"{v:.2f} ms", "Build time (ms)",
        ))
        charts.append(_flag_chart(
            "Build flags: static-CLAS build time (ms, cluster mode only)",
            "Cluster-mode only -- trad mode reports 0 here.  The build-flag knob propagates into the CLAS build via BUILD_BLAS_FROM_CLAS flags.",
            ("build_times_ms", "static_clas"),
            lambda v: f"{v:.2f} ms", "CLAS build (ms)",
        ))
        # Cluster mode: STATIC_CLAS bytes
        charts.append(_flag_chart(
            "Build flags: cluster CLAS bytes (cluster mode)",
            "How the BVH build-flag preference affects CLAS storage (the LEAF arrays before BLAS_FROM_CLAS).  Often flat -- CLAS encoding is mostly independent of the BLAS flag.",
            ("memory_bytes", "static_clas_actual"),
            fmt_bytes, "CLAS bytes",
        ))
        # Cluster mode: STATIC_BLAS bytes (BLAS-from-CLAS output)
        charts.append(_flag_chart(
            "Build flags: cluster BLAS bytes (cluster mode)",
            "Bytes of the cluster-path BLAS-from-CLAS output.  Flag-sensitive: FAST_TRACE typically larger (deeper BVH), FAST_BUILD smaller.",
            ("memory_bytes", "static_blas_total"),
            fmt_bytes, "BLAS bytes",
        ))
        # Trad mode: BLAS bytes
        charts.append(_flag_chart(
            "Build flags: traditional BLAS bytes (trad mode)",
            "DXR1 monolithic BLAS storage as a function of build-flag preference.  Same FAST_BUILD-smaller / FAST_TRACE-larger pattern is typical.",
            ("memory_bytes", "traditional_blas_actual"),
            fmt_bytes, "BLAS bytes",
        ))

    # =========================================================================
    # CATEGORY: Precision sweep (FLOAT32_3 truncate bits)
    # =========================================================================
    # NOTE: drop position_truncate_bits from the filter so runs WITH the field
    # set are picked up.
    # X axis: "bits KEPT" = 23 - bits_truncated.  So left-to-right means
    # INCREASING precision -- same reading direction as the COMPRESSED1 chart.
    # Default truncate=0 (full 23 bits) -> rightmost; truncate=20 (3 bits kept)
    # -> leftmost.
    prec_float_filter = {k: v for k, v in base_default.items() if k != "position_truncate_bits"}
    prec_float_filter["vertex_format"] = "float"
    prec_float_filter["compressed_bits"] = None
    runs_prec_f = []
    for r in match(prec_float_filter):
        if r.config.position_truncate_bits is None:
            continue
        runs_prec_f.append(r)
    if runs_prec_f:
        # --- Cluster bytes chart: 3 lines ----------------------------------
        # static_clas_actual    : the 8 static scene objects' CLAS, built once
        # animated_template     : the animated sphere's cluster TEMPLATES (rest-pose
        #                         topology), built once at setup
        # animated_pf_clas_actual: the per-frame INSTANTIATE output (ephemeral)
        # All three are CLAS-storage in bytes -- apples-to-apples y axis.  Showing
        # all three lets the user see which pool each precision knob actually moves
        # (FLOAT32_3: static + per-frame both move; templates flat.  COMPRESSED1:
        #  static moves, templates + per-frame both flat).
        c = Chart(
            title="FLOAT32_3 precision: cluster bytes -- static clusters vs templates",
            category="Precision (FLOAT32_3)",
            description=("Three CLAS-storage pools at the default scene, all in bytes "
                         "(apples-to-apples).  'static clusters' = the 8 static scene "
                         "objects' CLAS (built once at init).  'templates' = the "
                         "animated sphere's cluster TEMPLATES (rest-pose topology, "
                         "built once at setup).  'template instances' = the per-frame "
                         "INSTANTIATE output for the animated sphere (rebuilt every "
                         "frame).  X axis = position-mantissa bits KEPT "
                         "(= 23 - --position-truncate N).  Higher = more precision.  "
                         "Same direction as the COMPRESSED1 chart below."),
            chart_type="line", x_label="Bits kept (higher = more precision)",
            y_label="Bytes",
        )
        pts_static, pts_tmpl, pts_pf = [], [], []
        for r in runs_prec_f:
            b = r.config.position_truncate_bits
            bits_kept = 23 - b
            ys  = _g(r.data,"memory_bytes","static_clas_actual",default=0)
            yt  = _g(r.data,"memory_bytes","animated_template",default=0)
            ypf = _g(r.data,"memory_bytes","animated_pf_clas_actual",default=0)
            pts_static.append((bits_kept, ys,  f"{fmt_bytes(ys)}  (truncate={b}, kept={bits_kept})"))
            pts_tmpl.append((bits_kept,   yt,  f"{fmt_bytes(yt)}  (truncate={b}, kept={bits_kept})"))
            pts_pf.append((bits_kept,     ypf, f"{fmt_bytes(ypf)}  (truncate={b}, kept={bits_kept})"))
        c.series["static clusters"] = sorted(pts_static)
        c.series["templates"]       = sorted(pts_tmpl)
        c.series["template instances"]  = sorted(pts_pf)
        charts.append(c)

        # --- INSTANTIATE time chart (animated-only metric) -----------------
        c = Chart(
            title="FLOAT32_3 precision: per-frame INSTANTIATE time (templates)",
            category="Precision (FLOAT32_3)",
            description=("GPU microseconds spent each frame in INSTANTIATE_CLUSTER_TEMPLATES "
                         "for the animated sphere.  Cluster templates are decoded + per-frame "
                         "vertex positions written; precision affects encoding cost.  Animated-"
                         "only metric -- the static path runs INSTANTIATE once at init, not "
                         "per frame."),
            chart_type="line", x_label="Bits kept (higher = more precision)",
            y_label="INSTANTIATE (µs)",
        )
        pts = []
        for r in runs_prec_f:
            b = r.config.position_truncate_bits
            bits_kept = 23 - b
            y = _g(r.data,"frame_perf","pf_instantiate_us",default=0)
            pts.append((bits_kept, y, f"{fmt_us(y)}  (truncate={b})"))
        c.series["templates"] = sorted(pts)
        charts.append(c)

        # --- FPS chart -----------------------------------------------------
        c = Chart(
            title="FLOAT32_3 precision: FPS",
            category="Precision (FLOAT32_3)",
            description=("Scene-wide FPS as precision varies.  Extreme low precision "
                         "can collapse geometry and cause rays to miss -- FPS spikes "
                         "are NOT perf wins, they're rendering failures."),
            chart_type="line", x_label="Bits kept (higher = more precision)", y_label="FPS",
        )
        pts = []
        for r in runs_prec_f:
            b = r.config.position_truncate_bits
            bits_kept = 23 - b
            y = _g(r.data,"frame_perf","fps",default=0)
            pts.append((bits_kept, y, f"{y:.1f} fps  (truncate={b})"))
        c.series["FPS"] = sorted(pts)
        charts.append(c)

    # =========================================================================
    # CATEGORY: Precision sweep (COMPRESSED1 bits/component)
    # =========================================================================
    prec_cmp_filter = {k: v for k, v in base_default.items() if k != "compressed_bits"}
    prec_cmp_filter["vertex_format"] = "compressed"
    prec_cmp_filter["position_truncate_bits"] = None
    runs_prec_c = []
    for r in match(prec_cmp_filter):
        if r.config.compressed_bits is None:
            continue
        runs_prec_c.append(r)
    if runs_prec_c:
        # --- Cluster bytes chart: 3 lines (static + templates + per-frame CLAS) ---
        c = Chart(
            title="COMPRESSED1 precision: cluster bytes -- static clusters vs templates",
            category="Precision (COMPRESSED1)",
            description=("Three CLAS-storage pools at the default scene, all in bytes "
                         "(apples-to-apples).  'static clusters' = the 8 static scene "
                         "objects' CLAS.  'templates' = the animated sphere's cluster "
                         "TEMPLATES (rest-pose topology).  'template instances' = the per-frame "
                         "INSTANTIATE output for the animated sphere.  X axis = "
                         "--compressed-bits N (bits/component for the COMPRESSED1 "
                         "shared-exponent quantizer; valid 1..16).  Higher = more precision.  "
                         "Same direction as the FLOAT32_3 chart above."),
            chart_type="line", x_label="Bits / component (higher = more precision)",
            y_label="Bytes",
        )
        pts_static, pts_tmpl, pts_pf = [], [], []
        for r in runs_prec_c:
            b = r.config.compressed_bits
            ys  = _g(r.data,"memory_bytes","static_clas_actual",default=0)
            yt  = _g(r.data,"memory_bytes","animated_template",default=0)
            ypf = _g(r.data,"memory_bytes","animated_pf_clas_actual",default=0)
            pts_static.append((b, ys,  f"{fmt_bytes(ys)}  (cb={b})"))
            pts_tmpl.append((b,   yt,  f"{fmt_bytes(yt)}  (cb={b})"))
            pts_pf.append((b,     ypf, f"{fmt_bytes(ypf)}  (cb={b})"))
        c.series["static clusters"] = sorted(pts_static)
        c.series["templates"]       = sorted(pts_tmpl)
        c.series["template instances"]  = sorted(pts_pf)
        charts.append(c)

        # --- INSTANTIATE time chart (animated-only) ---
        c = Chart(
            title="COMPRESSED1 precision: per-frame INSTANTIATE time (templates)",
            category="Precision (COMPRESSED1)",
            description=("GPU microseconds spent each frame INSTANTIATEing the animated "
                         "sphere's cluster templates.  Animated-only metric."),
            chart_type="line", x_label="Bits / component (higher = more precision)",
            y_label="INSTANTIATE (µs)",
        )
        pts = []
        for r in runs_prec_c:
            b = r.config.compressed_bits
            y = _g(r.data,"frame_perf","pf_instantiate_us",default=0)
            pts.append((b, y, f"{fmt_us(y)}  (cb={b})"))
        c.series["templates"] = sorted(pts)
        charts.append(c)

        # --- FPS chart ---
        c = Chart(
            title="COMPRESSED1 precision: FPS",
            category="Precision (COMPRESSED1)",
            description=("Scene-wide FPS as precision varies.  Very low values (<4) can "
                         "collapse geometry; FPS spikes there are rendering failures, "
                         "not perf wins."),
            chart_type="line", x_label="Bits / component (higher = more precision)",
            y_label="FPS",
        )
        pts = []
        for r in runs_prec_c:
            b = r.config.compressed_bits
            y = _g(r.data,"frame_perf","fps",default=0)
            pts.append((b, y, f"{y:.1f} fps  (cb={b})"))
        c.series["FPS"] = sorted(pts)
        charts.append(c)

    return charts


# =====================================================================================
# HTML rendering
# =====================================================================================

# Distinct colour palette for series.  Sized for typical category counts (~6).
PALETTE = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#79c0ff",
           "#56d364", "#e3b341", "#ff7b72", "#d2a8ff", "#7ee787"]


def _embed_chartjs() -> str:
    """Returns Chart.js source inlined into the HTML so the report works
    fully offline.  Looks for chart.umd.min.js next to this script; falls
    back to a CDN <script src> if not present.
    """
    local = THIS_DIR / "chart.umd.min.js"
    if local.exists():
        return f"<script>{local.read_text(encoding='utf-8')}</script>"
    # CDN fallback (allows the report to still work on machines that have
    # internet; otherwise charts are blank with a console error).
    return '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>'


def render_html(charts: list[Chart], runs: list[RunResult], adapter_info: dict,
                sweep_id: str, total_wall_s: float, bench_seconds: float) -> str:
    # Group charts by category, preserving order of first appearance
    cat_order: list[str] = []
    cat_charts: dict[str, list[Chart]] = {}
    for c in charts:
        if c.category not in cat_charts:
            cat_order.append(c.category)
            cat_charts[c.category] = []
        cat_charts[c.category].append(c)

    ok = [r for r in runs if "_error" not in r.data]
    errs = [r for r in runs if "_error" in r.data]

    # Auto-flagged observations
    notes = find_anomalies(runs)
    notes_html = ""
    if notes:
        ni = "".join(
            f'<li class="note-{sev}"><b>[{sev.upper()}]</b> {msg}</li>'
            for sev, msg in notes
        )
        notes_html = f'''
<div class="notes-card">
  <h2>Notes &amp; observations</h2>
  <ul>{ni}</ul>
</div>
'''

    chart_js_blocks = []
    chart_html_blocks = []
    for cat in cat_order:
        chart_html_blocks.append(f'<h2 class="cat">{cat}</h2>')
        for i, c in enumerate(cat_charts[cat]):
            cid = f"chart_{cat_order.index(cat)}_{i}_{hashlib.md5(c.title.encode()).hexdigest()[:6]}"
            chart_html_blocks.append(f'''
<div class="chart-card">
  <h3>{c.title}</h3>
  <p class="desc">{c.description}</p>
  <div class="chart-wrap"><canvas id="{cid}"></canvas></div>
</div>
''')
            # Build datasets
            datasets = []
            for j, (label, pts) in enumerate(c.series.items()):
                color = PALETTE[j % len(PALETTE)]
                # For category x scale, x values become string labels (evenly
                # spaced regardless of numeric distance) -- otherwise 0/100/1k
                # collapse against the left edge on a linear scale.  For bar
                # charts same thing.  For linear/log x scales we pass numbers.
                if c.chart_type == "bar" or c.x_scale == "category":
                    data = [{"x": str(x), "y": y} for x, y, _t in pts]
                else:
                    data = [{"x": x, "y": y} for x, y, _t in pts]
                datasets.append({
                    "label": label,
                    "data": data,
                    "borderColor": color,
                    "backgroundColor": color + "88",
                    "tension": 0.15,
                    "pointRadius": 4 if c.chart_type == "line" else 0,
                    "borderWidth": 2,
                })
            # Tooltip text from third tuple element
            tip_map = {}
            for label, pts in c.series.items():
                tip_map[label] = {str(x): t for x, _y, t in pts}
            # Resolve x scale type: bar charts always use 'category';
            # line charts use Chart.x_scale, with x_is_log as legacy alias for 'log'.
            if c.chart_type == "bar":
                x_type = "category"
            elif c.x_is_log:
                x_type = "logarithmic"
            elif c.x_scale == "log":
                x_type = "logarithmic"
            elif c.x_scale == "category":
                x_type = "category"
            else:
                x_type = "linear"
            opts = {
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "x": {
                        "type": x_type,
                        "title": {"display": True, "text": c.x_label, "color": "#9da7b3"},
                        "ticks": {"color": "#9da7b3"},
                        "grid": {"color": "#22272e"},
                    },
                    "y": {
                        "type": "logarithmic" if c.y_is_log else "linear",
                        "title": {"display": True, "text": c.y_label, "color": "#9da7b3"},
                        "ticks": {"color": "#9da7b3"},
                        "grid": {"color": "#22272e"},
                    },
                },
                "plugins": {
                    "legend": {"labels": {"color": "#e6edf3"}},
                    "tooltip": {
                        "callbacks": {},  # custom callback added in JS below
                    },
                },
            }

            chart_js_blocks.append(f'''
(function() {{
  const tipMap = {json.dumps(tip_map)};
  const cfg = {{
    type: {json.dumps('bar' if c.chart_type=='bar' else 'line')},
    data: {{ datasets: {json.dumps(datasets)} }},
    options: {json.dumps(opts)},
  }};
  cfg.options.plugins.tooltip.callbacks = {{
    label: function(ctx) {{
      const lbl = ctx.dataset.label;
      const key = String(ctx.parsed.x);
      const t = (tipMap[lbl] && tipMap[lbl][key]) || (lbl + ': ' + ctx.parsed.y);
      return t;
    }}
  }};
  new Chart(document.getElementById({json.dumps(cid)}).getContext('2d'), cfg);
}})();
''')

    # Errors section
    err_html = ""
    if errs:
        err_rows = []
        for r in errs:
            err_rows.append(f"<tr><td>{r.config.slug()}</td><td>{r.data.get('_error','?')}</td></tr>")
        err_html = f'''
<div class="errors">
  <h2>Failed runs ({len(errs)})</h2>
  <table><thead><tr><th>Config</th><th>Error</th></tr></thead><tbody>
  {"".join(err_rows)}
  </tbody></table>
</div>
'''

    # Run table (per-config raw numbers)
    run_table_rows = []
    for r in ok:
        d = r.data
        run_table_rows.append(
            f"<tr>"
            f"<td><code>{r.config.slug()}</code></td>"
            f"<td>{r.config.geometry_mode}</td>"
            f"<td>{r.config.vertex_format}</td>"
            f"<td>{r.config.extra_instances}</td>"
            f"<td>{r.config.clas_alloc}</td>"
            f"<td>{r.config.build_flags}</td>"
            f"<td>{_g(d,'frame_perf','fps',default=0):.1f}</td>"
            f"<td>{_g(d,'frame_perf','ms_per_frame',default=0):.2f}</td>"
            f"<td>{_g(d,'build_times_ms','total',default=0):.1f}</td>"
            f"<td>{fmt_bytes(_g(d,'memory_bytes','total_as',default=0))}</td>"
            f"<td>{_g(d,'frame_perf','pf_tlas_us',default=0):.1f}</td>"
            f"<td>{_g(d,'frame_perf','pf_blas_us',default=0):.1f}</td>"
            f"<td>{_g(d,'frame_perf','pf_instantiate_us',default=0):.1f}</td>"
            f"<td>{r.wall_seconds:.1f}</td>"
            f"</tr>"
        )
    runs_table = '''
<div class="runs-table">
  <h2>All runs ({n} configs)</h2>
  <table><thead><tr>
    <th>Config slug</th><th>Geom</th><th>Vtx</th><th>Clones</th><th>CLAS alloc</th><th>Build flags</th>
    <th>FPS</th><th>ms/frame</th><th>Build ms</th><th>Total AS</th>
    <th>TLAS µs</th><th>Anim BLAS µs</th><th>INST µs</th><th>Wall s</th>
  </tr></thead><tbody>
  {rows}
  </tbody></table>
</div>
'''.format(n=len(ok), rows="\n".join(run_table_rows))

    # Adapter card
    adapter_card = f'''
<div class="adapter-card">
  <h2>Test system</h2>
  <table>
    <tr><th>Adapter</th><td>{adapter_info.get("description","?")}</td></tr>
    <tr><th>Driver</th><td>{adapter_info.get("driver_version","?") or "(unknown)"}</td></tr>
    <tr><th>VID/PID</th><td>{adapter_info.get("vendor_id","?"):04X} / {adapter_info.get("device_id","?"):04X}</td></tr>
    <tr><th>Dedicated VRAM</th><td>{adapter_info.get("dedicated_vram_bytes",0) / (1024**2):.0f} MB</td></tr>
    <tr><th>Clusters supported</th><td>{"yes" if adapter_info.get("clusters_supported") else "no (DXR1 only)"}</td></tr>
    <tr><th>Sweep id</th><td><code>{sweep_id}</code></td></tr>
    <tr><th>Runs / wall</th><td>{len(ok)} succeeded, {len(errs)} failed, total {total_wall_s:.0f} s ({len(ok)} × {bench_seconds:.1f} s bench window)</td></tr>
    <tr><th>Generated</th><td>{_dt.datetime.now().isoformat(timespec="seconds")}</td></tr>
  </table>
</div>
'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DXR2 Clustered Geometry Sweep -- {adapter_info.get("description","?")}</title>
{_embed_chartjs()}
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 0; padding: 24px;
         background: #0d1117; color: #e6edf3; }}
  h1   {{ margin: 0 0 4px 0; font-size: 22px; }}
  h2   {{ margin: 32px 0 12px 0; font-size: 18px; color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 4px; }}
  h2.cat {{ color: #f0c378; }}
  h3   {{ margin: 0 0 6px 0; font-size: 15px; color: #e6edf3; }}
  .desc {{ margin: 0 0 8px 0; color: #9da7b3; font-size: 12px; line-height: 1.4; }}
  .subtitle {{ color: #9da7b3; margin-bottom: 12px; }}
  .chart-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px;
                 padding: 14px; margin: 14px 0; }}
  .chart-wrap {{ position: relative; height: 320px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid #30363d; }}
  th     {{ color: #9da7b3; font-weight: 600; }}
  code   {{ font-family: Consolas, "Cascadia Code", monospace; color: #79c0ff; }}
  .adapter-card, .runs-table, .errors, .notes-card {{ background: #161b22; border: 1px solid #30363d;
                                          border-radius: 6px; padding: 14px; margin: 16px 0; }}
  .errors {{ border-color: #f85149; }}
  .notes-card ul {{ margin: 0; padding-left: 20px; }}
  .notes-card li {{ margin: 6px 0; font-size: 12px; line-height: 1.4; }}
  .note-warn b {{ color: #d29922; }}
  .note-info b {{ color: #58a6ff; }}
  .runs-table table {{ font-size: 11px; }}
</style>
</head>
<body>
<h1>DXR2 Clustered-Geometry Parameter Sweep</h1>
<div class="subtitle">Self-contained benchmark report. Hover bars/lines for raw values; click legend entries to toggle series.</div>

{adapter_card}

{notes_html}

{"".join(chart_html_blocks)}

{err_html}

{runs_table}

<script>
{"".join(chart_js_blocks)}
</script>
</body>
</html>
'''


# =====================================================================================
# Driver
# =====================================================================================

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default=str(DEFAULT_EXE), help="Path to sample exe")
    ap.add_argument("--out", default=None, help="Output HTML report path (default: results/<adapter>/<sweep_id>/report.html)")
    ap.add_argument("--bench-seconds", type=float, default=5.0, help="Measurement window per run (s)")
    ap.add_argument("--init-timeout", type=float, default=120.0, help="Init timeout per run (s); trad-mode 10K needs ~90s")
    ap.add_argument("--quick", action="store_true", help="Short matrix (skip 10K + precision sweeps)")
    ap.add_argument("--no-10k-trad", action="store_true", help="Skip the 10K traditional config (slow init)")
    ap.add_argument("--no-10k", action="store_true", help="Skip the 10K configs entirely")
    ap.add_argument("--verbose", action="store_true", help="Print each invocation")
    ap.add_argument("--no-run", action="store_true", help="Skip exe runs; reuse existing JSONs")
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT),
                    help="Root dir for per-machine results (default: benchmark/results/)")
    args = ap.parse_args(argv)

    exe = pathlib.Path(args.exe)
    if not exe.exists():
        print(f"ERROR: exe not found: {exe}", file=sys.stderr)
        return 2

    # Sweep id = timestamp + short hash of the matrix definition, so re-runs
    # with the same matrix can be diffed.
    sweep_id = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"sweep id: {sweep_id}")
    print(f"exe:      {exe}")
    print(f"bench:    {args.bench_seconds}s per run")

    # Build matrix
    include_10k = not (args.no_10k_trad or args.no_10k)
    matrix = build_sweep_matrix(quick=args.quick, include_10k_trad=include_10k)
    if args.no_10k:
        matrix = [c for c in matrix if c.extra_instances != 10000]
    print(f"matrix:   {len(matrix)} unique configs")

    # First we need adapter info -- run the default config once and grab the
    # adapter card.  If --no-run, look for any existing JSON to extract from.
    results_root = pathlib.Path(args.results_root)
    adapter_dir_placeholder = results_root / "_pending" / sweep_id
    adapter_dir_placeholder.mkdir(parents=True, exist_ok=True)

    runs: list[RunResult] = []
    t_total = time.time()

    if args.no_run:
        # Reuse: load any prior JSONs that match matrix slugs from the most
        # recent NON-EMPTY results dir.  (Earlier --no-run attempts can leave
        # empty placeholder dirs behind; skip them so we don't accidentally
        # report "missing JSON" against a stub.)
        existing_root_candidates = sorted(results_root.glob("*/*/"), reverse=True)
        existing_root_candidates = [d for d in existing_root_candidates if list(d.glob("*.json"))]
        if not existing_root_candidates:
            print("ERROR: --no-run but no non-empty existing results dirs", file=sys.stderr)
            return 2
        base = existing_root_candidates[0]
        print(f"reusing JSONs from {base}")
        for cfg in matrix:
            jp = base / f"{cfg.slug()}.json"
            if jp.exists():
                runs.append(RunResult(config=cfg, json_path=jp, data=json.loads(jp.read_text(encoding="utf-8")), wall_seconds=0.0))
            else:
                runs.append(RunResult(config=cfg, json_path=jp, data={"_error": "missing JSON"}, wall_seconds=0.0))
    else:
        for i, cfg in enumerate(matrix):
            t0 = time.time()
            print(f"[{i+1:>2}/{len(matrix)}] {cfg.slug():<50} ", end="", flush=True)
            r = run_one(exe, cfg, adapter_dir_placeholder, args.bench_seconds, args.init_timeout, args.verbose)
            elapsed = time.time() - t0
            if "_error" in r.data:
                print(f"FAIL ({elapsed:.1f}s) -- {r.data['_error']}")
            else:
                fps = _g(r.data, "frame_perf", "fps", default=0)
                tot = _g(r.data, "memory_bytes", "total_as", default=0)
                print(f"OK ({elapsed:.1f}s) -- {fps:.1f} fps, {fmt_bytes(tot)} AS")
            runs.append(r)

    total_wall = time.time() - t_total

    # Find adapter info from the first OK run
    ok_runs = [r for r in runs if "_error" not in r.data]
    if not ok_runs:
        print("ERROR: no successful runs; cannot render report", file=sys.stderr)
        for r in runs:
            print(f"  {r.config.slug()}: {r.data.get('_error','?')}")
        return 1
    adapter_info = ok_runs[0].data.get("adapter", {})
    adapter_desc = adapter_info.get("description", "unknown")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", adapter_desc).strip("_")[:48] or "unknown"
    final_dir = results_root / safe / sweep_id
    final_dir.mkdir(parents=True, exist_ok=True)

    # Move JSONs from placeholder dir into the per-adapter dir
    for r in runs:
        if r.json_path.exists() and r.json_path.parent == adapter_dir_placeholder:
            new_path = final_dir / r.json_path.name
            r.json_path.replace(new_path)
            r.json_path = new_path
    if adapter_dir_placeholder.exists():
        try:
            adapter_dir_placeholder.rmdir()
            (adapter_dir_placeholder.parent).rmdir()
        except OSError:
            pass

    # Build + render
    print(f"\nbuilding charts...")
    charts = build_charts(runs)
    print(f"  {len(charts)} charts across {len(set(c.category for c in charts))} categories")

    html_out = pathlib.Path(args.out) if args.out else (final_dir / "report.html")
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(charts, runs, adapter_info, sweep_id, total_wall, args.bench_seconds)
    html_out.write_text(html, encoding="utf-8")
    print(f"  HTML: {html_out}  ({html_out.stat().st_size/1024:.1f} KB)")
    print(f"  JSONs: {final_dir}/  ({len([r for r in runs if r.json_path.exists()])} files)")
    print(f"\nDONE.  Total wall: {total_wall:.0f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
