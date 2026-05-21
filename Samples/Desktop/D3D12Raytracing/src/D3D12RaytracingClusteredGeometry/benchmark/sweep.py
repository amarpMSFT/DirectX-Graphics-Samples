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
    scales = [0, 100, 1000] + ([10000] if include_10k_trad and not quick else [])
    for n in scales:
        runs.append(RunConfig(geometry_mode="clusters",    extra_instances=n))
        # Skip the trad 10K config unless explicitly opted in -- 60-90 s init,
        # dominates the sweep wall-clock.
        if not (n == 10000 and not include_10k_trad):
            runs.append(RunConfig(geometry_mode="traditional", extra_instances=n))

    # --- 2. Orthogonal at default scene size (clusters, extra=0) ----------
    runs.append(RunConfig(vertex_format="compressed"))

    for ca in ("implicit", "get-sizes"):  # compact is the default
        runs.append(RunConfig(clas_alloc=ca))

    for bf in ("none", "fast-build"):  # fast-trace is the default
        runs.append(RunConfig(build_flags=bf))

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
    x_is_log: bool = False
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
        x_is_log=False, y_is_log=True,
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
        x_is_log=False, y_is_log=True,
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
        x_is_log=False, y_is_log=False,
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
        x_is_log=False, y_is_log=True,
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
        x_is_log=False, y_is_log=True,
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

    # --- Per-frame INSTANTIATE (cluster-only) -----------------------------
    c = Chart(
        title="Per-frame INSTANTIATE_CLUSTER_TEMPLATES time vs scene size (cluster mode)",
        category="Scaling",
        description=("Cluster-mode-only: GPU time for the per-frame INSTANTIATE "
                     "op that materialises animated CLAS from rest-pose templates.  "
                     "Sensitive to vertex format + precision since it re-encodes "
                     "per-frame.  Lower is better."),
        chart_type="line", x_label="Extra instances (clones)", y_label="INSTANTIATE (µs)",
        x_is_log=False, y_is_log=True,
    )
    pts = []
    for r in match({**scale_filter, "geometry_mode": "clusters", "trad_alloc": "compact"}):
        x = r.config.extra_instances
        y = _g(r.data, "frame_perf", "pf_instantiate_us", default=0)
        pts.append((x, y, fmt_us(y)))
    pts.sort()
    if pts:
        c.series["clusters"] = pts
        charts.append(c)

    # --- Cluster + triangle count vs scene size (sanity / cross-check) ----
    c = Chart(
        title="Cluster count and triangle count vs scene size",
        category="Scaling",
        description=("Total clusters in the active geometry mode (cluster mode "
                     "reports real clusters; trad mode reports 0 since it has "
                     "no concept of clusters at the BVH level).  Triangle count "
                     "is geometry-mode-independent.  Use as a sanity check "
                     "that your sweep actually scaled."),
        chart_type="line", x_label="Extra instances (clones)", y_label="Count",
        x_is_log=False, y_is_log=True,
    )
    for mode in ("clusters", "traditional"):
        for field_, label_suffix in (("total_clusters", " clusters"), ("total_triangles", " triangles")):
            if mode == "traditional" and field_ == "total_clusters":
                continue
            pts = []
            for r in match({**scale_filter, "geometry_mode": mode, "trad_alloc": "compact"}):
                x = r.config.extra_instances
                y = _g(r.data, "scene", field_, default=0)
                pts.append((x, y, f"{y:,}"))
            pts.sort()
            if pts:
                c.series[f"{mode}: {field_}"] = pts
    if any(c.series.values()):
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
    # CATEGORY: BVH build flags
    # =========================================================================
    rows = []
    for bf in ("none", "fast-build", "fast-trace"):
        m = match({**base_default, "vertex_format": "float", "build_flags": bf})
        if m:
            rows.append((bf, m[0]))
    if len(rows) >= 2:
        c = Chart(
            title="BVH build flags: FPS + build time + BLAS bytes (cluster mode, default scene)",
            category="Build flags",
            description=("Effect of the cluster-path BUILD_BLAS_FROM_CLAS flag "
                         "(NONE / FAST_BUILD / FAST_TRACE).  FAST_TRACE typically "
                         "produces the best traversal perf (highest FPS) at the cost "
                         "of slower build; FAST_BUILD is the reverse.  NONE lets the "
                         "driver pick, usually = FAST_TRACE."),
            chart_type="bar", x_label="Build flag", y_label="Value",
        )
        c.series["FPS"] = [(bf, _g(r.data,"frame_perf","fps",default=0), f"{_g(r.data,'frame_perf','fps',default=0):.1f}") for bf, r in rows]
        c.series["BLAS build (ms)"] = [(bf, _g(r.data,"build_times_ms","static_blas",default=0), f"{_g(r.data,'build_times_ms','static_blas',default=0):.2f} ms") for bf, r in rows]
        c.series["BLAS bytes"] = [(bf, _g(r.data,"memory_bytes","static_blas_total",default=0), fmt_bytes(_g(r.data,"memory_bytes","static_blas_total",default=0))) for bf, r in rows]
        charts.append(c)

    # =========================================================================
    # CATEGORY: Precision sweep (FLOAT32_3 truncate bits)
    # =========================================================================
    # NOTE: drop position_truncate_bits from the filter so runs WITH the field
    # set are picked up (base_default has it set to None for orthogonal-cards;
    # the precision sweep is the one case where we want only the runs that DID
    # set it).
    prec_float_filter = {k: v for k, v in base_default.items() if k != "position_truncate_bits"}
    prec_float_filter["vertex_format"] = "float"
    prec_float_filter["compressed_bits"] = None
    pts_mem = []
    pts_inst = []
    pts_fps  = []
    for r in match(prec_float_filter):
        b = r.config.position_truncate_bits
        if b is None:
            continue
        pts_mem.append((b, _g(r.data,"memory_bytes","static_clas_actual",default=0), fmt_bytes(_g(r.data,"memory_bytes","static_clas_actual",default=0))))
        pts_inst.append((b, _g(r.data,"frame_perf","pf_instantiate_us",default=0), fmt_us(_g(r.data,"frame_perf","pf_instantiate_us",default=0))))
        pts_fps.append((b,  _g(r.data,"frame_perf","fps",default=0), f"{_g(r.data,'frame_perf','fps',default=0):.1f} fps"))
    if pts_mem:
        for chart_title, y_lab, lab, pts, ylog in [
            ("FLOAT32_3 precision sweep: CLAS bytes",          "CLAS bytes",        "CLAS bytes",        pts_mem,  False),
            ("FLOAT32_3 precision sweep: INSTANTIATE time",    "INSTANTIATE (µs)", "INSTANTIATE (µs)", pts_inst, False),
            ("FLOAT32_3 precision sweep: FPS",                  "FPS",               "FPS",               pts_fps,  False),
        ]:
            c = Chart(
                title=chart_title,
                category="Precision (FLOAT32_3)",
                description=("--position-truncate N zeroes the low N bits of each "
                             "FLOAT32_3 position mantissa, increasing CLAS quantizer "
                             "efficiency.  X axis = bits zeroed (0 = full precision; "
                             "23 = signs+exponent only).  Extreme truncations (>16) "
                             "may collapse geometry to a point -- FPS can spike "
                             "because rays miss everything; don't read it as a perf win."),
                chart_type="line", x_label="Bits truncated", y_label=y_lab, y_is_log=ylog,
            )
            c.series[lab] = sorted(pts)
            charts.append(c)

    # =========================================================================
    # CATEGORY: Precision sweep (COMPRESSED1 bits/component)
    # =========================================================================
    prec_cmp_filter = {k: v for k, v in base_default.items() if k != "compressed_bits"}
    prec_cmp_filter["vertex_format"] = "compressed"
    prec_cmp_filter["position_truncate_bits"] = None
    pts_mem = []; pts_inst = []; pts_fps = []
    for r in match(prec_cmp_filter):
        b = r.config.compressed_bits
        if b is None:
            continue
        pts_mem.append((b, _g(r.data,"memory_bytes","static_clas_actual",default=0), fmt_bytes(_g(r.data,"memory_bytes","static_clas_actual",default=0))))
        pts_inst.append((b, _g(r.data,"frame_perf","pf_instantiate_us",default=0), fmt_us(_g(r.data,"frame_perf","pf_instantiate_us",default=0))))
        pts_fps.append((b, _g(r.data,"frame_perf","fps",default=0), f"{_g(r.data,'frame_perf','fps',default=0):.1f} fps"))
    if pts_mem:
        for chart_title, y_lab, lab, pts in [
            ("COMPRESSED1 precision sweep: CLAS bytes",       "CLAS bytes",        "CLAS bytes",        pts_mem),
            ("COMPRESSED1 precision sweep: INSTANTIATE time", "INSTANTIATE (µs)", "INSTANTIATE (µs)", pts_inst),
            ("COMPRESSED1 precision sweep: FPS",              "FPS",               "FPS",               pts_fps),
        ]:
            c = Chart(
                title=chart_title,
                category="Precision (COMPRESSED1)",
                description=("--compressed-bits N sets bits/component for the "
                             "COMPRESSED1 shared-exponent quantizer.  Higher = "
                             "more precision; valid range 1..16.  Very low values "
                             "(<4) collapse geometry; FPS spikes can mean rays miss."),
                chart_type="line", x_label="Bits / component", y_label=y_lab,
            )
            c.series[lab] = sorted(pts)
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
                # For bar charts the x is a string label; for line it's numeric.
                if c.chart_type == "bar":
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
            opts = {
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "x": {
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
            if c.x_is_log:
                opts["scales"]["x"]["type"] = "logarithmic"

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
        # recent results dir
        existing_root_candidates = sorted(results_root.glob("*/*/"), reverse=True)
        if not existing_root_candidates:
            print("ERROR: --no-run but no existing results", file=sys.stderr)
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
