#!/usr/bin/env python3
"""
compare.py - cross-machine comparison of DXR2 sweep reports.

Takes 2+ sweep result directories (each = one machine's run), overlays
the same per-chart metrics across machines so adapter/driver differences
pop visually.  Output is a self-contained HTML report.

Usage:
    python compare.py path\\to\\sweep_dir_A path\\to\\sweep_dir_B [--out compare.html]
    python compare.py results\\rtx4090\\20260521T000141 results\\warp\\20260521T010000

Each input directory must contain *.json files produced by sweep.py.
Charts are matched by config slug across machines -- only configs that ALL
machines ran end up in the comparison (use --partial to include configs
that only a subset ran).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any

THIS_DIR = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class MachineRun:
    """One JSON result loaded from one machine's sweep dir."""
    slug: str            # filename stem (e.g. "default", "n-1000")
    data: dict[str, Any]


@dataclass
class MachineReport:
    """All runs for one machine."""
    label: str           # short label used in chart series (e.g. "RTX 4090")
    dir: pathlib.Path
    adapter: dict[str, Any]
    runs: dict[str, MachineRun]  # keyed by slug

    @property
    def slugs(self) -> set[str]:
        return set(self.runs.keys())


def load_machine_report(d: pathlib.Path, label_override: str | None) -> MachineReport:
    """Load a machine's sweep directory. Adapter info is taken from the first
    OK JSON (they should all agree)."""
    if not d.is_dir():
        raise SystemExit(f"not a directory: {d}")
    runs: dict[str, MachineRun] = {}
    adapter: dict[str, Any] | None = None
    for jp in sorted(d.glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  WARN: failed to parse {jp.name}: {e}", file=sys.stderr)
            continue
        if "_error" in data:
            continue
        if adapter is None:
            adapter = data.get("adapter", {})
        runs[jp.stem] = MachineRun(slug=jp.stem, data=data)
    if not runs:
        raise SystemExit(f"no usable JSON files found in {d}")
    label = label_override or (adapter.get("description", d.name) if adapter else d.name)
    return MachineReport(label=label, dir=d, adapter=adapter or {}, runs=runs)


# ---------------------------------------------------------------------------
# Helpers (mirrored from sweep.py)
# ---------------------------------------------------------------------------

def _g(d: dict, *path, default=None):
    x = d
    for p in path:
        if not isinstance(x, dict) or p not in x:
            return default
        x = x[p]
    return x

def fmt_bytes(n: float) -> str:
    if n >= 1024 ** 3: return f"{n / 1024**3:.2f} GB"
    if n >= 1024 ** 2: return f"{n / 1024**2:.2f} MB"
    if n >= 1024:      return f"{n / 1024:.1f} KB"
    return f"{n:.0f} B"

def fmt_us(n: float) -> str:
    if n >= 1000: return f"{n / 1000:.2f} ms"
    return f"{n:.1f} µs"


# Each "ChartDef" is a recipe for one comparison chart.  It pulls a SCALAR
# metric out of a result JSON, and groups runs into points on the x axis
# based on the config it cares about.  Series = machines.
@dataclass
class ChartDef:
    title: str
    category: str
    description: str
    chart_type: str           # "line" or "bar"
    x_label: str
    y_label: str
    y_is_log: bool = False
    # x_keys: ordered list of (slug, x_value) pairs to plot. Each machine's
    # series shows the y value for the matching slug.
    x_keys: list[tuple[str, Any]] = field(default_factory=list)
    # metric_path: tuple of dict keys to extract the scalar from result data
    metric_path: tuple[str, ...] = ()
    # value_formatter: optional fn(float) -> str for tooltips
    value_formatter: Any = None


def build_chart_defs(machines: list[MachineReport]) -> list[ChartDef]:
    """Define the cross-machine charts based on the configs found in the
    machine reports.  Only configs that all-or-most machines ran get charted.
    """
    # Find common slugs (intersection)
    all_slugs = set()
    for m in machines:
        all_slugs |= m.slugs
    common_slugs = set.intersection(*[m.slugs for m in machines]) if machines else set()

    defs: list[ChartDef] = []

    # --- Scaling: extra-instances on X -----------------------------------
    # Cluster mode runs: 'default' (=0), 'n-100', 'n-1000', 'n-10000'
    cluster_xs = [("default", 0), ("n-100", 100), ("n-1000", 1000), ("n-10000", 10000)]
    cluster_xs = [(s, x) for s, x in cluster_xs if s in all_slugs]
    trad_xs    = [("geom-traditional", 0), ("geom-traditional_n-100", 100),
                  ("geom-traditional_n-1000", 1000), ("geom-traditional_n-10000", 10000)]
    trad_xs    = [(s, x) for s, x in trad_xs if s in all_slugs]

    for mode_label, xs in (("clusters", cluster_xs), ("traditional", trad_xs)):
        if len(xs) < 2:
            continue
        defs.append(ChartDef(
            title=f"Total AS memory vs scene size ({mode_label})",
            category="Scaling -- memory",
            description=("Compares total acceleration-structure bytes across "
                         "machines at the same scene scale.  Differences here "
                         "are usually adapter-architecture / driver-allocator "
                         "specific (e.g. NVIDIA vs Intel vs WARP)."),
            chart_type="line", x_label="Extra instances", y_label="Total AS (bytes)",
            y_is_log=True, x_keys=xs,
            metric_path=("memory_bytes", "total_as"),
            value_formatter=fmt_bytes,
        ))
        defs.append(ChartDef(
            title=f"Static build time vs scene size ({mode_label})",
            category="Scaling -- build time",
            description=("Wall-clock ms to build the static AS pipeline.  "
                         "Where machines diverge most is in the parallel "
                         "compaction passes."),
            chart_type="line", x_label="Extra instances", y_label="Build time (ms)",
            y_is_log=True, x_keys=xs,
            metric_path=("build_times_ms", "total"),
            value_formatter=lambda v: f"{v:.1f} ms",
        ))
        defs.append(ChartDef(
            title=f"Steady-state FPS vs scene size ({mode_label})",
            category="Scaling -- FPS",
            description=("Higher is better.  Reveals adapter-class differences "
                         "(consumer vs workstation vs WARP) more sharply than "
                         "memory does."),
            chart_type="line", x_label="Extra instances", y_label="FPS",
            x_keys=xs, metric_path=("frame_perf", "fps"),
            value_formatter=lambda v: f"{v:.1f} fps",
        ))
        defs.append(ChartDef(
            title=f"Per-frame TLAS rebuild µs vs scene size ({mode_label})",
            category="Scaling -- per-frame ops",
            description="Microseconds GPU spent rebuilding TLAS each frame.  Lower is better.",
            chart_type="line", x_label="Extra instances", y_label="TLAS rebuild (µs)",
            y_is_log=True, x_keys=xs,
            metric_path=("frame_perf", "pf_tlas_us"),
            value_formatter=fmt_us,
        ))
        defs.append(ChartDef(
            title=f"Per-frame anim BLAS rebuild µs vs scene size ({mode_label})",
            category="Scaling -- per-frame ops",
            description=("Cluster mode: BUILD_BLAS_FROM_CLAS.  Traditional: full "
                         "DXR1 BLAS rebuild.  Lower is better."),
            chart_type="line", x_label="Extra instances", y_label="Anim BLAS (µs)",
            y_is_log=True, x_keys=xs,
            metric_path=("frame_perf", "pf_blas_us"),
            value_formatter=fmt_us,
        ))

    # --- Orthogonal slice at default scene: CLAS alloc -------------------
    clas_xs = [("clas-implicit", "implicit"), ("clas-get-sizes", "get-sizes"), ("default", "compact")]
    clas_xs = [(s, x) for s, x in clas_xs if s in all_slugs]
    if len(clas_xs) >= 2:
        defs.append(ChartDef(
            title="CLAS alloc strategy: actual bytes",
            category="Orthogonal -- CLAS alloc",
            description=("How CLAS allocation strategy varies across machines "
                         "at the default scene.  Implicit lets driver pick; "
                         "GetSizes does explicit 2-pass; Compact does post-build "
                         "compaction."),
            chart_type="bar", x_label="CLAS alloc mode", y_label="CLAS bytes",
            x_keys=clas_xs, metric_path=("memory_bytes", "static_clas_actual"),
            value_formatter=fmt_bytes,
        ))
        defs.append(ChartDef(
            title="CLAS alloc strategy: build time",
            category="Orthogonal -- CLAS alloc",
            description="Wall-clock ms to build the static CLAS pipeline. Lower is better.",
            chart_type="bar", x_label="CLAS alloc mode", y_label="Build time (ms)",
            x_keys=clas_xs, metric_path=("build_times_ms", "static_clas"),
            value_formatter=lambda v: f"{v:.1f} ms",
        ))

    # --- Build flags ------------------------------------------------------
    bf_xs = [("bf-none", "none"), ("bf-fast-build", "fast-build"), ("default", "fast-trace")]
    bf_xs = [(s, x) for s, x in bf_xs if s in all_slugs]
    if len(bf_xs) >= 2:
        defs.append(ChartDef(
            title="BVH build flags: FPS",
            category="Orthogonal -- build flags",
            description=("Higher is better.  FAST_TRACE typically wins for traversal "
                         "perf.  Per-machine ratios can reveal driver bias."),
            chart_type="bar", x_label="Build flag", y_label="FPS",
            x_keys=bf_xs, metric_path=("frame_perf", "fps"),
            value_formatter=lambda v: f"{v:.1f}",
        ))
        defs.append(ChartDef(
            title="BVH build flags: BLAS bytes",
            category="Orthogonal -- build flags",
            description=("BLAS bytes per build flag.  Cluster path always "
                         "calls BLAS_FROM_CLAS so the flag affects the "
                         "BLAS layout."),
            chart_type="bar", x_label="Build flag", y_label="BLAS bytes",
            x_keys=bf_xs, metric_path=("memory_bytes", "static_blas_total"),
            value_formatter=fmt_bytes,
        ))

    # --- Vertex format ----------------------------------------------------
    vfm_xs = [("default", "FLOAT32_3"), ("vtx-compressed", "COMPRESSED1")]
    vfm_xs = [(s, x) for s, x in vfm_xs if s in all_slugs]
    if len(vfm_xs) >= 2:
        defs.append(ChartDef(
            title="Vertex format: CLAS actual bytes",
            category="Orthogonal -- vertex format",
            description=("COMPRESSED1 typically saves 2-4x on CLAS bytes vs "
                         "FLOAT32_3.  Compare ratios across machines."),
            chart_type="bar", x_label="Vertex format", y_label="CLAS bytes",
            x_keys=vfm_xs, metric_path=("memory_bytes", "static_clas_actual"),
            value_formatter=fmt_bytes,
        ))
        defs.append(ChartDef(
            title="Vertex format: INSTANTIATE µs",
            category="Orthogonal -- vertex format",
            description="Per-frame INSTANTIATE time. COMPRESSED1 may add small overhead.",
            chart_type="bar", x_label="Vertex format", y_label="INSTANTIATE (µs)",
            x_keys=vfm_xs, metric_path=("frame_perf", "pf_instantiate_us"),
            value_formatter=fmt_us,
        ))

    # --- Precision sweeps: COMPRESSED1 bits/component --------------------
    cb_slugs = sorted([s for s in all_slugs if s.startswith("vtx-compressed_cb-")],
                      key=lambda s: int(s.rsplit("-", 1)[1]))
    cb_xs = [(s, int(s.rsplit("-", 1)[1])) for s in cb_slugs]
    if len(cb_xs) >= 3:
        defs.append(ChartDef(
            title="COMPRESSED1 precision: CLAS bytes vs bits/component",
            category="Precision (COMPRESSED1)",
            description="Bytes saved as bits/component decreases. Lower bits = more compression but more precision loss.",
            chart_type="line", x_label="Bits / component", y_label="CLAS bytes",
            x_keys=cb_xs, metric_path=("memory_bytes", "static_clas_actual"),
            value_formatter=fmt_bytes,
        ))
        defs.append(ChartDef(
            title="COMPRESSED1 precision: FPS vs bits/component",
            category="Precision (COMPRESSED1)",
            description="Watch for FPS spikes at very low bits -- can indicate geometry collapse making rays miss.",
            chart_type="line", x_label="Bits / component", y_label="FPS",
            x_keys=cb_xs, metric_path=("frame_perf", "fps"),
            value_formatter=lambda v: f"{v:.1f} fps",
        ))

    # --- Precision sweeps: FLOAT32_3 truncate bits ----------------------
    tr_slugs = sorted([s for s in all_slugs if s.startswith("trunc-")],
                      key=lambda s: int(s.rsplit("-", 1)[1]))
    tr_xs = [(s, int(s.rsplit("-", 1)[1])) for s in tr_slugs]
    if len(tr_xs) >= 3:
        defs.append(ChartDef(
            title="FLOAT32_3 precision: CLAS bytes vs truncate bits",
            category="Precision (FLOAT32_3)",
            description="Bytes after CLAS quantizer.  Higher truncate = lower mantissa bits = better compression.",
            chart_type="line", x_label="Bits truncated", y_label="CLAS bytes",
            x_keys=tr_xs, metric_path=("memory_bytes", "static_clas_actual"),
            value_formatter=fmt_bytes,
        ))

    return defs


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

# Use a different palette than sweep.py so it's visually obvious this is a
# multi-machine comparison report.  Each machine gets one color, used
# consistently across all charts in the report.
PALETTE = ["#58a6ff", "#f0883e", "#3fb950", "#bc8cff", "#f85149", "#56d364", "#79c0ff"]


def _embed_chartjs() -> str:
    local = THIS_DIR / "chart.umd.min.js"
    if local.exists():
        return f"<script>{local.read_text(encoding='utf-8')}</script>"
    return '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>'


def render_html(machines: list[MachineReport], defs: list[ChartDef]) -> str:
    # Map each machine to a stable color
    machine_color = {m.label: PALETTE[i % len(PALETTE)] for i, m in enumerate(machines)}

    # Group by category
    cat_order: list[str] = []
    cat_charts: dict[str, list[ChartDef]] = {}
    for d in defs:
        if d.category not in cat_charts:
            cat_order.append(d.category)
            cat_charts[d.category] = []
        cat_charts[d.category].append(d)

    chart_js_blocks = []
    chart_html_blocks = []
    for cat in cat_order:
        chart_html_blocks.append(f'<h2 class="cat">{cat}</h2>')
        for i, cdef in enumerate(cat_charts[cat]):
            cid = f"cmp_{cat_order.index(cat)}_{i}_{hashlib.md5(cdef.title.encode()).hexdigest()[:6]}"
            chart_html_blocks.append(f'''
<div class="chart-card">
  <h3>{cdef.title}</h3>
  <p class="desc">{cdef.description}</p>
  <div class="chart-wrap"><canvas id="{cid}"></canvas></div>
</div>
''')
            # Build per-machine dataset
            datasets = []
            tip_map = {}
            for m in machines:
                color = machine_color[m.label]
                data = []
                tips = {}
                for slug, xv in cdef.x_keys:
                    if slug not in m.runs:
                        continue  # machine didn't run this config; skip the point
                    yv = _g(m.runs[slug].data, *cdef.metric_path, default=None)
                    if yv is None:
                        continue
                    key = str(xv) if cdef.chart_type == "bar" else xv
                    data.append({"x": key if cdef.chart_type == "bar" else xv, "y": yv})
                    tips[str(xv)] = (cdef.value_formatter(yv) if cdef.value_formatter else f"{yv}")
                if not data:
                    continue
                tip_map[m.label] = tips
                datasets.append({
                    "label": m.label,
                    "data": data,
                    "borderColor": color,
                    "backgroundColor": color + "88",
                    "tension": 0.15,
                    "pointRadius": 5 if cdef.chart_type == "line" else 0,
                    "borderWidth": 2,
                })
            if not datasets:
                continue
            opts = {
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "x": {
                        "title": {"display": True, "text": cdef.x_label, "color": "#9da7b3"},
                        "ticks": {"color": "#9da7b3"},
                        "grid": {"color": "#22272e"},
                    },
                    "y": {
                        "type": "logarithmic" if cdef.y_is_log else "linear",
                        "title": {"display": True, "text": cdef.y_label, "color": "#9da7b3"},
                        "ticks": {"color": "#9da7b3"},
                        "grid": {"color": "#22272e"},
                    },
                },
                "plugins": {
                    "legend": {"labels": {"color": "#e6edf3"}},
                    "tooltip": {"callbacks": {}},
                },
            }
            chart_js_blocks.append(f'''
(function() {{
  const tipMap = {json.dumps(tip_map)};
  const cfg = {{
    type: {json.dumps('bar' if cdef.chart_type=='bar' else 'line')},
    data: {{ datasets: {json.dumps(datasets)} }},
    options: {json.dumps(opts)},
  }};
  cfg.options.plugins.tooltip.callbacks = {{
    label: function(ctx) {{
      const lbl = ctx.dataset.label;
      const key = String(ctx.parsed.x);
      const t = (tipMap[lbl] && tipMap[lbl][key]) || (lbl + ': ' + ctx.parsed.y);
      return lbl + ': ' + t;
    }}
  }};
  new Chart(document.getElementById({json.dumps(cid)}).getContext('2d'), cfg);
}})();
''')

    # Machine cards (one per machine, shown side by side)
    machine_cards = ""
    for m in machines:
        a = m.adapter
        machine_cards += f'''
<div class="machine-card" style="border-left:4px solid {machine_color[m.label]}">
  <h3>{m.label}</h3>
  <table>
    <tr><th>Adapter</th><td>{a.get("description","?")}</td></tr>
    <tr><th>Driver</th><td>{a.get("driver_version","?") or "(unknown)"}</td></tr>
    <tr><th>VID/PID</th><td>{a.get("vendor_id",0):04X} / {a.get("device_id",0):04X}</td></tr>
    <tr><th>VRAM</th><td>{a.get("dedicated_vram_bytes",0)/1024**2:.0f} MB</td></tr>
    <tr><th>Clusters</th><td>{"yes" if a.get("clusters_supported") else "no"}</td></tr>
    <tr><th>Configs</th><td>{len(m.runs)}</td></tr>
    <tr><th>Source dir</th><td><code>{m.dir.name}</code></td></tr>
  </table>
</div>
'''

    # Headline delta table: for a few key metrics at default config,
    # show per-machine values + relative delta vs first machine.
    headline_metrics = [
        ("default",                  ("frame_perf",   "fps"),               "FPS (default)",        lambda v: f"{v:.1f}"),
        ("default",                  ("memory_bytes", "total_as"),          "Total AS (default)",   fmt_bytes),
        ("default",                  ("build_times_ms", "total"),           "Build ms (default)",   lambda v: f"{v:.1f}"),
        ("default",                  ("frame_perf",   "pf_tlas_us"),        "TLAS µs (default)",    fmt_us),
        ("default",                  ("frame_perf",   "pf_blas_us"),        "Anim BLAS µs (default)", fmt_us),
        ("n-1000",                   ("frame_perf",   "fps"),               "FPS (n=1000 cluster)", lambda v: f"{v:.1f}"),
        ("n-1000",                   ("memory_bytes", "total_as"),          "Total AS (n=1000 c)",  fmt_bytes),
        ("geom-traditional_n-1000",  ("frame_perf",   "fps"),               "FPS (n=1000 trad)",    lambda v: f"{v:.1f}"),
        ("geom-traditional_n-1000",  ("memory_bytes", "total_as"),          "Total AS (n=1000 t)",  fmt_bytes),
    ]
    headline_rows = []
    for slug, path, label, fmt in headline_metrics:
        if not any(slug in m.runs for m in machines):
            continue
        cells = []
        base = None
        for m in machines:
            if slug not in m.runs:
                cells.append("<td>-</td>")
                continue
            v = _g(m.runs[slug].data, *path, default=None)
            if v is None:
                cells.append("<td>-</td>")
                continue
            if base is None:
                base = v
                cells.append(f"<td>{fmt(v)}</td>")
            else:
                if base == 0:
                    rel = "n/a"
                else:
                    rel = (v - base) / base * 100
                    rel = f"{rel:+.1f}%"
                cls = "delta-up" if isinstance(rel, str) and rel.startswith("+") else ("delta-down" if isinstance(rel, str) and rel.startswith("-") else "")
                cells.append(f'<td>{fmt(v)} <span class="{cls}">({rel})</span></td>')
        headline_rows.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")

    machine_header_row = "<th></th>" + "".join(f'<th style="color:{machine_color[m.label]}">{m.label}</th>' for m in machines)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DXR2 Cross-Machine Comparison</title>
{_embed_chartjs()}
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 0; padding: 24px; background: #0d1117; color: #e6edf3; }}
  h1   {{ margin: 0 0 4px 0; font-size: 22px; }}
  h2   {{ margin: 32px 0 12px 0; font-size: 18px; color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 4px; }}
  h2.cat {{ color: #f0c378; }}
  h3   {{ margin: 0 0 6px 0; font-size: 15px; color: #e6edf3; }}
  .desc {{ margin: 0 0 8px 0; color: #9da7b3; font-size: 12px; line-height: 1.4; }}
  .machines-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .machine-card {{ flex: 1; min-width: 280px; background: #161b22; border: 1px solid #30363d;
                   border-radius: 6px; padding: 12px 16px; }}
  .chart-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px;
                 padding: 14px; margin: 14px 0; }}
  .chart-wrap {{ position: relative; height: 320px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid #30363d; }}
  th     {{ color: #9da7b3; font-weight: 600; }}
  code   {{ font-family: Consolas, "Cascadia Code", monospace; color: #79c0ff; }}
  .headline {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; margin: 16px 0; }}
  .delta-up   {{ color: #f0883e; }}
  .delta-down {{ color: #3fb950; }}
</style>
</head>
<body>
<h1>DXR2 Clustered-Geometry Cross-Machine Comparison</h1>
<div class="subtitle">{len(machines)} machines, {sum(len(m.runs) for m in machines)} total runs.  Generated {_dt.datetime.now().isoformat(timespec="seconds")}.</div>

<div class="machines-row">
{machine_cards}
</div>

<div class="headline">
  <h2>Headline numbers (delta vs first machine)</h2>
  <table>
    <thead><tr>{machine_header_row}</tr></thead>
    <tbody>
    {"".join(headline_rows)}
    </tbody>
  </table>
</div>

{"".join(chart_html_blocks)}

<script>
{"".join(chart_js_blocks)}
</script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="2+ sweep result directories (each = one machine)")
    ap.add_argument("--label", action="append", default=[],
                    help="Override label for the i-th directory (repeat for multiple)")
    ap.add_argument("--out", default="compare.html", help="Output HTML path")
    args = ap.parse_args(argv)

    if len(args.dirs) < 2:
        print("ERROR: need at least 2 sweep result directories", file=sys.stderr)
        return 2

    machines: list[MachineReport] = []
    for i, d in enumerate(args.dirs):
        label = args.label[i] if i < len(args.label) else None
        print(f"loading {d}...", end=" ", flush=True)
        m = load_machine_report(pathlib.Path(d), label)
        print(f"{len(m.runs)} runs, adapter={m.adapter.get('description','?')!r}")
        machines.append(m)

    defs = build_chart_defs(machines)
    print(f"building {len(defs)} comparison charts...")

    html = render_html(machines, defs)
    out = pathlib.Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
