#!/usr/bin/env python3
"""Colorful interactive companion to scripts/plot_raw_results.py.

Does not overwrite the matplotlib PNGs in figures/. Writes HTML under
figures/interactive/. Same data rules: include every sweep, facet by
(sweep, mission, propulsion), never share a y-axis across families.
No scoring or ranking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures" / "interactive"

SWEEP_CSVS = [
    RESULTS / "my_pilot" / "results.csv",
    RESULTS / "my_pilot_nuclear" / "results.csv",
    RESULTS / "electric_vs_nuclear" / "results.csv",
]
SMOKE_JSONL = RESULTS / "smoke" / "results.jsonl"

# Bright categorical palettes — identity colors, not a good/bad scale.
AGENT_COLORS = {
    "random": "#FF6B6B",
    "max_thrust": "#FF9F1C",
    "prograde": "#FFD93D",
    "edelbaum": "#6BCB77",
    "bang_bang": "#4D96FF",
    "ppo": "#9B5DE5",
    "sac": "#F15BB5",
    "td3": "#00BBF9",
    "pid": "#00F5D4",
}
REASON_COLORS = {
    "timeout": "#4D96FF",
    "out_of_propellant": "#FF9F1C",
    "hardware_failure": "#6BCB77",
    "safety_violation": "#9B5DE5",
    "diverged": "#FF6B6B",
}
HOVER = [
    "sweep",
    "family",
    "mission",
    "propulsion",
    "agent",
    "seed",
    "status",
    "trip_time_days",
    "wear_fraction",
    "constraint_violations",
    "termination_reason_mode",
    "success_rate",
]


def load_sweeps() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in SWEEP_CSVS:
        if path.exists():
            frames.append(
                pd.read_csv(path).copy().assign(source_file=str(path.relative_to(ROOT)))
            )
    if SMOKE_JSONL.exists():
        frames.append(
            pd.read_json(SMOKE_JSONL, lines=True).assign(
                source_file=str(SMOKE_JSONL.relative_to(ROOT))
            )
        )
    data = pd.concat(frames, ignore_index=True, sort=False).copy()
    data["panel"] = (
        data["family"].astype(str)
        + "  ·  "
        + data["propulsion"].astype(str)
        + "  ·  "
        + data["mission"].astype(str)
        + "  ·  "
        + data["sweep"].astype(str)
    )
    data["family"] = data["family"].astype(str)
    return data


def _agent_map(data: pd.DataFrame) -> dict[str, str]:
    mapping = dict(AGENT_COLORS)
    extra = px.colors.qualitative.Prism
    i = 0
    for agent in sorted(data["agent"].unique()):
        if agent not in mapping:
            mapping[agent] = extra[i % len(extra)]
            i += 1
    return mapping


def _style(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=20, color="#1B1B3A")),
        paper_bgcolor="#FFF8F0",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Trebuchet MS, Segoe UI, sans-serif", color="#1B1B3A"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#FFD6A5", borderwidth=1),
        hoverlabel=dict(bgcolor="#1B1B3A", font_size=12, font_color="white"),
        margin=dict(t=80, l=60, r=30, b=60),
    )
    fig.update_xaxes(showgrid=False, tickangle=-30)
    fig.update_yaxes(showgrid=True, gridcolor="#F0E6DD", zeroline=False, matches=None)
    return fig


def strip_metric(data: pd.DataFrame, y: str, title: str) -> go.Figure:
    colors = _agent_map(data)
    fig = px.strip(
        data.sort_values(["family", "mission", "propulsion", "sweep", "agent", "seed"]),
        x="agent",
        y=y,
        color="agent",
        color_discrete_map=colors,
        facet_col="panel",
        facet_col_wrap=3,
        hover_data=HOVER,
        stripmode="overlay",
        title=title,
    )
    n_rows = int(max(1, (data["panel"].nunique() + 2) // 3))
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.4, color="white")))
    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("panel=", "")))
    fig.update_layout(height=max(420, 320 * n_rows))
    return _style(fig, title)


def termination_bars(data: pd.DataFrame) -> go.Figure:
    counts = (
        data.groupby(
            ["family", "panel", "agent", "termination_reason_mode"], dropna=False
        )
        .size()
        .reset_index(name="n_seeds")
    )
    fig = px.bar(
        counts.sort_values(["family", "panel", "agent"]),
        x="agent",
        y="n_seeds",
        color="termination_reason_mode",
        color_discrete_map=REASON_COLORS,
        facet_col="panel",
        facet_col_wrap=3,
        hover_data=["family", "panel", "agent", "termination_reason_mode", "n_seeds"],
        title="termination_reason_mode — seed counts (not a ranking)",
    )
    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("panel=", "")))
    n_rows = int(max(1, (data["panel"].nunique() + 2) // 3))
    fig.update_layout(height=max(420, 320 * n_rows))
    return _style(fig, "termination_reason_mode — seed counts (not a ranking)")


def termination_sunburst(data: pd.DataFrame) -> go.Figure:
    counts = (
        data.groupby(
            ["family", "mission", "propulsion", "sweep", "termination_reason_mode"]
        )
        .size()
        .reset_index(name="n")
    )
    fig = px.sunburst(
        counts,
        path=["family", "mission", "propulsion", "sweep", "termination_reason_mode"],
        values="n",
        color="family",
        color_discrete_map={"electric": "#4D96FF", "nuclear": "#FF9F1C"},
        title="Outcome mix nested by family → mission → propulsion → sweep → reason",
    )
    fig.update_traces(hovertemplate="%{id}<br>n_seeds=%{value}<extra></extra>")
    return _style(
        fig,
        "Outcome mix nested by family → mission → propulsion → sweep → reason",
    )


def overview_scatter(data: pd.DataFrame) -> go.Figure:
    colors = _agent_map(data)
    fig = px.scatter(
        data,
        x="trip_time_days",
        y="wear_fraction",
        color="agent",
        color_discrete_map=colors,
        symbol="termination_reason_mode",
        facet_col="family",
        hover_data=HOVER,
        title="Per-seed trip_time_days vs wear_fraction (axes not shared across families)",
    )
    fig.update_traces(marker=dict(size=11, opacity=0.8, line=dict(width=0.5, color="white")))
    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None)
    return _style(
        fig,
        "Per-seed trip_time_days vs wear_fraction (axes not shared across families)",
    )


def overview_table(data: pd.DataFrame) -> go.Figure:
    show = data[
        [
            "family",
            "sweep",
            "mission",
            "propulsion",
            "agent",
            "seed",
            "trip_time_days",
            "termination_reason_mode",
            "wear_fraction",
            "constraint_violations",
            "success_rate",
            "status",
        ]
    ].sort_values(["family", "sweep", "mission", "propulsion", "agent", "seed"])
    header_fill = ["#9B5DE5"] * len(show.columns)
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{c}</b>" for c in show.columns],
                    fill_color=header_fill,
                    font=dict(color="white", size=12),
                    align="left",
                    height=32,
                ),
                cells=dict(
                    values=[show[c] for c in show.columns],
                    fill_color=[
                        ["#FDE2FF" if i % 2 == 0 else "#FFF8F0" for i in range(len(show))]
                    ]
                    * len(show.columns),
                    align="left",
                    font=dict(size=11, color="#1B1B3A"),
                    height=26,
                ),
            )
        ]
    )
    return _style(fig, "All seeds — sortable raw table (same rows as the PNG figures)")


def pipeline_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Experiment pipeline</title>
  <style>
    body {
      margin: 0; font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background: linear-gradient(135deg, #FFF8F0, #CDB4DB 40%, #A0C4FF);
      color: #1B1B3A; min-height: 100vh;
    }
    h1 { text-align: center; padding-top: 1.5rem; }
    .row { display: flex; justify-content: center; gap: 0.6rem; flex-wrap: wrap; padding: 1.5rem 6vw; align-items: center; }
    .box {
      width: 160px; min-height: 110px; border-radius: 18px; padding: 0.85rem;
      border: 3px solid #1B1B3A; text-align: center; font-weight: 700;
      box-shadow: 5px 5px 0 #1B1B3A;
    }
    .arrow { font-size: 2rem; font-weight: 900; }
    .c1 { background: #4D96FF; }
    .c2 { background: #9B5DE5; color: white; }
    .c3 { background: #F15BB5; color: white; }
    .c4 { background: #FF9F1C; }
    .c5 { background: #6BCB77; }
    .c6 { background: #FFD93D; width: 280px; }
    .note { text-align: center; max-width: 40rem; margin: 0 auto 2rem; }
  </style>
</head>
<body>
  <h1>Experiment pipeline</h1>
  <p class="note">Methodology only — not results. Same step loop as the original PNG.</p>
  <div class="row">
    <div class="box c1">Mission setup<br>reset + observe</div>
    <div class="arrow">→</div>
    <div class="box c2">Agent<br>selects 5-D action</div>
    <div class="arrow">→</div>
    <div class="box c3">Propulsion model<br>thrust, ṁ, wear</div>
    <div class="arrow">→</div>
    <div class="box c4">Vehicle<br>power + orbit</div>
    <div class="arrow">→</div>
    <div class="box c5">Termination check</div>
  </div>
  <div class="row">
    <div class="arrow">↘</div>
    <div class="box c6">Recorded outcome<br>trip_time_days · termination_reason_mode<br>wear_fraction · constraint_violations</div>
  </div>
</body>
</html>
"""


def write_text(name: str, text: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(text)
    return path


def write_html(name: str, fig: go.Figure, include_js: bool = True) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.write_html(
        path,
        include_plotlyjs="cdn" if include_js else False,
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )
    return path


def write_index(pages: list[tuple[str, str, str]]) -> Path:
    cards = []
    for href, title, blurb in pages:
        cards.append(
            f"""
            <a class="card" href="{href}">
              <h2>{title}</h2>
              <p>{blurb}</p>
            </a>
            """
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Propulsion RL — interactive figures</title>
  <style>
    body {{
      margin: 0; font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background: radial-gradient(circle at 10% 10%, #FFD6A5, transparent 40%),
                  radial-gradient(circle at 90% 0%, #CDB4DB, transparent 35%),
                  radial-gradient(circle at 80% 80%, #A0C4FF, transparent 40%),
                  #FFF8F0;
      color: #1B1B3A;
    }}
    header {{
      padding: 2.2rem 8vw 1rem;
    }}
    h1 {{ font-size: 2.4rem; margin: 0 0 0.4rem; }}
    .note {{ max-width: 52rem; line-height: 1.45; }}
    .grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1.1rem; padding: 0 8vw 3rem;
    }}
    .card {{
      display: block; text-decoration: none; color: inherit;
      background: white; border: 3px solid #1B1B3A; border-radius: 18px;
      padding: 1.1rem 1.2rem; box-shadow: 6px 6px 0 #9B5DE5;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .card:nth-child(2) {{ box-shadow: 6px 6px 0 #4D96FF; }}
    .card:nth-child(3) {{ box-shadow: 6px 6px 0 #FF9F1C; }}
    .card:nth-child(4) {{ box-shadow: 6px 6px 0 #F15BB5; }}
    .card:nth-child(5) {{ box-shadow: 6px 6px 0 #6BCB77; }}
    .card:nth-child(6) {{ box-shadow: 6px 6px 0 #FF6B6B; }}
    .card:nth-child(7) {{ box-shadow: 6px 6px 0 #00BBF9; }}
    .card:hover {{ transform: translate(-3px, -3px); }}
    h2 {{ margin: 0 0 0.35rem; font-size: 1.15rem; }}
    p {{ margin: 0; color: #3D3D6B; }}
  </style>
</head>
<body>
  <header>
    <h1>Interactive companion figures</h1>
    <p class="note">
      These HTML charts sit beside the original matplotlib PNGs in <code>figures/</code>
      and do not replace them. Hover a point for the raw seed. Facets are
      <b>propulsion · mission · sweep</b>, so <code>hall_hermes</code> on
      <code>earth_mars_cargo</code> (electric_vs_nuclear) stays separate from
      <code>hall_hermes</code> on <code>mars_crew_fast</code> (my_pilot_nuclear)
      and <code>leo_geo_transfer</code> (my_pilot). Electric and nuclear y-axes
      are not shared. No scores or rankings.
    </p>
  </header>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    path = OUT / "index.html"
    path.write_text(html)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_sweeps()
    data[HOVER + ["panel"]].to_csv(OUT / "00_interactive_source.csv", index=False)

    figs = [
        (
            "01_trip_time_days.html",
            "1. Trip time",
            "Strip plot of every seed. Facet = propulsion · mission · sweep.",
            strip_metric(
                data,
                "trip_time_days",
                "trip_time_days by agent — every seed, faceted by propulsion · mission · sweep",
            ),
        ),
        (
            "02_termination_reason.html",
            "2. Termination mix",
            "Stacked seed counts of termination_reason_mode.",
            termination_bars(data),
        ),
        (
            "02_termination_sunburst.html",
            "2b. Termination sunburst",
            "Same outcomes nested family → mission → propulsion → sweep.",
            termination_sunburst(data),
        ),
        (
            "03_wear_fraction.html",
            "3. Wear fraction",
            "Strip plot of every seed.",
            strip_metric(
                data,
                "wear_fraction",
                "wear_fraction by agent — every seed, faceted by propulsion · mission · sweep",
            ),
        ),
        (
            "04_constraint_violations.html",
            "4. Constraint violations",
            "Strip plot of every seed.",
            strip_metric(
                data,
                "constraint_violations",
                "constraint_violations by agent — every seed, faceted by propulsion · mission · sweep",
            ),
        ),
        (
            "06_overview_scatter.html",
            "6. Overview scatter",
            "Each seed as a point: trip time vs wear, hover for the rest.",
            overview_scatter(data),
        ),
        (
            "06_overview_table.html",
            "6b. Overview table",
            "Full raw seed table, filterable in the Plotly bar.",
            overview_table(data),
        ),
    ]

    pages = []
    for href, title, blurb, fig in figs:
        write_html(href, fig)
        pages.append((href, title, blurb))
        print("wrote", OUT / href)

    write_text("05_experiment_pipeline.html", pipeline_html())
    pages.insert(
        5,
        (
            "05_experiment_pipeline.html",
            "5. Pipeline",
            "Colorful methodology diagram, not results.",
        ),
    )
    print("wrote", OUT / "05_experiment_pipeline.html")

    write_index(pages)
    print("wrote", OUT / "index.html")
    print("plotlyjs available:", bool(get_plotlyjs()))
    # Confirm hall_hermes is not merged across missions
    hermes = data.loc[data["propulsion"] == "hall_hermes", ["sweep", "mission", "panel"]]
    print("\nhall_hermes panels (must stay distinct):")
    print(hermes.drop_duplicates().sort_values(["mission", "sweep"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
