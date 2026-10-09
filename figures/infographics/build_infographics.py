#!/usr/bin/env python3
"""Build Mars-crew infographics from nuclear_long2/results.csv only.

Writes HTML + 300 dpi PNG under figures/infographics/.
Does not modify results/, configs/, plot_results.py, or existing figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "nuclear_long2" / "results.csv"
OUT = Path(__file__).resolve().parent
EXISTING = ROOT / "figures"
TRIP_LIMIT = 220.0

PROP_LABEL = {
    "hall_hermes": "Hall HERMeS",
    "ion_next": "Ion NEXT",
    "nep_brayton": "NEP Brayton",
    "ntp_nerva": "NTP NERVA",
    "ntp_pewee": "NTP Pewee",
}
AGENT_LABEL = {
    "random": "random",
    "max_thrust": "max thrust",
    "prograde": "prograde",
    "ppo": "PPO",
    "sac": "SAC",
}


def stats(df: pd.DataFrame) -> dict:
    n = len(df)
    n_ok = int((df["status"] == "ok").sum())
    n_fail_status = int((df["status"] != "ok").sum())
    n_success_cells = int((df["success_rate"] > 0).sum())
    success_pct = 100.0 * float(df["success_rate"].mean())

    g = (
        df.groupby(["propulsion", "agent"], observed=True)
        .agg(
            progress_mean=("progress", "mean"),
            progress_std=("progress", "std"),
            trip_mean=("trip_time_days", "mean"),
            trip_std=("trip_time_days", "std"),
            wear_mean=("wear_fraction", "mean"),
            n=("progress", "size"),
        )
        .reset_index()
        .sort_values("progress_mean", ascending=False)
    )
    top = g.iloc[0]
    top_seed = df.loc[df["progress"].idxmax()]
    shortest = df.loc[df["trip_time_days"].idxmin()]

    term = df["termination_reason_mode"].value_counts()
    n_timeout = int(term.get("timeout", 0))
    n_prop = int(term.get("out_of_propellant", 0))
    n_hw = int(term.get("hardware_failure", 0))

    by_prop = df.groupby("propulsion", observed=True)["progress"].mean().sort_values(ascending=False)
    by_agent = df.groupby("agent", observed=True)["progress"].mean().sort_values(ascending=False)

    electric = df[df["family"] == "electric"]["progress"].mean()
    nuclear = df[df["family"] == "nuclear"]["progress"].mean()

    scripted = df[df["learns"] == False]["progress"].mean()
    learned = df[df["learns"] == True]["progress"].mean()

    cost_finite = int(df["cost_per_kg_delivered"].notna().sum())

    ranking = []
    for i, row in g.iterrows():
        ranking.append(
            {
                "rank": len(ranking) + 1,
                "propulsion": PROP_LABEL[str(row["propulsion"])],
                "propulsion_key": str(row["propulsion"]),
                "agent": AGENT_LABEL[str(row["agent"])],
                "agent_key": str(row["agent"]),
                "progress": float(row["progress_mean"]),
                "progress_std": float(row["progress_std"] if pd.notna(row["progress_std"]) else 0.0),
                "trip": float(row["trip_mean"]),
                "n": int(row["n"]),
            }
        )

    return {
        "n": n,
        "n_ok": n_ok,
        "n_fail_status": n_fail_status,
        "n_success_cells": n_success_cells,
        "success_pct": success_pct,
        "n_timeout": n_timeout,
        "n_propellant": n_prop,
        "n_hardware": n_hw,
        "trip_limit": TRIP_LIMIT,
        "top_progress": float(top["progress_mean"]),
        "top_progress_std": float(top["progress_std"] if pd.notna(top["progress_std"]) else 0.0),
        "top_agent": AGENT_LABEL[str(top["agent"])],
        "top_propulsion": PROP_LABEL[str(top["propulsion"])],
        "top_trip": float(top["trip_mean"]),
        "best_seed_progress": float(top_seed["progress"]),
        "best_seed_agent": AGENT_LABEL[str(top_seed["agent"])],
        "best_seed_propulsion": PROP_LABEL[str(top_seed["propulsion"])],
        "best_seed_id": int(top_seed["seed"]),
        "best_seed_trip": float(top_seed["trip_time_days"]),
        "best_seed_reason": str(top_seed["termination_reason_mode"]),
        "shortest_trip": float(shortest["trip_time_days"]),
        "shortest_agent": AGENT_LABEL[str(shortest["agent"])],
        "shortest_propulsion": PROP_LABEL[str(shortest["propulsion"])],
        "shortest_reason": str(shortest["termination_reason_mode"]),
        "electric_progress": float(electric),
        "nuclear_progress": float(nuclear),
        "scripted_progress": float(scripted),
        "learned_progress": float(learned),
        "cost_finite": cost_finite,
        "payload_unique": [float(x) for x in sorted(df["payload_delivered_kg"].unique())],
        "ranking": ranking,
        "by_prop": {PROP_LABEL[k]: float(v) for k, v in by_prop.items()},
        "by_agent": {AGENT_LABEL[k]: float(v) for k, v in by_agent.items()},
        "mean_trip_by_prop": {
            PROP_LABEL[k]: float(v)
            for k, v in df.groupby("propulsion", observed=True)["trip_time_days"].mean().items()
        },
    }


CSS = """
:root {
  --bg: #070b16;
  --panel: #12192b;
  --ink: #f4f7ff;
  --muted: #9aa6c4;
  --line: #2a3554;
  --accent: #56B4E9;
  --warn: #E69F00;
  --ok: #009E73;
  --hot: #D55E00;
  --pink: #CC79A7;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  background: var(--bg); color: var(--ink);
  font-family: "Trebuchet MS", "Segoe UI", sans-serif;
}
body.slide { width: 1600px; height: 900px; overflow: hidden; }
body.poster { width: 2480px; min-height: 3508px; }
.stars {
  position: absolute; inset: 0;
  background:
    radial-gradient(1px 1px at 12% 18%, #fff 50%, transparent 51%),
    radial-gradient(1px 1px at 28% 72%, #cde 50%, transparent 51%),
    radial-gradient(1px 1px at 63% 22%, #fff 50%, transparent 51%),
    radial-gradient(1px 1px at 81% 58%, #9cf 50%, transparent 51%),
    radial-gradient(1px 1px at 47% 41%, #fff 50%, transparent 51%),
    radial-gradient(2px 2px at 91% 12%, #fff 50%, transparent 51%);
  opacity: 0.55; pointer-events: none;
}
.wrap { position: relative; z-index: 1; padding: 36px 44px 28px; height: 100%; }
h1 { font-size: 42px; margin: 0 0 6px; letter-spacing: 0.02em; }
h2 { font-size: 22px; margin: 0; color: var(--accent); font-weight: 700; }
.kicker { color: var(--warn); font-size: 14px; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 8px; }
.row { display: flex; gap: 18px; }
.card {
  background: rgba(18,25,43,0.92);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px 20px;
}
.big { font-size: 64px; font-weight: 800; line-height: 1; color: var(--warn); }
.big.ok { color: var(--ok); }
.big.bad { color: var(--hot); }
.big.blue { color: var(--accent); }
.label { color: var(--muted); font-size: 15px; margin-top: 8px; line-height: 1.35; }
.note { color: var(--muted); font-size: 13px; line-height: 1.4; }
.foot { position: absolute; bottom: 18px; left: 44px; right: 44px;
  color: var(--muted); font-size: 13px; display: flex; justify-content: space-between; }
.route { position: relative; height: 210px; margin-top: 8px; }
.planet { position: absolute; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 14px; font-weight: 700; }
.earth { width: 92px; height: 92px; left: 40px; top: 54px; background: #2E6BFF; box-shadow: 0 0 24px #2E6BFF88; }
.mars { width: 72px; height: 72px; right: 48px; top: 68px; background: #C1440E; box-shadow: 0 0 24px #C1440E88; }
.path { position: absolute; left: 132px; right: 128px; top: 98px; height: 0;
  border-top: 3px dashed #56B4E9; }
.limit {
  position: absolute; left: 132px; right: 128px; top: 36px;
  border-top: 3px solid var(--hot);
}
.limit span { position: absolute; top: -26px; right: 0; color: var(--hot); font-weight: 800; font-size: 16px; }
.ship { position: absolute; top: 82px; left: 38%; font-size: 28px; }
.bar-row { display: grid; grid-template-columns: 56px 210px 1fr 140px; gap: 10px; align-items: center; margin: 8px 0; }
.bar-track { height: 22px; background: #1c2740; border-radius: 99px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; }
.rank { font-size: 28px; font-weight: 800; color: var(--warn); }
.arrow { font-size: 22px; font-weight: 800; }
.up { color: var(--ok); }
.down { color: var(--hot); }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.grid2 { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 16px; margin-top: 16px; }
img.chart { width: 100%; border-radius: 12px; border: 1px solid var(--line); background: #fff; }
"""


def html_glance(s: dict) -> str:
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body class="slide">
<div class="stars"></div>
<div class="wrap">
  <div class="kicker">nuclear_long2 · mars_crew_fast · 100 completed cells</div>
  <h1>Mars crew transfer at a glance</h1>
  <h2>Nobody arrived. The 220-day clock stopped every electric and NEP Brayton timeout; NTP stages emptied their tanks in hours.</h2>
  <div class="row" style="margin-top:22px">
    <div class="card" style="flex:1">
      <div class="big bad">0%</div>
      <div class="label">Success rate<br>{s['n_success_cells']} of {s['n']} cells had success_rate &gt; 0.<br>All {s['n_ok']} cells finished with status=ok (0 runner crashes).</div>
    </div>
    <div class="card" style="flex:1">
      <div class="big blue">{s['top_progress']:.3f}</div>
      <div class="label">Highest mean transfer progress (0–1)<br>{s['top_propulsion']} + {s['top_agent']}<br>mean of 4 seeds ± {s['top_progress_std']:.3f}</div>
    </div>
    <div class="card" style="flex:1">
      <div class="big">{s['trip_limit']:.0f}<span style="font-size:28px"> days</span></div>
      <div class="label">Hard crew time limit in the mission code<br>{s['n_timeout']} cells ended as timeout on this wall.<br>Cost per kg delivered: missing in {s['n'] - s['cost_finite']} / {s['n']} rows (no payload delivered).</div>
    </div>
  </div>
  <div class="card" style="margin-top:18px">
    <div class="route">
      <div class="planet earth">EARTH</div>
      <div class="path"></div>
      <div class="limit"><span>220-day limit — arrival after this is failure</span></div>
      <div class="ship">◆</div>
      <div class="planet mars">MARS</div>
    </div>
    <div class="note">Shortest elapsed time in the CSV is {s['shortest_trip']:.4f} days ({s['shortest_propulsion']} + {s['shortest_agent']}), but that cell ended <b>{s['shortest_reason']}</b> — it is not a successful transit. Best single-seed progress is {s['best_seed_progress']:.4f} ({s['best_seed_propulsion']} + {s['best_seed_agent']}, seed {s['best_seed_id']}), still a {s['best_seed_reason']} at {s['best_seed_trip']:.1f} days.</div>
  </div>
  <div class="foot"><span>Source: results/nuclear_long2/results.csv · config: configs/nuclear_long.yaml</span><span>Payload delivered = 0 kg in every row</span></div>
</div></body></html>"""


def html_winners(s: dict) -> str:
    rows = []
    max_p = max(r["progress"] for r in s["ranking"]) or 1.0
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    for r in s["ranking"][:8]:
        width = 100.0 * r["progress"] / max_p
        color = colors[(r["rank"] - 1) % len(colors)]
        rows.append(
            f"""<div class="bar-row">
              <div class="rank">#{r['rank']}</div>
              <div><b>{r['propulsion']}</b><br><span class="note">{r['agent']} · n={r['n']} seeds</span></div>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%;background:{color}"></div></div>
              <div style="text-align:right"><b>{r['progress']:.3f}</b><br><span class="note">progress<br>{r['trip']:.2f} days</span></div>
            </div>"""
        )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{CSS}
.bar-row {{ margin: 7px 0; }}
</style></head>
<body class="slide">
<div class="stars"></div>
<div class="wrap">
  <div class="kicker">Ranked by mean transfer progress — not by trip time</div>
  <h1>Closest configs vs the rest</h1>
  <h2>Success rate is 0% for every pairing. Rank is “how far the trajectory got,” not “who landed.”</h2>
  <div class="grid2">
    <div class="card">{''.join(rows)}</div>
    <div>
      <div class="card">
        <div class="label">Why trip time is not the ranking key</div>
        <p class="note" style="font-size:16px;color:var(--ink)">NTP NERVA / Pewee have the smallest trip_time_days (hours, not months) because they hit <b>out_of_propellant</b>. That is a failed burn-out, not a fast arrival. They are listed below NEP Brayton on this board.</p>
      </div>
      <div class="card" style="margin-top:16px">
        <div class="big blue">#{1}</div>
        <div class="label">Top mean progress: <b>{s['top_propulsion']} + {s['top_agent']}</b><br>{s['top_progress']:.4f} ± {s['top_progress_std']:.4f}<br>mean trip {s['top_trip']:.1f} days (limit {s['trip_limit']:.0f})</div>
      </div>
      <div class="card" style="margin-top:16px">
        <div class="label">Honest gaps</div>
        <p class="note" style="font-size:16px;color:var(--ink)">Runner failures: {s['n_fail_status']}. Successful arrivals: {s['n_success_cells']}. $/kg delivered: not defined ({s['cost_finite']} finite values in {s['n']} rows).</p>
      </div>
    </div>
  </div>
  <div class="foot"><span>Source: results/nuclear_long2/results.csv</span><span>25 pairings × 4 seeds = 100 cells</span></div>
</div></body></html>"""


def html_levers(s: dict) -> str:
    d_family = s["nuclear_progress"] - s["electric_progress"]
    d_learn = s["learned_progress"] - s["scripted_progress"]
    prop_items = "".join(
        f"<div class='bar-row'><div></div><div>{name}</div>"
        f"<div class='bar-track'><div class='bar-fill' style='width:{100*val/max(s['by_prop'].values()):.1f}%;background:#56B4E9'></div></div>"
        f"<div>{val:.3f}</div></div>"
        for name, val in s["by_prop"].items()
    )
    agent_items = "".join(
        f"<div class='bar-row'><div></div><div>{name}</div>"
        f"<div class='bar-track'><div class='bar-fill' style='width:{100*val/max(s['by_agent'].values()):.1f}%;background:#E69F00'></div></div>"
        f"<div>{val:.3f}</div></div>"
        for name, val in s["by_agent"].items()
    )
    fam_arrow = "up" if d_family > 0 else "down"
    learn_arrow = "up" if d_learn > 0 else "down"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{CSS}
.bar-row {{ grid-template-columns: 0 170px 1fr 90px; }}
.lever {{ display:flex; gap:14px; align-items:flex-start; margin:12px 0; }}
.badge {{ min-width:72px; text-align:center; padding:8px; border-radius:12px; background:#1c2740; }}
</style></head>
<body class="slide">
<div class="stars"></div>
<div class="wrap">
  <div class="kicker">Swept parameters in nuclear_long.yaml: propulsion × agent × seed</div>
  <h1>What actually changes the outcome</h1>
  <h2>Mission, cost model, and time limit were held fixed. Only hardware, controller, and seed varied.</h2>
  <div class="row" style="margin-top:18px">
    <div class="card" style="flex:1.15">
      <div class="label">Mean progress by propulsion</div>
      {prop_items}
      <div class="label" style="margin-top:10px">Mean progress by controller</div>
      {agent_items}
    </div>
    <div class="card" style="flex:0.85">
      <div class="lever">
        <div class="badge"><div class="arrow {fam_arrow}">{'▲' if d_family>0 else '▼'}</div>family</div>
        <div><b>Nuclear vs electric</b><div class="note">Mean progress nuclear {s['nuclear_progress']:.4f} vs electric {s['electric_progress']:.4f} (difference {d_family:+.4f}). Electric cells all timed out at {s['trip_limit']:.0f} days.</div></div>
      </div>
      <div class="lever">
        <div class="badge"><div class="arrow {learn_arrow}">{'▲' if d_learn>0 else '▼'}</div>agent</div>
        <div><b>Learned (PPO/SAC) vs scripted</b><div class="note">Mean progress learned {s['learned_progress']:.4f} vs scripted {s['scripted_progress']:.4f} (difference {d_learn:+.4f}). Still 0 successful arrivals.</div></div>
      </div>
      <div class="lever">
        <div class="badge"><div class="arrow down">■</div>limit</div>
        <div><b>220-day crew clock (not swept)</b><div class="note">{s['n_timeout']} timeouts, {s['n_propellant']} out of propellant, {s['n_hardware']} hardware failures. Seed is a repeat, not a physics knob (4 repeats per pairing).</div></div>
      </div>
      <p class="note">Do not read NTP’s tiny trip times as “faster is better”: mean trip is {s['mean_trip_by_prop']['NTP NERVA']:.3f} days (NERVA) and {s['mean_trip_by_prop']['NTP Pewee']:.3f} days (Pewee) because those cells emptied the tanks.</p>
    </div>
  </div>
  <div class="foot"><span>Arrows compare recorded CSV means only</span><span>No $/kg: {s['cost_finite']} finite cost_per_kg_delivered values</span></div>
</div></body></html>"""


def html_poster(s: dict) -> str:
    heat = EXISTING / "nuclear_long2_progress_heatmap.png"
    lines = EXISTING / "nuclear_long2_progress_lines.png"
    endings = EXISTING / "nuclear_long2_endings_by_system.png"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{CSS}
body.poster .wrap {{ padding: 72px 80px 80px; }}
body.poster h1 {{ font-size: 72px; }}
body.poster .big {{ font-size: 96px; }}
body.poster .label {{ font-size: 26px; color: #d5dcf0; }}
body.poster .kicker {{ font-size: 20px; }}
body.poster .note {{ font-size: 22px; }}
body.poster .foot {{ font-size: 20px; bottom: 36px; }}
.poster-kpis {{ display:grid; grid-template-columns: repeat(4,1fr); gap: 24px; margin: 28px 0 36px; }}
.charts {{ display:grid; grid-template-columns: 1fr 1fr; gap: 28px; }}
.charts img {{ width: 100%; }}
.wide {{ grid-column: 1 / -1; }}
</style></head>
<body class="poster">
<div class="stars"></div>
<div class="wrap">
  <div class="kicker">High-school research poster · AI controllers × spacecraft propulsion · crewed Mars clock</div>
  <h1>Crewed Mars transfer:<br>100 cells, 0 arrivals</h1>
  <h2>nuclear_long2 sweep · mars_crew_fast · 5 propulsion systems × 5 controllers × 4 seeds</h2>
  <div class="poster-kpis">
    <div class="card"><div class="big bad">0%</div><div class="label">Success rate<br>{s['n_success_cells']}/{s['n']} cells</div></div>
    <div class="card"><div class="big">{s['trip_limit']:.0f}</div><div class="label">Day crew limit<br>{s['n_timeout']} timeouts</div></div>
    <div class="card"><div class="big blue">{s['top_progress']:.3f}</div><div class="label">Best mean progress<br>{s['top_propulsion']} + {s['top_agent']}</div></div>
    <div class="card"><div class="big ok">{s['n_ok']}</div><div class="label">Cells with status=ok<br>runner crashes: {s['n_fail_status']}</div></div>
  </div>
  <div class="charts">
    <div class="card"><div class="label">Transfer progress by pairing</div><img class="chart" src="{heat}"></div>
    <div class="card"><div class="label">Progress vs controller</div><img class="chart" src="{lines}"></div>
    <div class="card wide"><div class="label">How cells ended, by propulsion</div><img class="chart" src="{endings}"></div>
  </div>
  <div class="card" style="margin-top:28px">
    <div class="label">What the numbers mean</div>
    <p class="note" style="font-size:22px;color:var(--ink);line-height:1.45">
      Progress is a 0–1 score of remaining Δv to the arrival geometry, not a map distance.
      Payload delivered is 0 kg in every row, so cost per kg is undefined ({s['cost_finite']} finite values).
      Shortest clock time is {s['shortest_trip']:.4f} days ({s['shortest_propulsion']} + {s['shortest_agent']}) ending as {s['shortest_reason']} — not a landing.
      Hardware failures: {s['n_hardware']} cells, all on NEP Brayton.
    </p>
  </div>
  <div class="foot"><span>Data: results/nuclear_long2/results.csv · Config: configs/nuclear_long.yaml</span><span>Charts reused from figures/nuclear_long2_*.png (not edited)</span></div>
</div></body></html>"""


def write_html(name: str, html: str) -> Path:
    path = OUT / name
    path.write_text(html, encoding="utf-8")
    print("wrote", path)
    return path


def export_pngs(jobs: list[tuple[Path, Path, dict]]) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("Need playwright: python3 -m pip install playwright && python3 -m playwright install chromium") from exc

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for html_path, png_path, opts in jobs:
            w, h, scale = opts["w"], opts["h"], opts["scale"]
            page = browser.new_page(
                viewport={"width": w, "height": h},
                device_scale_factor=scale,
            )
            page.goto(html_path.resolve().as_uri(), wait_until="networkidle", timeout=60000)
            page.screenshot(path=str(png_path), full_page=True, type="png")
            page.close()
            from PIL import Image

            im = Image.open(png_path)
            im.save(png_path, dpi=(300, 300))
            print("wrote", png_path, im.size)
        browser.close()


def main() -> int:
    df = pd.read_csv(CSV)
    s = stats(df)
    (OUT / "computed_stats.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    glance = write_html("01_mars_at_a_glance.html", html_glance(s))
    winners = write_html("02_closest_vs_rest.html", html_winners(s))
    levers = write_html("03_what_changes_outcome.html", html_levers(s))
    poster = write_html("04_summary_poster.html", html_poster(s))
    # 1600x900 CSS px × 3.125 scale ≈ 5000×2812 ≈ 16:9 at 300 dpi on a 16.67"×9.37" slide.
    # A4: 794×1123 CSS px × 3.125 ≈ 2481×3510 at 300 dpi.
    export_pngs(
        [
            (glance, OUT / "01_mars_at_a_glance.png", {"w": 1600, "h": 900, "scale": 3.125}),
            (winners, OUT / "02_closest_vs_rest.png", {"w": 1600, "h": 900, "scale": 3.125}),
            (levers, OUT / "03_what_changes_outcome.png", {"w": 1600, "h": 900, "scale": 3.125}),
            (poster, OUT / "04_summary_poster.png", {"w": 794, "h": 1123, "scale": 3.125}),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
