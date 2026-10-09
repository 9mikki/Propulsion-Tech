#!/usr/bin/env python3
"""Charts for the newest completed experiment sweeps.

Newest Mars run:     results/nuclear_long2/results.csv  (config: configs/nuclear_long.yaml)
Newest LEO-GEO run:  results/electric_long/results.csv  (config: configs/electric_long.yaml)

Older same-experiment runs (nuclear_long, my_pilot_long_gate) are not plotted.
Git-LFS pointer files (my_pilot, my_pilot_nuclear, electric_vs_nuclear, smoke)
are not usable data and are not plotted.

Does not write to results/ or configs/. Does not overwrite figures/nuclear_long2_*.png
or figures/infographics/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DPI = 300

# Okabe–Ito colorblind-safe palette
AGENT_COLORS = {
    "random": "#000000",
    "max_thrust": "#E69F00",
    "prograde": "#56B4E9",
    "edelbaum": "#F0E442",
    "ppo": "#009E73",
    "sac": "#CC79A7",
}
PROP_COLORS = {
    "hall_hermes": "#0072B2",
    "hall_spt100": "#56B4E9",
    "ion_next": "#009E73",
    "ion_nstar": "#000000",
    "nep_brayton": "#E69F00",
    "nep_kilopower": "#CC79A7",
    "ntp_nerva": "#D55E00",
    "ntp_pewee": "#F0E442",
}
REASON_COLORS = {
    "success": "#009E73",
    "timeout": "#0072B2",
    "out_of_propellant": "#E69F00",
    "hardware_failure": "#D55E00",
    "safety_violation": "#CC79A7",
    "diverged": "#000000",
}
REASON_LABEL = {
    "success": "Reached mission target",
    "timeout": "Hit the time limit",
    "out_of_propellant": "Ran out of propellant",
    "hardware_failure": "Hardware failed",
    "safety_violation": "Safety violation",
    "diverged": "Trajectory diverged",
}
AGENT_LABEL = {
    "random": "random",
    "max_thrust": "max thrust",
    "prograde": "prograde",
    "edelbaum": "Edelbaum",
    "ppo": "PPO",
    "sac": "SAC",
}
PROP_LABEL = {
    "hall_hermes": "Hall HERMeS\n(electric)",
    "hall_spt100": "Hall SPT-100\n(electric)",
    "ion_next": "Ion NEXT\n(electric)",
    "ion_nstar": "Ion NSTAR\n(electric)",
    "nep_brayton": "NEP Brayton\n(nuclear electric)",
    "nep_kilopower": "NEP Kilopower\n(nuclear electric)",
    "ntp_nerva": "NTP NERVA\n(nuclear thermal)",
    "ntp_pewee": "NTP Pewee\n(nuclear thermal)",
}

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.28,
    }
)


@dataclass(frozen=True)
class Experiment:
    name: str
    csv: Path
    outdir: Path
    title_mission: str
    agents: tuple[str, ...]
    propulsion: tuple[str, ...]
    trip_limit_days: float | None
    trip_limit_label: str
    heatmap_vmax: float
    progress_ymax: float
    trip_ymax: float
    trip_note: str


MARS = Experiment(
    name="mars",
    csv=ROOT / "results" / "nuclear_long2" / "results.csv",
    outdir=ROOT / "figures" / "mars",
    title_mission="crewed Mars transit (mars_crew_fast)",
    agents=("random", "max_thrust", "prograde", "ppo", "sac"),
    propulsion=("hall_hermes", "ion_next", "nep_brayton", "ntp_nerva", "ntp_pewee"),
    trip_limit_days=220.0,
    trip_limit_label="220-day crew time limit",
    heatmap_vmax=0.45,
    progress_ymax=0.55,
    trip_ymax=255.0,
    trip_note=(
        "NTP NERVA and NTP Pewee bars are near 0 days\n"
        "(every cell ran out of propellant in under 2 days).\n"
        "Short trip time is not a successful arrival."
    ),
)

LEO = Experiment(
    name="leo_geo",
    csv=ROOT / "results" / "electric_long" / "results.csv",
    outdir=ROOT / "figures" / "leo_geo",
    title_mission="LEO → GEO transfer (leo_geo_transfer)",
    agents=("random", "max_thrust", "prograde", "edelbaum", "ppo", "sac"),
    propulsion=("hall_hermes", "hall_spt100", "ion_next", "ion_nstar", "nep_kilopower"),
    trip_limit_days=1095.0,
    trip_limit_label="1095-day mission clock",
    heatmap_vmax=1.0,
    progress_ymax=1.08,
    trip_ymax=1250.0,
    trip_note=(
        "NEP Kilopower bars are short because the stage ran out of\n"
        "propellant, not because it arrived. Hall SPT-100 ended in\n"
        "hardware failure around 380 days. Short trip time ≠ success."
    ),
)


def load(exp: Experiment) -> pd.DataFrame:
    if not exp.csv.exists():
        raise SystemExit(f"Missing {exp.csv}")
    head = exp.csv.read_text(encoding="utf-8", errors="replace")[:80]
    if head.startswith("version https://git-lfs.github.com"):
        raise SystemExit(f"{exp.csv} is a Git LFS pointer, not a results table")
    df = pd.read_csv(exp.csv)
    df["agent"] = pd.Categorical(df["agent"], exp.agents, ordered=True)
    df["propulsion"] = pd.Categorical(df["propulsion"], exp.propulsion, ordered=True)
    return df


def _save(fig: plt.Figure, exp: Experiment, name: str) -> Path:
    exp.outdir.mkdir(parents=True, exist_ok=True)
    path = exp.outdir / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("wrote", path)
    return path


def _seed_means(df: pd.DataFrame, prop: str, agent: str, col: str) -> np.ndarray:
    return df.loc[(df["propulsion"] == prop) & (df["agent"] == agent), col].to_numpy(
        dtype=float
    )


def fig_grouped_bars(
    df: pd.DataFrame,
    exp: Experiment,
    *,
    column: str,
    ylabel: str,
    title: str,
    filename: str,
    ymax: float,
    mark_limit: bool = False,
    footnote: str = "",
    footnote_xy: tuple[float, float] = (0.99, 0.97),
    footnote_ha: str = "right",
) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    x = np.arange(len(exp.propulsion))
    width = 0.78 / max(len(exp.agents), 1)
    offsets = (np.arange(len(exp.agents)) - (len(exp.agents) - 1) / 2) * width
    rng = np.random.default_rng(0)
    for off, agent in zip(offsets, exp.agents):
        means, stds = [], []
        for i, prop in enumerate(exp.propulsion):
            vals = _seed_means(df, prop, agent, column)
            means.append(float(np.nanmean(vals)) if len(vals) else np.nan)
            stds.append(float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0)
            if len(vals):
                jitter = rng.uniform(-0.02, 0.02, size=len(vals))
                ax.scatter(
                    np.full(len(vals), x[i] + off) + jitter,
                    vals,
                    color=AGENT_COLORS[agent],
                    s=18,
                    zorder=3,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.4,
                )
        ax.bar(
            x + off,
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=2.2,
            color=AGENT_COLORS[agent],
            edgecolor="#333333",
            linewidth=0.4,
            label=AGENT_LABEL[agent],
            zorder=2,
        )
    if mark_limit and exp.trip_limit_days is not None:
        ax.axhline(
            exp.trip_limit_days,
            color="#D55E00",
            linestyle="--",
            linewidth=2.0,
            label=exp.trip_limit_label,
            zorder=1,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([PROP_LABEL[p] for p in exp.propulsion])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Propulsion system")
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    if footnote:
        fx, fy = footnote_xy
        ax.text(
            fx,
            fy,
            footnote,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha=footnote_ha,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BBBBBB"),
        )
    ax.legend(
        title="Controller",
        ncol=min(4, len(exp.agents) + int(mark_limit)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=True,
        fontsize=10,
    )
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, exp, filename)


def fig_progress_lines(df: pd.DataFrame, exp: Experiment) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    x = np.arange(len(exp.agents))
    for prop in exp.propulsion:
        means, lo, hi = [], [], []
        for agent in exp.agents:
            vals = _seed_means(df, prop, agent, "progress")
            m = float(np.mean(vals)) if len(vals) else np.nan
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            means.append(m)
            lo.append(m - s)
            hi.append(m + s)
        ax.plot(
            x,
            means,
            marker="o",
            linewidth=2.4,
            markersize=8,
            markeredgecolor="#333333",
            markeredgewidth=0.6,
            color=PROP_COLORS[prop],
            label=PROP_LABEL[prop].replace("\n", " "),
        )
        ax.fill_between(x, lo, hi, color=PROP_COLORS[prop], alpha=0.14, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABEL[a] for a in exp.agents], rotation=15, ha="right")
    ax.set_xlabel("Controller (agent)")
    ax.set_ylabel("Transfer progress (0 = start, 1 = arrival geometry)")
    ax.set_ylim(0, exp.progress_ymax)
    n_ok = int((df["status"] == "ok").sum())
    n_win = int((df["success_rate"] > 0).sum())
    ax.set_title(
        f"How close each pairing got on the {exp.title_mission}\n"
        f"(mean ± 1 s.d. over 4 seeds; {n_win} of {n_ok} cells had any successful eval episode)"
    )
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, loc="best", fontsize=9)
    _save(fig, exp, "02_progress_lines.png")


def fig_progress_heatmap(df: pd.DataFrame, exp: Experiment) -> None:
    pivot = (
        df.pivot_table(index="propulsion", columns="agent", values="progress", aggfunc="mean")
        .reindex(index=list(exp.propulsion), columns=list(exp.agents))
    )
    fig, ax = plt.subplots(figsize=(11.2, 6.8))
    im = ax.imshow(
        pivot.to_numpy(),
        cmap="cividis",
        vmin=0.0,
        vmax=exp.heatmap_vmax,
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(exp.agents)))
    ax.set_yticks(np.arange(len(exp.propulsion)))
    ax.set_xticklabels([AGENT_LABEL[a] for a in exp.agents])
    ax.set_yticklabels([PROP_LABEL[p].replace("\n", " ") for p in exp.propulsion])
    ax.set_xlabel("Controller (agent)")
    ax.set_ylabel("Propulsion system")
    mid = 0.55 * exp.heatmap_vmax
    for i in range(len(exp.propulsion)):
        for j in range(len(exp.agents)):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                ax.text(j, i, "—", ha="center", va="center", color="black", fontsize=12)
                continue
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color="white" if val > mid else "black",
                fontsize=11,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean transfer progress (dimensionless, 0–1)")
    ax.set_title(
        f"Transfer progress heatmap, {exp.title_mission}\n"
        "Each cell is the mean of 4 independent seeds (blank would mean a missing run)"
    )
    _save(fig, exp, "03_progress_heatmap.png")


def fig_termination_pie(df: pd.DataFrame, exp: Experiment) -> None:
    counts = df["termination_reason_mode"].value_counts()
    order = [k for k in REASON_LABEL if k in counts.index]
    for k in counts.index:
        if k not in order:
            order.append(k)
    sizes = [int(counts[k]) for k in order]
    colors = [REASON_COLORS.get(k, "#999999") for k in order]
    total = float(sum(sizes))
    fig, ax = plt.subplots(figsize=(10.8, 7.2))

    def _autopct(p: float) -> str:
        n = p / 100.0 * total
        return f"{p:.0f}%\n({n:.0f} cells)" if p >= 8 else ""

    wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=_autopct,
        startangle=90,
        pctdistance=0.58,
        wedgeprops=dict(linewidth=1.5, edgecolor="white"),
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_color("white")
        t.set_fontweight("bold")
    ax.legend(
        wedges,
        [f"{REASON_LABEL.get(k, k)} (n={counts[k]})" for k in order],
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        frameon=True,
    )
    n_ok = int((df["status"] == "ok").sum())
    n_fail = int((df["status"] != "ok").sum())
    extra = f"; {n_fail} cells failed to run" if n_fail else ""
    ax.set_title(
        f"How the {n_ok} completed cells ended on the {exp.title_mission}\n"
        f"(most common outcome among 4 evaluation episodes{extra})"
    )
    _save(fig, exp, "04_termination_pie.png")


def fig_endings_by_system(df: pd.DataFrame, exp: Experiment) -> None:
    reasons = [k for k in REASON_LABEL if k in set(df["termination_reason_mode"])]
    for k in df["termination_reason_mode"].unique():
        if k not in reasons:
            reasons.append(str(k))
    counts = (
        df.groupby(["propulsion", "termination_reason_mode"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=list(exp.propulsion))
    )
    for r in reasons:
        if r not in counts.columns:
            counts[r] = 0
    counts = counts[reasons]
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    x = np.arange(len(exp.propulsion))
    bottom = np.zeros(len(exp.propulsion))
    n_per = int(len(exp.agents) * 4)
    for r in reasons:
        vals = counts[r].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=REASON_COLORS.get(r, "#999999"),
            edgecolor="white",
            label=REASON_LABEL.get(r, r),
            width=0.7,
        )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([PROP_LABEL[p] for p in exp.propulsion])
    ax.set_ylabel(f"Number of cells (out of {n_per} per system)")
    ax.set_ylim(0, n_per + 4)
    ax.set_title(
        f"How cells ended, by propulsion system\n"
        f"({n_per} cells per system = {len(exp.agents)} controllers × 4 seeds)"
    )
    ax.legend(
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, exp, "06_endings_by_system.png")


def plot_mars(df: pd.DataFrame, exp: Experiment) -> None:
    fig_grouped_bars(
        df,
        exp,
        column="trip_time_days",
        ylabel="Trip time (days)",
        title=(
            "Mean trip time on the crewed Mars mission\n"
            "(error bars = s.d. across 4 seeds; dots = each seed)"
        ),
        filename="01_trip_time.png",
        ymax=exp.trip_ymax,
        mark_limit=True,
        footnote=exp.trip_note,
        footnote_xy=(0.99, 0.55),
        footnote_ha="right",
    )
    fig_progress_lines(df, exp)
    fig_progress_heatmap(df, exp)
    fig_termination_pie(df, exp)
    fig_grouped_bars(
        df,
        exp,
        column="wear_fraction",
        ylabel="Thruster / engine wear fraction (0–1)",
        title=(
            "Engine wear at the end of the evaluation episodes\n"
            "(mean ± 1 s.d. over 4 seeds; 1.0 means the hardware is fully worn)"
        ),
        filename="05_wear.png",
        ymax=1.12,
    )
    fig_endings_by_system(df, exp)


def plot_leo(df: pd.DataFrame, exp: Experiment) -> None:
    fig_grouped_bars(
        df,
        exp,
        column="success_rate",
        ylabel="Success rate (fraction of 4 eval episodes)",
        title=(
            "Success rate on the LEO → GEO transfer\n"
            "(error bars = s.d. across 4 seeds; dots = each seed; missing = none)"
        ),
        filename="01_success_rate.png",
        ymax=1.12,
        footnote=(
            "Only Hall HERMeS + Edelbaum produced any arrivals\n"
            "(seeds 0 and 3: 4/4; seeds 1 and 2: 3/4)."
        ),
        footnote_xy=(0.98, 0.97),
        footnote_ha="right",
    )
    fig_progress_lines(df, exp)
    fig_progress_heatmap(df, exp)
    fig_termination_pie(df, exp)
    fig_grouped_bars(
        df,
        exp,
        column="trip_time_days",
        ylabel="Trip time (days)",
        title=(
            "Mean trip time on the LEO → GEO transfer\n"
            "(error bars = s.d. across 4 seeds; dots = each seed)"
        ),
        filename="05_trip_time.png",
        ymax=exp.trip_ymax,
        mark_limit=True,
        footnote=exp.trip_note,
        footnote_xy=(0.22, 0.97),
        footnote_ha="left",
    )
    fig_endings_by_system(df, exp)


def _summarize(df: pd.DataFrame, exp: Experiment) -> None:
    n_fail = int((df["status"] != "ok").sum())
    n_success = int((df["success_rate"] > 0).sum())
    n_cost = int(df["cost_per_kg_delivered"].notna().sum())
    print(f"Loaded {len(df)} rows from {exp.csv.relative_to(ROOT)}")
    print(f"  mission: {sorted(df['mission'].unique().tolist())}")
    print(f"  status!=ok cells: {n_fail}")
    print(f"  cells with success_rate > 0: {n_success}")
    print(
        f"  cost_per_kg_delivered finite: {n_cost} / {len(df)} "
        "(NaN when payload_delivered_kg is 0)"
    )
    print(f"  writing to {exp.outdir.relative_to(ROOT)}/")


def main() -> int:
    print("Plotting newest Mars sweep (nuclear_long2) and newest LEO-GEO sweep (electric_long).")
    print("Skipping older same-experiment runs: nuclear_long, my_pilot_long_gate.")
    print(
        "No usable tables for my_pilot, my_pilot_nuclear, electric_vs_nuclear, smoke "
        "(Git LFS pointers)."
    )
    mars_df = load(MARS)
    _summarize(mars_df, MARS)
    plot_mars(mars_df, MARS)

    leo_df = load(LEO)
    _summarize(leo_df, LEO)
    plot_leo(leo_df, LEO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
