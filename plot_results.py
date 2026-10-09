#!/usr/bin/env python3
"""Charts for the nuclear_long2 Mars-crew sweep.

Reads ONLY results/nuclear_long2/results.csv (config: configs/nuclear_long.yaml).
Does not write to results/ or configs/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "results" / "nuclear_long2" / "results.csv"
FIGDIR = ROOT / "figures"
TRIP_LIMIT_DAYS = 220.0
DPI = 300

# Okabe–Ito colorblind-safe palette
AGENT_COLORS = {
    "random": "#000000",
    "max_thrust": "#E69F00",
    "prograde": "#56B4E9",
    "ppo": "#009E73",
    "sac": "#CC79A7",
}
PROP_COLORS = {
    "hall_hermes": "#0072B2",
    "ion_next": "#56B4E9",
    "nep_brayton": "#E69F00",
    "ntp_nerva": "#009E73",
    "ntp_pewee": "#D55E00",
}
REASON_COLORS = {
    "timeout": "#0072B2",
    "out_of_propellant": "#E69F00",
    "hardware_failure": "#009E73",
}

AGENTS = ["random", "max_thrust", "prograde", "ppo", "sac"]
PROPULSION = ["hall_hermes", "ion_next", "nep_brayton", "ntp_nerva", "ntp_pewee"]
PROP_LABEL = {
    "hall_hermes": "Hall HERMeS\n(electric)",
    "ion_next": "Ion NEXT\n(electric)",
    "nep_brayton": "NEP Brayton\n(nuclear electric)",
    "ntp_nerva": "NTP NERVA\n(nuclear thermal)",
    "ntp_pewee": "NTP Pewee\n(nuclear thermal)",
}
AGENT_LABEL = {
    "random": "random",
    "max_thrust": "max thrust",
    "prograde": "prograde",
    "ppo": "PPO",
    "sac": "SAC",
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
        "savefig.pad_inches": 0.25,
    }
)


def load() -> pd.DataFrame:
    if not CSV.exists():
        raise SystemExit(f"Missing {CSV}")
    df = pd.read_csv(CSV)
    df["agent"] = pd.Categorical(df["agent"], AGENTS, ordered=True)
    df["propulsion"] = pd.Categorical(df["propulsion"], PROPULSION, ordered=True)
    return df


def _save(fig: plt.Figure, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("wrote", path)
    return path


def fig_trip_time(df: pd.DataFrame) -> None:
    """Grouped bars of mean trip time with seed points and the 220-day limit."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharey=False)
    families = [
        ("electric", "Electric propulsion (Hall HERMeS, Ion NEXT)"),
        ("nuclear", "Nuclear propulsion (NEP Brayton, NTP NERVA, NTP Pewee)"),
    ]
    x = np.arange(len(AGENTS))
    width = 0.16

    for ax, (family, subtitle) in zip(axes, families):
        props = [p for p in PROPULSION if df.loc[df["propulsion"] == p, "family"].iloc[0] == family]
        n = len(props)
        offsets = (np.arange(n) - (n - 1) / 2) * width
        for off, prop in zip(offsets, props):
            sub = df[df["propulsion"] == prop]
            means, stds = [], []
            rng = np.random.default_rng(0)
            for i, agent in enumerate(AGENTS):
                vals = sub.loc[sub["agent"] == agent, "trip_time_days"].to_numpy()
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                jitter = rng.uniform(-0.03, 0.03, size=len(vals))
                ax.scatter(
                    np.full(len(vals), x[i] + off) + jitter,
                    vals,
                    color=PROP_COLORS[prop],
                    s=22,
                    zorder=3,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.4,
                )
            ax.bar(
                x + off,
                means,
                width=width * 0.95,
                yerr=stds,
                capsize=3,
                color=PROP_COLORS[prop],
                edgecolor="white",
                label=PROP_LABEL[prop].replace("\n", " "),
                zorder=2,
            )
        ax.axhline(
            TRIP_LIMIT_DAYS,
            color="#D55E00",
            linestyle="--",
            linewidth=2.0,
            label="220-day crew time limit",
            zorder=1,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([AGENT_LABEL[a] for a in AGENTS], rotation=20, ha="right")
        ax.set_xlabel("Controller (agent)")
        ax.set_title(subtitle, fontsize=13)
        ax.grid(axis="y", alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend().remove()

    axes[0].set_ylabel("Trip time (days)")
    axes[0].set_ylim(0, 255)
    axes[1].set_ylim(0, 255)
    axes[1].text(
        0.02,
        0.12,
        "NTP NERVA and NTP Pewee bars are near 0 days\n"
        "(every cell ran out of propellant in under 2 days).",
        transform=axes[1].transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BBBBBB"),
    )
    merged_h, merged_l, seen = [], [], set()
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab in seen:
                continue
            seen.add(lab)
            merged_h.append(h)
            merged_l.append(lab)
    fig.legend(
        merged_h,
        merged_l,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fontsize=10,
    )
    fig.suptitle(
        "Mean trip time on the crewed Mars mission\n"
        "(error bars = standard deviation across 4 seeds; dots = each seed)",
        y=1.03,
    )
    fig.tight_layout()
    _save(fig, "nuclear_long2_trip_time.png")


def fig_progress_lines(df: pd.DataFrame) -> None:
    """How mean transfer progress changes across controllers, one line per system."""
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    x = np.arange(len(AGENTS))
    for prop in PROPULSION:
        means, lo, hi = [], [], []
        for agent in AGENTS:
            vals = df.loc[(df["propulsion"] == prop) & (df["agent"] == agent), "progress"]
            m = float(vals.mean())
            s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            means.append(m)
            lo.append(m - s)
            hi.append(m + s)
        ax.plot(
            x,
            means,
            marker="o",
            linewidth=2.4,
            markersize=8,
            color=PROP_COLORS[prop],
            label=PROP_LABEL[prop].replace("\n", " "),
        )
        ax.fill_between(x, lo, hi, color=PROP_COLORS[prop], alpha=0.15, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABEL[a] for a in AGENTS])
    ax.set_xlabel("Controller (agent)")
    ax.set_ylabel("Transfer progress (0 = start, 1 = arrival geometry)")
    ax.set_ylim(0, 0.55)
    ax.set_title(
        "How close each pairing got to Mars arrival geometry\n"
        "(mean ± 1 standard deviation over 4 seeds; 0 of 100 cells arrived)"
    )
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, loc="upper left", fontsize=9)
    _save(fig, "nuclear_long2_progress_lines.png")


def fig_progress_heatmap(df: pd.DataFrame) -> None:
    """Two swept parameters: propulsion × agent."""
    pivot = (
        df.pivot_table(index="propulsion", columns="agent", values="progress", aggfunc="mean")
        .reindex(index=PROPULSION, columns=AGENTS)
    )
    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    im = ax.imshow(pivot.to_numpy(), cmap="cividis", vmin=0.0, vmax=0.45, aspect="auto")
    ax.set_xticks(np.arange(len(AGENTS)))
    ax.set_yticks(np.arange(len(PROPULSION)))
    ax.set_xticklabels([AGENT_LABEL[a] for a in AGENTS])
    ax.set_yticklabels([PROP_LABEL[p].replace("\n", " ") for p in PROPULSION])
    ax.set_xlabel("Controller (agent)")
    ax.set_ylabel("Propulsion system")
    for i in range(len(PROPULSION)):
        for j in range(len(AGENTS)):
            val = pivot.iloc[i, j]
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color="white" if val > 0.22 else "black",
                fontsize=12,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean transfer progress (dimensionless, 0–1)")
    ax.set_title(
        "Transfer progress heatmap (propulsion × controller)\n"
        "Each cell is the mean of 4 independent seeds"
    )
    _save(fig, "nuclear_long2_progress_heatmap.png")


def fig_termination_pie(df: pd.DataFrame) -> None:
    """Part-of-whole: how the 100 completed cells ended."""
    counts = df["termination_reason_mode"].value_counts()
    order = [k for k in ("timeout", "out_of_propellant", "hardware_failure") if k in counts]
    sizes = [int(counts[k]) for k in order]
    labels = {
        "timeout": "Hit 220-day time limit",
        "out_of_propellant": "Ran out of propellant",
        "hardware_failure": "Hardware failed",
    }
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=[REASON_COLORS[k] for k in order],
        autopct=lambda p: f"{p:.0f}%\n({p / 100.0 * sum(sizes):.0f} cells)",
        startangle=90,
        pctdistance=0.62,
        wedgeprops=dict(linewidth=1.5, edgecolor="white"),
    )
    for t in autotexts:
        t.set_fontsize(12)
        t.set_color("white")
        t.set_fontweight("bold")
    ax.legend(
        wedges,
        [f"{labels[k]} (n={counts[k]})" for k in order],
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        frameon=True,
    )
    ax.set_title(
        "How the 100 completed cells ended\n"
        "(most common outcome among 4 evaluation episodes; success rate was 0% in every cell)"
    )
    _save(fig, "nuclear_long2_termination_pie.png")


def fig_wear_bars(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    x = np.arange(len(PROPULSION))
    width = 0.15
    offsets = (np.arange(len(AGENTS)) - (len(AGENTS) - 1) / 2) * width
    for off, agent in zip(offsets, AGENTS):
        means, stds = [], []
        for prop in PROPULSION:
            vals = df.loc[(df["propulsion"] == prop) & (df["agent"] == agent), "wear_fraction"]
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        ax.bar(
            x + off,
            means,
            width=width * 0.95,
            yerr=stds,
            capsize=2.5,
            color=AGENT_COLORS[agent],
            edgecolor="white",
            label=AGENT_LABEL[agent],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([PROP_LABEL[p] for p in PROPULSION])
    ax.set_ylabel("Thruster / engine wear fraction (0–1)")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Propulsion system")
    ax.set_title("")
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.suptitle(
        "Engine wear at the end of the evaluation episodes\n"
        "(mean ± 1 standard deviation over 4 seeds; 1.0 means the hardware is fully worn)",
        y=1.01,
    )
    ax.legend(
        title="Controller",
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=True,
    )
    fig.tight_layout()
    _save(fig, "nuclear_long2_wear.png")


def fig_termination_by_system(df: pd.DataFrame) -> None:
    """Stacked counts: ending reason vs propulsion (shows the NTP vs EP split)."""
    reasons = ["timeout", "out_of_propellant", "hardware_failure"]
    counts = (
        df.groupby(["propulsion", "termination_reason_mode"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=PROPULSION)
    )
    for r in reasons:
        if r not in counts.columns:
            counts[r] = 0
    counts = counts[reasons]

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    x = np.arange(len(PROPULSION))
    bottom = np.zeros(len(PROPULSION))
    labels = {
        "timeout": "Hit 220-day time limit",
        "out_of_propellant": "Ran out of propellant",
        "hardware_failure": "Hardware failed",
    }
    for r in reasons:
        vals = counts[r].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=REASON_COLORS[r],
            edgecolor="white",
            label=labels[r],
            width=0.7,
        )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([PROP_LABEL[p] for p in PROPULSION])
    ax.set_ylabel("Number of cells (out of 20 per system)")
    ax.set_ylim(0, 24)
    ax.set_title(
        "How cells ended, by propulsion system\n"
        "(20 cells per system = 5 controllers × 4 seeds; none reported a successful arrival)"
    )
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    _save(fig, "nuclear_long2_endings_by_system.png")


def main() -> int:
    df = load()
    n_fail = int((df["status"] != "ok").sum())
    n_success = int((df["success_rate"] > 0).sum())
    print(f"Loaded {len(df)} rows from {CSV.relative_to(ROOT)}")
    print(f"  status!=ok cells: {n_fail}")
    print(f"  cells with success_rate > 0: {n_success}")
    print(
        "  cost_per_kg_delivered finite values: "
        f"{int(df['cost_per_kg_delivered'].notna().sum())} / {len(df)} "
        "(all missing because payload_delivered_kg is 0 when arrival fails)"
    )
    fig_trip_time(df)
    fig_progress_lines(df)
    fig_progress_heatmap(df)
    fig_termination_pie(df)
    fig_wear_bars(df)
    fig_termination_by_system(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
