#!/usr/bin/env python3
"""Descriptive plots of propulsion-RL experiment outputs.

No scoring, ranking, or composite metrics. Electric and nuclear families
are drawn on separate axes because they were run on different missions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"

SWEEP_CSVS = [
    RESULTS / "my_pilot" / "results.csv",
    RESULTS / "my_pilot_nuclear" / "results.csv",
    RESULTS / "electric_vs_nuclear" / "results.csv",
]
SMOKE_JSONL = RESULTS / "smoke" / "results.jsonl"

METRIC_COLS = [
    "trip_time_days",
    "wear_fraction",
    "constraint_violations",
    "termination_reason_mode",
    "success_rate",
    "status",
]

# Distinct categorical colors for termination reasons — not a good/bad scale.
REASON_COLORS = {
    "timeout": "#4C78A8",
    "out_of_propellant": "#F58518",
    "hardware_failure": "#54A24B",
    "safety_violation": "#B279A2",
    "diverged": "#E45756",
}


def _load_sweeps() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in SWEEP_CSVS:
        if not path.exists():
            continue
        df = pd.read_csv(path).copy()
        df = df.assign(source_file=str(path.relative_to(ROOT)))
        frames.append(df)
    if SMOKE_JSONL.exists():
        smoke = pd.read_json(SMOKE_JSONL, lines=True)
        smoke = smoke.assign(source_file=str(SMOKE_JSONL.relative_to(ROOT)))
        frames.append(smoke)
    if not frames:
        raise SystemExit("No sweep result files found.")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data = data.copy()
    data["facet"] = (
        data["propulsion"].astype(str)
        + "\n"
        + data["mission"].astype(str)
        + "\n["
        + data["sweep"].astype(str)
        + "]"
    )
    data["family"] = data["family"].astype(str)
    return data


def _write_table(name: str, df: pd.DataFrame) -> Path:
    path = FIGDIR / name
    df.to_csv(path, index=False)
    return path


def inventory(data: pd.DataFrame) -> None:
    print("=" * 72)
    print("EXPERIMENT RESULT FILES (repo-wide, excluding per-cell dumps)")
    print("=" * 72)

    extra = [
        RESULTS / "my_pilot" / "matrix.csv",
        RESULTS / "my_pilot_nuclear" / "matrix.csv",
        RESULTS / "electric_vs_nuclear" / "matrix.csv",
        RESULTS / "smoke" / "matrix.csv",
        RESULTS / "my_pilot" / "excluded.csv",
        RESULTS / "my_pilot_nuclear" / "excluded.csv",
        RESULTS / "electric_vs_nuclear" / "excluded.csv",
    ]
    for path in SWEEP_CSVS + [SMOKE_JSONL] + extra:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            df = pd.read_json(path, lines=True)
        else:
            df = pd.read_csv(path)
        print(f"\n{path.relative_to(ROOT)}")
        print(f"  rows={len(df)}  cols={len(df.columns)}")
        print("  headers:")
        for col in df.columns:
            print(f"    - {col}")
        header_path = FIGDIR / f"00_headers_{path.parent.name}_{path.name}.txt"
        header_path.write_text(
            f"{path.relative_to(ROOT)}\nrows={len(df)}\ncols={len(df.columns)}\n"
            + "\n".join(df.columns)
            + "\n"
        )

    print("\n" + "=" * 72)
    print("COMBINED SWEEP ROWS USED FOR FIGURES")
    print("=" * 72)
    print(f"n_rows={len(data)}")
    print("\nCounts by family × propulsion × mission × sweep:")
    counts = (
        data.groupby(["family", "propulsion", "mission", "sweep"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["family", "mission", "propulsion", "sweep"])
    )
    print(counts.to_string(index=False))
    _write_table("00_inventory_counts.csv", counts)

    print("\nSchema columns present in the combined frame:")
    for col in data.columns:
        print(f"  - {col}")


def _family_order(data: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(data["family"].tolist()))
    preferred = ["electric", "nuclear"]
    return [f for f in preferred if f in present] + [f for f in present if f not in preferred]


def _facet_order(subset: pd.DataFrame) -> list[str]:
    rows = (
        subset[["sweep", "propulsion", "mission", "facet"]]
        .drop_duplicates()
        .sort_values(["sweep", "mission", "propulsion"])
    )
    return rows["facet"].tolist()


WRAP = 4


def _family_row_plan(data: pd.DataFrame) -> list[tuple[str, list[str], int]]:
    plan = []
    for family in _family_order(data):
        facets = _facet_order(data[data["family"] == family])
        n_rows = int(np.ceil(len(facets) / WRAP)) if facets else 1
        plan.append((family, facets, n_rows))
    return plan


def _strip_box_by_family(
    data: pd.DataFrame,
    y: str,
    filename: str,
    ylabel: str,
    table_name: str,
) -> None:
    plan = _family_row_plan(data)
    total_rows = sum(block[2] for block in plan)
    fig, axes = plt.subplots(
        total_rows,
        WRAP,
        figsize=(3.3 * WRAP + 1.0, 4.0 * total_rows),
        squeeze=False,
        sharey=False,
    )
    fig.suptitle(
        f"{ylabel} by agent — each point is one seed; "
        "electric and nuclear are on separate rows (not a shared scale)",
        y=1.01,
        fontsize=11,
    )

    table_rows = []
    row_cursor = 0
    for family, facets, n_rows in plan:
        subset = data[data["family"] == family]
        for local_row in range(n_rows):
            r = row_cursor + local_row
            for c in range(WRAP):
                ax = axes[r][c]
                idx = local_row * WRAP + c
                if idx >= len(facets):
                    ax.axis("off")
                    continue
                cell = subset[subset["facet"] == facets[idx]].copy()
                agents = sorted(cell["agent"].unique())
                sns.boxplot(
                    data=cell,
                    x="agent",
                    y=y,
                    order=agents,
                    color="#D9D9D9",
                    fliersize=0,
                    width=0.55,
                    ax=ax,
                )
                sns.stripplot(
                    data=cell,
                    x="agent",
                    y=y,
                    order=agents,
                    color="#2F2F2F",
                    size=5,
                    jitter=0.18,
                    ax=ax,
                )
                mission = cell["mission"].iloc[0]
                propulsion = cell["propulsion"].iloc[0]
                sweep = cell["sweep"].iloc[0]
                ax.set_title(
                    f"{family}: {propulsion}\n{mission}  [{sweep}]",
                    fontsize=8,
                )
                ax.set_xlabel("")
                ax.set_ylabel(ylabel if c == 0 else "")
                ax.tick_params(axis="x", rotation=40, labelsize=8)
                ax.grid(axis="y", alpha=0.3)
                for _, row in cell.iterrows():
                    table_rows.append(
                        {
                            "family": family,
                            "propulsion": propulsion,
                            "mission": mission,
                            "sweep": sweep,
                            "agent": row["agent"],
                            "seed": row["seed"],
                            y: row[y],
                        }
                    )
        row_cursor += n_rows

    fig.tight_layout()
    out = FIGDIR / filename
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    tbl = pd.DataFrame(table_rows).sort_values(
        ["family", "sweep", "mission", "propulsion", "agent", "seed"]
    )
    tpath = _write_table(table_name, tbl)
    print(f"\n--- {filename}  raw values ({tpath.name}) ---")
    print(tbl.to_string(index=False))


def plot_termination(data: pd.DataFrame) -> None:
    plan = _family_row_plan(data)
    total_rows = sum(block[2] for block in plan)
    fig, axes = plt.subplots(
        total_rows,
        WRAP,
        figsize=(3.3 * WRAP + 1.0, 4.0 * total_rows),
        squeeze=False,
        sharey=False,
    )
    fig.suptitle(
        "termination_reason_mode counts by agent — one count per seed; "
        "electric and nuclear on separate rows",
        y=1.01,
        fontsize=11,
    )

    reasons = sorted(data["termination_reason_mode"].dropna().astype(str).unique())
    table_rows = []
    legend_ax = None
    row_cursor = 0
    for family, facets, n_rows in plan:
        subset = data[data["family"] == family]
        for local_row in range(n_rows):
            r = row_cursor + local_row
            for c in range(WRAP):
                ax = axes[r][c]
                idx = local_row * WRAP + c
                if idx >= len(facets):
                    ax.axis("off")
                    continue
                cell = subset[subset["facet"] == facets[idx]]
                agents = sorted(cell["agent"].unique())
                counts = (
                    cell.groupby(["agent", "termination_reason_mode"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(index=agents, fill_value=0)
                )
                for reason in reasons:
                    if reason not in counts.columns:
                        counts[reason] = 0
                counts = counts[reasons]
                bottom = np.zeros(len(agents))
                x = np.arange(len(agents))
                for reason in reasons:
                    vals = counts[reason].to_numpy(dtype=float)
                    ax.bar(
                        x,
                        vals,
                        bottom=bottom,
                        color=REASON_COLORS.get(reason, "#9A9A9A"),
                        label=reason,
                        width=0.7,
                    )
                    bottom += vals
                ax.set_xticks(x)
                ax.set_xticklabels(agents, rotation=40, ha="right", fontsize=8)
                mission = cell["mission"].iloc[0]
                propulsion = cell["propulsion"].iloc[0]
                sweep = cell["sweep"].iloc[0]
                ax.set_title(
                    f"{family}: {propulsion}\n{mission}  [{sweep}]",
                    fontsize=8,
                )
                ax.set_ylabel("seed count" if c == 0 else "")
                ax.grid(axis="y", alpha=0.3)
                legend_ax = ax
                for agent in agents:
                    for reason in reasons:
                        n = int(counts.loc[agent, reason])
                        if n == 0:
                            continue
                        table_rows.append(
                            {
                                "family": family,
                                "propulsion": propulsion,
                                "mission": mission,
                                "sweep": sweep,
                                "agent": agent,
                                "termination_reason_mode": reason,
                                "n_seeds": n,
                            }
                        )
        row_cursor += n_rows

    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=len(reasons))

    fig.tight_layout()
    out = FIGDIR / "02_termination_reason_by_agent.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    tbl = pd.DataFrame(table_rows)
    tpath = _write_table("02_termination_reason_counts.csv", tbl)
    print(f"\n--- 02_termination_reason_by_agent.png  raw counts ({tpath.name}) ---")
    print(tbl.to_string(index=False))

    seed_level = data[
        [
            "family",
            "propulsion",
            "mission",
            "sweep",
            "agent",
            "seed",
            "termination_reason_mode",
        ]
    ].sort_values(["family", "sweep", "mission", "propulsion", "agent", "seed"])
    _write_table("02_termination_reason_by_seed.csv", seed_level)
    print("\nPer-seed termination_reason_mode:")
    print(seed_level.to_string(index=False))


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    ax.set_title("Experiment step loop (methodology — not results)", pad=12)

    boxes = [
        (0.3, 1.7, 1.9, 1.5, "Mission setup\nreset env\nobserve state"),
        (2.6, 1.7, 1.9, 1.5, "Agent\nselects 5-D action\n(throttle, Isp\nknob, steering,\nthermal margin)"),
        (4.9, 1.7, 2.1, 1.5, "Propulsion model\ndecode → thrust,\nṁ, wear, heat"),
        (7.4, 1.7, 2.0, 1.5, "Vehicle\npower bus +\norbit dynamics"),
        (9.8, 1.7, 2.3, 1.5, "Termination check\nmission.terminated\nor step-limit\ntruncation"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor="#F4F4F4",
                edgecolor="#333333",
                linewidth=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        ymid = boxes[i][1] + boxes[i][3] / 2
        ax.add_patch(
            FancyArrowPatch(
                (x0 + 0.05, ymid),
                (x1 - 0.05, ymid),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#333333",
                lw=1.2,
            )
        )

    ax.add_patch(
        FancyBboxPatch(
            (4.6, 0.25),
            3.4,
            1.1,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor="#EEF3F8",
            edgecolor="#333333",
            linewidth=1.2,
        )
    )
    ax.text(
        6.3,
        0.8,
        "Recorded outcome per seed\ntrip_time_days, termination_reason_mode,\nwear_fraction, constraint_violations",
        ha="center",
        va="center",
        fontsize=8,
    )
    ax.add_patch(
        FancyArrowPatch(
            (10.95, 1.7),
            (6.3, 1.4),
            arrowstyle="-|>",
            mutation_scale=12,
            color="#333333",
            lw=1.2,
            connectionstyle="arc3,rad=0.25",
        )
    )
    fig.tight_layout()
    out = FIGDIR / "05_experiment_pipeline.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--- {out.name}  (methodology diagram, no numeric table) ---")


def _wrap_nums(values: list[float], per_line: int = 4) -> str:
    parts = [f"{v:.4g}" for v in values]
    lines = [", ".join(parts[i : i + per_line]) for i in range(0, len(parts), per_line)]
    return "\n".join(lines)


def plot_overview_matrix(data: pd.DataFrame) -> None:
    """One overview table per family. Cell text is raw seed values, not scores."""
    written = []
    for family in _family_order(data):
        subset = data[data["family"] == family]
        row_keys = (
            subset[["propulsion", "mission", "sweep"]]
            .drop_duplicates()
            .sort_values(["sweep", "mission", "propulsion"])
        )
        agents = sorted(subset["agent"].unique())
        fig_w = max(3.6 * len(agents) + 3.2, 12)
        fig_h = max(2.6 * len(row_keys) + 1.6, 6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(0, len(agents))
        ax.set_ylim(0, len(row_keys))
        ax.invert_yaxis()
        ax.set_xticks(np.arange(len(agents)) + 0.5)
        ax.set_xticklabels(agents, fontsize=10)
        ax.set_yticks(np.arange(len(row_keys)) + 0.5)
        ax.set_yticklabels(
            [f"{r.propulsion}\n{r.mission}\n[{r.sweep}]" for r in row_keys.itertuples()],
            fontsize=9,
        )
        ax.set_title(
            f"Overview — {family} family (raw per-seed values, no scoring)",
            loc="left",
        )
        ax.tick_params(length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")

        for i, row in enumerate(row_keys.itertuples()):
            for j, agent in enumerate(agents):
                cell = subset[
                    (subset["propulsion"] == row.propulsion)
                    & (subset["mission"] == row.mission)
                    & (subset["sweep"] == row.sweep)
                    & (subset["agent"] == agent)
                ].sort_values("seed")
                ax.add_patch(
                    plt.Rectangle(
                        (j, i),
                        1,
                        1,
                        fill=True,
                        facecolor="#FAFAFA",
                        edgecolor="#BBBBBB",
                        linewidth=0.7,
                    )
                )
                if cell.empty:
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        "—",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="#888888",
                    )
                    continue
                reasons = ", ".join(
                    f"{k}×{int(v)}"
                    for k, v in cell["termination_reason_mode"].value_counts().items()
                )
                text = (
                    f"trip_time_days\n{_wrap_nums(list(cell['trip_time_days']))}\n"
                    f"term: {reasons}\n"
                    f"wear_fraction\n{_wrap_nums(list(cell['wear_fraction']))}"
                )
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    family="DejaVu Sans",
                    linespacing=1.25,
                )

        fig.tight_layout()
        out_name = f"06_overview_matrix_{family}.png"
        fig.savefig(FIGDIR / out_name, dpi=170, bbox_inches="tight")
        plt.close(fig)
        written.append(out_name)

    overview = data[
        [
            "family",
            "propulsion",
            "mission",
            "sweep",
            "agent",
            "seed",
            "trip_time_days",
            "termination_reason_mode",
            "wear_fraction",
            "constraint_violations",
        ]
    ].sort_values(["family", "sweep", "mission", "propulsion", "agent", "seed"])
    tpath = _write_table("06_overview_matrix_raw.csv", overview)
    print(f"\n--- overview matrices {written}  raw cell values ({tpath.name}) ---")
    print(overview.to_string(index=False))


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    data = _load_sweeps()
    inventory(data)

    plot_cols = [
        "sweep",
        "family",
        "propulsion",
        "mission",
        "agent",
        "seed",
        *METRIC_COLS,
        "source_file",
    ]
    plot_cols = [c for c in plot_cols if c in data.columns]
    used = data[plot_cols].copy()
    _write_table("00_combined_plot_source.csv", used)

    _strip_box_by_family(
        data,
        y="trip_time_days",
        filename="01_trip_time_days_by_agent.png",
        ylabel="trip_time_days",
        table_name="01_trip_time_days_by_seed.csv",
    )
    plot_termination(data)
    _strip_box_by_family(
        data,
        y="wear_fraction",
        filename="03_wear_fraction_by_agent.png",
        ylabel="wear_fraction",
        table_name="03_wear_fraction_by_seed.csv",
    )
    _strip_box_by_family(
        data,
        y="constraint_violations",
        filename="04_constraint_violations_by_agent.png",
        ylabel="constraint_violations",
        table_name="04_constraint_violations_by_seed.csv",
    )
    plot_pipeline()
    plot_overview_matrix(data)

    print("\nWrote figures to", FIGDIR)
    for p in sorted(FIGDIR.iterdir()):
        print(" ", p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
