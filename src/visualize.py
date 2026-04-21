"""
visualize.py
------------
Generates a set of PNG charts from the cleaned Sun Run 2026 dataset.

All charts land in `charts/` at the repo root.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "clean" / "sunrun2026_clean.csv"
CHARTS = ROOT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

AGE_ORDER = [
    "15-19", "16-18", "19-24", "25-29", "30-34", "35-39",
    "40-44", "45-49", "50-54", "55-59", "60-64", "65-69",
    "70-74", "75-79", "80-84", "85-89", "90-94", "95+",
]
SEX_PALETTE = {"M": "#1f77b4", "F": "#d62728", "X": "#2ca02c"}


def load() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH)
    # Filter obvious outliers (sub-25min is impossible; >4hr 10K is extremely rare)
    df = df[(df["chip_time_min"] >= 25) & (df["chip_time_min"] <= 240)]
    return df


# ---------------------------------------------------------------------------
# 1. Distribution of finish times by sex
# ---------------------------------------------------------------------------
def fig_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for sex, label in [("M", "Men"), ("F", "Women"), ("X", "Non-binary")]:
        sub = df[df["sex"] == sex]
        ax.hist(sub["chip_time_min"], bins=80, alpha=0.55,
                label=f"{label} (n={len(sub):,})", color=SEX_PALETTE[sex])
    ax.set_xlabel("Chip time (minutes)")
    ax.set_ylabel("Number of finishers")
    ax.set_title("Sun Run 2026 — Finish-time distribution by sex")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "01_time_distribution.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Median time vs age group (line plot, M / F / X)
# ---------------------------------------------------------------------------
def fig_median_by_age(df: pd.DataFrame) -> None:
    med = (
        df.groupby(["age_group", "sex"])["chip_time_min"]
        .median()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    for sex, label in [("M", "Men"), ("F", "Women")]:
        sub = med[med["sex"] == sex]
        sub = sub.set_index("age_group").reindex(AGE_ORDER).dropna()
        ax.plot(sub.index, sub["chip_time_min"], marker="o",
                label=label, color=SEX_PALETTE[sex], linewidth=2.5)
    ax.set_xlabel("Age group")
    ax.set_ylabel("Median chip time (min)")
    ax.set_title("Median 10K time by age group — Sun Run 2026")
    plt.xticks(rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "02_median_by_age.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Gender gap (F-M median minutes) by age group
# ---------------------------------------------------------------------------
def fig_gender_gap(df: pd.DataFrame) -> None:
    pivot = (
        df[df["sex"].isin(["M", "F"])]
        .groupby(["age_group", "sex"])["chip_time_min"]
        .median()
        .unstack()
        .reindex(AGE_ORDER)
        .dropna()
    )
    pivot["gap"] = pivot["F"] - pivot["M"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(pivot.index, pivot["gap"], color="#6c3483")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Median F - M time (min)")
    ax.set_title("Gender gap in finish time grows with age — Sun Run 2026")
    plt.xticks(rotation=45)
    for i, v in enumerate(pivot["gap"]):
        ax.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(CHARTS / "03_gender_gap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Start-delay vs chip time (hexbin) — the "corralling effect"
# ---------------------------------------------------------------------------
def fig_start_delay(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["start_delay_sec", "chip_time_min"])
    sub = sub[sub["start_delay_sec"] <= 2400]  # drop extreme tail for readability

    fig, ax = plt.subplots(figsize=(10, 7))
    hb = ax.hexbin(sub["start_delay_sec"] / 60, sub["chip_time_min"],
                   gridsize=60, cmap="viridis", mincnt=1, bins="log")
    ax.set_xlabel("Start delay (min, gun − chip)")
    ax.set_ylabel("Chip time (min)")
    ax.set_title("Later start = slower finish (corral effect)")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(count)")
    fig.tight_layout()
    fig.savefig(CHARTS / "04_start_delay_vs_chip.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Bib number vs chip time
# ---------------------------------------------------------------------------
def fig_bib_vs_time(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["bib", "chip_time_min"])
    sub = sub[sub["bib"] <= 60000]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hexbin(sub["bib"], sub["chip_time_min"], gridsize=60,
              cmap="magma", mincnt=1, bins="log")
    ax.set_xlabel("Bib number")
    ax.set_ylabel("Chip time (min)")
    ax.set_title("Bib numbers are correlated with finish time (seeding)")
    fig.tight_layout()
    fig.savefig(CHARTS / "05_bib_vs_time.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Top 15 cities — participation vs median time
# ---------------------------------------------------------------------------
def fig_top_cities(df: pd.DataFrame) -> None:
    agg = (
        df.groupby("city")
        .agg(n=("chip_time_min", "size"),
             median=("chip_time_min", "median"))
        .sort_values("n", ascending=False)
        .head(15)
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(agg["city"], agg["n"], color="#8ecae6", alpha=0.8, label="Participants")
    ax1.set_xlabel("City")
    ax1.set_ylabel("Participants")
    plt.xticks(rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(agg["city"], agg["median"], color="#023047", marker="o",
             linewidth=2.5, label="Median time")
    ax2.set_ylabel("Median chip time (min)")

    ax1.set_title("Top 15 cities: participation and median finish time")
    fig.tight_layout()
    fig.savefig(CHARTS / "06_top_cities.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Box plot of chip time by age group and sex
# ---------------------------------------------------------------------------
def fig_box_by_age(df: pd.DataFrame) -> None:
    sub = df[df["sex"].isin(["M", "F"])].copy()
    sub["age_group"] = pd.Categorical(sub["age_group"], categories=AGE_ORDER, ordered=True)
    sub = sub.dropna(subset=["age_group"])

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=sub, x="age_group", y="chip_time_min", hue="sex",
        palette=SEX_PALETTE, ax=ax, fliersize=1, showfliers=False,
    )
    ax.set_xlabel("Age group")
    ax.set_ylabel("Chip time (min)")
    ax.set_title("Finish-time spread by age group and sex (outliers hidden)")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(CHARTS / "07_box_by_age.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Correlation heatmap
# ---------------------------------------------------------------------------
AGE_MIDPOINTS = {
    "15-19": 17, "16-18": 17, "19-24": 22, "25-29": 27, "30-34": 32,
    "35-39": 37, "40-44": 42, "45-49": 47, "50-54": 52, "55-59": 57,
    "60-64": 62, "65-69": 67, "70-74": 72, "75-79": 77, "80-84": 82,
    "85-89": 87, "90-94": 92, "95+": 97,
}


def fig_correlation_heatmap(df: pd.DataFrame) -> None:
    sub = df.copy()
    sub["age_mid"] = sub["age_group"].map(AGE_MIDPOINTS)
    sub["sex_code"] = sub["sex"].map({"M": 0, "F": 1, "X": 2})
    cols = ["chip_time_min", "km_pace_min", "start_delay_sec",
            "bib", "age_mid", "sex_code", "sex_place"]
    corr = sub[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearman rank correlation between key variables")
    fig.tight_layout()
    fig.savefig(CHARTS / "08_correlation_heatmap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Cumulative finish curve
# ---------------------------------------------------------------------------
def fig_cumulative(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for sex, label in [("M", "Men"), ("F", "Women")]:
        sub = np.sort(df.loc[df["sex"] == sex, "chip_time_min"].values)
        y = np.arange(1, len(sub) + 1) / len(sub)
        ax.plot(sub, y, label=f"{label} (n={len(sub):,})",
                color=SEX_PALETTE[sex], linewidth=2.5)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Chip time (min)")
    ax.set_ylabel("Fraction of finishers ≤ t")
    ax.set_title("Cumulative finish-time distribution by sex")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "09_cumulative.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load()
    print(f"Loaded {len(df):,} rows for plotting")
    fig_distribution(df)
    fig_median_by_age(df)
    fig_gender_gap(df)
    fig_start_delay(df)
    fig_bib_vs_time(df)
    fig_top_cities(df)
    fig_box_by_age(df)
    fig_correlation_heatmap(df)
    fig_cumulative(df)
    print(f"Wrote 9 charts -> {CHARTS}/")


if __name__ == "__main__":
    main()
