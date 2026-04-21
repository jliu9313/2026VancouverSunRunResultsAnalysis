"""
analyze.py
----------
Summary-statistics and correlation analysis for the Vancouver Sun Run 2026.

Reads the clean CSV produced by `parse_data.py` and writes tidy
summary tables to `reports/` plus a plain-text findings dump.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "clean" / "sunrun2026_clean.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


# Age-group midpoint helper (for numeric correlations)
AGE_MIDPOINTS = {
    "00-14": 10, "15-19": 17, "16-18": 17, "19-24": 22,
    "25-29": 27, "30-34": 32, "35-39": 37, "40-44": 42,
    "45-49": 47, "50-54": 52, "55-59": 57, "60-64": 62,
    "65-69": 67, "70-74": 72, "75-79": 77, "80-84": 82,
    "85-89": 87, "90-94": 92, "95+": 97, "3+": 65,
}


def load() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH)
    df["age_mid"] = df["age_group"].map(AGE_MIDPOINTS)
    return df


def overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    tbl = (
        df.groupby("sex")["chip_time_min"]
        .describe()
        .round(2)
    )
    return tbl


def by_age_group(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["sex", "age_group"])
        .agg(
            n=("chip_time_min", "size"),
            mean_min=("chip_time_min", "mean"),
            median_min=("chip_time_min", "median"),
            std_min=("chip_time_min", "std"),
            fastest_min=("chip_time_min", "min"),
        )
        .round(2)
        .reset_index()
    )


def top_cities(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return (
        df.groupby("city")
        .agg(
            n=("chip_time_min", "size"),
            mean_min=("chip_time_min", "mean"),
            median_min=("chip_time_min", "median"),
        )
        .sort_values("n", ascending=False)
        .head(n)
        .round(2)
        .reset_index()
    )


def correlation_block(df: pd.DataFrame) -> dict:
    """Pearson & Spearman correlations for the interesting pairs."""
    sub = df.dropna(subset=["age_mid", "chip_time_min", "start_delay_sec", "bib"])
    pairs = [
        ("age_mid", "chip_time_min"),
        ("bib", "chip_time_min"),
        ("start_delay_sec", "chip_time_min"),
        ("sex_place", "chip_time_min"),
    ]
    out = {}
    for a, b in pairs:
        pear = stats.pearsonr(sub[a], sub[b])
        spear = stats.spearmanr(sub[a], sub[b])
        out[f"{a} vs {b}"] = {
            "pearson_r": round(float(pear.statistic), 4),
            "pearson_p": float(pear.pvalue),
            "spearman_rho": round(float(spear.statistic), 4),
            "spearman_p": float(spear.pvalue),
            "n": len(sub),
        }
    return out


def sex_gap_by_age(df: pd.DataFrame) -> pd.DataFrame:
    """How does the M-F median time gap evolve across age groups?"""
    pivot = (
        df[df["sex"].isin(["M", "F"])]
        .groupby(["age_group", "sex"])["chip_time_min"]
        .median()
        .unstack()
    )
    pivot["gap_min"] = pivot["F"] - pivot["M"]
    pivot["gap_pct"] = 100 * pivot["gap_min"] / pivot["M"]
    return pivot.round(2).reset_index()


def country_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("country")
        .agg(
            n=("chip_time_min", "size"),
            median_min=("chip_time_min", "median"),
            fastest_min=("chip_time_min", "min"),
        )
        .sort_values("n", ascending=False)
        .round(2)
        .reset_index()
    )


def start_delay_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket runners by start-delay (seconds between gun and chip start)
    and look at their median chip time. If bigger delay correlates with
    slower chip time, it supports the 'corralling effect' causal story
    (slower runners start further back and also get held up by crowding).
    """
    sub = df.dropna(subset=["start_delay_sec", "chip_time_min"]).copy()
    sub["delay_bucket"] = pd.cut(
        sub["start_delay_sec"],
        bins=[-1, 30, 120, 300, 600, 1200, 4000],
        labels=["0-30s", "30s-2m", "2-5m", "5-10m", "10-20m", "20m+"],
    )
    return (
        sub.groupby("delay_bucket", observed=True)
        .agg(
            n=("chip_time_min", "size"),
            median_chip_min=("chip_time_min", "median"),
            median_pace_min=("km_pace_min", "median"),
        )
        .round(2)
        .reset_index()
    )


def fade_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compare average pace by start-delay bucket but within-group
    (same age group, same sex), to isolate the 'crowding' component."""
    sub = df.dropna(subset=["start_delay_sec", "km_pace_min", "age_group"]).copy()
    sub["delay_bucket"] = pd.cut(
        sub["start_delay_sec"],
        bins=[-1, 60, 300, 900, 4000],
        labels=["<1m", "1-5m", "5-15m", "15m+"],
    )
    return (
        sub.groupby(["age_group", "sex", "delay_bucket"], observed=True)["km_pace_min"]
        .median()
        .unstack()
        .round(2)
        .reset_index()
    )


def write_report(df: pd.DataFrame) -> None:
    lines: list[str] = []
    P = lines.append

    P("# Vancouver Sun Run 2026 — Automated Findings\n")
    P(f"Total finishers parsed: {len(df):,}")
    P(f"Men: {(df['sex']=='M').sum():,}   Women: {(df['sex']=='F').sum():,}   "
      f"Non-binary: {(df['sex']=='X').sum():,}")
    P(f"Distinct cities: {df['city'].nunique():,}   Countries: {df['country'].nunique()}\n")

    P("## Overall chip-time (minutes) by sex")
    P(overall_summary(df).to_string())
    P("")

    P("## Median time by age group")
    P(by_age_group(df).to_string(index=False))
    P("")

    P("## Top 20 cities by participation")
    P(top_cities(df).to_string(index=False))
    P("")

    P("## Country summary (top 15)")
    P(country_summary(df).head(15).to_string(index=False))
    P("")

    P("## Correlations")
    corrs = correlation_block(df)
    for k, v in corrs.items():
        P(f"  {k}: pearson r = {v['pearson_r']:+.3f}, "
          f"spearman rho = {v['spearman_rho']:+.3f}  (n={v['n']:,})")
    P("")

    P("## Male vs Female median gap by age group")
    P(sex_gap_by_age(df).to_string(index=False))
    P("")

    P("## Start-delay effect (proxy for corral / crowding)")
    P(start_delay_effect(df).to_string(index=False))
    P("")

    P("## Within-group pace by start delay (crowding isolation)")
    P(fade_analysis(df).to_string(index=False))
    P("")

    (REPORTS / "findings.txt").write_text("\n".join(lines))

    # Also write structured CSVs for reuse.
    by_age_group(df).to_csv(REPORTS / "by_age_group.csv", index=False)
    top_cities(df, 50).to_csv(REPORTS / "top_cities.csv", index=False)
    sex_gap_by_age(df).to_csv(REPORTS / "sex_gap_by_age.csv", index=False)
    start_delay_effect(df).to_csv(REPORTS / "start_delay_effect.csv", index=False)
    country_summary(df).to_csv(REPORTS / "country_summary.csv", index=False)

    pd.DataFrame(correlation_block(df)).T.to_csv(REPORTS / "correlations.csv")


def main() -> None:
    df = load()
    write_report(df)
    print(f"Wrote analysis reports to {REPORTS}/")


if __name__ == "__main__":
    main()
