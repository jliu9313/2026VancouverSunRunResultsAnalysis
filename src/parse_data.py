"""
parse_data.py
-------------
Parses the messy fixed-width text dump from the Vancouver Sun Run 2026
results (sportstats.one format) into a clean pandas DataFrame and
saves it as a CSV for downstream analysis.

The raw file (`data/raw/SunRun2026Data.txt`) contains:
  * repeating page headers ("Vancouver Sun Run", column names, rules)
  * blank lines / page breaks
  * rows with an optional Final Place
  * rows with an optional KM Pace
  * rows where the country has no province prefix (international runners)

We use a robust regex (anchored on the "Place in Sex" X/Y column and the
age-group category) to pull out each field.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "SunRun2026Data.txt"
CLEAN_DIR = Path(__file__).resolve().parents[1] / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_PATH = CLEAN_DIR / "sunrun2026_clean.csv"

# ---------------------------------------------------------------------------
# Regex: every valid result row contains "X/Y  A/B  CATEGORY" near the end,
# where CATEGORY looks like "M30-34", "F25-29", "M80+", etc.
# ---------------------------------------------------------------------------
ROW_RE = re.compile(
    r"""
    ^\s*
    (?:(?P<final_place>\d+)\s+)?                         # optional Final Place
    (?:(?P<gun_time>\d{1,2}:\d{2}(?::\d{2})?)\s+)?        # optional Gun Time
    (?P<chip_time>\d{1,2}:\d{2}(?::\d{2})?)\s+            # Chip Time (always present)
    (?:(?P<km_pace>\d{1,2}:\d{2})\s+)?                    # optional KM Pace
    (?P<bib>\d+)\s+                                       # Bib No.
    (?P<rest>.+?)                                         # Name + City + Prov/CTRY
    \s+(?P<sex_place>\d+)/(?P<sex_total>\d+)\s+           # Place in Sex
    (?P<cat_place>\d+)/(?P<cat_total>\d+)\s+              # Place in Category
    (?P<category>                                         # Age group / category
        Non-Binary
        | [MF]\d{1,2}(?:-\d{1,2})?\+?                     # M30-34, F25-29, M80+, M15- (truncated)
        | [MF]\d{1,2}-                                    # truncated trailing dash
    )
    \s*$
    """,
    re.VERBOSE,
)

PROV_RE = re.compile(r"^(?P<city>.+?)\s{2,}(?P<prov>[A-Z]{2})?\s*(?P<ctry>[A-Z]{3})\s*$")


def _time_to_seconds(t: str | None) -> float | None:
    """Convert 'MM:SS' or 'H:MM:SS' to seconds (float). None -> None."""
    if not t:
        return None
    parts = t.split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + int(s)


def _split_name_city_country(rest: str) -> tuple[str, str, str | None, str]:
    """Split the 'rest' chunk into (name, city, province, country).

    `rest` is everything between the bib number and the sex-place column,
    e.g. 'Justin KENT              Burnaby            BC CAN'
    """
    rest = rest.rstrip()

    # Strip trailing country (3 letters) and optional province (2 letters)
    # by walking backwards.
    tokens = rest.rsplit(None, 2)
    country = None
    province = None
    if len(tokens) >= 2 and re.fullmatch(r"[A-Z]{3}", tokens[-1]):
        country = tokens[-1]
        if re.fullmatch(r"[A-Z]{2}", tokens[-2]):
            province = tokens[-2]
            remainder = " ".join(tokens[:-2])
        else:
            remainder = " ".join(tokens[:-1])
    else:
        remainder = rest

    # Now `remainder` is 'Name                 City'. The separator is
    # typically 2+ spaces. If no such separator, put everything in name.
    m = re.match(r"^(?P<name>.+?)\s{2,}(?P<city>.+?)\s*$", remainder)
    if m:
        name = m.group("name").strip()
        city = m.group("city").strip()
    else:
        name = remainder.strip()
        city = ""

    return name, city, province, country


def parse_file(path: Path = RAW_PATH) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            m = ROW_RE.match(line)
            if not m:
                continue

            name, city, province, country = _split_name_city_country(m.group("rest"))

            rows.append(
                {
                    "final_place": int(m.group("final_place")) if m.group("final_place") else None,
                    "gun_time_sec": _time_to_seconds(m.group("gun_time")),
                    "chip_time_sec": _time_to_seconds(m.group("chip_time")),
                    "km_pace_sec": _time_to_seconds(m.group("km_pace")),
                    "bib": int(m.group("bib")),
                    "name": name,
                    "city": city,
                    "province": province,
                    "country": country,
                    "sex_place": int(m.group("sex_place")),
                    "sex_total": int(m.group("sex_total")),
                    "cat_place": int(m.group("cat_place")),
                    "cat_total": int(m.group("cat_total")),
                    "category": m.group("category"),
                }
            )

    df = pd.DataFrame(rows)

    # Normalise truncated age groups (e.g. 'M15-' -> 'M15-19', 'F08' -> 'F08-14').
    # These appear where the tool that generated the PDF/txt truncated the
    # long category string. We use best-guess canonical names.
    TRUNC_MAP = {
        "M15-": "M15-19", "F15-": "F15-19",
        "M15": "M15-19", "F15": "F15-19", "F08": "F00-14",
        "M80-": "M80+", "F80-": "F80+",
        "F95": "F95+", "M3": "M3+",  # rare edge cases
    }
    df["category"] = df["category"].replace(TRUNC_MAP)

    # Derive useful columns
    df["sex"] = df["category"].apply(
        lambda c: "X" if c == "Non-Binary" else c[0]
    )
    df["age_group"] = df["category"].apply(
        lambda c: c if c == "Non-Binary" else c[1:]
    )
    df["chip_time_min"] = df["chip_time_sec"] / 60
    df["gun_time_min"] = df["gun_time_sec"] / 60
    # If KM pace is missing, derive from chip time (10 km race).
    df["km_pace_sec"] = df["km_pace_sec"].fillna(df["chip_time_sec"] / 10)
    df["km_pace_min"] = df["km_pace_sec"] / 60
    df["start_delay_sec"] = df["gun_time_sec"] - df["chip_time_sec"]

    return df


def main() -> None:
    df = parse_file()
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Parsed {len(df):,} rows -> {CLEAN_PATH}")
    print(df.head())
    print("\nColumn dtypes:")
    print(df.dtypes)


if __name__ == "__main__":
    main()
