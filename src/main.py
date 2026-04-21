"""
main.py
-------
End-to-end pipeline: parse -> analyze -> visualize.
Run from the repo root:   python src/main.py
"""

from __future__ import annotations

from src import parse_data, analyze, visualize  # type: ignore


def main() -> None:
    print("[1/3] Parsing raw fixed-width results...")
    parse_data.main()
    print("[2/3] Computing statistics & correlations...")
    analyze.main()
    print("[3/3] Rendering charts...")
    visualize.main()
    print("Done.")


if __name__ == "__main__":
    main()
