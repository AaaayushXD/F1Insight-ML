"""
Create minimal results.csv in raw_dataset from qualifying for demo/training when API collection is not possible.
Uses qualifying position as proxy for finish position (for pipeline validation only).
Run once, then run clean_data and train.
"""

from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw_dataset"
CLEAN = Path(__file__).resolve().parent.parent / "data" / "cleaned_dataset"


def main():
    qual_path = RAW / "qualifying.csv"
    if not qual_path.exists():
        print("qualifying.csv not found in raw_dataset. Run data collection first.")
        return
    df = pd.read_csv(qual_path)
    driver_col = "Driver_driverId" if "Driver_driverId" in df.columns else "driverId"
    constructor_col = "Constructor_constructorId" if "Constructor_constructorId" in df.columns else "constructorId"
    pos_col = "position"
    if pos_col not in df.columns:
        print("No position column in qualifying.")
        return
    results = pd.DataFrame({
        "season": df["season"],
        "round": df["round"],
        driver_col: df[driver_col],
        constructor_col: df[constructor_col],
        "grid": df[pos_col],
        "position": df[pos_col],
        "points": 0,
        "laps": 50,
        "Status_statusId": "1",
    })
    out_path = RAW / "results.csv"
    results.to_csv(out_path, index=False)
    print(f"Wrote {len(results)} rows to {out_path} (demo: finish_position = qualifying position).")
    print("Run: python -m app.scripts.clean_data && python -m app.ml.train")


if __name__ == "__main__":
    main()
