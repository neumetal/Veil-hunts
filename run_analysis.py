"""
run_analysis.py
Top-level CLI runner for the Valley-Adjusted Fog Probability Grid.

MODES
-----
Full pipeline (data fetch + scoring + outputs):
    python run_analysis.py --start 2025-11-14 --end 2025-11-16

Skip heatmap generation (CSV only):
    python run_analysis.py --start 2025-11-14 --end 2025-11-16 --no-heatmaps

Heatmap from an existing CSV (no data fetch):
    python run_analysis.py --csv scans_export/fog_data_20251114_20251116.csv --timestamp "2025-11-15 05:00"

Timeline chart from an existing CSV:
    python run_analysis.py --csv scans_export/fog_data_xxx.csv --timeline

No arguments -> defaults to the previous 7 days:
    python run_analysis.py
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

# -- Resolve backend package ---------------------------------------------------
_BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, _BACKEND)

import pandas as pd
from grid_utils   import generate_grid, fetch_elevations, classify_valley_points
from api_client   import fetch_all_weather
from veil_logic   import build_master_dataframe, print_verification_stats
from map_service  import generate_heatmap, generate_daily_summary_chart

EXPORT_DIR = r"C:\veil_finder_project\scans_export"


# -----------------------------------------------------------------------------
# Full analysis pipeline
# -----------------------------------------------------------------------------
def run_full_analysis(
    start_date: str,
    end_date: str,
    generate_maps: bool = True,
) -> tuple[pd.DataFrame, str]:
    """
    Execute the complete fog probability analysis pipeline:
      1. Generate spatial grid
      2. Fetch elevation data
      3. Classify valley points
      4. Fetch hourly weather data from Open-Meteo
      5. Score every grid point / hour
      6. Save master CSV
      7. Print verification stats
      8. Generate heatmaps for top fog hours + timeline chart

    Returns (master_df, csv_path).
    """
    print()
    print("+======================================================+")
    print("|   VALLEY-ADJUSTED FOG PROBABILITY GRID  v2.0        |")
    print("|   Center: Doylestown, PA  (40.31degN, 75.13degW)        |")
    print(f"|   Date range : {start_date}  ->  {end_date:<26}|")
    print("+======================================================+")

    # -- Step 1: Grid ---------------------------------------------------------
    print("\n[1/5]  Generating spatial grid ...")
    points = generate_grid()
    print(f"       {len(points)} grid points at 0.05deg resolution within 40-mile radius.")

    # -- Step 2: Elevation -----------------------------------------------------
    print("\n[2/5]  Fetching elevation data ...")
    elevations = fetch_elevations(points)

    # -- Step 3: Valley classification -----------------------------------------
    print("[3/5]  Classifying valley points ...")
    valley_cls, mean_elev, std_elev, threshold = classify_valley_points(elevations)

    # -- Step 4: Weather data --------------------------------------------------
    print("[4/5]  Fetching hourly weather data ...")
    weather_data = fetch_all_weather(points, start_date, end_date)

    if not weather_data:
        print("\n  ERROR: No weather data was retrieved. Exiting.")
        sys.exit(1)

    # -- Step 5: Score ---------------------------------------------------------
    print("[5/5]  Calculating fog scores and building master DataFrame ...")
    master_df = build_master_dataframe(weather_data, elevations, valley_cls)

    # -- Save CSV --------------------------------------------------------------
    os.makedirs(EXPORT_DIR, exist_ok=True)
    csv_name = f"fog_data_{start_date.replace('-','')}_{end_date.replace('-','')}.csv"
    csv_path = os.path.join(EXPORT_DIR, csv_name)
    master_df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Master CSV saved -> {csv_path}")
    print(f"    Rows: {len(master_df):,}   Columns: {master_df.shape[1]}")

    # -- Verification stats + peak hours ---------------------------------------
    peak_timestamps = print_verification_stats(master_df)

    # -- Outputs ---------------------------------------------------------------
    if generate_maps:
        print("  Generating heatmaps for top 3 fog hours ...")
        for ts in peak_timestamps[:3]:
            generate_heatmap(master_df, ts)

        print("  Generating timeline summary chart ...")
        generate_daily_summary_chart(master_df)

    print(f"\n  ✓ All outputs saved to: {EXPORT_DIR}")
    print()
    return master_df, csv_path


# -----------------------------------------------------------------------------
# CSV-only modes
# -----------------------------------------------------------------------------
def run_from_csv_heatmap(csv_path: str, timestamp: str, output: str | None = None):
    """Load existing master CSV and generate a heatmap for one timestamp."""
    print(f"  Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    result = generate_heatmap(df, timestamp, output_path=output)
    if result:
        print(f"  ✓ Done: {result}")


def run_from_csv_timeline(csv_path: str, output: str | None = None):
    """Load existing master CSV and generate the full timeline chart."""
    print(f"  Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    result = generate_daily_summary_chart(df, output_path=output)
    print(f"  ✓ Done: {result}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Valley-Adjusted Fog Probability Grid — Doylestown, PA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Full pipeline args
    parser.add_argument("--start",       metavar="YYYY-MM-DD", help="Start date for data fetch")
    parser.add_argument("--end",         metavar="YYYY-MM-DD", help="End date for data fetch")
    parser.add_argument("--no-heatmaps", action="store_true",  help="Skip PNG generation (CSV only)")

    # CSV-only args
    parser.add_argument("--csv",       metavar="PATH",      help="Existing master CSV (skips data fetch)")
    parser.add_argument("--timestamp", metavar="TIMESTAMP", help='Timestamp to plot, e.g. "2025-11-15 05:00"')
    parser.add_argument("--timeline",  action="store_true", help="Generate timeline chart instead of heatmap")
    parser.add_argument("--output",    metavar="PATH",      help="Override output PNG filename")

    args = parser.parse_args()

    # -- Routing ---------------------------------------------------------------
    if args.csv:
        if not os.path.isfile(args.csv):
            parser.error(f"CSV file not found: {args.csv}")
        if args.timeline:
            run_from_csv_timeline(args.csv, args.output)
        elif args.timestamp:
            run_from_csv_heatmap(args.csv, args.timestamp, args.output)
        else:
            parser.error("With --csv, also provide --timestamp or --timeline.")

    else:
        # Full pipeline — default to last 7 days if no dates given
        end   = args.end   or datetime.now().strftime("%Y-%m-%d")
        start = args.start or (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        if not args.start or not args.end:
            print(f"  No dates specified. Defaulting to last 7 days: {start} -> {end}")

        run_full_analysis(start, end, generate_maps=not args.no_heatmaps)


if __name__ == "__main__":
    main()
