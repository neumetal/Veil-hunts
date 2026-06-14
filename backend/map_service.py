"""
map_service.py
Generates static PNG heatmaps and timeline charts for the Fog Probability Grid.

Color scale:
    Score  0  ->  White  (Clear)
    Score  5  ->  Yellow (Moderate)
    Score 10  ->  Cyan   (Fog)
    Score 12  ->  Aqua   (Confirmed fog via weather code)

CLI usage:
    python map_service.py --csv scans_export/fog_data_xxx.csv --timestamp "2025-11-15 05:00"
    python map_service.py --csv scans_export/fog_data_xxx.csv --timestamp "2025-11-15 05:00" --output custom_name.png
    python map_service.py --csv scans_export/fog_data_xxx.csv --timeline
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

# -- Constants ----------------------------------------------------------------
EXPORT_DIR = r"C:\veil_finder_project\scans_export"
CENTER_LAT  = 40.31
CENTER_LON  = -75.13
DARK_BG     = "#0d1117"
PANEL_BG    = "#161b22"

# Custom colormap: White -> Yellow -> Cyan -> Aquamarine
FOG_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "fog_scale",
    [
        (0.00, "#ffffff"),   # 0  — Clear
        (0.42, "#ffef00"),   # 5  — Moderate
        (0.83, "#00e5ff"),   # 10 — Fog
        (1.00, "#00ffc8"),   # 12 — Confirmed fog
    ],
    N=256,
)
SCORE_VMIN, SCORE_VMAX = 0, 12


# -----------------------------------------------------------------------------
# Heatmap
# -----------------------------------------------------------------------------
def generate_heatmap(
    df: pd.DataFrame,
    timestamp,
    output_path: str | None = None,
) -> str | None:
    """
    Generate a static PNG heatmap for a specific timestamp.

    Args:
        df          : Master DataFrame with [Timestamp, Lat, Lon, Fog_Score, IsValley, ...]
        timestamp   : str or pd.Timestamp to filter on
        output_path : Override the save path (otherwise auto-named in EXPORT_DIR)

    Returns:
        Absolute path of the saved PNG, or None if the timestamp has no data.
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)

    ts    = pd.Timestamp(timestamp)
    frame = df[df["Timestamp"] == ts].copy()

    if frame.empty:
        print(f"  WARNING: No data for timestamp {ts} — heatmap skipped.")
        return None

    # -- Figure setup ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 10.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    # -- Scatter plot ----------------------------------------------------------
    sc = ax.scatter(
        frame["Lon"],
        frame["Lat"],
        c=frame["Fog_Score"],
        cmap=FOG_CMAP,
        vmin=SCORE_VMIN,
        vmax=SCORE_VMAX,
        s=170,
        alpha=0.92,
        edgecolors="none",
        zorder=3,
    )

    # -- Valley point outlines -------------------------------------------------
    valleys = frame[frame["IsValley"] == True]
    if not valleys.empty:
        ax.scatter(
            valleys["Lon"],
            valleys["Lat"],
            marker="v",
            s=60,
            facecolors="none",
            edgecolors="white",
            linewidths=0.6,
            alpha=0.45,
            zorder=4,
        )

    # -- Center marker (Doylestown) --------------------------------------------
    ax.plot(CENTER_LON, CENTER_LAT, "w*", markersize=16, zorder=6)
    ax.plot(CENTER_LON, CENTER_LAT, "wo", markersize=22, zorder=5, alpha=0.2)

    # -- Colorbar --------------------------------------------------------------
    cbar = plt.colorbar(sc, ax=ax, fraction=0.028, pad=0.015, aspect=30)
    cbar.set_label("Fog Probability Score", color="white", fontsize=11, labelpad=12)
    cbar.set_ticks([0, 5, 10, 12])
    cbar.set_ticklabels(["0  Clear", "5  Moderate", "10  Fog", "12  Confirmed"])
    cbar.ax.yaxis.set_tick_params(color="white", length=4)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9.5)
    cbar.outline.set_edgecolor("#444")

    # -- Legend ----------------------------------------------------------------
    legend_elements = [
        Patch(facecolor="#ffffff", edgecolor="#999", label="Score 0 — Clear"),
        Patch(facecolor="#ffef00", edgecolor="none", label="Score 5 — Moderate"),
        Patch(facecolor="#00e5ff", edgecolor="none", label="Score 10 — Fog"),
        Patch(facecolor="#00ffc8", edgecolor="none", label="Score 12 — Confirmed Fog"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="none",
               markersize=9, linewidth=0, label="Valley Point"),
        Line2D([0], [0], marker="*", color="w", markersize=12,
               linewidth=0, label="Doylestown, PA"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        facecolor=PANEL_BG,
        labelcolor="white",
        edgecolor="#444",
        fontsize=9.5,
        framealpha=0.9,
    )

    # -- Grid & axes styling ---------------------------------------------------
    ax.grid(True, color="#2a2a2a", linewidth=0.6, alpha=0.8)
    ax.tick_params(colors="#aaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2fdeg"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2fdeg"))

    # -- Titles & annotations --------------------------------------------------
    ts_str     = ts.strftime("%A, %B %-d %Y  .  %H:%M ET")
    avg_score  = frame["Fog_Score"].mean()
    fog_pct    = 100 * (frame["Fog_Score"] >= 10).sum() / len(frame)
    valley_n   = int(frame["IsValley"].sum())

    ax.set_title(
        "Valley-Adjusted Fog Probability Grid",
        color="white", fontsize=16, fontweight="bold", pad=18, loc="left",
    )
    ax.set_title(
        ts_str,
        color="#aaa", fontsize=11, pad=18, loc="right",
    )
    ax.set_xlabel("Longitude", color="#aaa", fontsize=10, labelpad=8)
    ax.set_ylabel("Latitude",  color="#aaa", fontsize=10, labelpad=8)

    fig.text(
        0.5, 0.01,
        f"Grid avg score: {avg_score:.1f}   .   Fog coverage: {fog_pct:.0f}%   .   Valley pts: {valley_n}   .   Center: Doylestown PA",
        ha="center", color="#555", fontsize=8.5,
    )

    plt.tight_layout(rect=[0, 0.025, 1, 1])

    # -- Save -----------------------------------------------------------------
    if output_path is None:
        ts_tag      = ts.strftime("%Y%m%d_%H%M")
        output_path = os.path.join(EXPORT_DIR, f"fog_heatmap_{ts_tag}.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Heatmap -> {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# Timeline chart
# -----------------------------------------------------------------------------
def generate_daily_summary_chart(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> str:
    """
    Generate a time-series area chart of grid-average Fog Score over the full
    date range. Saves as PNG and returns the output path.
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)

    hourly = (
        df.groupby("Timestamp")["Fog_Score"]
        .mean()
        .reset_index(name="Avg_Fog_Score")
        .sort_values("Timestamp")
    )

    fig, ax = plt.subplots(figsize=(17, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    ax.fill_between(hourly["Timestamp"], hourly["Avg_Fog_Score"], alpha=0.25, color="#00e5ff")
    ax.plot(hourly["Timestamp"], hourly["Avg_Fog_Score"], color="#00e5ff", linewidth=1.8)

    ax.axhline(5,  color="#ffef00", linestyle="--", linewidth=1.0, alpha=0.55, label="Moderate threshold (5)")
    ax.axhline(10, color="#00e5ff", linestyle="--", linewidth=1.0, alpha=0.55, label="Fog threshold (10)")

    ax.set_ylim(bottom=0)
    ax.set_title("Grid-Average Fog Probability Score — Full Date Range",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Timestamp (ET)", color="#aaa", fontsize=10)
    ax.set_ylabel("Avg Fog Score",  color="#aaa", fontsize=10)
    ax.tick_params(colors="#aaa", labelsize=8.5)
    ax.legend(facecolor=PANEL_BG, labelcolor="white", edgecolor="#444", fontsize=9)
    ax.grid(True, color="#2a2a2a", linewidth=0.6, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(EXPORT_DIR, "fog_score_timeline.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Timeline chart -> {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fog heatmaps / timeline charts from a master CSV."
    )
    parser.add_argument("--csv",       required=True, help="Path to master fog CSV")
    parser.add_argument("--timestamp", help='Heatmap timestamp, e.g. "2025-11-15 05:00"')
    parser.add_argument("--timeline",  action="store_true", help="Generate timeline chart instead of heatmap")
    parser.add_argument("--output",    help="Override output PNG path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["Timestamp"])

    if args.timeline:
        generate_daily_summary_chart(df, output_path=args.output)
    elif args.timestamp:
        generate_heatmap(df, args.timestamp, output_path=args.output)
    else:
        parser.error("Provide --timestamp for a heatmap or --timeline for the summary chart.")