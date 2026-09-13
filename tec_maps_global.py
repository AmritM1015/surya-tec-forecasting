"""
tec_maps_global.py — Fetch and plot global TEC maps from CEDAR Madrigal

A TEC MAP is a 2-D snapshot of the entire Earth's ionosphere at a single
5-minute timestamp: one TEC value per 1°×1° grid cell, covering all
latitudes (−90° to +90°) and all longitudes (−180° to +180°).

This contrasts with a TEC TIME SERIES (tec_timeseries_global.py), which computes
the global median TEC across all cells at each timestep:
  - Time series: collapse space → one median value per timestep
  - Map:         collapse time  → one TEC value per cell at one timestep

Both products come from the same Madrigal binned file (instrument 8000,
gpsYYMMDDg.hdf5). The difference is only in what you hold fixed:
  - Time series: vary time, summarise space   → 1 row per timestep
  - Map:         fix time,  vary (lat, lon)   → 1 row per grid cell

No spatial filter is applied — the full global grid is fetched and plotted.

8 sample maps are extracted at 3-hour intervals across the query day,
capturing the full diurnal cycle: midnight, pre-dawn, dawn, morning,
noon, afternoon, evening, and night.

Requirements:
    pip install madrigalWeb pandas matplotlib numpy

Usage:
    python3 tec_maps_global.py

Outputs:
    tec_maps_global_data.csv    — full global dataset (all cells, all timesteps)
    tec_maps_global_8panel.png  — 8-panel figure, one global map per sample time
    tec_map_global_HHMM.png     — individual map PNG for each of the 8 samples
"""

import datetime
import sys

# ── user info required by Madrigal Rules-of-the-Road ──────────────────────────
USER_NAME        = "Amrit Madabushi"
USER_EMAIL       = "am2699@njit.edu"
USER_AFFILIATION = "NJIT"

# ── query parameters ───────────────────────────────────────────────────────────
MADRIGAL_URL  = "https://cedar.openmadrigal.org"
INST_CODE     = 8000
START_DATE    = datetime.datetime(2023, 1, 1)
END_DATE      = datetime.datetime(2023, 1, 1)

# No spatial filter — fetch the entire globe
FILTER_STR    = ""

PARAMETERS    = "gdlat,glon,tec"
FILE_FILTER   = "binned"

# ── 8 sample map times (UTC hours on START_DATE) ──────────────────────────────
SAMPLE_HOURS  = [0, 3, 6, 9, 12, 15, 18, 21]
SAMPLE_LABELS = [
    "00:00 UTC\n(midnight)",
    "03:00 UTC\n(pre-dawn)",
    "06:00 UTC\n(dawn)",
    "09:00 UTC\n(morning)",
    "12:00 UTC\n(noon)",
    "15:00 UTC\n(afternoon)",
    "18:00 UTC\n(evening)",
    "21:00 UTC\n(night)",
]

OUTPUT_CSV    = "tec_maps_global_data.csv"
OUTPUT_8PANEL = "tec_maps_global_8panel.png"

# ──────────────────────────────────────────────────────────────────────────────


def check_deps():
    missing = []
    for pkg in ("madrigalWeb", "pandas", "matplotlib", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with:  pip install {' '.join(missing)}")
        sys.exit(1)


def parse_isprint_text(text, parm_names):
    """Parse whitespace-delimited ASCII returned by isprintService."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("year"):
            continue
        parts = line.split()
        if len(parts) < len(parm_names):
            continue
        rows.append(parts[:len(parm_names)])
    return rows


def nearest_timestamp(data, target_hour):
    """Return the timestamp in data closest to target_hour UTC on START_DATE."""
    target = datetime.datetime(
        START_DATE.year, START_DATE.month, START_DATE.day, target_hour, 2, 30
    )
    unique_times = data["datetime"].dropna().unique()
    if len(unique_times) == 0:
        return None
    deltas = [abs((t - target).total_seconds()) for t in unique_times]
    return unique_times[deltas.index(min(deltas))]


def plot_map(ax, map_df, title):
    """Plot one global TEC map on the given axes."""
    import numpy as np

    lats = sorted(map_df["gdlat"].unique())
    lons = sorted(map_df["glon"].unique())
    tec_grid = map_df.pivot(index="gdlat", columns="glon", values="tec")

    # pcolormesh needs cell edges, not centers — shift by 0.5°
    lat_edges = [l - 0.5 for l in lats] + [lats[-1] + 0.5]
    lon_edges = [l - 0.5 for l in lons] + [lons[-1] + 0.5]
    LON, LAT = np.meshgrid(lon_edges, lat_edges)

    tec_arr = tec_grid.reindex(index=lats, columns=lons).values

    im = ax.pcolormesh(LON, LAT, tec_arr,
                       cmap="viridis", vmin=0, vmax=60,
                       shading="flat")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    # Draw equator and prime meridian for reference
    ax.axhline(0, color="white", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="white", linewidth=0.4, alpha=0.5)
    return im


def main():
    check_deps()

    import madrigalWeb.madrigalWeb as mw
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    print(f"Connecting to {MADRIGAL_URL} ...")
    access = mw.MadrigalData(MADRIGAL_URL)

    # ── 1. find experiments ────────────────────────────────────────────────────
    print(f"Searching instrument {INST_CODE} "
          f"{START_DATE.date()} (CEDAR only) ...")

    experiments = access.getExperiments(
        INST_CODE,
        START_DATE.year, START_DATE.month, START_DATE.day, 0, 0, 0,
        END_DATE.year,   END_DATE.month,   END_DATE.day,   23, 59, 59,
        local=1,
    )

    if not experiments:
        print("No experiments found.")
        return

    print(f"Found {len(experiments)} experiment(s).")

    # ── 2. collect binned files ────────────────────────────────────────────────
    chosen_files = []
    for exp in experiments:
        for f in access.getExperimentFiles(exp.id):
            if FILE_FILTER.lower() in f.kindatdesc.lower():
                chosen_files.append(f)

    if not chosen_files:
        print("No binned files found.")
        return

    print(f"\nQuerying {len(chosen_files)} file(s) [{FILE_FILTER}] — NO spatial filter (global):")
    for f in chosen_files:
        print(f"  {f.name.split('/')[-1]}  [{f.kindatdesc}]")

    # ── 3. fetch — no spatial filter ──────────────────────────────────────────
    all_cols = ["year", "month", "day", "hour", "min", "sec"] + PARAMETERS.split(",")
    parm_str = ",".join(all_cols)

    frames = []
    for f in chosen_files:
        fname = f.name.split("/")[-1]
        print(f"\nFetching {fname} (global — this may take several minutes) ...")
        try:
            result = access.isprint(
                f.name, parm_str, FILTER_STR,
                USER_NAME, USER_EMAIL, USER_AFFILIATION,
            )
            rows = parse_isprint_text(result, all_cols)
            if rows:
                df = pd.DataFrame(rows, columns=all_cols)
                frames.append(df)
                print(f"  -> {len(df)} records")
            else:
                print("  -> empty result")
        except Exception as e:
            print(f"  -> error: {e}")

    if not frames:
        print("\nNo data retrieved.")
        return

    # ── 4. assemble and clean ──────────────────────────────────────────────────
    data = pd.concat(frames, ignore_index=True)

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    def to_dt(row):
        try:
            return datetime.datetime(
                int(row["year"]), int(row["month"]), int(row["day"]),
                int(row["hour"]), int(row["min"]),   int(row["sec"]),
            )
        except Exception:
            return None

    data["datetime"] = data.apply(to_dt, axis=1)
    data = data.dropna(subset=["datetime"]).sort_values(["datetime", "gdlat", "glon"]).reset_index(drop=True)
    data.loc[data["tec"] > 1e20, "tec"] = float("nan")

    valid = data["tec"].dropna()
    print(f"\nTotal records     : {len(data)}")
    print(f"Time range        : {data['datetime'].min()}  ->  {data['datetime'].max()}")
    print(f"Unique lat cells  : {data['gdlat'].nunique()}")
    print(f"Unique lon cells  : {data['glon'].nunique()}")
    if len(valid):
        print(f"TEC (TECU)        : min={valid.min():.2f}  "
              f"median={valid.median():.2f}  max={valid.max():.2f}")

    # ── 5. save CSV ────────────────────────────────────────────────────────────
    data.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")

    # ── 6. select 8 sample timestamps ─────────────────────────────────────────
    print("\nSelecting 8 sample map times (3-hour intervals) ...")
    sample_times = []
    for h in SAMPLE_HOURS:
        t = nearest_timestamp(data, h)
        if t is not None:
            sample_times.append(t)
            print(f"  {h:02d}:00 UTC target -> nearest record: {t.strftime('%H:%M:%S')}")

    if not sample_times:
        print("Could not find sample times.")
        return

    # ── 7. 8-panel figure ─────────────────────────────────────────────────────
    print("\nPlotting 8-panel global TEC map figure ...")
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(
        f"Global TEC Maps — Full Earth  |  {START_DATE.strftime('%Y-%m-%d')}\n"
        f"Lat −90°–+90°, Lon −180°–+180°  |  colormap: viridis, 0–60 TECU  |  "
        f"white lines: equator and prime meridian",
        fontsize=11, y=0.99
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.15)

    last_im = None
    for idx, (t, label) in enumerate(zip(sample_times, SAMPLE_LABELS)):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        map_df = data[data["datetime"] == t].copy()
        if map_df.empty:
            ax.set_title(f"{label}\n(no data)", fontsize=9)
            continue
        im = plot_map(ax, map_df, label)
        last_im = im

        if idx % 4 == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=7)
        if idx >= 4:
            ax.set_xlabel("Longitude (°E)", fontsize=7)

        # Save individual map
        ind_fig, ind_ax = plt.subplots(figsize=(10, 5))
        plot_map(ind_ax, map_df,
                 f"Global TEC  {t.strftime('%Y-%m-%d %H:%M')} UTC")
        ind_ax.set_xlabel("Longitude (°E)", fontsize=9)
        ind_ax.set_ylabel("Latitude (°N)", fontsize=9)
        ind_fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0, 60), cmap="viridis"),
            ax=ind_ax, label="TEC (TECU)", shrink=0.85
        )
        fname_ind = f"tec_map_global_{t.strftime('%H%M')}.png"
        ind_fig.tight_layout()
        ind_fig.savefig(fname_ind, dpi=150)
        plt.close(ind_fig)
        print(f"  Saved -> {fname_ind}")

    # Shared colorbar for 8-panel figure
    if last_im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 60), cmap="viridis")
        fig.colorbar(sm, cax=cbar_ax, label="TEC (TECU)")

    fig.savefig(OUTPUT_8PANEL, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {OUTPUT_8PANEL}")
    # Open with: xdg-open tec_maps_global_8panel.png


if __name__ == "__main__":
    main()
