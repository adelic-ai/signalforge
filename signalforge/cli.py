"""
signalforge.cli

Command-line interface for SignalForge.

Usage
-----
    python -m signalforge <command> [options]

Commands
--------
    demo            Run the built-in EEG seizure demo (no arguments needed)
    run             Run the pipeline on a CSV file
    plan            Show a SamplingPlan for a domain
    neighborhood    Show the p-adic arithmetic viewing box for an integer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
EXAMPLES = HERE.parent / "examples"


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

def cmd_demo(args: argparse.Namespace) -> int:
    """Run the built-in EEG seizure detection demo."""
    csv_path = DOCS / "eeg_chbmit" / "chb01_03_eeg_rms.csv"
    if not csv_path.exists():
        print(f"Demo data not found: {csv_path}")
        print("Run docs/eeg_chbmit/edf_to_helix_signal.py to generate it.")
        return 1

    import pandas as pd
    from .pipeline.canonical import CanonicalRecord, OrderType
    from .pipeline.binned import materialize
    from .pipeline.surface import measure
    from .pipeline.feature import engineer
    from .domains import eeg

    SFREQ = 256
    SEIZURE_START_S = 2996
    SEIZURE_END_S   = 3036

    print("=" * 60)
    print("SignalForge — EEG Seizure Detection Demo")
    print("=" * 60)
    print(f"Dataset  : CHB-MIT chb01_03  ({SFREQ} Hz, 1 hour, 23 channels)")
    print(f"Signal   : RMS across all channels → scalar time series")
    print(f"Seizure  : {SEIZURE_START_S}–{SEIZURE_END_S} s  (ground truth annotation)")
    print(f"Method   : unsupervised p-adic multiscale z-score")
    print(f"Training : none")
    print()

    df = pd.read_csv(csv_path)
    t_secs = df["t_sec"].to_numpy(dtype=np.float64)
    rms    = df["eeg_rms"].to_numpy(dtype=np.float64)
    sample_indices = np.round(t_secs * SFREQ).astype(np.int64)

    t0 = time.perf_counter()
    records = [
        CanonicalRecord(
            primary_order=int(si),
            order_type=OrderType.SEQUENCE,
            channel="eeg", metric="rms", value=float(v),
            seq_order=int(si),
        )
        for si, v in zip(sample_indices, rms)
    ]
    plan    = eeg.sampling_plan()
    binned  = materialize(records, plan, agg_funcs={"eeg": {"rms": "mean"}})
    surfaces = measure(binned, plan, profile="continuous")
    tensors  = [engineer(s, plan) for s in surfaces]
    elapsed = time.perf_counter() - t0

    ft = tensors[0]
    windows_s = tuple(w // SFREQ for w in plan.windows)
    time_axis = np.array(ft.time_axis)

    seiz_bin_start = (SEIZURE_START_S * SFREQ) // plan.cbin
    seiz_bin_end   = (SEIZURE_END_S   * SFREQ) // plan.cbin
    t_start = int(np.searchsorted(time_axis, seiz_bin_start))
    t_end   = int(np.searchsorted(time_axis, seiz_bin_end))

    zscore_mat = ft.values["mean_zscore"]

    print(f"{'Scale':>8}   {'z_max':>7}   {'z_mean':>7}   {'signal'}")
    print("-" * 50)
    for si, ws in enumerate(windows_s):
        row = zscore_mat[si, t_start:t_end]
        valid = row[np.isfinite(row)]
        if len(valid) == 0:
            continue
        zmax  = valid.max()
        zmean = valid.mean()
        bar   = "█" * min(int(abs(zmax)), 20) if zmax > 0 else "░" * min(int(abs(zmax)), 20)
        sign  = "+" if zmax > 0 else "-"
        print(f"{ws:>6}s   {zmax:>7.2f}   {zmean:>7.2f}   {sign}{bar}")

    peak_z  = zscore_mat[:, t_start:t_end]
    valid   = peak_z[np.isfinite(peak_z)]
    overall = valid.max() if len(valid) > 0 else float("nan")

    print()
    print(f"Peak z-score in seizure window : {overall:.2f} σ")
    print(f"Pipeline time                  : {elapsed:.1f} s")
    print(f"Surface shape                  : {surfaces[0].shape}  (scales × time bins)")
    print(f"Feature tensor shape           : {ft.shape}  (scales × time bins)")
    print(f"Features computed              : {len(ft.feature_names)}")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    """Run the pipeline on a CSV file."""
    domain = args.domain.lower()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return 1

    script = EXAMPLES / f"run_{domain}.py"
    if not script.exists():
        print(f"No run script for domain {domain!r}. Available: intermagnet, eeg, equities, timeseries")
        return 1

    import runpy
    import sys as _sys
    _sys.argv = [str(script), "--csv", str(csv_path)]
    if args.out:
        _sys.argv += ["--out", args.out]
    runpy.run_path(str(script), run_name="__main__")
    return 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> int:
    """Show a SamplingPlan for a domain."""
    domain = args.domain.lower()

    try:
        if domain == "intermagnet":
            from .domains import intermagnet
            plan = intermagnet.sampling_plan()
        elif domain == "intermagnet-yearly":
            from .domains import intermagnet
            plan = intermagnet.sampling_plan_yearly()
        elif domain == "eeg":
            from .domains import eeg
            plan = eeg.sampling_plan()
        elif domain == "equities":
            from .domains import equities
            plan = equities.sampling_plan()
        elif domain == "equities-daily":
            from .domains import equities
            plan = equities.sampling_plan_daily()
        elif domain == "timeseries":
            from .domains import timeseries
            plan = timeseries.sampling_plan()
        else:
            print(f"Unknown domain {domain!r}.")
            print("Available: intermagnet, intermagnet-yearly, eeg, equities, equities-daily, timeseries")
            return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    print(f"SamplingPlan  — {domain}")
    print(f"  horizon       : {plan.horizon:,} ({plan.horizon // plan.cbin} bins)")
    print(f"  grain         : {plan.grain}")
    print(f"  cbin          : {plan.cbin}")
    print(f"  prime_basis   : {plan.prime_basis}")
    print(f"  n_windows     : {len(plan.windows)}")
    print()
    print(f"  {'window':>12}   {'hop':>8}   {'n_values':>8}   coordinate")
    print(f"  {'-'*12}   {'-'*8}   {'-'*8}   ----------")
    for w, h, nv, coord in zip(plan.windows, plan.hops, plan.n_values, plan.coordinates):
        print(f"  {w:>12,}   {h:>8,}   {nv:>8,}   {coord}")
    return 0


# ---------------------------------------------------------------------------
# neighborhood
# ---------------------------------------------------------------------------

def cmd_neighborhood(args: argparse.Namespace) -> int:
    """Show the p-adic arithmetic viewing box."""
    from .lattice import neighborhood as nb_fn

    anchor = args.anchor
    radius = args.radius
    basis  = args.basis  # may be None

    try:
        nb = nb_fn(anchor, radius, basis=basis)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    print(f"Neighborhood(anchor={anchor}, radius={radius})")
    print(f"  integers : {nb.integers[0]} .. {nb.integers[-1]}  ({len(nb.integers)} values)")
    print(f"  primes   : {nb.prime_basis}")
    print()
    nb.show()
    return 0


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def _auto_ingest(csv_path: Path) -> list:
    """Auto-detect CSV format and ingest into CanonicalRecords."""
    import pandas as pd
    from .pipeline.canonical import CanonicalRecord, OrderType

    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    # Two-column CSV: treat as generic timeseries
    if len(cols) == 2:
        date_col, value_col = cols
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col])
        return [
            CanonicalRecord(
                primary_order=i,
                order_type=OrderType.SEQUENCE,
                channel="value",
                metric="value",
                value=float(row[value_col]),
                seq_order=i,
            )
            for i, row in df.iterrows()
        ]

    # Multi-column: look for known patterns
    lower = {c.lower(): c for c in cols}

    # timestamp + ticker + metric + value (equities format)
    if all(k in lower for k in ("timestamp", "ticker", "metric", "value")):
        from .domains import equities
        return equities.ingest(str(csv_path))

    # timestamp + station + component + value (intermagnet format)
    if all(k in lower for k in ("timestamp", "station", "component", "value")):
        from .domains import intermagnet
        return intermagnet.ingest(str(csv_path))

    # t_sec + eeg_rms (EEG format)
    if all(k in lower for k in ("t_sec", "eeg_rms")):
        from .domains import eeg
        return eeg.ingest(str(csv_path))

    # Fallback: use first column as index, second as value
    if len(cols) >= 2:
        value_col = cols[1]
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col])
        return [
            CanonicalRecord(
                primary_order=i,
                order_type=OrderType.SEQUENCE,
                channel=value_col,
                metric="value",
                value=float(row[value_col]),
                seq_order=i,
            )
            for i, row in df.iterrows()
        ]

    print(f"Cannot auto-detect format for {csv_path.name}")
    return []


def cmd_load(args: argparse.Namespace) -> int:
    """Show a summary of a CSV file."""
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return 1

    import pandas as pd

    t0 = time.perf_counter()
    records = _auto_ingest(csv_path)
    elapsed = time.perf_counter() - t0

    if not records:
        return 1

    channels = sorted({r.channel for r in records})
    keys_all = set()
    for r in records:
        if hasattr(r, 'keys') and r.keys:
            keys_all.update(r.keys.keys())

    orders = [r.primary_order for r in records]
    min_o, max_o = min(orders), max(orders)

    # Estimate grain
    from .lattice.sampling import grain_from_orders
    grain = grain_from_orders(orders)

    print(f"  file      : {csv_path.name}")
    print(f"  records   : {len(records):,}")
    print(f"  channels  : {channels}")
    if keys_all:
        print(f"  keys      : {sorted(keys_all)}")
    print(f"  range     : {min_o:,} → {max_o:,}  (span: {max_o - min_o:,})")
    print(f"  est grain : {grain}")
    print(f"  ({elapsed:.2f}s)")
    return 0


# ---------------------------------------------------------------------------
# surface
# ---------------------------------------------------------------------------

def cmd_surface(args: argparse.Namespace) -> int:
    """Load CSV, build surface, show anomaly summary. Optional heatmap."""
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return 1

    t_total = time.perf_counter()

    # Ingest
    t0 = time.perf_counter()
    records = _auto_ingest(csv_path)
    if not records:
        return 1
    t_ingest = time.perf_counter() - t0

    channels = sorted({r.channel for r in records})
    print(f"  file      : {csv_path.name}")
    print(f"  records   : {len(records):,}  channels={channels}  ({t_ingest:.2f}s)")

    # Build graph
    from .graph import Input, Bin, Measure, Baseline, Residual, Pipeline

    x = Input()
    agg = {ch: {"value": "mean"} for ch in channels}
    b = Bin(agg_funcs=agg)(x)
    m = Measure(profile="continuous")(b)

    # Default output is the measured surface
    output = m

    # Apply baseline and/or residual if requested
    if args.baseline:
        bl_kwargs = {"method": args.baseline}
        if args.baseline == "ewma":
            bl_kwargs["alpha"] = args.alpha
        else:
            bl_kwargs["window"] = args.window
        bl = Baseline(**bl_kwargs)(m)

        if args.residual:
            output = Residual(mode=args.residual)(m, bl)
        else:
            output = bl
    elif args.residual:
        print("  --residual requires --baseline")
        return 1

    pipe = Pipeline(x, output)

    # Resolve — use explicit overrides if provided, else derive from data
    resolve_kwargs = {}
    if args.horizon:
        resolve_kwargs["horizon"] = args.horizon
    if args.grain:
        resolve_kwargs["grain"] = args.grain

    pipe.resolve(records=records, **resolve_kwargs)
    plan = pipe.plan

    print(f"  horizon   : {plan.horizon:,}  grain={plan.grain}  cbin={plan.cbin}")
    print(f"  windows   : {plan.windows}")
    print(f"  basis     : {plan.prime_basis}")

    # Build
    t0 = time.perf_counter()
    result = pipe.build(records)
    surfaces = result.value
    t_build = time.perf_counter() - t0

    print(f"  surfaces  : {len(surfaces)}  ({t_build:.2f}s)")
    for s in surfaces:
        finite = np.isfinite(list(s.values.values())[0])
        print(f"    {s.channel:12s}  shape={s.shape}  coverage={finite.mean():.1%}")

    # Anomaly summary
    # Pick the first available value array from the surface
    print()
    label = "peak |z| per scale"
    if args.baseline and args.residual:
        label = f"{args.residual} residual vs {args.baseline} baseline"
    elif args.baseline:
        label = f"{args.baseline} baseline"
    print(f"  Anomaly summary ({label}):")
    for s in surfaces:
        # Use "mean" if available, else first value array
        arr = s.values.get("mean")
        if arr is None:
            arr = next(iter(s.values.values()), None)
        if arr is None:
            continue
        print(f"    {s.channel}")
        scale_peaks = []
        for i, w in enumerate(plan.windows):
            row = arr[i]
            finite = row[np.isfinite(row)]
            if len(finite) == 0:
                continue
            mu = np.mean(finite)
            sigma = np.std(finite)
            if sigma == 0:
                continue
            z = (finite - mu) / sigma
            peak_idx = int(np.argmax(np.abs(z)))
            peak = float(np.max(np.abs(z)))
            scale_peaks.append((peak, w, peak_idx))
        scale_peaks.sort(reverse=True)
        for peak, w, idx in scale_peaks[:8]:
            bar = "█" * min(int(peak), 30)
            print(f"      scale={w:>6}  peak |z| = {peak:>6.2f}σ  {bar}")

    total = time.perf_counter() - t_total
    print(f"\n  total: {total:.2f}s")

    # Heatmap
    if args.hm or args.save:
        # Try to read dates from the CSV for the time axis
        import pandas as pd
        df = pd.read_csv(csv_path)
        date_col = df.columns[0]
        dates = None
        try:
            dates = pd.to_datetime(df[date_col])
            # Drop rows with missing values to match records
            value_col = df.columns[1] if len(df.columns) == 2 else None
            if value_col:
                mask = pd.to_numeric(df[value_col], errors="coerce").notna()
                dates = dates[mask].reset_index(drop=True)
        except Exception:
            pass

        subtitle = ""
        if args.baseline and args.residual:
            subtitle = f"{args.residual} residual vs {args.baseline}"
        elif args.baseline:
            subtitle = f"baseline: {args.baseline}"

        _render_heatmap(surfaces, plan, csv_path.name, dates,
                        subtitle=subtitle, save_path=args.save)

    return 0


def _render_heatmap(surfaces: list, plan, filename: str = "", dates=None,
                    subtitle: str = "", save_path: str = None) -> None:
    """Render a z-score heatmap of the mean surface for each channel."""
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    for s in surfaces:
        # Use "mean" if available, else first value array
        arr = s.values.get("mean")
        if arr is None:
            arr = next(iter(s.values.values()), None)
        if arr is None:
            continue

        fig, ax = plt.subplots(figsize=(14, 6))
        finite_arr = np.where(np.isfinite(arr), arr, np.nan)

        # Compute z-scores per scale
        z = np.full_like(finite_arr, np.nan)
        for i in range(finite_arr.shape[0]):
            row = finite_arr[i]
            valid = row[np.isfinite(row)]
            if len(valid) > 0 and np.std(valid) > 0:
                z[i] = (row - np.nanmean(row)) / np.nanstd(row)

        n_scales, n_time = z.shape

        # Time axis: use dates if available (allow partial coverage)
        if dates is not None and len(dates) > 0:
            last_idx = min(len(dates) - 1, n_time - 1)
            extent = [
                mdates.date2num(dates.iloc[0]),
                mdates.date2num(dates.iloc[last_idx]),
                -0.5,
                n_scales - 0.5,
            ]
            im = ax.imshow(
                z,
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-4, vmax=4,
                origin="lower",
                extent=extent,
            )
            ax.xaxis_date()
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate(rotation=45)
        else:
            im = ax.imshow(
                z,
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-4, vmax=4,
                origin="lower",
            )
            ax.set_xlabel("Time (bins)")

        ax.set_ylabel("Window scale")
        ax.set_yticks(range(n_scales))
        ax.set_yticklabels([str(w) for w in plan.windows], fontsize=7)

        title = f"SignalForge — {s.channel}"
        if filename:
            title += f"  ({filename})"
        if subtitle:
            title += f"\n{subtitle}"
        ax.set_title(title, fontsize=13, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(
            "z-score (σ)\nDeviation of windowed mean from scale baseline",
            fontsize=9,
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved: {save_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="signalforge",
        description="SignalForge — p-adic multiscale signal processing pipeline",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # demo
    p_demo = sub.add_parser("demo", help="Run the EEG seizure detection demo")

    # run
    p_run = sub.add_parser("run", help="Run the pipeline on a CSV file")
    p_run.add_argument("domain", choices=["intermagnet", "eeg", "equities", "timeseries"],
                       help="Domain (determines SamplingPlan and ingest)")
    p_run.add_argument("csv", help="Input CSV file path")
    p_run.add_argument("--out", default=None, help="Output directory for CSV export")

    # plan
    p_plan = sub.add_parser("plan", help="Show a SamplingPlan for a domain")
    p_plan.add_argument("domain",
                        choices=["intermagnet", "intermagnet-yearly", "eeg", "equities", "equities-daily", "timeseries"],
                        help="Domain name")

    # load
    p_load = sub.add_parser("load", help="Show a summary of a CSV file")
    p_load.add_argument("csv", help="Input CSV file path")

    # surface
    p_surf = sub.add_parser("surface", help="Build and display a surface from a CSV")
    p_surf.add_argument("csv", help="Input CSV file path")
    p_surf.add_argument("-hm", action="store_true", help="Render heatmap")
    p_surf.add_argument("--horizon", type=int, default=None, help="Explicit horizon")
    p_surf.add_argument("--grain", type=int, default=None, help="Explicit grain")
    p_surf.add_argument("--baseline", default=None,
                        choices=["ewma", "median", "rolling_mean"],
                        help="Baseline method")
    p_surf.add_argument("--alpha", type=float, default=0.1,
                        help="EWMA smoothing factor (default: 0.1)")
    p_surf.add_argument("--window", type=int, default=20,
                        help="Baseline window size for median/rolling_mean (default: 20)")
    p_surf.add_argument("--residual", default=None,
                        choices=["difference", "ratio", "z"],
                        help="Residual mode (requires --baseline)")
    p_surf.add_argument("--save", default=None,
                        help="Save heatmap to file instead of displaying")

    # neighborhood
    p_nb = sub.add_parser("neighborhood", help="Show the p-adic arithmetic viewing box")
    p_nb.add_argument("anchor", type=int, help="Center integer")
    p_nb.add_argument("radius", type=int, nargs="?", default=6,
                      help="Half-width (default: 6)")
    p_nb.add_argument("--basis", type=int, nargs="+", default=None,
                      help="Explicit prime basis (default: auto)")

    args = parser.parse_args(argv)

    dispatch = {
        "demo":         cmd_demo,
        "run":          cmd_run,
        "plan":         cmd_plan,
        "load":         cmd_load,
        "surface":      cmd_surface,
        "neighborhood": cmd_neighborhood,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
