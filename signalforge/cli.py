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
    from .signal import CanonicalRecord, OrderType
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
    from .signal import CanonicalRecord, OrderType

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


def _fmt_basis(basis: dict) -> str:
    """Format prime basis as readable factorization: 2^3 x 3^2 x 5"""
    if not basis:
        return "1"
    parts = []
    for p in sorted(basis):
        e = basis[p]
        parts.append(f"{p}^{e}" if e > 1 else str(p))
    return " x ".join(parts)


def _fmt_windows(windows: tuple, max_show: int = 8) -> str:
    """Format window list, collapsing middle if too long."""
    if len(windows) <= max_show:
        return ", ".join(str(w) for w in windows)
    half = max_show // 2
    head = ", ".join(str(w) for w in windows[:half])
    tail = ", ".join(str(w) for w in windows[-half:])
    return f"{head}, ... , {tail}  ({len(windows)} total)"


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

    from .lattice.sampling import grain_from_orders
    grain = grain_from_orders(orders)

    # Compute a default plan to show what the lattice looks like
    from .lattice.sampling import SamplingPlan
    span = max_o - min_o
    horizon = max(span, 360)
    try:
        plan = SamplingPlan(horizon, grain)
    except Exception:
        plan = None

    print()
    print(f"  SignalForge  {csv_path.name}")
    print(f"  {'─' * 40}")
    print(f"  records   {len(records):,}")
    print(f"  channels  {', '.join(channels)}")
    if keys_all:
        print(f"  keys      {', '.join(sorted(keys_all))}")
    print(f"  span      {min_o:,} .. {max_o:,}  ({max_o - min_o:,})")
    print(f"  grain     {grain}  (estimated)")
    if plan:
        print(f"  basis     {_fmt_basis(plan.prime_basis)}")
        print(f"  scales    {len(plan.windows)}  [{plan.windows[0]} .. {plan.windows[-1]}]")
    print(f"  {'─' * 40}")
    print(f"  {elapsed:.2f}s")
    print()
    print(f"  Next:")
    print(f"    sf surface {csv_path} -hm")
    if plan and len(plan.windows) > 12:
        print(f"    sf surface {csv_path} -hm --max-window {plan.windows[-1]}")
    print()
    return 0


# ---------------------------------------------------------------------------
# surface
# ---------------------------------------------------------------------------

def cmd_surface(args: argparse.Namespace) -> int:
    """Load CSV, build surface, show anomaly summary. Optional heatmap."""
    # Resolve CSV path: explicit arg > workspace config
    csv_arg = getattr(args, "csv", None)
    ws = _find_workspace()
    ws_config = _load_workspace(ws) if ws else {}

    if csv_arg:
        csv_path = Path(csv_arg)
    elif "csv" in ws_config:
        csv_path = Path(ws_config["csv"])
    else:
        print("  No CSV specified and no workspace found.")
        print("  Usage: sf surface <csv> or cd into a workspace.")
        return 1

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return 1

    # Inherit workspace defaults if not overridden on CLI
    if args.max_window is None and "max_window" in ws_config:
        args.max_window = ws_config["max_window"]
    if args.grain is None and "grain" in ws_config:
        args.grain = ws_config["grain"]

    t_total = time.perf_counter()

    # Ingest
    t0 = time.perf_counter()
    records = _auto_ingest(csv_path)
    if not records:
        return 1
    t_ingest = time.perf_counter() - t0

    # Zoom — slice records by index or date
    zoom_label = ""
    if args.start_date or args.end_date:
        import pandas as pd
        df = pd.read_csv(csv_path)
        dates = pd.to_datetime(df[df.columns[0]], errors="coerce")
        value_col = df.columns[1] if len(df.columns) == 2 else None
        if value_col:
            mask = pd.to_numeric(df[value_col], errors="coerce").notna()
            dates = dates[mask].reset_index(drop=True)

        start_idx = 0
        end_idx = len(records)
        if args.start_date:
            sd = pd.Timestamp(args.start_date)
            start_idx = int((dates >= sd).idxmax()) if (dates >= sd).any() else 0
        if args.end_date:
            ed = pd.Timestamp(args.end_date)
            end_idx = int((dates <= ed)[::-1].idxmax()) + 1 if (dates <= ed).any() else len(records)

        records = records[start_idx:end_idx]
        # Re-index primary_order from 0
        for i, r in enumerate(records):
            r.primary_order = i
            if hasattr(r, 'seq_order') and r.seq_order is not None:
                r.seq_order = i
        zoom_label = f"  zoom      : {args.start_date or 'start'} → {args.end_date or 'end'} ({len(records):,} records)"
    elif args.start is not None or args.end is not None:
        s = args.start or 0
        e = args.end or len(records)
        records = records[s:e]
        for i, r in enumerate(records):
            r.primary_order = i
            if hasattr(r, 'seq_order') and r.seq_order is not None:
                r.seq_order = i
        zoom_label = f"  zoom      : bin {s} → {e} ({len(records):,} records)"

    channels = sorted({r.channel for r in records})

    # Build graph — signal path: Input → Measure (no explicit Bin)
    from .graph import Input, Measure, Baseline, Residual, Pipeline

    x = Input()
    m = Measure()(x)

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

    # Resolve — derive horizon from max_window if given
    resolve_kwargs = {}
    if args.max_window:
        resolve_kwargs["windows"] = [args.max_window]
    if args.grain:
        resolve_kwargs["grain"] = args.grain

    pipe.resolve(records=records, **resolve_kwargs)
    plan = pipe.plan

    # Build
    t0 = time.perf_counter()
    result = pipe.build(records)
    surfaces = result.value
    t_build = time.perf_counter() - t0

    # --- Output ---
    print()
    print(f"  SignalForge  {csv_path.name}")
    print(f"  {'─' * 50}")
    print(f"  records   {len(records):,}  ({t_ingest:.2f}s)")
    print(f"  channels  {', '.join(channels)}")
    if zoom_label:
        print(zoom_label)
    print(f"  horizon   {plan.horizon:,}   basis {_fmt_basis(plan.prime_basis)}")
    print(f"  grain     {plan.grain}   cbin {plan.cbin}")
    print(f"  scales    {_fmt_windows(plan.windows)}")
    print(f"  {'─' * 50}")

    for s in surfaces:
        first_arr = next(iter(s.data.values()))
        finite = np.isfinite(first_arr)
        n_s, n_t = s.shape
        print(f"  {s.channel}  {n_s} scales x {n_t} bins  {finite.mean():.0%} coverage  ({t_build:.2f}s)")

    # Anomaly summary
    print()
    if args.baseline and args.residual:
        label = f"{args.residual} vs {args.baseline}"
    elif args.baseline:
        label = args.baseline
    else:
        label = "z-score"
    print(f"  Anomaly  ({label})")
    print(f"  {'─' * 50}")

    _zoom_suggestion = None
    for s in surfaces:
        arr = s.data.get("mean")
        if arr is None:
            arr = next(iter(s.data.values()), None)
        if arr is None:
            continue

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
            # Map peak_idx back to time position
            time_pos = peak_idx  # bin position relative to start
            scale_peaks.append((peak, w, time_pos))

        scale_peaks.sort(reverse=True)

        # Show top anomalies with location
        for peak, w, tpos in scale_peaks[:6]:
            bar_len = min(int(peak * 2), 30)
            bar = "█" * bar_len
            print(f"    {w:>6}  {peak:>5.1f}σ  {bar}")

        # Where is the strongest anomaly?
        # Build zoom suggestion around peak
        _zoom_suggestion = None
        if scale_peaks:
            top_peak, top_scale, top_bin = scale_peaks[0]
            # Try to resolve bin to date
            _peak_date = None
            _dates_zoomed = None
            try:
                import pandas as pd
                _df = pd.read_csv(csv_path)
                _first_col = _df[_df.columns[0]]
                # Only parse as dates if the column isn't purely numeric
                _is_numeric = pd.to_numeric(_first_col, errors="coerce").notna().all()
                if _is_numeric:
                    raise ValueError("numeric first column, skip date parsing")
                _dates = pd.to_datetime(_first_col, errors="coerce")
                if _dates.isna().all():
                    raise ValueError("no valid dates")
                _value_col = _df.columns[1] if len(_df.columns) == 2 else None
                if _value_col:
                    _mask = pd.to_numeric(_df[_value_col], errors="coerce").notna()
                    _dates = _dates[_mask].reset_index(drop=True)
                # Apply same zoom as records
                if args.start_date or args.end_date:
                    _s_idx = 0
                    _e_idx = len(_dates)
                    if args.start_date:
                        _sd = pd.Timestamp(args.start_date)
                        _s_idx = int((_dates >= _sd).idxmax()) if (_dates >= _sd).any() else 0
                    if args.end_date:
                        _ed = pd.Timestamp(args.end_date)
                        _e_idx = int((_dates <= _ed)[::-1].idxmax()) + 1 if (_dates <= _ed).any() else len(_dates)
                    _dates_zoomed = _dates.iloc[_s_idx:_e_idx].reset_index(drop=True)
                elif args.start is not None or args.end is not None:
                    _s = args.start or 0
                    _e = args.end or len(_dates)
                    _dates_zoomed = _dates.iloc[_s:_e].reset_index(drop=True)
                else:
                    _dates_zoomed = _dates
                if top_bin < len(_dates_zoomed):
                    _peak_date = _dates_zoomed.iloc[top_bin]
            except Exception:
                pass

            if _peak_date is not None:
                print(f"  peak at {_peak_date.strftime('%Y-%m-%d')}, scale {top_scale}")
                # Zoom window: ~10% of data centered on peak
                margin = max(len(records) // 10, top_scale * 2)
                zoom_start = max(0, top_bin - margin)
                zoom_end = min(len(records), top_bin + margin)
                try:
                    _zoom_start_date = _dates_zoomed.iloc[zoom_start].strftime('%Y-%m-%d')
                    _zoom_end_date = _dates_zoomed.iloc[min(zoom_end, len(_dates_zoomed) - 1)].strftime('%Y-%m-%d')
                    _zoom_suggestion = f"sf surface {csv_path} -hm --start-date {_zoom_start_date} --end-date {_zoom_end_date}"
                except Exception:
                    pass
            else:
                print(f"  peak at bin {top_bin}, scale {top_scale}")
                margin = max(len(records) // 10, top_scale * 2)
                zoom_start = max(0, top_bin - margin)
                zoom_end = min(len(records), top_bin + margin)
                _zoom_suggestion = f"sf surface {csv_path} -hm --start {zoom_start} --end {zoom_end}"

    total = time.perf_counter() - t_total
    print(f"  {'─' * 50}")
    print(f"  {total:.2f}s")

    # Save to workspace if available
    name = getattr(args, "name", None)
    if ws and name:
        exp_dir = ws / "output" / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        for s in surfaces:
            for agg_name, arr in s.data.items():
                np.save(exp_dir / f"{s.channel}_{agg_name}.npy", arr)
        import json
        meta = {
            "channel": [s.channel for s in surfaces],
            "horizon": plan.horizon,
            "grain": plan.grain,
            "cbin": plan.cbin,
            "windows": list(plan.windows),
            "baseline": args.baseline,
            "residual": args.residual,
            "csv": str(csv_path),
        }
        with open(exp_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  saved     {exp_dir}/")

    # Suggestions (suppressed by "suggestions": false in sf.json)
    show_hints = ws_config.get("suggestions", True) if ws_config else True
    if show_hints:
        if not args.hm:
            print()
            cmd = f"sf surface {csv_path}"
            if args.max_window:
                cmd += f" --max-window {args.max_window}"
            print(f"  Add -hm for heatmap:  {cmd} -hm")
        if not args.baseline:
            print(f"  Try a baseline:       sf surface {csv_path} -hm --baseline ewma --residual z")
        if _zoom_suggestion and not (args.start_date or args.end_date or args.start is not None):
            print(f"  Zoom into peak:       {_zoom_suggestion}")
        if ws and not name:
            print(f"  Save this run:        add --name <label>")
    print()

    # Heatmap
    if args.hm or args.save:
        # Try to read dates from the CSV for the time axis
        import pandas as pd
        df = pd.read_csv(csv_path)
        date_col = df.columns[0]
        dates = None
        try:
            dates = pd.to_datetime(df[date_col])
            value_col = df.columns[1] if len(df.columns) == 2 else None
            if value_col:
                mask = pd.to_numeric(df[value_col], errors="coerce").notna()
                dates = dates[mask].reset_index(drop=True)
            # Apply same zoom slice to dates
            if args.start_date or args.end_date:
                start_idx = 0
                end_idx = len(dates)
                if args.start_date:
                    sd = pd.Timestamp(args.start_date)
                    start_idx = int((dates >= sd).idxmax()) if (dates >= sd).any() else 0
                if args.end_date:
                    ed = pd.Timestamp(args.end_date)
                    end_idx = int((dates <= ed)[::-1].idxmax()) + 1 if (dates <= ed).any() else len(dates)
                dates = dates.iloc[start_idx:end_idx].reset_index(drop=True)
            elif args.start is not None or args.end is not None:
                s = args.start or 0
                e = args.end or len(dates)
                dates = dates.iloc[s:e].reset_index(drop=True)
        except Exception:
            pass

        subtitle = ""
        if args.baseline and args.residual:
            subtitle = f"{args.residual} residual vs {args.baseline}"
        elif args.baseline:
            subtitle = f"baseline: {args.baseline}"

        # If named experiment, auto-save heatmap there too
        save_target = args.save
        if not save_target and name and ws:
            save_target = str(ws / "output" / name / "heatmap.png")

        _render_heatmap(surfaces, plan, csv_path.name, dates,
                        subtitle=subtitle, save_path=save_target)

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
        arr = s.data.get("mean")
        if arr is None:
            arr = next(iter(s.data.values()), None)
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

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# workspace (sf init)
# ---------------------------------------------------------------------------

_SF_CONFIG = "sf.json"


def _find_workspace() -> Path | None:
    """Walk up from cwd looking for sf.json."""
    p = Path.cwd()
    for _ in range(10):
        if (p / _SF_CONFIG).exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return None


def _load_workspace(ws: Path) -> dict:
    import json
    with open(ws / _SF_CONFIG) as f:
        return json.load(f)


def _save_workspace(ws: Path, config: dict) -> None:
    import json
    with open(ws / _SF_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  config: {ws / _SF_CONFIG}")


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a workspace directory."""
    import json

    ws = Path(args.name)
    csv_path = Path(args.csv) if args.csv else None

    if ws.exists() and (ws / _SF_CONFIG).exists():
        print(f"  Workspace already exists: {ws}")
        return 1

    ws.mkdir(parents=True, exist_ok=True)
    (ws / "cache").mkdir(exist_ok=True)
    (ws / "output").mkdir(exist_ok=True)

    config = {
        "version": 1,
        "name": args.name,
    }

    if csv_path:
        if not csv_path.exists():
            print(f"  File not found: {csv_path}")
            return 1
        config["csv"] = str(csv_path.resolve())

        records = _auto_ingest(csv_path)
        if records:
            channels = sorted({r.channel for r in records})
            from .lattice.sampling import grain_from_orders
            orders = [r.primary_order for r in records]
            grain = grain_from_orders(orders)
            config["summary"] = {
                "records": len(records),
                "channels": channels,
                "estimated_grain": grain,
            }

    if args.max_window:
        config["max_window"] = args.max_window
    if args.grain:
        config["grain"] = args.grain

    _save_workspace(ws, config)

    print()
    print(f"  SignalForge  workspace")
    print(f"  {'─' * 40}")
    print(f"  created   {ws}/")
    print(f"  cache     {ws}/cache/")
    print(f"  output    {ws}/output/")
    if csv_path:
        print(f"  data      {csv_path.name}")
        if "summary" in config:
            s = config["summary"]
            print(f"  records   {s['records']:,}")
            print(f"  channels  {', '.join(s['channels'])}")
            print(f"  grain     {s['estimated_grain']}  (estimated)")
    print(f"  {'─' * 40}")
    print()
    print(f"  cd {ws} && sf surface -hm")
    print()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show workspace status."""
    ws = _find_workspace()
    if ws is None:
        print("  No workspace found. Run: sf init <name> --csv <file>")
        return 1

    config = _load_workspace(ws)

    print()
    print(f"  SignalForge  {config.get('name', ws.name)}")
    print(f"  {'─' * 40}")
    print(f"  workspace {ws}")

    # Data
    csv = config.get("csv")
    if csv:
        csv_path = Path(csv)
        exists = "ok" if csv_path.exists() else "MISSING"
        print(f"  data      {csv_path.name}  ({exists})")
    else:
        print(f"  data      (none)")

    summary = config.get("summary", {})
    if summary:
        print(f"  records   {summary.get('records', '?'):,}")
        channels = summary.get("channels", [])
        print(f"  channels  {', '.join(channels)}")
        print(f"  grain     {summary.get('estimated_grain', '?')}")

    # Defaults
    mw = config.get("max_window")
    gr = config.get("grain")
    if mw or gr:
        parts = []
        if mw:
            parts.append(f"max-window={mw}")
        if gr:
            parts.append(f"grain={gr}")
        print(f"  defaults  {', '.join(parts)}")

    # Cache
    cache_dir = ws / "cache"
    if cache_dir.exists():
        cached = list(cache_dir.glob("*.npz"))
        if cached:
            print(f"  cached    {len(cached)} surface(s)")
            for c in sorted(cached)[:5]:
                print(f"            {c.stem}")
            if len(cached) > 5:
                print(f"            ... and {len(cached) - 5} more")
        else:
            print(f"  cached    (empty)")

    # Output / experiments
    output_dir = ws / "output"
    if output_dir.exists():
        experiments = [d for d in sorted(output_dir.iterdir()) if d.is_dir()]
        if experiments:
            print(f"  runs      {len(experiments)}")
            for e in experiments[:8]:
                files = list(e.iterdir())
                print(f"            {e.name}/  ({len(files)} files)")
            if len(experiments) > 8:
                print(f"            ... and {len(experiments) - 8} more")

    print(f"  {'─' * 40}")
    print()

    # Suggestions
    if not csv:
        print(f"  No data linked. Re-init with: sf init {ws.name} --csv <file>")
    else:
        print(f"  sf surface -hm")
    print()
    return 0


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

_INSPECT_ENTRIES = {
    # --- Baselines ---
    "ewma": {
        "category": "baseline",
        "name": "Exponentially Weighted Moving Average",
        "formula": "s_t = alpha * x_t + (1 - alpha) * s_{t-1}",
        "params": "alpha: smoothing factor in (0, 1]. Higher = more responsive.",
        "description":
            "Tracks a drifting baseline by weighting recent values more heavily.\n"
            "  Effective memory is ~1/alpha observations. At alpha=0.1, the baseline\n"
            "  reflects roughly the last 10 points; at alpha=0.01, the last 100.",
        "use_when":
            "Slowly changing mean: trending markets, gradual load shifts, seasonal drift.\n"
            "  Not ideal for periodic baselines (use median for those).",
        "cli": "sf surface data.csv -hm --baseline ewma --alpha 0.1",
    },
    "median": {
        "category": "baseline",
        "name": "Rolling Median Filter",
        "formula": "s_t = median(x_{t-w/2}, ..., x_{t+w/2})",
        "params": "window: number of observations in the centered window.",
        "description":
            "Robust to spikes and outliers. A single extreme value doesn't move\n"
            "  the baseline. Produces a smooth estimate of the 'typical' value.",
        "use_when":
            "Sharp transient spikes that shouldn't contaminate the baseline.\n"
            "  Good for bursty event data, noisy sensors, outlier-heavy signals.",
        "cli": "sf surface data.csv -hm --baseline median --window 20",
    },
    "rolling_mean": {
        "category": "baseline",
        "name": "Rolling Mean",
        "formula": "s_t = mean(x_{t-w+1}, ..., x_t)",
        "params": "window: number of observations in the trailing window.",
        "description":
            "Simple trailing average. Every point in the window contributes equally.\n"
            "  Responds to level shifts after ~window observations.",
        "use_when":
            "Straightforward smoothing when you don't need outlier robustness\n"
            "  (use median) or exponential decay (use ewma).",
        "cli": "sf surface data.csv -hm --baseline rolling_mean --window 20",
    },
    # --- Residuals ---
    "difference": {
        "category": "residual",
        "name": "Difference Residual",
        "formula": "r_t = x_t - baseline_t",
        "params": "Applied via --residual difference",
        "description":
            "Absolute deviation from baseline. Preserves units.\n"
            "  A residual of 10 means 10 units above baseline.",
        "use_when":
            "Absolute magnitude matters: event counts where '50 extra logins'\n"
            "  is meaningful regardless of the base rate.",
        "cli": "sf surface data.csv -hm --baseline ewma --residual difference",
    },
    "ratio": {
        "category": "residual",
        "name": "Ratio Residual",
        "formula": "r_t = x_t / baseline_t",
        "params": "Applied via --residual ratio",
        "description":
            "Multiplicative deviation. A ratio of 2.0 means double the baseline.\n"
            "  Scale-invariant: 2x spike means the same at base 10 or 10,000.",
        "use_when":
            "Proportional deviation matters more than absolute. Common for count\n"
            "  data, throughput, or metrics where the baseline varies widely.",
        "cli": "sf surface data.csv -hm --baseline ewma --residual ratio",
    },
    "z": {
        "category": "residual",
        "name": "Z-Score Residual",
        "formula": "r_t = (x_t - baseline_t) / std(baseline)",
        "params": "Applied via --residual z",
        "description":
            "Deviation normalized by the baseline's standard deviation.\n"
            "  Measures 'how unusual' relative to the baseline's own variability.",
        "use_when":
            "Unitless anomaly score that accounts for normal variability.\n"
            "  A z of 3 means the same whether the signal is VIX or Kerberos TGTs.",
        "cli": "sf surface data.csv -hm --baseline ewma --residual z",
    },
    # --- Concepts ---
    "horizon": {
        "category": "concept",
        "name": "Horizon",
        "formula": "H = lcm(W union {g})",
        "params": "Derived from windows and grain, or set explicitly with --max-window.",
        "description":
            "The outer boundary of the coordinate space. All valid windows divide it.\n"
            "  May be larger than the data — that's fine. It's a normalization factor\n"
            "  that ensures all windows share the same lattice structure.",
        "use_when":
            "You want to understand why certain window sizes are available and others\n"
            "  aren't. The horizon's prime factorization determines the lattice.",
        "cli": "sf plan equities",
    },
    "grain": {
        "category": "concept",
        "name": "Grain",
        "formula": "g = estimated from inter-event intervals",
        "params": "Set explicitly with --grain, or auto-derived from data.",
        "description":
            "Finest resolution the data supports. Measured from the actual spacing\n"
            "  between events. Below grain, there's no data to analyze.\n"
            "  cbin (computational bin) is the smallest divisor of horizon >= grain.",
        "use_when":
            "You want to understand the resolution floor. sf load shows the estimated\n"
            "  grain. Override with --grain if you know your data's true cadence.",
        "cli": "sf load data.csv",
    },
    "surface": {
        "category": "concept",
        "name": "Surface",
        "formula": "S(scale, time) = agg(signal[t:t+window]) for each window in plan",
        "params": "Shape is (n_scales x n_time_bins).",
        "description":
            "A 2D measurement: time on one axis, scale on the other. Each cell is\n"
            "  the aggregated signal within that window at that position.\n"
            "  Same plan = same grid = surfaces from different signals are comparable.\n"
            "  A Surface is also a LatticeSignal — you can measure surfaces of surfaces.",
        "use_when":
            "Always. The surface is SignalForge's primary output. Everything else\n"
            "  (baselines, residuals, features) operates on surfaces.",
        "cli": "sf surface data.csv -hm",
    },
    "lattice": {
        "category": "concept",
        "name": "Lattice",
        "formula": "n = p1^a1 * p2^a2 * ... * pk^ak",
        "params": "Every positive integer has a unique prime factorization.",
        "description":
            "The divisibility lattice of the integers. Every window that divides the\n"
            "  horizon is a point on this lattice. The prime factorization of the\n"
            "  horizon determines how many scales exist and how they relate.\n"
            "  Richer factorization = more scales = finer multiscale analysis.",
        "use_when":
            "You want to understand why SignalForge chooses certain window sizes.\n"
            "  sf neighborhood <n> shows the lattice around any integer.",
        "cli": "sf neighborhood 360",
    },
}


def cmd_inspect(args: argparse.Namespace) -> int:
    """Show documentation for a method, operator, or concept."""
    name = getattr(args, "name", None)

    # No argument: show everything
    if name is None:
        print()
        print(f"  SignalForge  inspect")
        print(f"  {'─' * 40}")

        categories = {}
        for key, entry in _INSPECT_ENTRIES.items():
            cat = entry.get("category", "other")
            categories.setdefault(cat, []).append((key, entry))

        cat_labels = {"baseline": "Baselines", "residual": "Residuals", "concept": "Concepts"}
        for cat in ["baseline", "residual", "concept"]:
            if cat not in categories:
                continue
            print(f"\n  {cat_labels.get(cat, cat)}")
            for key, entry in categories[cat]:
                print(f"    {key:<15s} {entry['name']}")

        print(f"\n  {'─' * 40}")
        print(f"  sf inspect <name> for details")
        print()
        return 0

    name = name.lower()
    entry = _INSPECT_ENTRIES.get(name)

    if entry is None:
        print(f"  Unknown: {name!r}")
        print(f"  Available: {', '.join(sorted(_INSPECT_ENTRIES))}")
        return 1

    cat = entry.get("category", "")
    print()
    print(f"  {entry['name']}  ({cat})")
    print(f"  {'─' * 50}")
    print(f"  {entry['formula']}")
    print(f"  {entry['params']}")
    print()
    print(f"  {entry['description']}")
    print()
    print(f"  When:  {entry['use_when']}")
    print()
    print(f"  Example:")
    print(f"    {entry['cli']}")
    print()
    return 0


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

    # init
    p_init = sub.add_parser("init", help="Initialize a workspace directory")
    p_init.add_argument("name", help="Workspace directory name")
    p_init.add_argument("--csv", default=None, help="Path to CSV data file")
    p_init.add_argument("--max-window", type=int, default=None,
                        help="Default max analysis window")
    p_init.add_argument("--grain", type=int, default=None, help="Default grain")

    # status
    p_status = sub.add_parser("status", help="Show workspace status")

    # inspect
    p_insp = sub.add_parser("inspect", help="Show docs for a method, operator, or concept")
    p_insp.add_argument("name", nargs="?", default=None,
                        help="Name to inspect (e.g. ewma, z, lattice). Omit to list all.")

    # load
    p_load = sub.add_parser("load", help="Show a summary of a CSV file")
    p_load.add_argument("csv", help="Input CSV file path")

    # surface
    p_surf = sub.add_parser("surface", help="Build and display a surface from a CSV")
    p_surf.add_argument("csv", nargs="?", default=None,
                        help="Input CSV file path (optional if in a workspace)")
    p_surf.add_argument("-hm", action="store_true", help="Render heatmap")
    p_surf.add_argument("--max-window", type=int, default=None,
                        help="Largest analysis window (horizon derived automatically)")
    p_surf.add_argument("--grain", type=int, default=None, help="Finest resolution")
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
    p_surf.add_argument("--start", type=int, default=None,
                        help="Start index (bin number) for zoom")
    p_surf.add_argument("--end", type=int, default=None,
                        help="End index (bin number) for zoom")
    p_surf.add_argument("--start-date", default=None,
                        help="Start date for zoom (e.g. 2008-01-01)")
    p_surf.add_argument("--end-date", default=None,
                        help="End date for zoom (e.g. 2009-12-31)")
    p_surf.add_argument("--save", default=None,
                        help="Save heatmap to file instead of displaying")
    p_surf.add_argument("--name", default=None,
                        help="Save this run as a named experiment in the workspace")

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
        "init":         cmd_init,
        "status":       cmd_status,
        "inspect":      cmd_inspect,
        "load":         cmd_load,
        "surface":      cmd_surface,
        "neighborhood": cmd_neighborhood,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
