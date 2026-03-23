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
        print(f"No run script for domain {domain!r}. Available: intermagnet, eeg")
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
        else:
            print(f"Unknown domain {domain!r}.")
            print("Available: intermagnet, intermagnet-yearly, eeg")
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
    p_run.add_argument("domain", choices=["intermagnet", "eeg"],
                       help="Domain (determines SamplingPlan and ingest)")
    p_run.add_argument("csv", help="Input CSV file path")
    p_run.add_argument("--out", default=None, help="Output directory for CSV export")

    # plan
    p_plan = sub.add_parser("plan", help="Show a SamplingPlan for a domain")
    p_plan.add_argument("domain",
                        choices=["intermagnet", "intermagnet-yearly", "eeg"],
                        help="Domain name")

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
        "neighborhood": cmd_neighborhood,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
