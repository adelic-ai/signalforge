# Operators Guide

Operators are functions you insert into the pipeline via `.transform()` after
`.materialize()` and before `.measure()`. They work on `list[BinnedRecord]` —
the binned, channel-organized representation of your data before the multiscale
surface is built.

All built-in operators are in `signalforge.pipeline.operators`, imported as `ops`:

```python
import signalforge as sf
import signalforge.pipeline.operators as ops
```

---

## Built-in operators

### Cleaning

**`ops.clip(low, high)`** — hard clamp on bin values.

```python
.transform(ops.clip(low=0.0, high=1000.0))
```

**`ops.winsorize(lower=0.01, upper=0.99)`** — replace extreme values with
percentile bounds, computed per (channel, metric). Less aggressive than clip
when you don't know the valid range in advance.

```python
.transform(ops.winsorize(0.01, 0.99))
```

**`ops.fill_gaps(value=0.0)`** — insert fill bins for any gap in the sequence.
Useful for sparse event data where missing bins should be treated as zero rather
than absent.

```python
.transform(ops.fill_gaps(0.0))
```

### Channel selection

**`ops.drop_channels(*names)`** — remove named channels entirely.

```python
.transform(ops.drop_channels("noise", "artifact"))
```

**`ops.keep_channels(*names)`** — keep only the named channels.

```python
.transform(ops.keep_channels("flux", "temperature"))
```

**`ops.drop_sparse(min_coverage=0.1)`** — drop channels where fewer than
`min_coverage` fraction of bins are populated. Removes channels that are
too sparse to produce meaningful surfaces.

```python
.transform(ops.drop_sparse(min_coverage=0.5))
```

---

## Derived channels

Derived channels let you compute new channels from existing ones. The derived
channel flows through `measure` and `engineer` exactly like any source channel —
it gets its own multiscale surface and its own z-scores.

There are two operators depending on whether your computation needs one bin at
a time or the full time axis.

---

### `ops.derive` — per-bin derivation

Use when your function only needs the values at a single point in time.

`fn` receives a `dict[str, float]` — the current bin's channel values — and
returns a single `float`.

```python
ops.derive(name, fn, metric="derived")
```

**Ratio of two channels:**

```python
.transform(ops.derive(
    "flux_ratio",
    lambda ch: ch["flux_x"] / max(ch["flux_z"], 1.0)
))
```

**Difference:**

```python
.transform(ops.derive(
    "component_delta",
    lambda ch: ch["flux_x"] - ch["flux_z"]
))
```

**Weighted composite:**

```python
.transform(ops.derive(
    "fused_score",
    lambda ch: 0.6 * ch["flux"] + 0.4 * ch["temperature"]
))
```

**Normalized difference:**

```python
.transform(ops.derive(
    "norm_diff",
    lambda ch: (ch["A"] - ch["B"]) / max(ch["B"], 1e-9)
))
```

**Per-source ratio** (when keys distinguish sources — e.g. station or host):

```python
.transform(ops.derive(
    "signal_to_noise",
    lambda ch: ch["signal"] / max(ch["noise"], 1e-9)
))
```

Bins where any channel referenced in `fn` is missing are silently skipped.

---

### `ops.derive_temporal` — full time-axis derivation

Use when your function needs to look across time — lags, rolling windows,
cumulative statistics.

`fn` receives a `dict[str, np.ndarray]` — one array per channel, aligned to a
common bin index axis with `NaN` where a channel has no record — and returns a
`np.ndarray` of the same length.

```python
ops.derive_temporal(name, fn, metric="derived")
```

**Lagged difference:**

```python
import numpy as np

.transform(ops.derive_temporal(
    "flux_lag5",
    lambda ch: ch["flux_x"] - np.roll(ch["flux_x"], 5)
))
```

Note: `np.roll` wraps at the boundary. To suppress the wrap-around, mask the
first N positions:

```python
lambda ch: np.concatenate([
    np.full(5, np.nan),
    ch["flux_x"][5:] - ch["flux_x"][:-5]
])
```

**Rolling residual (subtract local mean):**

```python
.transform(ops.derive_temporal(
    "flux_residual",
    lambda ch: ch["flux_x"] - np.convolve(
        np.nan_to_num(ch["flux_x"]), np.ones(10) / 10, mode="same"
    )
))
```

**Cross-channel lag comparison:**

```python
.transform(ops.derive_temporal(
    "component_lag3",
    lambda ch: ch["flux_x"] - np.roll(ch["flux_z"], 3)
))
```

Output positions where the result is `NaN` are dropped and not emitted as
records.

---

## Chaining operators

Operators chain naturally — each `.transform()` receives the output of the
previous one. Derived channels from an earlier `.transform()` are available
to later ones.

```python
bundle = (
    sf.acquire("intermagnet", path)
      .materialize()
      .transform(ops.winsorize(0.01, 0.99))
      .transform(ops.derive(
          "flux_ratio",
          lambda ch: ch["flux_x"] / max(ch["flux_z"], 1.0)
      ))
      .transform(ops.derive_temporal(
          "ratio_lag10",
          lambda ch: ch["flux_ratio"] - np.roll(ch["flux_ratio"], 10)
      ))
      .transform(ops.keep_channels("flux_ratio", "ratio_lag10"))
      .measure()
      .engineer()
      .assemble()
      .run()
)
```

The second `derive_temporal` references `flux_ratio` — the channel created
by the first `derive`. Chaining works because each operator returns the full
updated record list.

---

## Writing a custom operator

If the built-in operators don't cover your case, pass any callable to
`.transform()` directly. It receives `list[BinnedRecord]` and must return
`list[BinnedRecord]`.

```python
def my_operator(binned):
    # ... your logic ...
    return binned

pipeline.transform(my_operator)
```

See `signalforge/pipeline/operators.py` for the `BinnedRecord` field reference
and the patterns used by the built-in operators.
