# Your Data

How to get your own data into SignalForge.

## Simplest path

A two-column CSV — first column is the index, second is the value:

```csv
date,temperature
2024-01-01,22.5
2024-01-02,23.1
2024-01-03,21.8
```

```bash
sf surface mydata.csv -hm
```

That's it. SF auto-detects the format, estimates the resolution, and builds the surface.

## Schema — when you need more control

For multi-column data, SF can infer the structure:

```bash
sf schema mydata.csv
```

```
  Schema: mydata
  ──────────────────────────────────────────────────
    timestamp            ordered       (primary)
    sensor_type          categorical
    location             categorical
    reading              numeric       (value)
  ──────────────────────────────────────────────────
```

Correct anything it got wrong:

```bash
sf schema mydata.csv --group-by location --channel sensor_type --save mydata.schema.json
```

Use it:

```bash
sf surface mydata.csv --schema mydata.schema.json -hm
```

The schema file is reusable — same format, different data, same schema.

## What SF expects

Your data maps to five concepts:

| Your data | SF concept | What it does |
|-----------|-----------|--------------|
| timestamp or sequence number | ordered axis | Orders your data |
| measurement category | channel | Each category gets its own surface |
| numeric observation | value | What you're surfacing |
| grouping entity | keys (group-by) | Per-entity surfaces on the same grid |
| time vs sequence | ordering type | Estimated automatically |

Everything else — resolution, scales, windows — is derived from the data.

## Multi-channel data

If your CSV has multiple measurement types, set the channel axis:

```bash
sf schema sensors.csv --channel sensor_type --save sensors.schema.json
sf surface sensors.csv --schema sensors.schema.json -hm
```

Each channel gets its own surface. All surfaces share the same grid — directly comparable.

## Per-entity analysis

Group by an entity column to get one surface set per entity:

```bash
sf schema machines.csv --group-by machine_id --save machines.schema.json
sf surface machines.csv --schema machines.schema.json -hm
```

## Event data

For logs and event records — each row is an event, not a periodic measurement.

Two-column CSV (timestamp + count) works directly. For multi-column event data, use `sf schema` to identify channels and keys.

**Time-ordered** — timestamps as the index. Gaps between events are real (the system was idle).

**Event-ordered** — sequence numbers as the index. Gaps disappear. The pattern matters, not the timing.

Both are valid. SF estimates the ordering from the data.

## Native formats

For binary or proprietary formats (EDF, netCDF, etc.), write a small parser that reads the format and produces Records via a Schema. See the built-in domain adapters in `signalforge/domains/` for examples.

For CSV data, no parser needed — `sf schema` + `sf surface --schema` handles it.

## Large data

- **100k records**: instant
- **1M records**: under a second
- **10M records**: a few seconds
- **100M+**: chunk the data, process segments

See the [Python API](python-api.md) for programmatic chunking.

## What's next

- [CLI reference](cli.md) — all commands and flags
- [Python API](python-api.md) — chaining and DAG composition
- [Concepts](concepts.md) — how the lattice works (for the curious)
