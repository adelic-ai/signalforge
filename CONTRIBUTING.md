# Contributing to SignalForge

There are two ways to contribute: adding a domain, or improving the core pipeline.

---

## Adding a domain

This is the primary contribution path. A domain is a single Python file in
`signalforge/domains/` that declares a `SamplingPlan` for a specific data source.

Read **[docs/domain_guide.md](docs/domain_guide.md)** before starting. The guide
walks through the three decisions you need to make and includes a complete example.

### Steps

1. Fork the repo and create a branch: `git checkout -b domain/yourname`
2. Create `signalforge/domains/yourname.py`
3. Add a docstring that names the domain, units, sampling rate, and a reference
4. Add an example script to `examples/run_yourname.py`
5. Run the existing tests: `uv run pytest`
6. Open a PR with a brief description of the data source and why the window
   selection makes sense for the domain

The PR description should answer:
- What is the data source?
- What is `horizon` and why?
- What is `grain` and why?
- Which windows are anchors and what is their significance in the field?

---

## Improving the core pipeline

The pipeline stages (Stages 1–5) live in `signalforge/pipeline/` and
`signalforge/lattice/`. Changes here affect every domain.

Before opening a PR for a pipeline change:

- All existing tests must pass: `uv run pytest`
- New behavior should have tests in `tests/`
- Do not introduce domain semantics into any pipeline or lattice module —
  domain knowledge lives exclusively in `signalforge/domains/`

---

## Development setup

```bash
git clone https://github.com/adelic/signalforge
cd signalforge
uv sync --dev
uv run pytest
```

---

## Code style

- Python 3.12+
- No external dependencies beyond numpy and pandas in core pipeline
- Immutable artifacts: use `__slots__`
- No datetime objects in the pipeline — ordering is integers only
- Type annotations on all public functions

---

## License

By submitting a PR you agree that your contribution will be licensed under the
same terms as the project (BUSL-1.1, converting to Apache 2.0 on 2029-03-22).
