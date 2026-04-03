"""
signalforge.domains

Domain-specific ingest functions.

Each domain module implements:

    ingest(path: str) -> list[CanonicalRecord]

Ingest transforms raw source data into domain-agnostic CanonicalRecords.
All domain knowledge stops here. The graph derives sampling geometry
from the data and user declarations — domains do not define plans.

Shipped domains
---------------
eeg         — clinical EEG (CHB-MIT format)
intermagnet — geomagnetic observatory (INTERMAGNET standard)
equities    — equity price data (yfinance format)
timeseries  — generic two-column CSV (date, value)

Adding a domain
---------------
Create a module in this package. Implement ingest(). No registration
or plugin machinery required — import directly.

    from signalforge.domains import timeseries
    records = timeseries.ingest("data.csv")
"""
