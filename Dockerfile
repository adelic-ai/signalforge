FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY signalforge/ signalforge/

# Install dependencies (no dev extras)
RUN uv sync --no-dev --frozen

# Default command shows help; override with any signalforge subcommand
ENTRYPOINT ["uv", "run", "signalforge"]
CMD ["--help"]
