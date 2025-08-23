# --------- BASE STAGE ---------
FROM python:3.11-slim AS base


# Install system dependencies (including libpq-dev for psycopg2)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root --no-interaction

# --------- FASTAPI IMAGE ---------
FROM base AS fastapi

# Copy all app code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --------- WORKER IMAGE (slim) ---------
FROM base AS worker

# Copy worker script and any other modules it needs
COPY app/worker.py ./worker.py
# If it imports other files in app/, copy them too
COPY app/configs/config.py ./config.py
COPY app/utils.py ./utils.py

# Run worker
CMD ["poetry", "run", "python", "worker.py"]
