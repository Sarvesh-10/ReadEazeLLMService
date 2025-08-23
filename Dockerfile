# --------- BASE STAGE ---------
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y curl build-essential

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

# Only copy the worker script and necessary modules
COPY worker.py .  
# If worker imports other modules, copy them too, e.g.:
# COPY utils/ ./utils/
# COPY config.py .

# Run worker
CMD ["poetry", "run", "python", "worker.py"]
