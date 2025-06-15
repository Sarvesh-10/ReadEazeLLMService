FROM python:3.11-slim

# Install system dependencies and curl
RUN apt-get update && apt-get install -y curl build-essential

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# Install dependencies without installing the package itself (no-root)
RUN poetry install --no-root --no-interaction

# Copy the rest of the application code
COPY . .

# Expose the port your FastAPI app will run on
EXPOSE 8000

# Run the app using Poetry (adjust the module path if needed)
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
