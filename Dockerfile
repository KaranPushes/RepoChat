FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies including Node.js and npm for frontend
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nodejs \
    npm \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend, install npm packages, and build production assets
COPY frontend/package*.json ./frontend/
RUN npm install --prefix frontend
COPY frontend/ ./frontend/
RUN npm run build --prefix frontend

# Copy backend code, prompts, and config
COPY repochat/ ./repochat/
COPY prompts/ ./prompts/
COPY pyproject.toml .

EXPOSE 8000

CMD ["uvicorn", "repochat.api:app", "--host", "0.0.0.0", "--port", "8000"]
