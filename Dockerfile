FROM python:3.13-slim

# 1. Pasang uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. Setup environment
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1
# Memaksa uv menggunakan Python sistem (3.12) bukan download lagi
ENV UV_PYTHON_PREFERENCE=only-system

WORKDIR /app

# 3. Copy config
COPY pyproject.toml uv.lock ./

# 4. Install dependencies
# uv akan otomatis bikin .venv di /app/.venv
RUN uv sync --frozen --no-dev --no-install-project

# 5. Tambahkan .venv/bin ke PATH supaya uvicorn bisa dipanggil langsung
ENV PATH="/app/.venv/bin:$PATH"

# 6. Copy kodingan
COPY . .

# 7. Jalankan langsung (karena bin sudah di PATH)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]