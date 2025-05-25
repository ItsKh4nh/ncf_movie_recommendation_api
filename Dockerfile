# ---- Builder Stage ----
FROM python:3.10-slim AS builder

WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv
# Activate venv for subsequent RUN commands in this stage
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install dependencies into venv
RUN pip install --no-cache-dir -r requirements.txt

# ---- Final Stage ----
FROM python:3.10-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Add venv to PATH in the final image
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code and model files
COPY app.py .
COPY output/ncf_model.pt output/
COPY output/movie_mappings.pkl output/

# Expose API port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]