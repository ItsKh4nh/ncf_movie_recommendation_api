FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create models directory
RUN mkdir -p models

# Note: Model files (ncf_model.pt and movie_data.pkl) should be
# mounted as a volume or copied separately

# Expose API port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 