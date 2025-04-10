FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY app.py .
COPY models/ncf_model.pt models/
COPY models/movie_data.pkl models/

# Expose API port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]