# Movie Recommendation API

This API serves personalized movie recommendations using a Neural Collaborative Filtering model.

## Prerequisites

- Docker
- Trained model files:
  - `models/ncf_model.pt` - The trained PyTorch model
  - `models/movie_data.pkl` - Movie data for recommendations

## Setup

1. Make sure you have the trained model files ready. These files should be generated from the `collaborative.py` script after training.

2. Place the model files in a local `models` directory.

3. Build the Docker image:

   ```bash
   docker build -t movie-recommender .
   ```

4. Run the Docker container:

   ```bash
   docker run -p 8000:8000 -v $(pwd)/models:/app/models movie-recommender
   ```

   - Note: For Windows Command Prompt, use:
     ```
     docker run -p 8000:8000 -v %cd%/models:/app/models movie-recommender
     ```
   - For Windows PowerShell, use:
     ```
     docker run -p 8000:8000 -v ${PWD}/models:/app/models movie-recommender
     ```

## API Usage

### Get Recommendations

```
GET /api/recommendations?user_id={user_id}&top_k={top_k}
```

Parameters:

- `user_id`: User ID for which recommendations should be generated
- `top_k` (optional): Number of recommendations to return (default: 20)

Example:

```
curl http://localhost:8000/api/recommendations?user_id=1
```

Response:

```json
{
  "user_id": 1,
  "recommendations": [
    {
      "id": 1234,
      "title": "Movie Title",
      "score": 0.9123
    },
    ...
  ],
  "processing_time_ms": 120.45
}
```

## API Documentation

Once the server is running, visit:

- http://localhost:8000/docs - SwaggerUI API documentation
- http://localhost:8000/redoc - ReDoc API documentation
