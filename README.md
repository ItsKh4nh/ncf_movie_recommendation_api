# Movie Recommendation API

This API serves personalized movie recommendations using a Neural Collaborative Filtering model trained on the MovieLens 32M dataset.

## Endpoints

The API is publicly available at:

```
https://api-cineworld.onrender.com
```

### Get Recommendations

```
GET /recommendations?user_id={user_id}&top_k={top_k}
```

Parameters:

- `user_id`: ID of the user for which recommendations should be generated for
- `top_k` (optional): Number of movie recommendations to return (default: 10)

Example:

```
curl https://api-cineworld.onrender.com/recommendations?user_id=1&top_k=10
```

Response:

```json
{
  "user_id": 1,
  "recommendations": [
    {
      "movie_id": 1234,
      "title": "Movie Title",
      "score": 0.9123
    },
    ...
  ],
  "processing_time_ms": 120.45
}
```

### API Documentation

For detailed API documentation, visit:

- https://api-cineworld.onrender.com/docs - SwaggerUI API documentation
- https://api-cineworld.onrender.com/redoc - ReDoc API documentation

## Local Development

### Prerequisites

- Docker
- Trained model files:
  - `output/ncf_model.pt` - The trained Neural Collaborative Filtering model in PyTorch format
  - `output/movie_mappings.pkl` - Reference data for the movies

### Setup

1. Make sure you have the trained model files ready. These files should be generated from the `collaborative.py` script after training.

2. Place the model files in a local `output` directory.

3. Build the Docker image:

   ```bash
   docker build -t movie-recommender .
   ```

4. Run the Docker container:

   ```bash
   docker run -p 8000:8000 -v $(pwd)/output:/app/output movie-recommender
   ```

   - Note: For Windows Command Prompt, use:
     ```
     docker run -p 8000:8000 -v %cd%/output:/app/output movie-recommender
     ```
   - For Windows PowerShell, use:
     ```
     docker run -p 8000:8000 -v ${PWD}/output:/app/output movie-recommender
     ```
