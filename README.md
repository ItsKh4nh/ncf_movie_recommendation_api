# NCF Movie Recommendation API

## Project Overview
This is a movie recommendation system that utilizes Neural Collaborative Filtering (NCF) to provide personalized movie recommendations. The model is implemented based on the [NCF architecture](https://arxiv.org/abs/1708.05031) proposed by He et al. The model is trained on the [MovieLens 32M dataset](https://grouplens.org/datasets/movielens/) and deployed as a RESTful API.

This API serves as the recommendation engine for my main project [CineWorld movie streaming platform](https://github.com/ItsKh4nh/cineworld_v2), check it out if you interested.

## Results Achieved

The model achieved impressive performance metrics:

- **Hit Ratio@10**: 0.9588  
  Measures how often the model successfully includes a movie the user would actually interact within the top 10 movie recommendations. A score of 0.9588 means that 95.88% of the time, a relevant movie appears in the user's top 10 list.

- **NDCG@10**: 0.7406  
  NDCG measures not just whether good recommendations appear in the top 10, but how well they are ranked (with higher positions being more important). A score of 0.7406 indicates strong ranking performance, with the most relevant items appearing nearer to the top of recommendations.


## Project Structure
```
├── app.py                 # FastAPI application for serving recommendations
├── docker-compose.yml     # Docker Compose configuration for local deployment
├── Dockerfile             # Docker configuration for containerization
├── output/                # Directory for trained model artifacts
│   ├── neumf_model.pt     # Trained Neural Collaborative Filtering model
│   └── movie_mappings.pkl # Movie ID to title mappings
├── requirements.txt       # Python dependencies
├── training.py            # Training script
└── training.ipynb         # Jupyter notebook version
```

## Technologies Used
- **FastAPI**: Modern, fast web framework for building APIs
- **PyTorch**: Deep learning framework for model development
- **PyTorch Lightning**: Lightweight PyTorch wrapper for high-performance AI research
- **NumPy/Pandas**: Data manipulation and analysis
- **Pickle**: Serialization and deserialization of trained model
- **Docker**: Containerization for consistent deployment

## Installation and Setup

### Prerequisites
- Python 3.10+
- Docker (recommended for deployment)
- Trained model files in the `output` directory:
  - `neumf_model.pt`: The trained Neural Collaborative Filtering model
  - `movie_mappings.pkl`: Movie ID to title mappings

### Option 1: Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/ItsKh4nh/NCF_recommendation_api.git
   cd NCF_recommendation_api
   ```

2. Make sure you have trained model files in the `output` directory.

3. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. The API will be available at `http://localhost:8000`

### Option 2: Direct Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ItsKh4nh/NCF_recommendation_api.git
   cd NCF_recommendation_api
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure you have the trained model files in the `output` directory.

5. Start the API server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## Training the Model
If you want to train the model yourself:

1. Download the [MovieLens 32M dataset](https://grouplens.org/datasets/movielens/)
2. Run the training script:
   ```bash
   python training.py
   ```
or
   ```bash
   jupyter notebook training.ipynb
   ```

3. The trained model will be saved to the `output` directory.

## API Access Guide

The API is publicly available at:
```
https://api-cineworld.onrender.com
```

### Get Recommendations

```
GET /recommendations?user_id={user_id}&top_k={top_k}
```

Parameters:
- `user_id`: ID of the user for which recommendations should be generated
- `top_k` (Optional): Number of movie recommendations to return (default: 10)

Example:
```bash
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
- [SwaggerUI API documentation](https://api-cineworld.onrender.com/docs)
- [ReDoc API documentation](https://api-cineworld.onrender.com/redoc)