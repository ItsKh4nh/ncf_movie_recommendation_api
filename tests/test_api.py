import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os


def test_read_root(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "CineWorld Recommendation API is currently running"}


def test_recommendations_without_model(client):
    """Test recommendations endpoint when model is not loaded."""
    with patch("app.model", None):
        response = client.get("/recommendations?user_id=1&top_k=5")
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]


def test_recommendations_with_invalid_user_id(client):
    """Test recommendations endpoint with user_id exceeding model limits."""
    # Mock model with limited users
    mock_model = MagicMock()
    mock_model.user_embedding.num_embeddings = 100

    mock_mappings = {"movie_ids": list(range(10)), "movie_titles": {i: f"Movie {i}" for i in range(10)}}

    with patch("app.model", mock_model), patch("app.movie_mappings", mock_mappings):
        response = client.get("/recommendations?user_id=150&top_k=5")
        assert response.status_code == 400
        assert "User ID must be less than 100" in response.json()["detail"]


def test_recommendations_valid_request(client):
    """Test recommendations endpoint with valid request."""
    # Mock model
    mock_model = MagicMock()
    mock_model.user_embedding.num_embeddings = 1000
    mock_model.parameters.return_value = [MagicMock(device="cpu")]

    # Mock model output
    import torch
    import numpy as np

    mock_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    mock_model.return_value.cpu.return_value.numpy.return_value = mock_scores

    mock_mappings = {
        "movie_ids": [1, 2, 3, 4, 5],
        "movie_titles": {1: "Movie 1", 2: "Movie 2", 3: "Movie 3", 4: "Movie 4", 5: "Movie 5"},
    }

    with patch("app.model", mock_model), patch("app.movie_mappings", mock_mappings):
        response = client.get("/recommendations?user_id=1&top_k=3")
        assert response.status_code == 200

        data = response.json()
        assert data["user_id"] == 1
        assert len(data["recommendations"]) == 3
        assert "processing_time_ms" in data

        # Check that recommendations are sorted by score (descending)
        scores = [rec["score"] for rec in data["recommendations"]]
        assert scores == sorted(scores, reverse=True)


def test_recommendations_default_top_k(client):
    """Test recommendations endpoint with default top_k value."""
    mock_model = MagicMock()
    mock_model.user_embedding.num_embeddings = 1000
    mock_model.parameters.return_value = [MagicMock(device="cpu")]

    # Mock 15 movies to test default top_k=10
    import torch

    mock_scores = torch.tensor([0.9 - i * 0.05 for i in range(15)])
    mock_model.return_value.cpu.return_value.numpy.return_value = mock_scores

    mock_mappings = {"movie_ids": list(range(15)), "movie_titles": {i: f"Movie {i}" for i in range(15)}}

    with patch("app.model", mock_model), patch("app.movie_mappings", mock_mappings):
        response = client.get("/recommendations?user_id=1")
        assert response.status_code == 200

        data = response.json()
        assert len(data["recommendations"]) == 10  # Default top_k
