import pytest
from fastapi.testclient import TestClient
import os
import tempfile
import pickle
import torch
import numpy as np
from app import app, NCF


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model_files():
    """Create mock model files for testing."""
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Create mock model checkpoint
        num_users, num_items = 1000, 5000
        model = NCF(num_users=num_users, num_items=num_items)

        checkpoint = {"num_users": num_users, "num_items": num_items, "model_state_dict": model.state_dict()}

        model_path = os.path.join(output_dir, "ncf_model.pt")
        torch.save(checkpoint, model_path)

        # Create mock movie mappings
        movie_mappings = {
            "movie_ids": list(range(100)),  # First 100 movies
            "movie_titles": {i: f"Movie {i}" for i in range(100)},
        }

        mappings_path = os.path.join(output_dir, "movie_mappings.pkl")
        with open(mappings_path, "wb") as f:
            pickle.dump(movie_mappings, f)

        # Temporarily change working directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        yield output_dir

        os.chdir(original_cwd)
