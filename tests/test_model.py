import pytest
import torch
from app import NCF


def test_ncf_model_initialization():
    """Test NCF model initialization."""
    num_users, num_items = 100, 200
    embedding_dim = 16

    model = NCF(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim)

    assert model.user_embedding.num_embeddings == num_users
    assert model.item_embedding.num_embeddings == num_items
    assert model.user_embedding.embedding_dim == embedding_dim
    assert model.item_embedding.embedding_dim == embedding_dim


def test_ncf_model_forward():
    """Test NCF model forward pass."""
    model = NCF(num_users=100, num_items=200, embedding_dim=8)

    # Test with batch of users and items
    user_input = torch.tensor([0, 1, 2])
    item_input = torch.tensor([10, 11, 12])

    output = model(user_input, item_input)

    assert output.shape == (3, 1)  # 3 samples, 1 output each
    assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output range


def test_ncf_model_single_prediction():
    """Test NCF model with single user-item pair."""
    model = NCF(num_users=100, num_items=200)

    user_input = torch.tensor([5])
    item_input = torch.tensor([25])

    output = model(user_input, item_input)

    assert output.shape == (1, 1)
    assert 0 <= output.item() <= 1
