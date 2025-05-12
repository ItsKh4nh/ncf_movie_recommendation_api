from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import pickle
import numpy as np
import time
import torch.nn as nn
import pytorch_lightning as pl

# Setup FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="API for personalized movie recommendations using Neural Collaborative Filtering",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model definition
class NCF(pl.LightningModule):
    """Neural Collaborative Filtering (NCF)"""

    def __init__(
        self,
        num_users,
        num_items,
        ratings=None,
        all_movie_ids=None,
        embedding_dim=8,
        lr=0.001,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items, embedding_dim=embedding_dim
        )
        self.fc1 = nn.Linear(in_features=embedding_dim * 2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_movie_ids = all_movie_ids
        self.lr = lr

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        pred = nn.Sigmoid()(self.output(vector))
        return pred


# Load model and movie data
model = None
movie_mappings = None


def load_model():
    global model, movie_mappings

    # Set paths
    model_path = os.path.join("output", "ncf_model.pt")
    movie_mappings_path = os.path.join("output", "movie_mappings.pkl")

    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(movie_mappings_path):
        raise FileNotFoundError(
            f"Model files not found. Ensure {model_path} and {movie_mappings_path} exist."
        )

    # Load model
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model = NCF(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        ratings=None,
        all_movie_ids=None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load movie data
    with open(movie_mappings_path, "rb") as f:
        movie_mappings = pickle.load(f)

    return {"status": "Model loaded successfully"}


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/")
def read_root():
    return {"message": "CineWorld Recommendation API is currently running"}


@app.get("/recommendations")
def get_recommendations(user_id: int, top_k: int = 10):
    if model is None or movie_mappings is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please try again later."
        )

    start_time = time.time()

    # Get movie info
    all_movie_ids = movie_mappings["movie_ids"]
    movie_titles = movie_mappings["movie_titles"]

    # Limit user_id to what model supports
    max_users = model.user_embedding.num_embeddings
    if user_id >= max_users:
        raise HTTPException(
            status_code=400, detail=f"User ID must be less than {max_users}"
        )

    # Predict scores
    device = next(model.parameters()).device
    user_tensor = torch.tensor([user_id] * len(all_movie_ids)).to(device)
    item_tensor = torch.tensor(all_movie_ids).to(device)

    with torch.no_grad():
        scores = np.squeeze(model(user_tensor, item_tensor).cpu().numpy())

    # Get top-k recommendations
    top_indices = np.argsort(scores)[::-1][:top_k]
    recommendations = [
        {
            "movie_id": int(all_movie_ids[i]),
            "title": movie_titles.get(int(all_movie_ids[i])),
            "score": float(scores[i]),
        }
        for i in top_indices
    ]

    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "processing_time_ms": round((time.time() - start_time) * 1000, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
