# 1. IMPORTS AND SETUP
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping

np.random.seed(123)

# 2. DATA PREPARATION
ratings = pd.read_csv("/kaggle/input/tmdb-movie-dataset/movielens.csv")
ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

# Split into Train/Test sets - last interacted movie as Test item
ratings["rank_latest"] = ratings.groupby(["user_id"])["timestamp"].rank(
    method="first", ascending=False
)
train_ratings = ratings[ratings["rank_latest"] != 1]
test_ratings = ratings[ratings["rank_latest"] == 1]

# Create a mapping of movie_id to title
movie_id_to_title = (
    ratings[["movie_id", "title"]]
    .drop_duplicates()
    .set_index("movie_id")
    .to_dict()["title"]
)

# Prepare final training data
train_ratings = train_ratings[["user_id", "movie_id", "rating"]]
test_ratings = test_ratings[["user_id", "movie_id", "rating"]]
train_ratings.loc[:, "rating"] = 1  # Convert to implicit feedback (1 = interaction)

print("Data preparation completed.")


# 3. DATASET CLASS
# Get a list of all movie_ids
all_movie_ids = ratings["movie_id"].unique()


class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset with On-the-fly Negative Sampling"""

    def __init__(self, ratings, all_movie_ids, num_negatives=4):
        self.ratings = ratings
        self.all_movie_ids = all_movie_ids
        self.num_negatives = num_negatives
        self.user_item_set = set(zip(ratings["user_id"], ratings["movie_id"]))

        self.user_items = {}
        for u, item in self.user_item_set:
            if u not in self.user_items:
                self.user_items[u] = set()
            self.user_items[u].add(item)

        # Create user and item lists for positive samples only
        self.users = ratings["user_id"].values
        self.items = ratings["movie_id"].values
        self.labels = np.ones(len(self.users))

    def __len__(self):
        return len(self.users) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        # Determine if this is a positive or negative sample
        base_idx = idx // (1 + self.num_negatives)
        offset = idx % (1 + self.num_negatives)

        user = self.users[base_idx]

        if offset == 0:  # Positive sample
            item = self.items[base_idx]
            label = 1.0
        else:  # Generate negative sample
            interacted_items = self.user_items.get(user, set())
            while True:
                item = np.random.choice(self.all_movie_ids)
                if item not in interacted_items:
                    break
            label = 0.0

        return torch.tensor(user), torch.tensor(item), torch.tensor(label)


# 4. MODEL DEFINITION
class NCF(pl.LightningModule):
    """Neural Collaborative Filtering (NCF)"""

    def __init__(
        self, num_users, num_items, ratings, all_movie_ids, embedding_dim=8, lr=0.001
    ):
        super().__init__()
        print(f"Initializing NCF model")
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

        self.save_hyperparameters(ignore=["ratings", "all_movie_ids"])
        print("Model initialized successfully")

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        pred = nn.Sigmoid()(self.output(vector))
        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(
            MovieLensTrainDataset(self.ratings, self.all_movie_ids),
            batch_size=512,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )


# 5. TRAINING
def train_model():
    num_users = ratings["user_id"].max() + 1
    num_items = ratings["movie_id"].max() + 1

    model = NCF(num_users, num_items, train_ratings, all_movie_ids)

    # Setup GPU Acceleration
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Early Stopping to prevent Overfitting
    early_stop_callback = EarlyStopping(monitor="train_loss", patience=3, mode="min")

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[TQDMProgressBar(refresh_rate=20), early_stop_callback],
        logger=False,
        enable_checkpointing=False,
    )

    print(f"Training on: {trainer.accelerator}")
    print("Training in progress...")
    trainer.fit(model)
    print("Training completed.")

    print("Saving necessary files...")
    os.makedirs("output", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
        },
        "output/ncf_model.pt",
    )

    with open("output/movie_mappings.pkl", "wb") as f:
        pickle.dump({"movie_ids": all_movie_ids, "movie_titles": movie_id_to_title}, f)
    return model


# 6. EVALUATION
def evaluate_model(model):
    # User-item pairs for testing
    test_user_item_set = set(zip(test_ratings["user_id"], test_ratings["movie_id"]))

    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby("user_id")["movie_id"].apply(list).to_dict()

    # Evaluate metrics
    K = 10
    device = next(model.parameters()).device
    hits = []

    for u, item in tqdm(test_user_item_set, desc="Evaluating"):
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_movie_ids) - set(interacted_items)

        # Sample 99 negative items + 1 positive for evaluation
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
        test_items = selected_not_interacted + [item]

        user_tensor = torch.tensor([u] * 100).to(device)
        item_tensor = torch.tensor(test_items).to(device)

        with torch.no_grad():
            predicted_labels = np.squeeze(model(user_tensor, item_tensor).cpu().numpy())

        # Get indices of top K items
        top_indices = np.argsort(predicted_labels)[::-1][:K].tolist()
        top_items = [test_items[i] for i in top_indices]

        # Hit Ratio: Check if positive item is in top K
        hit = 1.0 if item in top_items else 0.0
        hits.append(hit)

    # Calculate final metrics
    hit_ratio = np.mean(hits)

    print("Evaluation completed.")
    print(f"Hit Ratio @ {K}:  {hit_ratio:.4f}")

    metrics = {
        "hit_ratio": hit_ratio,
    }

    return metrics


# 7. RECOMMENDATION FUNCTION
def recommend(
    user_id,
    top_k=10,
    model_path="output/ncf_model.pt",
    movie_mappings_path="output/movie_mappings.pkl",
):
    """Get movie recommendations for a specific user"""
    print(f"Getting recommendations for user_id: {user_id}")

    print("Loading model...", end=" ")
    checkpoint = torch.load(model_path)
    model = NCF(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        ratings=None,
        all_movie_ids=None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Done")

    print("Loading movie mappings...", end=" ")
    with open(movie_mappings_path, "rb") as f:
        movie_mappings = pickle.load(f)
    print("Done")

    all_movie_ids = movie_mappings["movie_ids"]
    movie_titles = movie_mappings["movie_titles"]

    print(f"Computing scores...")

    # Predict scores
    device = next(model.parameters()).device
    user_tensor = torch.tensor([user_id] * len(all_movie_ids)).to(device)
    item_tensor = torch.tensor(all_movie_ids).to(device)

    with torch.no_grad():
        scores = np.squeeze(model(user_tensor, item_tensor).cpu().numpy())

    # Get top-k recommendations
    print(f"Finding top {top_k} movies...")
    top_indices = np.argsort(scores)[::-1][:top_k]
    recommendations = [
        {
            "id": int(all_movie_ids[i]),
            "title": movie_titles.get(int(all_movie_ids[i])),
            "score": float(scores[i]),
        }
        for i in top_indices
    ]

    print("Recommendations generated.")
    return recommendations


# 8. MAIN EXECUTION
if __name__ == "__main__":
    model = train_model()
    metrics = evaluate_model(model)

    user_id = 100  # Test with user_id 100
    recs = recommend(user_id=user_id, top_k=10)

    print(f"\nRecommended movies for user_id: {user_id}")
    for i, movie in enumerate(recs, 1):
        print(f"{i}. {movie['title']} (Score: {movie['score']:.5f})")
