##############################
# 1. IMPORTS AND SETUP
##############################
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import time
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

np.random.seed(123)

##############################
# 2. DATA PREPARATION
##############################
start_time_data_prep = time.time()
ratings = pd.read_csv(
    "/kaggle/input/movielens-32m/ratings.csv", parse_dates=["timestamp"]
)

ratings.sample(5)

ratings["rank_latest"] = ratings.groupby(["user_id"])["timestamp"].rank(
    method="first", ascending=False
)

train_ratings = ratings[ratings["rank_latest"] != 1]
test_ratings = ratings[ratings["rank_latest"] == 1]

# drop columns that we no longer need
train_ratings = train_ratings[["user_id", "movie_id", "rating"]]
test_ratings = test_ratings[["user_id", "movie_id", "rating"]]

train_ratings.loc[:, "rating"] = 1

train_ratings.sample(5)

# Get a list of all movie IDs
all_movie_ids = ratings["movie_id"].unique()

# Placeholders that will hold the training data
users, items, labels = [], [], []

# This is the set of items that each user has interaction with
user_item_set = set(zip(train_ratings["user_id"], train_ratings["movie_id"]))

# 4:1 ratio of negative to positive samples
num_negatives = 4

for u, i in tqdm(user_item_set):
    users.append(u)
    items.append(i)
    labels.append(1)  # items that the user has interacted with are positive
    for _ in range(num_negatives):
        # randomly select an item
        negative_item = np.random.choice(all_movie_ids)
        # check that the user has not interacted with this item
        while (u, negative_item) in user_item_set:
            negative_item = np.random.choice(all_movie_ids)
        users.append(u)
        items.append(negative_item)
        labels.append(0)  # items not interacted with are negative

end_time_data_prep = time.time()
print(f"Data Preparation took: {end_time_data_prep - start_time_data_prep:.2f} seconds")


##############################
# 3. DATASET CLASS
##############################
class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movie_ids (list): List containing all movie_ids

    """

    def __init__(self, ratings, all_movie_ids):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movie_ids)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_movie_ids):
        start_time_dataset = time.time()
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings["user_id"], ratings["movie_id"]))

        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movie_ids)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movie_ids)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        end_time_dataset = time.time()
        print(
            f"Dataset Generation (get_dataset) took: {end_time_dataset - start_time_dataset:.2f} seconds"
        )
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


##############################
# 4. MODEL DEFINITION
##############################
class NCF(pl.LightningModule):
    """Neural Collaborative Filtering (NCF)

    Args:
        num_users (int): Number of unique users
        num_items (int): Number of unique items
        ratings (pd.DataFrame): Dataframe containing the movie ratings for training
        all_movie_ids (list): List containing all movie_ids (train + test)
    """

    def __init__(self, num_users, num_items, ratings, all_movie_ids):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_movie_ids = all_movie_ids

    def forward(self, user_input, item_input):

        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(
            MovieLensTrainDataset(self.ratings, self.all_movie_ids),
            batch_size=512,
            num_workers=4,
        )


##############################
# 5. TRAINING
##############################
start_time_training = time.time()
num_users = ratings["user_id"].max() + 1
num_items = ratings["movie_id"].max() + 1

all_movie_ids = ratings["movie_id"].unique()

model = NCF(num_users, num_items, train_ratings, all_movie_ids)

trainer = pl.Trainer(
    max_epochs=5,
    gpus=1,
    reload_dataloaders_every_epoch=True,
    progress_bar_refresh_rate=50,
    logger=False,
    checkpoint_callback=False,
)

trainer.fit(model)

end_time_training = time.time()
print(f"Model Training took: {end_time_training - start_time_training:.2f} seconds")

##############################
# 6. EVALUATION
##############################
start_time_eval = time.time()
# User-item pairs for testing
test_user_item_set = set(zip(test_ratings["user_id"], test_ratings["movie_id"]))

# Dict of all items that are interacted with by each user
user_interacted_items = ratings.groupby("user_id")["movie_id"].apply(list).to_dict()

hits = []
for u, i in tqdm(test_user_item_set):
    interacted_items = user_interacted_items[u]
    not_interacted_items = set(all_movie_ids) - set(interacted_items)
    selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
    test_items = selected_not_interacted + [i]

    predicted_labels = np.squeeze(
        model(torch.tensor([u] * 100), torch.tensor(test_items)).detach().numpy()
    )

    top10_items = [
        test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()
    ]

    if i in top10_items:
        hits.append(1)
    else:
        hits.append(0)

end_time_eval = time.time()
print(f"Evaluation took: {end_time_eval - start_time_eval:.2f} seconds")
print("The Hit Ratio @ 10 is {:.2f}".format(np.average(hits)))


##############################
# 7. TEST
##############################
def get_movie_recommendations(
    user_id, model, all_movie_ids, user_interacted_items, top_k=10
):
    """
    Get movie recommendations for a specific user.

    Args:
        user_id (int): The ID of the user to get recommendations for
        model (NCF): The trained model
        all_movie_ids (array): Array of all movie IDs
        user_interacted_items (dict): Dictionary of items each user has interacted with
        top_k (int, optional): Number of recommendations to return. Defaults to 10.

    Returns:
        list: List of top_k movie IDs recommended for the user
    """
    start_time = time.time()

    # Check if user_id exists in the dataset
    if user_id not in user_interacted_items:
        print(f"User {user_id} not found in the dataset")
        return []

    # Get items the user has already interacted with
    interacted_items = user_interacted_items[user_id]

    # Get items the user has not interacted with
    not_interacted_items = list(set(all_movie_ids) - set(interacted_items))

    # Predict scores for all non-interacted items
    batch_size = 1024  # Process in batches to avoid memory issues
    all_predictions = []

    for i in range(0, len(not_interacted_items), batch_size):
        batch_items = not_interacted_items[i : i + batch_size]
        user_tensor = torch.tensor([user_id] * len(batch_items))
        item_tensor = torch.tensor(batch_items)

        with torch.no_grad():  # No need to compute gradients
            predictions = model(user_tensor, item_tensor).detach().numpy().squeeze()
            all_predictions.extend(predictions)

    # Combine items and their predicted scores
    item_pred_pairs = list(zip(not_interacted_items, all_predictions))

    # Sort by prediction score (highest first)
    item_pred_pairs.sort(key=lambda x: x[1], reverse=True)

    # Get top-k items
    top_k_items = [item for item, _ in item_pred_pairs[:top_k]]

    end_time = time.time()
    print(f"Recommendation generation took: {end_time - start_time:.2f} seconds")

    return top_k_items


# Function to load movie information from links.csv
def load_movie_info():
    """
    Load movie information from links.csv file

    Returns:
        dict: Dictionary mapping movie_id to (tmdb_id, title)
    """
    start_time = time.time()
    print("Loading movie information from links.csv...")

    try:
        # Load links.csv file
        links_df = pd.read_csv("links.csv")

        # Create a dictionary mapping movie_id to (tmdb_id, title)
        movie_info = {}
        for _, row in links_df.iterrows():
            movie_info[row["movie_id"]] = (row["tmdb_id"], row["title"])

        end_time = time.time()
        print(
            f"Loaded information for {len(movie_info)} movies in {end_time - start_time:.2f} seconds"
        )

        return movie_info, links_df

    except Exception as e:
        print(f"Error loading movie information: {e}")
        return {}, None


##############################
# 8. SAVE MODEL AND DATA
##############################
def save_outputs(model, links_df):
    """
    Save model and links data to output folder for later use

    Args:
        model (NCF): Trained NCF model
        links_df (DataFrame): DataFrame containing links data
    """
    start_time = time.time()
    print("\nSaving model and data to output folder...")

    # Create output directory if it doesn't exist
    if not os.path.exists("output"):
        os.makedirs("output")
        print("Created output directory")

    # Save the trained model
    try:
        model_path = os.path.join("output", "ncf_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Save links data as pickle file
    try:
        if links_df is not None:
            links_path = os.path.join("output", "links.pkl")
            with open(links_path, "wb") as f:
                pickle.dump(links_df, f)
            print(f"Links data saved to {links_path}")
        else:
            print("Links data not available to save")
    except Exception as e:
        print(f"Error saving links data: {e}")

    end_time = time.time()
    print(f"Saving completed in {end_time - start_time:.2f} seconds")


# Example usage
if __name__ == "__main__":
    # Load movie information
    movie_info, links_df = load_movie_info()

    # Test with a sample user (ensure the user exists in your dataset)
    sample_user_id = 1  # Replace with a valid user ID from your dataset

    print(f"\nGenerating movie recommendations for user {sample_user_id}:")
    recommendations = get_movie_recommendations(
        user_id=sample_user_id,
        model=model,
        all_movie_ids=all_movie_ids,
        user_interacted_items=user_interacted_items,
    )

    if recommendations:
        print(f"\nTop 10 movie recommendations for user {sample_user_id}:")
        for i, movie_id in enumerate(recommendations, 1):
            if movie_id in movie_info:
                tmdb_id, title = movie_info[movie_id]
                print(f"{i}. {title} (ID: {movie_id}, TMDB ID: {tmdb_id})")
            else:
                print(f"{i}. Movie ID: {movie_id} (Title not available)")
    # Save model and data for later use
    save_outputs(model, links_df)
