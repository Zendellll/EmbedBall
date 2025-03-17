"""
training.py
-----------
Handles preparing data for training (TensorDataset creation),
splitting into train/test, and the autoencoder training loop.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def extract_vectors_from_df(df) -> np.ndarray:
    """
    Extract columns from a DataFrame and convert to a numpy array.

    Args:
        df (pd.DataFrame): DataFrame containing features.

    Returns:
        np.ndarray: Array of shape (num_samples, num_features).
    """
    vector_cols = list(df.columns)  # get all df columns
    if not vector_cols:
        raise ValueError("No vector columns found in DataFrame.")
    return df[vector_cols].values


def prepare_data(df, test_size: float = 0.2) -> Tuple[torch.Tensor, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Prepare data with a train/test split as PyTorch Datasets.

    Args:
        df (pd.DataFrame): DataFrame containing all features.
        test_size (float): Proportion of data to be used as test set.

    Returns:
        (X_tensor, train_dataset, test_dataset)
        X_tensor: Full dataset as a tensor (features).
        train_dataset: PyTorch subset of training data.
        test_dataset: PyTorch subset of test data.
    """
    vectors = extract_vectors_from_df(df)
    X_tensor = torch.FloatTensor(vectors)
    dataset = TensorDataset(X_tensor, X_tensor)

    test_size_int = int(len(dataset) * test_size)
    train_size = len(dataset) - test_size_int

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size_int])
    return X_tensor, train_dataset, test_dataset


def plot_losses(train_losses, test_losses, file_path: str) -> None:
    """
    Plot training and test losses over epochs.

    Args:
        train_losses (list): Training loss history.
        test_losses (list): Test/validation loss history.
        file_path (str): Where to save the plot image.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()


def train_model(
    train_loader,
    model,
    test_loader=None,
    num_epochs=100,
    learning_rate=0.001,
    loss_plot_path="artifacts/loss_plot.plt",
    criterion=nn.MSELoss()
):
    """
    Train the autoencoder with an optional validation/test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_features, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch_features = batch_features.to(device)

            reconstructed, _ = model(batch_features)
            loss = criterion(reconstructed, batch_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        if test_loader is not None:
            model.eval()
            total_test_loss = 0

            with torch.no_grad():
                for batch_features, _ in test_loader:
                    batch_features = batch_features.to(device)
                    reconstructed, _ = model(batch_features)
                    loss = criterion(reconstructed, batch_features)
                    total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}")

    # Optional: plot losses if test_loader is provided
    if test_loader is not None:
        plot_losses(train_losses, test_losses, loss_plot_path)

    return model
