"""
evaluation.py
-------------
Functions for generating embeddings, comparing reconstructions, etc.
"""

import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple


def get_embeddings(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> np.ndarray:
    """
    Generate embeddings for all events in the given data_loader.

    Args:
        model (torch.nn.Module): Trained autoencoder model.
        data_loader (torch.utils.data.DataLoader): DataLoader for events.

    Returns:
        np.ndarray: Numpy array of shape (num_samples, latent_dim).
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating embeddings...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating embeddings"):
            # batch is (input, target), so we take only the input
            inputs = batch[0].to(device)
            latent = model.encode(inputs)
            embeddings.append(latent.cpu().numpy())

    all_embeddings = np.vstack(embeddings)
    logger.info(f"Generated embeddings of shape {all_embeddings.shape}")
    return all_embeddings


def compare_reconstructions(model: torch.nn.Module, sample_inputs: torch.Tensor, num_events: int = 5) -> None:
    """
    Compare original and reconstructed vectors for a few sample events.
    Prints mean error and shows a heatmap of differences.

    Args:
        model (torch.nn.Module): Trained autoencoder model.
        sample_inputs (torch.Tensor): Sample input events.
        num_events (int): Number of events to visualize.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Comparing reconstructions for {num_events} events...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    sample_inputs = sample_inputs.to(device)
    with torch.no_grad():
        reconstructed, _ = model(sample_inputs)

    original = sample_inputs.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    # Interleave original and reconstructed
    num_rows = num_events * 2
    interleaved = np.zeros((num_rows, original.shape[1]))
    for i in range(num_events):
        interleaved[i*2] = original[i]
        interleaved[i*2 + 1] = reconstructed[i]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 2 * num_rows))
    im = ax.imshow(interleaved, aspect='auto', cmap='RdYlBu')
    plt.colorbar(im, ax=ax)

    row_labels = []
    for i in range(num_events):
        row_labels.append(f"Event {i+1} Original")
        row_labels.append(f"Event {i+1} Reconstructed")
    ax.set_yticks(np.arange(num_rows))
    ax.set_yticklabels(row_labels)

    plt.title("Original vs. Reconstructed Event Vectors")
    plt.tight_layout()
    plt.show()

    # Log mean error
    for i in range(num_events):
        mean_err = np.mean(np.abs(original[i] - reconstructed[i]))
        logger.info(f"Event {i+1} Mean Reconstruction Error: {mean_err:.6f}")
