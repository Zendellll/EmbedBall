"""
visualization.py
----------------
Functions for dimensionality reduction (PCA, t-SNE, UMAP) and plotting embeddings.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import silhouette_score
from typing import Tuple


def sample_embeddings(embeddings: np.ndarray, labels: np.ndarray, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample a subset of embeddings (and corresponding labels) for large datasets.

    Args:
        embeddings (np.ndarray): Original embeddings array of shape (N, D).
        labels (np.ndarray): Corresponding labels of shape (N,).
        n_samples (int): Number of samples to keep.

    Returns:
        (sampled_embeddings, sampled_labels): The reduced dataset.
    """
    logger = logging.getLogger(__name__)
    if len(embeddings) <= n_samples:
        logger.info("Dataset size is smaller than n_samples; returning full set.")
        return embeddings, labels

    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    logger.info(f"Reduced dataset from {len(embeddings)} to {n_samples} samples")
    return embeddings[indices], labels[indices]


def plot_embedding(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Embedding Visualization",
    file_path: str = "artifacts/embedding_visualization.png"
) -> None:
    """
    Plot 2D embeddings with colors according to 'labels'.

    Args:
        embeddings_2d (np.ndarray): 2D array of shape (N, 2) from a dimensionality reduction.
        labels (np.ndarray): 1D array of labels.
        title (str): Title for the plot.
        file_path (str): File path for saving the figure.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Plotting 2D embedding '{title}'")

    unique_labels = np.unique(labels)
    cmap = cm.get_cmap("hsv", len(unique_labels))

    plt.figure(figsize=(10, 7))
    for i, lbl in enumerate(unique_labels):
        mask = (labels == lbl)
        color = cmap(i)
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            color=[color],
            alpha=0.6,
            label=str(lbl)
        )

    try:
        silhouette = silhouette_score(embeddings_2d, labels)
        subtitle = f"{title} (Silhouette: {silhouette:.3f})"
    except ValueError:
        # Silhouette can fail if there's only 1 cluster/label
        subtitle = title

    plt.title(subtitle)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()
    logger.info(f"Plot saved to {file_path}")
