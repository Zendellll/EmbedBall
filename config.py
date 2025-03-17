"""
config.py

Holds default configuration parameters for the Football AE pipeline.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Data settings
    csv_root_dir: str = "csv"
    test_size: float = 0.2

    # Model settings
    input_dim: int = 128  # Must match the number of columns in an embedded vector
    latent_dim: int = 32

    # Training settings
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 0.001

    # Artifacts (paths to save output)
    model_path: str = "artifacts/pretrained_autoencoder.pth"
    loss_plot_path: str = "artifacts/loss_plot.png"

    # Visualization
    pca_plot_path: str = "artifacts/pca_visualization.png"
    tsne_plot_path: str = "artifacts/tsne_visualization.png"
    umap_plot_path: str = "artifacts/umap_visualization.png"

    # Dimension reduction parameters
    tsne_perplexity: int = 30
    umap_n_neighbors: int = 30

