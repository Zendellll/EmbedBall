"""
main.py
-------
Orchestrates data loading, model creation, training, and evaluation.
"""

import logging
import os

# Third-party
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

# Local imports
from config import Config
from data_loading import load_match_events, cleanup_dataframe
from model import EventAutoencoder
from training import prepare_data, train_model
from evaluation import get_embeddings, compare_reconstructions
from visualization import plot_embedding, sample_embeddings

def setup_logging(log_level=logging.INFO) -> None:
    """
    Set up basic logging configuration.
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(config: Config) -> None:
    # Ensure the artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("Starting the Football Autoencoder pipeline...")

    # 1. Load and clean data
    df = load_match_events(csv_root_dir=config.csv_root_dir)
    df = cleanup_dataframe(df)
    logger.info(f"Data shape after cleanup: {df.shape}")

    # 2. Prepare train/test data
    X_tensor, train_dataset, test_dataset = prepare_data(df, test_size=config.test_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # 3. Define and train model
    input_dim = len(df.columns)  # Adjust if you want a subset of columns
    model = EventAutoencoder(input_dim=input_dim, latent_dim=config.latent_dim)
    model = train_model(
        train_loader,
        model,
        test_loader=test_loader,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        loss_plot_path=config.loss_plot_path
    )

    # 4. Save the trained model
    torch.save(model.state_dict(), config.model_path)
    logger.info(f"Model saved to {config.model_path}")

    # 5. Generate embeddings on the test set
    embeddings = get_embeddings(model, test_loader)

    # 6. Compare some reconstructions
    sample_indices = np.random.choice(len(X_tensor), size=config.num_events_to_compare, replace=False)
    sample_inputs = X_tensor[sample_indices]
    compare_reconstructions(model, sample_inputs, num_events=config.num_events_to_compare)

    # 7. Visualization / dimension reduction
    # First column in df represents event_type IDs or :
    labels = df.iloc[:, 0].to_numpy()

    # Sample embeddings for performance
    embeddings_sampled, labels_sampled = sample_embeddings(embeddings, labels, n_samples=2000)

    # 7a) PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings_sampled)
    plot_embedding(pca_result, labels_sampled, title="PCA Visualization", file_path=config.pca_plot_path)

    # 7b) t-SNE
    tsne = TSNE(n_components=2, perplexity=config.tsne_perplexity, random_state=42)
    tsne_result = tsne.fit_transform(embeddings_sampled)
    plot_embedding(tsne_result, labels_sampled, title="t-SNE Visualization", file_path=config.tsne_plot_path)

    # 7c) UMAP
    umap = UMAP(n_components=2, n_neighbors=config.umap_n_neighbors, random_state=42)
    umap_result = umap.fit_transform(embeddings_sampled)
    plot_embedding(umap_result, labels_sampled, title="UMAP Visualization", file_path=config.umap_plot_path)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    setup_logging(logging.INFO)

    # Load config
    config = Config()
    main(config)
