import logging
from typing import Tuple

import hydra
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np
from omegaconf import DictConfig
from sklearn.manifold import TSNE



logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_data(input_path: str) -> Tuple[np.array, np.array]:
    data = np.load(input_path)
    return data["embeddings"], data["labels"]

def plot_tsne_2d(
        embeddings: np.array, 
        labels: np.array, 
        sample_size: int,
        output_path: str,
        grid: bool
    ) -> None:
    n_samples = 20_000
    indices = np.random.choice(len(embeddings), sample_size, replace=False)
    embeddings_subset = embeddings[indices]
    labels_subset = labels[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_tsne = tsne.fit_transform(embeddings_subset)

    plt.figure(figsize=(10, 8))
    cmap = colormaps["coolwarm"].resampled(2)

    scatter = plt.scatter(
        embeddings_tsne[:, 0], embeddings_tsne[:, 1],
        c=labels_subset,
        cmap=cmap,
        alpha=0.6,
        edgecolors='k'
    )

    cmap = colormaps["coolwarm"].resampled(2)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='no click',
            markerfacecolor=cmap(0), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='click',
            markerfacecolor=cmap(1), markersize=10)
    ]
    plt.legend(handles=legend_elements, title="Labels")

    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.grid(grid)
    plt.tight_layout()
    plt.savefig(output_path)


def plot_tsne_3d(
        embeddings: np.array, 
        labels: np.array, 
        sample_size: int,
        output_path: str
    ) -> None:
    indices = np.random.choice(len(embeddings), sample_size, replace=False)
    embeddings_subset = embeddings[indices]
    labels_subset = labels[indices]

    tsne = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)
    embeddings_tsne = tsne.fit_transform(embeddings_subset)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    cmap = colormaps["coolwarm"].resampled(2)

    scatter = ax.scatter(
        embeddings_tsne[:, 0], embeddings_tsne[:, 1], embeddings_tsne[:, 2],
        c=labels_subset,
        cmap=cmap,
        alpha=0.6,
        edgecolors='k'
    )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='no click',
            markerfacecolor=cmap(0), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='click',
            markerfacecolor=cmap(1), markersize=10)
    ]
    ax.legend(handles=legend_elements, title="Labels")

    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_zlabel("component 3")
    plt.tight_layout()
    plt.savefig(output_path)



@hydra.main(config_path=".", config_name="conf", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(f"Loading data from {cfg['input_path']}")
    embeddings, labels = load_data(cfg["input_path"])

    logger.info(f"Starting 2D t-SNE plot with sample size {cfg['sample_size']}")
    plot_tsne_2d(
        embeddings, 
        labels, 
        cfg["sample_size"],
        cfg["2d_output_path"],
        cfg["grid"]
    )

    logger.info(f"Starting 3D t-SNE plot with sample size {cfg['sample_size']}")
    plot_tsne_3d(
        embeddings, 
        labels, 
        cfg["sample_size"],
        cfg["3d_output_path"]
    )


if __name__ == "__main__":
    main()