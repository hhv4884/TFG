import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
output_dir = "datasets"
os.makedirs(output_dir, exist_ok=True)
class_sep_values = [0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, sep in zip(axes, class_sep_values):

    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=sep,
        random_state=42
    )

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.set_title(f"class_sep = {sep}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

fig.suptitle("Comparación de datasets generados con función make_classification")
plt.tight_layout()

classification_path = os.path.join(output_dir, "make_classification.png")
plt.savefig(classification_path, dpi=300, bbox_inches='tight')
plt.close()

cluster_std_values = [1.0, 10.0, 100.0]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, std in zip(axes, cluster_std_values):

    X, y = make_blobs(
        n_samples=1000,
        centers=2,
        n_features=2,
        cluster_std=std,
        random_state=42
    )

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.set_title(f"cluster_std = {std}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

fig.suptitle("Comparación de datasets generados con función make_blobs")
plt.tight_layout()
blobs_path = os.path.join(output_dir, "make_blobs.png")
plt.savefig(blobs_path, dpi=300, bbox_inches='tight')
plt.close()