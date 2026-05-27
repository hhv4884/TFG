from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

def prueba():
    X, y = make_classification(
        n_samples=14000,
        n_features=2,
        class_sep=0.5,
        n_informative = 2,
        n_redundant = 0
    )

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("make_classification")
    plt.show()

if __name__ == "__main__":
    prueba()
