from core.load_dataset import load_mnist, load_base, load_fashion_mnist
from core.models import MLP_superior
import torch
from core.training import (train_model_superior_without_early_stop, train_model_superior_with_early_stop)
from core.utils import (compute_medians_linear, compute_medians_radial, compute_medians_moons)
from core.accuracy import (obtein_accuracies)

def train_mnist():
    print("TRAINING MODEL -> MNIST")
    hidden_size = [32, 64, 128]
    classes = [[0,7],[7,9]]
    early_stop = None
    inicializations = 15
    for size in hidden_size:
        for i in range(inicializations):
            for difficulty in classes:
                difficulty1 = difficulty[0]
                difficulty2 = difficulty[1]
                train_loader, test_loader = load_mnist(batch_size=128, classes=[difficulty1, difficulty2])
                for early_stop in [True, False]:
                    if early_stop is False:
                        path = f'training/superior_mnist/wo_es/dim_{size}/{difficulty1}_{difficulty2}/i_{i}'
                        print(path)
                    if early_stop is True:
                        path = f'training/superior_mnist/w_es/dim_{size}/{difficulty1}_{difficulty2}/i_{i}'
                        print(path)
                    model = MLP_superior(784, [size], 2)
                    model.to(device)
                    if early_stop is True:
                        train_model_superior_with_early_stop(model, train_loader, test_loader, path, device)
                    else:
                        train_model_superior_without_early_stop(model, train_loader, test_loader, path, device)

def train_fashion_mnist():
    print("TRAINING MODEL -> FASHION MNIST")
    hidden_size = [32, 64, 128]
    classes = [[2, 3],[1, 3]]
    early_stop = None
    inicializations = 15
    for size in hidden_size:
        for i in range(inicializations):
            for difficulty in classes:
                difficulty1 = difficulty[0]
                difficulty2 = difficulty[1]
                train_loader, test_loader = load_fashion_mnist(batch_size=128, classes=[difficulty1, difficulty2])
                for early_stop in [True, False]:
                    if early_stop is False:
                        path = f'training/superior_fashion_mnist/wo_es/dim_{size}/{difficulty1}_{difficulty2}/i_{i}'
                    if early_stop is True:
                        path = f'training/superior_fashion_mnist/w_es/dim_{size}/{difficulty1}_{difficulty2}/i_{i}'
                    model = MLP_superior(784, [size], 2)
                    model.to(device)
                    if early_stop is True:
                        train_model_superior_with_early_stop(model, train_loader, test_loader, path, device)
                    else:
                        train_model_superior_without_early_stop(model, train_loader, test_loader, path, device)

def train_case_linear_784_version():
    print("TRAINING MODEL -> Case - linear")
    hidden_size = [32, 64, 128]
    clusters = [1.0, 10.0, 100.0]
    early_stop = None
    inicializations = 15
    m_samples = 14000
    input_n = 784
    for size in hidden_size:
        for i in range(inicializations):
            for cluster in clusters:
                print(f"init {i}, cluster {cluster}, train_case_linear_784_version")
                train_loader, test_loader = load_base(
                    task="linear",
                    n_samples=m_samples,
                    cluster_number=cluster,
                    noise=0.0,
                    batch_size=128,
                    n_features=input_n
                )
                for early_stop in [True, False]:
                    if early_stop is False:
                        path = f'training/synthetic_datasets/case_linear_784_version_14000/wo_es/dim_{size}/cluster_{cluster}/i_{i}'
                    if early_stop is True:
                        path = f'training/synthetic_datasets/case_linear_784_version_14000/w_es/dim_{size}/cluster_{cluster}/i_{i}'
                    model = MLP_superior(input_n, [size], 2)
                    model.to(device)
                    if early_stop is True:
                        train_model_superior_with_early_stop(model, train_loader, test_loader, path, device)
                        obtein_accuracies(train_loader, test_loader, path, input_n, size, device)
                    else:
                        train_model_superior_without_early_stop(model, train_loader, test_loader, path, device)
                        obtein_accuracies(train_loader, test_loader, path, input_n, size, device)

def train_case_linear_2_version():
    print("TRAINING MODEL -> Case - linear")
    hidden_size = [32, 64, 128]
    clusters = [1.0, 10.0, 100.0]
    early_stop = None
    inicializations = 15
    m_samples = 14000
    input_n = 2
    for size in hidden_size:
        for i in range(inicializations):
            for cluster in clusters:
                print(f"init {i}, cluster {cluster}, train_case_linear_2_version")
                train_loader, test_loader = load_base(
                    task="linear",
                    n_samples=m_samples,
                    cluster_number=cluster,
                    noise=0.0,
                    batch_size=128,
                    n_features=input_n
                )
                for early_stop in [True, False]:
                    if early_stop is False:
                        path = f'training/synthetic_datasets/case_linear_2_version/wo_es/dim_{size}/cluster_{cluster}/i_{i}'
                    if early_stop is True:
                        path = f'training/synthetic_datasets/case_linear_2_version/w_es/dim_{size}/cluster_{cluster}/i_{i}'
                    model = MLP_superior(input_n, [size], 2)
                    model.to(device)
                    if early_stop is True:
                        train_model_superior_with_early_stop(model, train_loader, test_loader, path, device)
                        obtein_accuracies(train_loader, test_loader, path, input_n, size, device)
                    else:
                        train_model_superior_without_early_stop(model, train_loader, test_loader, path, device)
                        obtein_accuracies(train_loader, test_loader, path, input_n, size, device)

def train_case_radial():
    print("TRAINING MODEL -> Case - radial")
    hidden_size = [32, 64, 128]
    noises = [0.0, 0.05, 0.1, 0.3]
    early_stop = None
    inicializations = 15
    m_samples = 14000
    input_n = 2
    for size in hidden_size:
        for i in range(inicializations):
            for noise in noises:
                print(f"init {i}, noise {noise}, train_case_radial")
                train_loader, test_loader = load_base(
                    task="radial",
                    n_samples=m_samples,
                    cluster_number=None,
                    noise=noise,
                    batch_size=128,
                    n_features=input_n
                )
                for early_stop in [True, False]:
                    if early_stop is False:
                        path = f'training/synthetic_datasets/case_radial/wo_es/dim_{size}/noise_{noise}/i_{i}'
                    if early_stop is True:
                        path = f'training/synthetic_datasets/case_radial/w_es/dim_{size}/noise_{noise}/i_{i}'
                    model = MLP_superior(input_n, [size], 2)
                    model.to(device)
                    if early_stop is True:
                        train_model_superior_with_early_stop(model, train_loader, test_loader, path, device)
                        obtein_accuracies(train_loader, test_loader, path, input_n, size, device)
                    else:
                        train_model_superior_without_early_stop(model, train_loader, test_loader, path, device)
                        obtein_accuracies(train_loader, test_loader, path, input_n, size, device)

def train_case_moons():
    print("TRAINING MODEL -> Case - moons")
    hidden_size = [32, 64, 128]
    noises = [0.0, 0.05, 0.1, 0.3]
    early_stop = None
    inicializations = 15
    m_samples = 14000
    input_n = 2
    for size in hidden_size:
        for i in range(inicializations):
            for noise in noises:
                print(f"init {i}, noise {noise}, train_case_moons")
                train_loader, test_loader = load_base(
                    task="non_linear_non_radial",
                    n_samples=m_samples,
                    cluster_number=None,
                    noise=noise,
                    batch_size=128,
                    n_features=input_n
                )
                for early_stop in [True, False]:
                    if early_stop is False:
                        path = f'training/synthetic_datasets/case_moons/wo_es/dim_{size}/noise_{noise}/i_{i}'
                    if early_stop is True:
                        path = f'training/synthetic_datasets/case_moons/w_es/dim_{size}/noise_{noise}/i_{i}'
                    model = MLP_superior(input_n, [size], 2)
                    model.to(device)
                    if early_stop is True:
                        train_model_superior_with_early_stop(model, train_loader, test_loader, path, device)
                        obtein_accuracies(train_loader, test_loader, path, input_n, size, device)
                    else:
                        train_model_superior_without_early_stop(model, train_loader, test_loader, path, device)
                        obtein_accuracies(train_loader, test_loader, path, input_n, size, device)

def train_case_classification_784_version():
    print("TRAINING MODEL -> Case - classification")
    hidden_sizes = [32, 64, 128]
    class_sep = [2.0, 1.0, 0.5]
    early_stop = None
    inicializations = 15
    m_samples = 14000
    input_n = 784
    for size in hidden_sizes:
        for i in range(inicializations):
            for sep in class_sep:
                print(f"dim {size}, init {i}, class_sep {sep}, train_case_classification_784_version")
                train_loader, test_loader = load_base(
                    task="classification",
                    n_samples=m_samples,
                    class_sep=sep,
                    batch_size=128,
                    n_features=input_n
                )
                for early_stop in [True, False]:
                    if early_stop:
                        path = f"training/synthetic_datasets/case_classification_784_version/w_es/dim_{size}/class_sep_{sep}/i_{i}"
                    else:
                        path = f"training/synthetic_datasets/case_classification_784_version/wo_es/dim_{size}/class_sep_{sep}/i_{i}"
                    model = MLP_superior(input_n, [size], 2)
                    model.to(device)
                    if early_stop:
                        train_model_superior_with_early_stop(model, train_loader, test_loader, path, device)
                        obtein_accuracies(train_loader, test_loader, path, input_n, size, device)
                    else:
                        train_model_superior_without_early_stop(model, train_loader, test_loader, path, device)
                        obtein_accuracies(train_loader, test_loader, path, input_n, size, device)

def train_case_classification_2_version():
    print("TRAINING MODEL -> Case - classification")
    hidden_sizes = [32, 64, 128]
    class_sep = [0.5, 1.0, 2.0]
    early_stop = None
    inicializations = 15
    m_samples = 14000
    input_n = 2
    for size in hidden_sizes:
        for i in range(inicializations):
            for sep in class_sep:
                print(f"dim {size}, init {i}, sep {sep}, train_case_classification_784_version")
                train_loader, test_loader = load_base(
                    task="classification",
                    n_samples=m_samples,
                    class_sep=sep,
                    batch_size=128,
                    n_features=input_n
                )
                for early_stop in [True, False]:
                    if early_stop:
                        path = f"training/synthetic_datasets/case_classification_2_version/w_es/dim_{size}/class_sep_{sep}/i_{i}"
                    else:
                        path = f"training/synthetic_datasets/case_classification_2_version/wo_es/dim_{size}/class_sep_{sep}/i_{i}"
                    model = MLP_superior(input_n, [size], 2)
                    model.to(device)
                    if early_stop:
                        train_model_superior_with_early_stop(model, train_loader, test_loader, path, device)
                        obtein_accuracies(train_loader, test_loader, path, input_n, size, device)
                    else:
                        train_model_superior_without_early_stop(model, train_loader, test_loader, path, device)
                        obtein_accuracies(train_loader, test_loader, path, input_n, size, device)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Hello! Using {device} device :)')
    # train_mnist()
    # train_fashion_mnist()
    # train_case_linear_784_version()
    # train_case_linear_2_version()
    # train_case_radial()
    # train_case_moons()
    train_case_classification_784_version()
    train_case_classification_2_version()