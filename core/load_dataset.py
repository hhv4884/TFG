from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from .utils import clean_data
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def load_mnist(batch_size=128, classes = [0,1,2,3,4,5,6,7,8,9]):
    # operation out of place // in-place
    # por qué 0.5? -> normalize: (x-mean)/std
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="../data", train=True, download=True, transform=transform)
    clean_train_dataset = clean_data(train_dataset, classes)
    train_loader = torch.utils.data.DataLoader(dataset=clean_train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(root="../data", train=False, download=True, transform=transform)
    clean_test_dataset = clean_data(test_dataset, classes)
    test_loader = torch.utils.data.DataLoader(dataset=clean_test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_fashion_mnist(batch_size=128, classes = [0,1,2,3,4,5,6,7,8,9]):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.FashionMNIST(root="../data", train=True, download=True, transform=transform)
    labels = set()
    # for i, label in train_dataset:
    #     print(f'i {i}')
    # imagen
    #     labels.add(label)
    # print(f'Final labels: {labels}')
    # Final labels: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    # => FINAL:
    # 0 = T-shirt/top
    # 1 = Trouser
    # 2 = Pullover
    # 3 = Dress
    # 4 =  Coat
    # 5 = Sandal
    # 6 = Shirt
    # 7 = Sneaker
    # 8 = Bag
    # 9 =  Ankle boot

    # easy = [Dress, Pullover] => [2, 3]
    # hard = [Dres, Trousers] => [1, 3]
    clean_train_dataset = clean_data(train_dataset, classes)
    train_loader = torch.utils.data.DataLoader(dataset=clean_train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.FashionMNIST(root="../data", train=False, download=True, transform=transform)
    clean_test_dataset = clean_data(test_dataset, classes)
    test_loader = torch.utils.data.DataLoader(dataset=clean_test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_base(task, n_samples, class_sep, cluster_number=None, noise=0.0, batch_size=128, n_features=2):

    if task == "linear":
        x, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=2,
            cluster_std=cluster_number
        )

    elif task == "radial":
        x, y = make_circles(
            n_samples=n_samples,
            noise=noise
        )

    elif task == "non_linear_non_radial":
        x, y = make_moons(
            n_samples=n_samples,
            noise=noise
        )

    elif task == "classification":
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            class_sep=class_sep,
            n_informative = n_features,
            n_redundant = 0
        )

    else:
        raise ValueError("unknown task")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        shuffle=True
    )

    x_train = torch.FloatTensor(x_train)
    y_train = torch.tensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.tensor(y_test)
    print("Etiquetas de test:", y_test.tolist())
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader