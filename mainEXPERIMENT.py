import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from core.models import MLP_superior
from core.load_dataset import load_mnist, load_fashion_mnist
from core.utils import get_epoch_from_folder, load_model_from_txt, compute_accuracy

def run_experiment(problem, base_path, hidden_size, device, difficulty1, difficulty2):
    fractions = [1, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01, 0.005, 0.001]
    os.makedirs(base_path, exist_ok=True)
    log_path = os.path.join(base_path, "results_log.csv")
    if problem == "mnist":
        _, test_loader = load_mnist(batch_size=128, classes=[difficulty1, difficulty2])
    else:
        _, test_loader = load_fashion_mnist(batch_size=128, classes=[difficulty1, difficulty2])

    folders = [f for f in os.listdir(base_path) if f.startswith("i_")]
    folders.sort(key=lambda x: int(x.split("_")[1]))
    print(f"Processing {base_path} ({len(folders)} models)")

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        log_path = os.path.join(folder_path, "results_log.csv")

        try:
            model = MLP_superior(784, [hidden_size], 2)
            epoch = get_epoch_from_folder(folder_path)
            model = load_model_from_txt(model, folder_path, epoch, device)

            original_acc = compute_accuracy(model, test_loader, device, forward_method="base")

            with open(log_path, "w") as log_file:
                log_file.write(
                    "fraction,original,smallest,smallest_sign,random,random_sign,largest,largest_sign\n"
                )

                for fraction in fractions:
                    acc_smallest = compute_accuracy(
                        model, test_loader, device,
                        forward_method="threshold",
                        threshold_method='smallest',
                        fraction_non_zero=fraction,
                        to_signs=False
                    )
                    acc_smallest_sign = compute_accuracy(
                        model, test_loader, device,
                        forward_method="threshold",
                        threshold_method='smallest',
                        fraction_non_zero=fraction,
                        to_signs=True
                    )
                    acc_random = compute_accuracy(
                        model, test_loader, device,
                        forward_method="threshold",
                        threshold_method='random',
                        fraction_non_zero=fraction,
                        to_signs=False
                    )
                    acc_random_sign = compute_accuracy(
                        model, test_loader, device,
                        forward_method="threshold",
                        threshold_method='random',
                        fraction_non_zero=fraction,
                        to_signs=True
                    )
                    acc_largest = compute_accuracy(
                        model, test_loader, device,
                        forward_method="threshold",
                        threshold_method='highest',
                        fraction_non_zero=fraction,
                        to_signs=False
                    )
                    acc_largest_sign = compute_accuracy(
                        model, test_loader, device,
                        forward_method="threshold",
                        threshold_method='highest',
                        fraction_non_zero=fraction,
                        to_signs=True
                    )

                    log_file.write(
                        f"{fraction},{original_acc},{acc_smallest},{acc_smallest_sign},"
                        f"{acc_random},{acc_random_sign},{acc_largest},{acc_largest_sign}\n"
                    )

        except Exception as e:
            print(f"Error en {folder_path}: {e}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for es in ["w", "wo"]:
        for size in [32,64,128]:
            for difficulty in [[0,7],[7,9]]:
                base_path = f"training/real_datasets/mnist/{es}_es/dim_{size}/{difficulty[0]}_{difficulty[1]}"
                run_experiment("mnist",base_path,size, device, difficulty[0], difficulty[1])

    for es in ["w", "wo"]:
        for size in [32,64,128]:
            for difficulty in [[1,3],[2,3]]:
                base_path = f"training/real_datasets/fashion_mnist/{es}_es/dim_{size}/{difficulty[0]}_{difficulty[1]}"
                run_experiment("fashion-mnist",base_path,size, device, difficulty[0],difficulty[1])
