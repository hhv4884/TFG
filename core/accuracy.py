import os
from core.utils import (compute_accuracy, get_epoch_from_folder, load_model_from_txt)
from core.models import MLP_superior

def obtein_accuracies(train_loader, test_loader, base_path, input, hidden_size, device):
    fractions = [1, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01, 0.005, 0.001]
    os.makedirs(base_path, exist_ok=True)
    log_path = os.path.join(base_path, "results_log.csv")
    with open(log_path, "w") as log_file:
        log_file.write(
            "fraction,original,smallest,smallest_sign,random,random_sign,largest,largest_sign\n"
        )
        print(f"Processing {base_path}")
        model = MLP_superior(input, [hidden_size], 2)
        epoch = get_epoch_from_folder(base_path)
        model = load_model_from_txt(model, base_path, epoch, device)
        original_acc = compute_accuracy(model, test_loader, device, forward_method="base")

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