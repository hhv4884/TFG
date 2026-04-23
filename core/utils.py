import torch
from .new_dataset import NewDataset
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def save_logs(epoch, path):
    os.makedirs(path, exist_ok=True)
    log_file = f'{path}/log_epoch_{epoch}.txt'
    log_file = open(log_file, "w")
    return log_file

def clean_data(data, classes):
    # print(len(train_data)) -> 469, 469*128 = 60.032
    # print(len(test_data)) -> 79, 79*128 = 10.112
    # train_data -> DataLoader: map-style dataset
    prueba = NewDataset(data, classes)
    print(prueba.__len__())
    return NewDataset(data, classes)

# d e b u g
def get_epoch_from_folder(path):
    files = os.listdir(path)
    for f in files:
        match = re.search(r"epoch_(\d+)_layer\d+_edgelist\.txt", f)
        if match:
            return int(match.group(1))
    raise ValueError(f"No se encontraron archivos de epoch en {path}")

def extract_test_acc_from_log(log_path):
    with open(log_path, 'r') as f:
        line = f.read()
    match = re.search(r"Test Acc: ([\d\.]+)%", line)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"No se pudo extraer Test Acc de {log_path}")

def compute_medians_linear(base_path):
    results = {}

    for es_folder in os.listdir(base_path):  # w_es_14000, wo_es_14000
        es_path = os.path.join(base_path, es_folder)
        if not os.path.isdir(es_path):
            continue

        results[es_folder] = {}

        for dim_folder in os.listdir(es_path):  # dim_32, dim_64...
            dim_path = os.path.join(es_path, dim_folder)
            if not os.path.isdir(dim_path):
                continue

            for cluster_folder in os.listdir(dim_path):  # cluster_1.0
                cluster_path = os.path.join(dim_path, cluster_folder)
                if not os.path.isdir(cluster_path):
                    continue

                match = re.search(r"cluster_([\d\.]+)", cluster_folder)
                if not match:
                    continue

                cluster = match.group(1)
                accs = []

                for run_folder in os.listdir(cluster_path):  # i_0, i_1...
                    run_path = os.path.join(cluster_path, run_folder)

                    if not run_folder.startswith("i_"):
                        continue

                    try:
                        epoch = get_epoch_from_folder(run_path)
                        log_file = os.path.join(run_path, f"log_epoch_{epoch}.txt")

                        if not os.path.exists(log_file):
                            continue

                        acc = extract_test_acc_from_log(log_file)
                        accs.append(acc)

                    except Exception as e:
                        print(f"Error en {run_path}: {e}")
                        continue

                if len(accs) > 0:
                    results[es_folder][cluster] = np.median(accs)
                else:
                    results[es_folder][cluster] = None

    return results

def compute_medians_radial(base_path):
    results = {}
    for dim_folder in os.listdir(base_path):
        dim_path = os.path.join(base_path, dim_folder)
        if not os.path.isdir(dim_path):
            continue
        results[dim_folder] = {}
        for noise_folder in os.listdir(dim_path):
            noise_path = os.path.join(dim_path, noise_folder)
            if not os.path.isdir(noise_path):
                continue
            match_noise = re.search(r"noise_([\d\.]+)", noise_folder)
            if not match_noise:
                continue
            noise = match_noise.group(1)
            results[dim_folder][noise] = []
            for folder in os.listdir(noise_path):
                folder_path = os.path.join(noise_path, folder)
                if not os.path.isdir(folder_path):
                    continue
                if not folder.startswith("i_"):
                    continue
                try:
                    best_epoch = get_epoch_from_folder(folder_path)
                    log_file = os.path.join(folder_path, f"log_epoch_{best_epoch}.txt")
                    if not os.path.exists(log_file):
                        continue
                    acc = extract_test_acc_from_log(log_file)
                    results[dim_folder][noise].append(acc)
                except Exception as e:
                    print(f"Error en {folder_path}: {e}")
                    continue
        for noise in results[dim_folder]:
            accs = results[dim_folder][noise]
            if len(accs) > 0:
                results[dim_folder][noise] = np.median(accs)
            else:
                results[dim_folder][noise] = None
    return results

def compute_medians_moons(base_path):
    results = {}
    for dim_folder in os.listdir(base_path):
        dim_path = os.path.join(base_path, dim_folder)
        if not os.path.isdir(dim_path):
            continue
        results[dim_folder] = {}
        for folder in os.listdir(dim_path):
            folder_path = os.path.join(dim_path, folder)
            if not os.path.isdir(folder_path):
                continue
            match = re.search(r"noise_([\d\.]+)_i_(\d+)", folder)
            if not match:
                continue
            noise = match.group(1)
            try:
                best_epoch = get_epoch_from_folder(folder_path)
                log_file = os.path.join(folder_path, f"log_epoch_{best_epoch}.txt")
                if not os.path.exists(log_file):
                    continue
                acc = extract_test_acc_from_log(log_file)
                if noise not in results[dim_folder]:
                    results[dim_folder][noise] = []
                results[dim_folder][noise].append(acc)
            except Exception as e:
                print(f"Error en {folder_path}: {e}")
                continue
        for noise in results[dim_folder]:
            accs = results[dim_folder][noise]
            if len(accs) > 0:
                results[dim_folder][noise] = np.median(accs)
            else:
                results[dim_folder][noise] = None
    return results

def load_model_from_txt(model, path, epoch, device):

    def load_weights(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        header = lines[0]
        match = re.search(r'origin neurons: (\d+) \|\| number target neurons: (\d+)', header)
        in_features = int(match.group(1))
        out_features = int(match.group(2))
        weight_matrix = torch.zeros((out_features, in_features))
        for line in lines[1:]:
            match = re.search(r"L\d+_(\d+) L\d+_(\d+) -> ([\-\d\.eE]+)", line)
            if match:
                i = int(match.group(1))
                j = int(match.group(2))
                value = float(match.group(3))
                weight_matrix[j, i] = value
        return weight_matrix

    def load_bias(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        size = int(re.search(r'number of biases: (\d+)', lines[0]).group(1))
        bias = torch.zeros(size)
        for line in lines[1:]:
            match = re.search(r"L\d+_(\d+) -> ([\-\d\.eE]+)", line)
            if match:
                i = int(match.group(1))
                value = float(match.group(2))
                bias[i] = value
        return bias

    w0_path = os.path.join(path, f"epoch_{epoch}_layer0_edgelist.txt")
    b0_path = os.path.join(path, f"epoch_{epoch}_layer0_biaslist.txt")
    w1_path = os.path.join(path, f"epoch_{epoch}_layer1_edgelist.txt")
    b1_path = os.path.join(path, f"epoch_{epoch}_layer1_biaslist.txt")
    W0 = load_weights(w0_path)
    b0 = load_bias(b0_path)
    W1 = load_weights(w1_path)
    b1 = load_bias(b1_path)
    

    with torch.no_grad():
        model.net.input_layer.weight.copy_(W0)
        model.net.input_layer.bias.copy_(b0)
        model.net.last_layer.weight.copy_(W1)
        model.net.last_layer.bias.copy_(b1)

    model.to(device)
    model.eval()
    return model

def compute_accuracy(model, data_loader, device, forward_method="base",
                     threshold_method='smallest', fraction_non_zero=1.0,
                     to_signs=False):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if forward_method == "base":
                outputs = model(images)
            elif forward_method == "threshold":
                '''
                outputs, _ = model.threshold_weight_forward(
                    images,
                    method=threshold_method,
                    fraction_non_zero=fraction_non_zero,
                    to_signs=to_signs
                )
                '''
                outputs = model.threshold_weight_forward(
                    images,
                    method=threshold_method,
                    fraction_non_zero=fraction_non_zero,
                    to_signs=to_signs
                )
            else:
                raise ValueError(f"Unknown forward method: {forward_method}")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy