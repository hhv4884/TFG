import os
import pandas as pd
import torch
from pathlib import Path

def save_mnist_results(inicializations, hidden_size):
    print(f"\n=== RUNNING RESULTS (dim={hidden_size} ===")
    dfs_wo = []
    dfs_w = []
    difficulties = [[0,7],[7,9]]
    for j in difficulties:
        difficulty1 = j[0]
        difficulty2 = j[1]
        for i in range(inicializations):
            print(f"Inicialization = {i}, difficulty 1 = {difficulty1}, difficulty 2 = {difficulty2}")
            path_w = f"training/real_datasets/mnist/wo_es/dim_{hidden_size}/{difficulty1}_{difficulty2}"
            path_wo = f"training/real_datasets/mnist/wo_es/dim_{hidden_size}/{difficulty1}_{difficulty2}"
            og_path = Path(__file__).parent
            final_path_w = og_path/path_w/f"i_{i}"/ "results_log.csv"
            final_path_wo = og_path/path_wo/f"i_{i}"/ "results_log.csv"
            df_w = pd.read_csv(final_path_w)
            df_wo = pd.read_csv(final_path_wo)
            dfs_wo.append(df_wo)
            dfs_w.append(df_w)

        df_total_w = pd.concat(dfs_w, keys=range(len(dfs_w)), names=["model", "row"])
        df_total_w = df_total_w.reset_index(level="model")
        # print(df_total_w["run"].unique())

        df_total_wo = pd.concat(dfs_wo, keys=range(len(dfs_wo)), names=["model", "row"])
        df_total_wo = df_total_wo.reset_index(level="model")
        # print(df_total_wo["run"].unique())

        df_median_w = df_total_w.groupby("fraction").median(numeric_only=True)
        df_median_wo = df_total_wo.groupby("fraction").median(numeric_only=True)

        df_p25_w = df_total_w.groupby("fraction").quantile(0.25, numeric_only=True)
        df_p75_w = df_total_w.groupby("fraction").quantile(0.75, numeric_only=True)

        df_p25_wo = df_total_wo.groupby("fraction").quantile(0.25, numeric_only=True)
        df_p75_wo = df_total_wo.groupby("fraction").quantile(0.75, numeric_only=True)

        base_w = Path(f"results/real_datasets/mnist/w_es/dim_{hidden_size}/{difficulty1}_{difficulty2}")
        base_w.mkdir(parents=True, exist_ok=True)
        df_median_w.to_csv(base_w / "median_results.csv")
        df_p25_w.to_csv(base_w / "p25_results.csv")
        df_p75_w.to_csv(base_w / "p75_results.csv")

        base_wo = Path(f"results/real_datasets/mnist/wo_es/dim_{hidden_size}/{difficulty1}_{difficulty2}")
        base_wo.mkdir(parents=True, exist_ok=True)
        df_median_wo.to_csv(base_wo / "median_results.csv")
        df_p25_wo.to_csv(base_wo / "p25_results.csv")
        df_p75_wo.to_csv(base_wo / "p75_results.csv") 

def case_mnist():
    hidden_sizes = [32, 64, 128]
    inicializations = 15
    for i in hidden_sizes:
        save_mnist_results(inicializations, i)

def save_fashion_mnist_results(inicializations, hidden_size):
    print(f"\n=== RUNNING RESULTS (dim={hidden_size} ===")
    dfs_wo = []
    dfs_w = []
    difficulties = [[1,3],[2,3]]
    for j in difficulties:
        difficulty1 = j[0]
        difficulty2 = j[1]
        for i in range(inicializations):
            print(f"Inicialization = {i}, difficulty 1 = {difficulty1}, difficulty 2 = {difficulty2}")
            path_w = f"training/real_datasets/fashion_mnist/wo_es/dim_{hidden_size}/{difficulty1}_{difficulty2}"
            path_wo = f"training/real_datasets/fashion_mnist/wo_es/dim_{hidden_size}/{difficulty1}_{difficulty2}"
            og_path = Path(__file__).parent
            final_path_w = og_path/path_w/f"i_{i}"/ "results_log.csv"
            final_path_wo = og_path/path_wo/f"i_{i}"/ "results_log.csv"
            df_w = pd.read_csv(final_path_w)
            df_wo = pd.read_csv(final_path_wo)
            dfs_wo.append(df_wo)
            dfs_w.append(df_w)

        df_total_w = pd.concat(dfs_w, keys=range(len(dfs_w)), names=["model", "row"])
        df_total_w = df_total_w.reset_index(level="model")
        # print(df_total_w["run"].unique())

        df_total_wo = pd.concat(dfs_wo, keys=range(len(dfs_wo)), names=["model", "row"])
        df_total_wo = df_total_wo.reset_index(level="model")
        # print(df_total_wo["run"].unique())

        df_median_w = df_total_w.groupby("fraction").median(numeric_only=True)
        df_median_wo = df_total_wo.groupby("fraction").median(numeric_only=True)

        df_p25_w = df_total_w.groupby("fraction").quantile(0.25, numeric_only=True)
        df_p75_w = df_total_w.groupby("fraction").quantile(0.75, numeric_only=True)

        df_p25_wo = df_total_wo.groupby("fraction").quantile(0.25, numeric_only=True)
        df_p75_wo = df_total_wo.groupby("fraction").quantile(0.75, numeric_only=True)

        base_w = Path(f"results/real_datasets/fashion_mnist/w_es/dim_{hidden_size}/{difficulty1}_{difficulty2}")
        base_w.mkdir(parents=True, exist_ok=True)
        df_median_w.to_csv(base_w / "median_results.csv")
        df_p25_w.to_csv(base_w / "p25_results.csv")
        df_p75_w.to_csv(base_w / "p75_results.csv")

        base_wo = Path(f"results/real_datasets/fashion_mnist/wo_es/dim_{hidden_size}/{difficulty1}_{difficulty2}")
        base_wo.mkdir(parents=True, exist_ok=True)
        df_median_wo.to_csv(base_wo / "median_results.csv")
        df_p25_wo.to_csv(base_wo / "p25_results.csv")
        df_p75_wo.to_csv(base_wo / "p75_results.csv") 

def case_fashion_mnist():
    hidden_sizes = [32, 64, 128]
    inicializations = 15
    for i in hidden_sizes:
        save_fashion_mnist_results(inicializations, i)

def save_linear_results(inicializations, input_n, hidden_size, cluster):
    print(f"\n=== RUNNING RESULTS (dim={hidden_size}, cluster={cluster}) ===")
    dfs_wo = []
    dfs_w = []
    for i in range(inicializations):
        print(f"Inicialization = {i}")
        if (input_n == 784):
            path_w = f"training/synthetic_datasets/case_linear_784_version/w_es/dim_{hidden_size}/cluster_{cluster}"
            path_wo = f"training/synthetic_datasets/case_linear_784_version/wo_es/dim_{hidden_size}/cluster_{cluster}"
        else:
            path_w = f"training/synthetic_datasets/case_linear_2_version/w_es/dim_{hidden_size}/cluster_{cluster}"
            path_wo = f"training/synthetic_datasets/case_linear_2_version/wo_es/dim_{hidden_size}/cluster_{cluster}"
        og_path = Path(__file__).parent
        final_path_w = og_path/path_w/f"i_{i}"/ "results_log.csv"
        final_path_wo = og_path/path_wo/f"i_{i}"/ "results_log.csv"
        df_w = pd.read_csv(final_path_w)
        df_wo = pd.read_csv(final_path_wo)
        dfs_wo.append(df_wo)
        dfs_w.append(df_w)

    df_total_w = pd.concat(dfs_w, keys=range(len(dfs_w)), names=["model", "row"])
    df_total_w = df_total_w.reset_index(level="model")
    # print(df_total_w["run"].unique())

    df_total_wo = pd.concat(dfs_wo, keys=range(len(dfs_wo)), names=["model", "row"])
    df_total_wo = df_total_wo.reset_index(level="model")
    # print(df_total_wo["run"].unique())

    df_median_w = df_total_w.groupby("fraction").median(numeric_only=True)
    df_median_wo = df_total_wo.groupby("fraction").median(numeric_only=True)

    df_p25_w = df_total_w.groupby("fraction").quantile(0.25, numeric_only=True)
    df_p75_w = df_total_w.groupby("fraction").quantile(0.75, numeric_only=True)

    df_p25_wo = df_total_wo.groupby("fraction").quantile(0.25, numeric_only=True)
    df_p75_wo = df_total_wo.groupby("fraction").quantile(0.75, numeric_only=True)

    base_w = Path(f"results/synthetic_datasets/case_linear_{input_n}_version/w_es/dim_{hidden_size}/cluster_{cluster}")
    base_w.mkdir(parents=True, exist_ok=True)
    df_median_w.to_csv(base_w / "median_results.csv")
    df_p25_w.to_csv(base_w / "p25_results.csv")
    df_p75_w.to_csv(base_w / "p75_results.csv")

    base_wo = Path(f"results/synthetic_datasets/case_linear_{input_n}_version/wo_es/dim_{hidden_size}/cluster_{cluster}")
    base_wo.mkdir(parents=True, exist_ok=True)
    df_median_wo.to_csv(base_wo / "median_results.csv")
    df_p25_wo.to_csv(base_wo / "p25_results.csv")
    df_p75_wo.to_csv(base_wo / "p75_results.csv") 

def case_linear():
    hidden_sizes = [32, 64, 128]
    clusters = [1.0, 10.0, 100.0]
    inicializations = 15
    inputs = [784, 2]
    for i in hidden_sizes:
        for j in clusters:
            for input_n in inputs:
                save_linear_results(inicializations, input_n, i, j)

def save_radial_results(inicializations, hidden_size, noise):
    print(f"\n=== RUNNING RESULTS (dim={hidden_size}, noise={noise}) ===")
    dfs_wo = []
    dfs_w = []
    for i in range(inicializations):
        print(f"Inicialization = {i}")
        path_w = f"training/synthetic_datasets/case_radial/w_es/dim_{hidden_size}/noise_{noise}"
        path_wo = f"training/synthetic_datasets/case_radial/wo_es/dim_{hidden_size}/noise_{noise}"
        og_path = Path(__file__).parent
        final_path_w = og_path/path_w/f"i_{i}"/ "results_log.csv"
        final_path_wo = og_path/path_wo/f"i_{i}"/ "results_log.csv"
        df_w = pd.read_csv(final_path_w)
        df_wo = pd.read_csv(final_path_wo)
        dfs_wo.append(df_wo)
        dfs_w.append(df_w)

    df_total_w = pd.concat(dfs_w, keys=range(len(dfs_w)), names=["model", "row"])
    df_total_w = df_total_w.reset_index(level="model")
    # print(df_total_w["run"].unique())

    df_total_wo = pd.concat(dfs_wo, keys=range(len(dfs_wo)), names=["model", "row"])
    df_total_wo = df_total_wo.reset_index(level="model")
    # print(df_total_wo["run"].unique())

    df_median_w = df_total_w.groupby("fraction").median(numeric_only=True)
    df_median_wo = df_total_wo.groupby("fraction").median(numeric_only=True)

    df_p25_w = df_total_w.groupby("fraction").quantile(0.25, numeric_only=True)
    df_p75_w = df_total_w.groupby("fraction").quantile(0.75, numeric_only=True)

    df_p25_wo = df_total_wo.groupby("fraction").quantile(0.25, numeric_only=True)
    df_p75_wo = df_total_wo.groupby("fraction").quantile(0.75, numeric_only=True)

    base_w = Path(f"results/synthetic_datasets/case_radial/w_es/dim_{hidden_size}/noise_{noise}")
    base_w.mkdir(parents=True, exist_ok=True)
    df_median_w.to_csv(base_w / "median_results.csv")
    df_p25_w.to_csv(base_w / "p25_results.csv")
    df_p75_w.to_csv(base_w / "p75_results.csv")

    base_wo = Path(f"results/synthetic_datasets/case_radial/wo_es/dim_{hidden_size}/noise_{noise}")
    base_wo.mkdir(parents=True, exist_ok=True)
    df_median_wo.to_csv(base_wo / "median_results.csv")
    df_p25_wo.to_csv(base_wo / "p25_results.csv")
    df_p75_wo.to_csv(base_wo / "p75_results.csv") 

def case_radial():
    hidden_sizes = [32, 64, 128]
    noise = [0.0, 0.05, 0.1, 0.3]
    inicializations = 15
    for i in hidden_sizes:
        for j in noise:
            save_radial_results(inicializations, i, j)

def save_moons_results(inicializations, hidden_size, noise):
    print(f"\n=== RUNNING RESULTS (dim={hidden_size}, noise={noise}) ===")
    dfs_wo = []
    dfs_w = []
    for i in range(inicializations):
        print(f"Inicialization = {i}")
        path_w = f"training/synthetic_datasets/case_moons/w_es/dim_{hidden_size}/noise_{noise}"
        path_wo = f"training/synthetic_datasets/case_moons/wo_es/dim_{hidden_size}/noise_{noise}"
        og_path = Path(__file__).parent
        final_path_w = og_path/path_w/f"i_{i}"/ "results_log.csv"
        final_path_wo = og_path/path_wo/f"i_{i}"/ "results_log.csv"
        df_w = pd.read_csv(final_path_w)
        df_wo = pd.read_csv(final_path_wo)
        dfs_wo.append(df_wo)
        dfs_w.append(df_w)

    df_total_w = pd.concat(dfs_w, keys=range(len(dfs_w)), names=["model", "row"])
    df_total_w = df_total_w.reset_index(level="model")
    # print(df_total_w["run"].unique())

    df_total_wo = pd.concat(dfs_wo, keys=range(len(dfs_wo)), names=["model", "row"])
    df_total_wo = df_total_wo.reset_index(level="model")
    # print(df_total_wo["run"].unique())

    df_median_w = df_total_w.groupby("fraction").median(numeric_only=True)
    df_median_wo = df_total_wo.groupby("fraction").median(numeric_only=True)

    df_p25_w = df_total_w.groupby("fraction").quantile(0.25, numeric_only=True)
    df_p75_w = df_total_w.groupby("fraction").quantile(0.75, numeric_only=True)

    df_p25_wo = df_total_wo.groupby("fraction").quantile(0.25, numeric_only=True)
    df_p75_wo = df_total_wo.groupby("fraction").quantile(0.75, numeric_only=True)

    base_w = Path(f"results/synthetic_datasets/case_moons/w_es/dim_{hidden_size}/noise_{noise}")
    base_w.mkdir(parents=True, exist_ok=True)
    df_median_w.to_csv(base_w / "median_results.csv")
    df_p25_w.to_csv(base_w / "p25_results.csv")
    df_p75_w.to_csv(base_w / "p75_results.csv")

    base_wo = Path(f"results/synthetic_datasets/case_moons/wo_es/dim_{hidden_size}/noise_{noise}")
    base_wo.mkdir(parents=True, exist_ok=True)
    df_median_wo.to_csv(base_wo / "median_results.csv")
    df_p25_wo.to_csv(base_wo / "p25_results.csv")
    df_p75_wo.to_csv(base_wo / "p75_results.csv") 

def case_moons():
    hidden_sizes = [32, 64, 128]
    noise = [0.0, 0.05, 0.1, 0.3]
    inicializations = 15
    for i in hidden_sizes:
        for j in noise:
            save_moons_results(inicializations, i, j)

def save_classification_results(inicializations, input_n, hidden_size, noise):
    print(f"\n=== RUNNING RESULTS (dim={hidden_size}, class_sep={noise}) ===")
    dfs_wo = []
    dfs_w = []
    for i in range(inicializations):
        print(f"Inicialization = {i}")
        if (input_n == 784):
            path_w = f"training/synthetic_datasets/case_classification_784_version/w_es/dim_{hidden_size}/class_sep_{noise}"
            path_wo = f"training/synthetic_datasets/case_classification_784_version/wo_es/dim_{hidden_size}/class_sep_{noise}"
        else:
            path_w = f"training/synthetic_datasets/case_classification_2_version/w_es/dim_{hidden_size}/class_sep_{noise}"
            path_wo = f"training/synthetic_datasets/case_classification_2_version/wo_es/dim_{hidden_size}/class_sep_{noise}"
        og_path = Path(__file__).parent
        final_path_w = og_path/path_w/f"i_{i}"/ "results_log.csv"
        final_path_wo = og_path/path_wo/f"i_{i}"/ "results_log.csv"
        df_w = pd.read_csv(final_path_w)
        df_wo = pd.read_csv(final_path_wo)
        dfs_wo.append(df_wo)
        dfs_w.append(df_w)

    df_total_w = pd.concat(dfs_w, keys=range(len(dfs_w)), names=["model", "row"])
    df_total_w = df_total_w.reset_index(level="model")
    # print(df_total_w["run"].unique())

    df_total_wo = pd.concat(dfs_wo, keys=range(len(dfs_wo)), names=["model", "row"])
    df_total_wo = df_total_wo.reset_index(level="model")
    # print(df_total_wo["run"].unique())

    df_median_w = df_total_w.groupby("fraction").median(numeric_only=True)
    df_median_wo = df_total_wo.groupby("fraction").median(numeric_only=True)

    df_p25_w = df_total_w.groupby("fraction").quantile(0.25, numeric_only=True)
    df_p75_w = df_total_w.groupby("fraction").quantile(0.75, numeric_only=True)

    df_p25_wo = df_total_wo.groupby("fraction").quantile(0.25, numeric_only=True)
    df_p75_wo = df_total_wo.groupby("fraction").quantile(0.75, numeric_only=True)

    base_w = Path(f"results/synthetic_datasets/case_classification_{input_n}_version/w_es/dim_{hidden_size}/class_sep_{noise}")
    base_w.mkdir(parents=True, exist_ok=True)
    df_median_w.to_csv(base_w / "median_results.csv")
    df_p25_w.to_csv(base_w / "p25_results.csv")
    df_p75_w.to_csv(base_w / "p75_results.csv")

    base_wo = Path(f"results/synthetic_datasets/case_classification_{input_n}_version/wo_es/dim_{hidden_size}/class_sep_{noise}")
    base_wo.mkdir(parents=True, exist_ok=True)
    df_median_wo.to_csv(base_wo / "median_results.csv")
    df_p25_wo.to_csv(base_wo / "p25_results.csv")
    df_p75_wo.to_csv(base_wo / "p75_results.csv") 

def case_classification():
    hidden_sizes = [32, 64, 128]
    class_sep = [2.0, 1.0, 0.5]
    inicializations = 15
    inputs = [784, 2]
    for i in hidden_sizes:
        for j in class_sep:
            for input_n in inputs:
                save_classification_results(inicializations, input_n, i, j)

if __name__ == "__main__":
    # case_linear()
    # case_radial()
    # case_moons()
    case_classification()
    # case_mnist()
    # case_fashion_mnist()