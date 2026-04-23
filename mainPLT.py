import matplotlib.pyplot as plt
import os
import pandas as pd

def load_from_results(base_path):
    df_median = pd.read_csv(os.path.join(base_path, "median_results.csv"), index_col=0)
    df_p25 = pd.read_csv(os.path.join(base_path, "p25_results.csv"), index_col=0)
    df_p75 = pd.read_csv(os.path.join(base_path, "p75_results.csv"), index_col=0)
    return df_median, df_p25, df_p75

def plot_three_pruning(df_median, df_p25, df_p75, title, save_path):
    x = 1 - df_median.index.astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    configs = [
        ("smallest", "smallest_sign", "Smallest"),
        ("random", "random_sign", "Random"),
        ("largest", "largest_sign", "Largest"),
    ]

    for ax, (col, col_sign, name) in zip(axes, configs):

        ax.plot(x, df_median[col], label=col)
        ax.fill_between(x, df_p25[col], df_p75[col], alpha=0.2)

        ax.plot(x, df_median[col_sign], "--", label=f"{col} (sign)")
        ax.fill_between(x, df_p25[col_sign], df_p75[col_sign], alpha=0.2)

        ax.axhline(df_median["original"].iloc[0], linestyle=":", color="black", label="original")

        ax.set_title(name)
        ax.set_xlabel("fraction removed (p)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_case_w_vs_wo(base_path_w, base_path_wo, title_base, save_dir):

    df_median_w, df_p25_w, df_p75_w = load_from_results(base_path_w)
    df_median_wo, df_p25_wo, df_p75_wo = load_from_results(base_path_wo)

    plot_three_pruning(
        df_median_w, df_p25_w, df_p75_w,
        title=f"{title_base} (w_es)",
        save_path=os.path.join(save_dir, "w_es.png")
    )

    plot_three_pruning(
        df_median_wo, df_p25_wo, df_p75_wo,
        title=f"{title_base} (wo_es)",
        save_path=os.path.join(save_dir, "wo_es.png")
    )

def plot_all_mnist():
    hidden_sizes = [32, 64, 128]
    difficulties = [[0,7],[7,9]]

    for size in hidden_sizes:
        for d1, d2 in difficulties:

            base_w = f"results/real_datasets/mnist/w_es/dim_{size}/{d1}_{d2}"
            base_wo = f"results/real_datasets/mnist/wo_es/dim_{size}/{d1}_{d2}"

            save_dir = f"figures/real_datasets/mnist/dim_{size}/{d1}_{d2}"
            title = f"MNIST | dim={size} | {d1} vs {d2}"

            plot_case_w_vs_wo(base_w, base_wo, title, save_dir)

def plot_all_fashion():
    hidden_sizes = [32, 64, 128]
    difficulties = [[1,3],[2,3]]

    for size in hidden_sizes:
        for d1, d2 in difficulties:

            base_w = f"results/real_datasets/fashion_mnist/w_es/dim_{size}/{d1}_{d2}"
            base_wo = f"results/real_datasets/fashion_mnist/wo_es/dim_{size}/{d1}_{d2}"

            save_dir = f"figures/real_datasets/fashion_mnist/dim_{size}/{d1}_{d2}"
            title = f"Fashion-MNIST | dim={size} | {d1} vs {d2}"

            plot_case_w_vs_wo(base_w, base_wo, title, save_dir)

def plot_all_linear():
    hidden_sizes = [32, 64, 128]
    clusters = [1.0, 10.0, 100.0]
    inputs = [2, 784]

    for input_n in inputs:
        for size in hidden_sizes:
            for cluster in clusters:

                base_w = f"results/synthetic_datasets/case_linear_{input_n}_version/w_es/dim_{size}/cluster_{cluster}"
                base_wo = f"results/synthetic_datasets/case_linear_{input_n}_version/wo_es/dim_{size}/cluster_{cluster}"

                save_dir = f"figures/synthetic_datasets/case_linear_{input_n}_version/dim_{size}/cluster_{cluster}"
                title = f"Linear ({input_n}) | dim={size} | cluster={cluster}"

                plot_case_w_vs_wo(base_w, base_wo, title, save_dir)


def plot_all_radial():
    hidden_sizes = [32, 64, 128]
    noises = [0.0, 0.05, 0.1, 0.3]

    for size in hidden_sizes:
        for noise in noises:

            base_w = f"results/synthetic_datasets/case_radial/w_es/dim_{size}/noise_{noise}"
            base_wo = f"results/synthetic_datasets/case_radial/wo_es/dim_{size}/noise_{noise}"

            save_dir = f"figures/synthetic_datasets/case_radial/dim_{size}/noise_{noise}"
            title = f"Radial | dim={size} | noise={noise}"

            plot_case_w_vs_wo(base_w, base_wo, title, save_dir)

def plot_all_moons():
    hidden_sizes = [32, 64, 128]
    noises = [0.0, 0.05, 0.1, 0.3]

    for size in hidden_sizes:
        for noise in noises:

            base_w = f"results/synthetic_datasets/case_moons/w_es/dim_{size}/noise_{noise}"
            base_wo = f"results/synthetic_datasets/case_moons/wo_es/dim_{size}/noise_{noise}"

            save_dir = f"figures/synthetic_datasets/case_moons/dim_{size}/noise_{noise}"
            title = f"Moons | dim={size} | noise={noise}"

            plot_case_w_vs_wo(base_w, base_wo, title, save_dir)

def plot_all_classification():
    hidden_sizes = [32, 64, 128]
    noises = [0.0, 0.05, 0.1, 0.3]
    inputs = [2, 784]

    for input_n in inputs:
        for size in hidden_sizes:
            for noise in noises:

                base_w = f"results/synthetic_datasets/case_classification_{input_n}_version/w_es/dim_{size}/noise_{noise}"
                base_wo = f"results/synthetic_datasets/case_classification_{input_n}_version/wo_es/dim_{size}/noise_{noise}"

                save_dir = f"figures/synthetic_datasets/case_classification_{input_n}_version/dim_{size}/noise_{noise}"
                title = f"Classification ({input_n}) | dim={size} | noise={noise}"

                plot_case_w_vs_wo(base_w, base_wo, title, save_dir)

if __name__ == "__main__":

    print("\n===== GENERATING SYNTHETIC PLOTS =====")

    print("\n--- LINEAR ---")
    plot_all_linear()

    print("\n--- RADIAL ---")
    plot_all_radial()

    print("\n--- MOONS ---")
    plot_all_moons()

    print("\n--- CLASSIFICATION ---")
    plot_all_classification()

    print("\n===== DONE =====")