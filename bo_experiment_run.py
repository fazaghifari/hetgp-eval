import numpy as np

from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from src.mlhgp import MLHGP
from src.imlhgp import IMLHGP
from src.nnpehgp import NNPEHGP
from src.ksmlhgp import KSMLHGP
from src.ksimlhgp import KSIMLHGP

from src.bo import (
    BayesianOptimizer,
    AugmentedEIAcqFunction,
    LCBAcqFunction,
    CMAESAcqOptimizer,
    RandomAcqOptimizer,
    RAHBOAcqFunction,
    ANPEIAcqFunction,
)

## SUPPRESS ALL WARNINGS
import warnings

warnings.filterwarnings("ignore")


class F1Toy:
    """
    Toy function used in RAHBO paper
    """

    def __init__(self, add_noise=True):
        self.add_noise = add_noise

    @property
    def bounds(self):
        return np.array([[0], [1]])

    @property
    def minimum(self):
        return -0.5

    def noise_func(self, X):
        X = 20 * X - 10  # [0, 1] -> [-10, 10]
        return 1 / (1 + np.exp(-X)) / 10

    def __call__(self, X):
        target = 0.5 * np.sin(X * 20)
        if self.add_noise:
            rng = np.random.RandomState(1)
            target += rng.normal(
                np.zeros_like(X), self.noise_func(X), size=target.shape
            )
        return target.squeeze()


benchmark_f_dict = {"f1_toy": F1Toy()}


def build_model(model_name):
    base_kernel = ConstantKernel(1.0) * RBF(
        length_scale=0.1, length_scale_bounds=(0.01, 100)
    ) + WhiteKernel(noise_level=1, noise_level_bounds=(0.001, 10))

    noise_kernel = ConstantKernel(1.0) * RBF(
        length_scale=0.1, length_scale_bounds=(0.01, 100)
    ) + WhiteKernel(noise_level=1, noise_level_bounds=(0.001, 10))

    gp = GaussianProcessRegressor(kernel=base_kernel, n_restarts_optimizer=25)
    gp_noise = GaussianProcessRegressor(kernel=noise_kernel, n_restarts_optimizer=25)

    model_mapping = {
        "GP": gp,
        "KSMLHGP": KSMLHGP(model=gp, model_noise=gp_noise),
        "KSIMLHGP": KSIMLHGP(model=gp, model_noise=gp_noise),
        "MLHGP": MLHGP(model=gp, model_noise=gp_noise),
        "IMLHGP": IMLHGP(model=gp, model_noise=gp_noise),
        "NNPEHGP": NNPEHGP(model=gp, model_noise=gp_noise),
    }

    if model_name not in model_mapping:
        raise ValueError(
            f"Unsupported model name: {model_name}. Supported models are: {list(model_mapping.keys())}"
        )

    return model_mapping[model_name]


def build_acq_f(acq_f_params):
    if acq_f_params["name"] == "RAHBO":
        return RAHBOAcqFunction(
            alpha=acq_f_params["params"]["alpha"], beta=acq_f_params["params"]["beta"]
        )
    elif acq_f_params["name"] == "LCB":
        return LCBAcqFunction(beta=acq_f_params["params"]["beta"])

    elif acq_f_params["name"] == "AugmentedEI":
        return AugmentedEIAcqFunction()

    elif acq_f_params["name"] == "ANPEI":
        return ANPEIAcqFunction(alpha=acq_f_params["params"]["alpha"])

    else:
        raise ValueError(f"Unsupported acquisition function: {acq_f_params['name']}")


def build_acq_opt(acq_optimizer_name, bounds):
    if acq_optimizer_name == "CMAES":
        bo_optimizer = CMAESAcqOptimizer(n_initial=100, bounds=bounds)
    elif acq_optimizer_name == "Random":
        bo_optimizer = RandomAcqOptimizer(n_candidates=10000, bounds=bounds)
    return bo_optimizer


def run_bo_experiments(experiment_case):
    benchmark_f = benchmark_f_dict[experiment_case["benchmark_f"]]
    n_init = experiment_case["n_init"]
    n_iters = experiment_case["n_iters"]
    n_repeats = experiment_case["n_repeats"]

    init_datasets = []
    bounds = benchmark_f.bounds
    for _ in range(n_repeats):
        # sample initial observations for each repeat
        x_init = np.random.uniform(bounds[0], bounds[1], (n_init, bounds.shape[1]))
        y_init = benchmark_f(x_init)
        init_dataset = {"X": x_init, "y": y_init}
        init_datasets.append(init_dataset)

    bo_results = {}
    for method_name in experiment_case["methods"]:
        running_method = experiment_case["methods"][method_name]
        print("=====================================")
        if experiment_case["verbose"]:
            print(f"Running BO for method: {method_name}")
            print(f"Model: {running_method['model']}")
            print(f"Acquisition function: {running_method['acq_f']}")
            print(f"Acquisition optimizer: {running_method['bo_optimizer']}")

        results_repeat = []
        # Run BO for each repeat
        for i in range(n_repeats):
            if experiment_case["verbose"]:
                print(f"Running rep {i+1}/{n_repeats} of {method_name}")

            init_dataset = init_datasets[i]
            model = build_model(running_method["model"])
            acq_f = build_acq_f(running_method["acq_f"])
            bo_optimizer = build_acq_opt(running_method["bo_optimizer"], bounds)

            bo = BayesianOptimizer(
                model=model,
                acquisition_function=acq_f,
                true_function=benchmark_f,
                acq_optimizer=bo_optimizer,
                verbose=experiment_case["verbose"],
            )

            # Run BO
            result = bo.optimize(init_dataset, n_iter=n_iters)

            if experiment_case["verbose"]:
                print(f"\nResults for repeat {i}")
                print(result)
                best_idx = np.argmin(result["y"])
                print(
                    f"Best observation - X: {result['X'][best_idx]}, y: {result['y'][best_idx]}"
                )

            results_repeat.append(result)
        bo_results[method_name] = results_repeat

    return bo_results


def plot_results(bo_results, experiment_case, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    n_init = experiment_case["n_init"]

    fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axis

    for method_name, results in bo_results.items():
        cum_best_list = []
        for result in results:
            cum_best = np.minimum.accumulate(result["y"])
            cum_best_list.append(cum_best)

        cum_best_list = np.array(cum_best_list)
        mean = np.mean(cum_best_list, axis=0)[n_init - 1 :]
        std = np.std(cum_best_list, axis=0)[n_init - 1 :]
        ax.plot(mean, label=method_name)  # Use `ax.plot`
        ax.fill_between(
            range(len(mean)), mean - std, mean + std, alpha=0.2
        )  # Use `ax.fill_between`

    ax.legend()
    ax.set_title("Simple regrets")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best value")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")  # Use `fig.savefig`

    plt.show()  # Show the figure


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run BO experiments and save results.")
    parser.add_argument(
        "--config",
        type=str,
        default="bo_experiments_config.json",
        help="Path to the JSON configuration file (default: bo_experiments_config.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bo_results.png",
        help="Path to save the output PNG file (default: bo_results.png).",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        experiment_case = json.load(f)

    bo_results = run_bo_experiments(experiment_case)
    plot_results(bo_results, experiment_case, save_path=args.output)
