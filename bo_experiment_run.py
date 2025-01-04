import numpy as np
import pickle

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
    EIAcqFunction
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

    @property
    def minimizer(self):
        return 0.23562

    @property
    def alpha(self):
        return 1.0

    def noise_func(self, X):
        X = 20 * X - 10  # [0, 1] -> [-10, 10]
        return 1 / (1 + np.exp(-X))

    def func(self, X):
        return 0.5 * np.sin(X * 20)

    def mv(self, X):
        return self.func(X).squeeze() + (self.alpha * self.noise_func(X).squeeze())

    def __call__(self, X):
        target = self.func(X)
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

    elif acq_f_params["name"] == "EI":
        return EIAcqFunction()
    else:
        raise ValueError(f"Unsupported acquisition function: {acq_f_params['name']}")


def build_acq_opt(acq_optimizer_name, bounds):
    if acq_optimizer_name == "CMAES":
        bo_optimizer = CMAESAcqOptimizer(n_initial=10000, bounds=bounds)
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

    bo_results = {"experiment_config": experiment_case, "results": {}}

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
        bo_results["results"][method_name] = results_repeat

    return bo_results


def plot_results(bo_results, true_f, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    experiment_case = bo_results["experiment_config"]
    print("Plotting results...")
    print(f"Experiment config: {experiment_case}")

    true_opt = true_f.minimum
    n_init = experiment_case["n_init"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3), constrained_layout=True)
    axes = axes.flatten()  # Flatten the axes for easy indexing

    # looping each method
    for method_name, results in bo_results["results"].items():
        simple_regret_list = []
        simple_cum_regret_list = []
        risk_averse_cum_regret_list = []
        risk_averse_regret_list = []

        # looping for repetitions
        for result in results:
            # Mv = f(x) + alpha*f_variance(x)
            # use +alpha*f_variance(x) as lower the better thus higher f_variance(x) is worse

            MVxstar = true_f.mv(result["X"]).squeeze()

            MVtrue_opt = true_opt + (
                true_f.alpha * true_f.noise_func(true_f.minimizer).squeeze()
            )

            # Risk-averse regret
            risk_averse_regret = MVxstar - MVtrue_opt
            risk_averse_regret_list.append(risk_averse_regret)

            # Risk-averse cumulative regret
            risk_averse_cum_regret = np.cumsum(risk_averse_regret)
            risk_averse_cum_regret_list.append(risk_averse_cum_regret)

            # Simple regret f(x) - true_opt
            simple_regret = true_f.func(result["X"]).squeeze() - true_opt
            simple_regret_list.append(simple_regret)

            # Simple cumulative regret f(x) - true_opt
            simple_cum_regret = np.cumsum(simple_regret)
            simple_cum_regret_list.append(simple_cum_regret)

        risk_averse_cum_regret_list = np.array(risk_averse_cum_regret_list)
        risk_averse_cum_regret_mean = np.mean(risk_averse_cum_regret_list, axis=0)[
            n_init - 1 :
        ]
        risk_averse_cum_regret_std = np.std(risk_averse_cum_regret_list, axis=0)[
            n_init - 1 :
        ]
        risk_averse_cum_regret_se = risk_averse_cum_regret_std / np.sqrt(
            risk_averse_cum_regret_list.shape[0]
        )  # Standard error

        axes[0].plot(risk_averse_cum_regret_mean, label=method_name)
        axes[0].fill_between(
            range(len(risk_averse_cum_regret_mean)),
            risk_averse_cum_regret_mean - risk_averse_cum_regret_se,
            risk_averse_cum_regret_mean + risk_averse_cum_regret_se,
            alpha=0.2,
        )

        risk_averse_regret_list = np.array(risk_averse_regret_list)
        risk_averse_regret_mean = np.mean(risk_averse_regret_list, axis=0)[n_init - 1 :]
        risk_averse_regret_std = np.std(risk_averse_regret_list, axis=0)[n_init - 1 :]
        risk_averse_regret_se = risk_averse_regret_std / np.sqrt(
            risk_averse_regret_list.shape[0]
        )

        axes[1].plot(risk_averse_regret_mean, label=method_name)
        axes[1].fill_between(
            range(len(risk_averse_regret_mean)),
            risk_averse_regret_mean - risk_averse_regret_se,
            risk_averse_regret_mean + risk_averse_regret_se,
            alpha=0.2,
        )

        simple_regret_list = np.array(simple_regret_list)
        simple_regret_mean = np.mean(simple_regret_list, axis=0)[n_init - 1 :]
        simple_regret_std = np.std(simple_regret_list, axis=0)[n_init - 1 :]
        simple_regret_se = simple_regret_std / np.sqrt(simple_regret_list.shape[0])

        axes[2].plot(simple_regret_mean, label=method_name)  # Use `ax.plot`
        axes[2].fill_between(
            range(len(simple_regret_mean)),
            simple_regret_mean - simple_regret_se,
            simple_regret_mean + simple_regret_se,
            alpha=0.2,
        )  # Use `ax.fill_between`

        simple_cum_regret_list = np.array(simple_cum_regret_list)
        simple_cum_regret_mean = np.mean(simple_cum_regret_list, axis=0)[n_init - 1 :]
        simple_cum_regret_std = np.std(simple_cum_regret_list, axis=0)[n_init - 1 :]
        simple_cum_regret_se = simple_cum_regret_std / np.sqrt(
            simple_cum_regret_list.shape[0]
        )
        axes[3].plot(simple_cum_regret_mean, label=method_name)
        axes[3].fill_between(
            range(len(simple_cum_regret_mean)),
            simple_cum_regret_mean - simple_cum_regret_se,
            simple_cum_regret_mean + simple_cum_regret_se,
            alpha=0.2,
        )

    axes[0].legend()
    axes[0].set_title("Risk-averse cumulative regret")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Rt (mean ± SE)")

    axes[1].set_title("Risk-averse regret")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("$MV(xt)-MV(x_*)$ (mean ± SE)")

    axes[2].set_title("Simple regret")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("$f(xt)-f(x_*)$ (mean ± SE)")

    axes[3].set_title("Simple cumulative regret")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("cumm regret (mean ± SE)")

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
    parser.add_argument(
        "--result_file",
        type=str,
        default=None,
        help="Path to save the output pickle file (default: None).",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        experiment_case = json.load(f)

    if args.result_file is None:
        bo_results = run_bo_experiments(experiment_case)
        with open("bo_results.pkl", "wb") as f:
            pickle.dump(bo_results, f)
    else:
        with open(args.result_file, "rb") as f:
            bo_results = pickle.load(f)

    true_f = benchmark_f_dict[experiment_case["benchmark_f"]]
    plot_results(bo_results, true_f, save_path=args.output)
