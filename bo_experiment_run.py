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

    @property
    def minimizer(self):
        return 0.2356

    @property
    def alpha(self):
        return 1.0

    def noise_func(self, X):
        X = 20 * X - 10  # [0, 1] -> [-10, 10]
        return 1 / (1 + np.exp(-X)) / 10

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


def plot_results(bo_results, experiment_case, true_f, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    true_opt = true_f.minimum
    n_init = experiment_case["n_init"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes = axes.flatten()  # Flatten the axes for easy indexing

    # looping each method
    for method_name, results in bo_results.items():
        simple_regret_list = []
        Rt_list = []
        risk_averse_regrets_list = []

        # looping for repetitions
        for result in results:
            # Mv = f(x) + alpha*f_variance(x)
            # use +alpha*f_variance(x) as lower the better thus higher f_variance(x) is worse
            MVxstar = true_f(result["X"]) + (
                true_f.alpha * true_f.noise_func(result["X"]).squeeze()
            )
            MVtrue_opt = true_opt + (
                true_f.alpha * true_f.noise_func(true_f.minimizer).squeeze()
            )

            # Risk-averse cumulative regret
            risk_averse_cum_regrets = np.cumsum(MVxstar - MVtrue_opt)
            Rt_list.append(risk_averse_cum_regrets)

            # Risk-averse regret
            risk_averse_regrets = np.minimum.accumulate(MVxstar - MVtrue_opt)
            risk_averse_regrets_list.append(risk_averse_regrets)

            # Simple regret f(x) - true_opt
            simple_regret = np.minimum.accumulate(
                true_f.func(result["X"]).squeeze() - true_opt
            )
            simple_regret_list.append(simple_regret)

        Rt_list = np.array(Rt_list)
        Rt_mean = np.mean(Rt_list, axis=0)[n_init - 1 :]
        Rt_std = np.std(Rt_list, axis=0)[n_init - 1 :]
        axes[0].plot(Rt_mean, label=method_name)
        axes[0].fill_between(
            range(len(Rt_mean)), Rt_mean - Rt_std, Rt_mean + Rt_std, alpha=0.2
        )

        risk_averse_regrets_list = np.array(risk_averse_regrets_list)
        risk_averse_regrets_mean = np.mean(risk_averse_regrets_list, axis=0)[
            n_init - 1 :
        ]
        risk_averse_regrets_std = np.std(risk_averse_regrets_list, axis=0)[n_init - 1 :]
        axes[1].plot(risk_averse_regrets_mean, label=method_name)
        axes[1].fill_between(
            range(len(risk_averse_regrets_mean)),
            risk_averse_regrets_mean - risk_averse_regrets_std,
            risk_averse_regrets_mean + risk_averse_regrets_std,
            alpha=0.2,
        )

        simple_regret_list = np.array(simple_regret_list)
        simple_regret_mean = np.mean(simple_regret_list, axis=0)[n_init - 1 :]
        simple_regret_std = np.std(simple_regret_list, axis=0)[n_init - 1 :]
        axes[2].plot(simple_regret_mean, label=method_name)  # Use `ax.plot`
        axes[2].fill_between(
            range(len(simple_regret_mean)),
            simple_regret_mean - simple_regret_std,
            simple_regret_mean + simple_regret_std,
            alpha=0.2,
        )  # Use `ax.fill_between`

    axes[0].legend()
    axes[0].set_title("Risk-averse cumulative regrets")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Rt")

    axes[1].set_title("Risk-averse regrets")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Risk-averse Regrets")

    axes[2].set_title("Simple regrets")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Regrets")

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
    true_f = benchmark_f_dict[experiment_case["benchmark_f"]]
    plot_results(bo_results, experiment_case, true_f, save_path=args.output)
