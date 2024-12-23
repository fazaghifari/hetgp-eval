import numpy as np

from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from src.ksmlhgp import KSMLHGP
from src.bo import (
    BayesianOptimizer,
    AugmentedEIAcqFunction,
    LCBAcqFunction,
    CMAESAcqOptimizer,
    RandomAcqOptimizer,
    RAHBOAcqFunction,
)

## SUPPRESS ALL WARNINGS
import warnings

warnings.filterwarnings("ignore")


def g1(X):
    # logistic
    X = 20 * X - 10  # [0, 1] -> [-10, 10]
    return 1 / (1 + np.exp(-X)) / 10


def f1(X, add_noise=False):
    target = 0.5 * np.sin(X * 20)
    if add_noise:
        rng = np.random.RandomState(1)
        target += rng.normal(np.zeros_like(X), g1(X), size=target.shape)
    return target.squeeze()


benchmark_f_dict = {"f1_toy": f1}


def build_model(model_name):
    if model_name == "GP":
        base_kernel = ConstantKernel(1.0) * RBF(
            length_scale=0.1, length_scale_bounds=(0.01, 100)
        ) + WhiteKernel(noise_level=1, noise_level_bounds=(0.001, 10))
        noise_kernel = ConstantKernel(1.0) * RBF(
            length_scale=0.1, length_scale_bounds=(0.01, 100)
        ) + WhiteKernel(noise_level=1, noise_level_bounds=(0.001, 10))

        gp = GaussianProcessRegressor(kernel=base_kernel, n_restarts_optimizer=25)
        model = gp

    elif model_name == "KSMLHGP":
        base_kernel = ConstantKernel(1.0) * RBF(
            length_scale=0.1, length_scale_bounds=(0.01, 100)
        ) + WhiteKernel(noise_level=1, noise_level_bounds=(0.001, 10))
        noise_kernel = ConstantKernel(1.0) * RBF(
            length_scale=0.1, length_scale_bounds=(0.01, 100)
        ) + WhiteKernel(noise_level=1, noise_level_bounds=(0.001, 10))

        gp = GaussianProcessRegressor(kernel=base_kernel, n_restarts_optimizer=25)
        gp_noise = GaussianProcessRegressor(
            kernel=noise_kernel, n_restarts_optimizer=25
        )

        model = KSMLHGP(model=gp, model_noise=gp_noise)
    return model


def build_acq_f(acq_f_params):
    if acq_f_params["name"] == "RAHBO":
        acq_f = RAHBOAcqFunction(
            alpha=acq_f_params["params"]["alpha"], beta=acq_f_params["params"]["beta"]
        )
    elif acq_f_params["name"] == "LCB":
        acq_f = LCBAcqFunction(alpha=acq_f_params["params"]["alpha"])
    return acq_f


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
    bounds = np.array(experiment_case["bounds"])
    for _ in range(n_repeats):
        # sample initial observations for each repeat
        x_init = np.random.uniform(bounds[0], bounds[1], (n_init, bounds.shape[1]))
        y_init = benchmark_f(x_init, add_noise=True)
        init_dataset = {"X": x_init, "y": y_init}
        init_datasets.append(init_dataset)

    bo_results = {}
    for method_name in experiment_case["methods"]:
        running_method = experiment_case["methods"][method_name]
        if experiment_case["verbose"]:
            print("=====================================")
            print(f"Running BO for method: {method_name}")
            print(f"Model: {running_method['model']}")
            print(f"Acquisition function: {running_method['acq_f']}")
            print(f"Acquisition optimizer: {running_method['bo_optimizer']}")

        results_repeat = []
        # Run BO for each repeat
        for i in range(n_repeats):
            if experiment_case["verbose"]:
                print(f"Running rep {i+1}/{n_repeats}")

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
                print("=====================================")
            results_repeat.append(result)

        bo_results[method_name] = results_repeat


if __name__ == "__main__":
    import json

    with open("bo_experiments_config.json", "r") as f:
        experiment_case = json.load(f)
    run_bo_experiments(experiment_case)
