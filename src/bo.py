from typing import Any
import numpy as np
from cmaes import CMA
import copy


class BayesianOptimizer:
    def __init__(
        self, model, acquisition_function, true_function, acq_optimizer, verbose=True
    ) -> None:
        self.model = model
        self.acquisition_function = acquisition_function
        self.true_function = true_function
        self.acq_optimizer = acq_optimizer
        self.verbose = verbose

    def step(self, observations):
        self.model.fit(observations["X"], observations["y"])

        self.acquisition_function.initialize_acquisition_function(
            self.model, observations
        )
        x_cand, acq_val = self.acq_optimizer.optimize(self.acquisition_function)
        return x_cand, acq_val

    def add_point_to_observations(self, new_x, new_y, observations):
        # add new points to observations
        new_x = new_x.reshape(1, -1)
        observations["X"] = np.concatenate((observations["X"], new_x), axis=0)
        observations["y"] = np.append(observations["y"], new_y)
        return observations

    def optimize(self, observations, n_iter):
        current_observations = copy.deepcopy(observations)
        for i in range(n_iter):
            new_x, acq_val = self.step(current_observations)
            new_y = self.true_function(new_x)
            current_observations = self.add_point_to_observations(
                new_x, new_y, current_observations
            )
            if self.verbose:
                print(f"Iteration {i}: x={new_x}, y={new_y}, acq={acq_val}")
        return current_observations


class CMAESAcqOptimizer:
    """
    TODO: Implement CMAES optimizer
    usage:
    pip install cmaes
    https://pypi.org/project/cmaes/
    """

    def __init__(self, n_initial, bounds, n_generation=1000, cmaes_sigma=0.1) -> None:
        self.bounds = bounds
        self.xdim = bounds.shape[1]
        self.n_initial = n_initial
        self.n_generation = n_generation
        self.cmaes_sigma = cmaes_sigma

    def optimize(self, acquisition_function):
        # CMAES optimization
        start = np.random.uniform(self.bounds[0], self.bounds[1], self.xdim)  # [D]

        best_solution, best_value = self._cmaes_maximizer(
            acquisition_function,
            start,
            bounds=self.bounds,
            n_generation=self.n_generation,
            sigma=self.cmaes_sigma,
        )
        return best_solution, best_value

    def _cmaes_maximizer(self, obj_f, start, bounds, n_generation, sigma):
        if self.xdim == 1:
            # trick for 1d
            start = np.append(start, [0])
            obj_f_acq = lambda x: obj_f(np.array([x[0]]).reshape(-1, 1)) * -1
            bounds = np.concatenate((bounds.T, [[0, 1]]))
        else:
            obj_f_acq = lambda x: obj_f(x) * -1
            bounds = bounds.T

        optimizer = CMA(mean=start, sigma=sigma, bounds=bounds)
        best_solution = None
        for generation in range(n_generation):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = obj_f_acq(x)
                solutions.append((x, value))
            optimizer.tell(solutions)

            if generation == 0:
                best_solution = solutions[0]
            else:
                if best_solution[1] > solutions[0][1]:
                    best_solution = solutions[0]
        if self.xdim == 1:
            return best_solution[0][None, 0, None], best_solution[1]

        else:
            return best_solution[0][None, None], best_solution[1]


class RandomAcqOptimizer:
    def __init__(self, n_candidates, bounds) -> None:
        self.xdim = bounds.shape[1]
        self.n_candidates = n_candidates
        self.bounds = bounds

    def optimize(self, acquisition_function):
        # Random search
        x_cands = np.random.uniform(
            self.bounds[0], self.bounds[1], (self.n_candidates, self.bounds.shape[1])
        )  # [N, D]
        acq_values = acquisition_function(x_cands)  # [N]
        best_idx = np.argmax(acq_values)
        return x_cands[best_idx], acq_values[best_idx]


class AugmentedEIAcqFunction:
    """
    TODO: Implement Augmented Expected Improvement
    """

    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def initialize_acquisition_function(self, model, observations):
        # run this everytime model is updated
        return NotImplementedError

    def __call__(self, x_cand) -> Any:
        """
        TODO
        # Huang et al  Global optimization of stochastic black-box systems via sequential kriging meta-models. Journal of global optimization, 2006
        mean, variance = self._model.predict(tf.squeeze(x, -2))
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        ei = (self._eta - mean) * normal.cdf(self._eta) + variance * normal.prob(self._eta)
        augmentation = 1 - (tf.math.sqrt(self._noise_variance)) / (
            tf.math.sqrt(self._noise_variance + variance)
        )
        return ei * augmentation

        """
        return NotImplementedError


class RAHBOAcqFunction:
    def __init__(self, alpha=1, beta=0.5) -> None:
        self.alpha = alpha
        self.beta = beta

    def initialize_acquisition_function(self, model, observations):
        # run this everytime model is updated
        self.model = model

    def __call__(self, x_cand) -> Any:
        # Makartova et al Risk-averse Heteroscedastic Bayesian Optimization Neurips 2021
        mean_pred, stds_pred = self.model.predict(x_cand, return_std="multi")
        std_al, std_ep = stds_pred
        rahbo = -mean_pred + (self.beta * std_ep) - (self.alpha * (std_al**2))
        return rahbo
