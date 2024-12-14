from typing import Any
import numpy as np
from cmaes import CMA

class BayesianOptimizer:
    def __init__(
        self, model, acquisition_function, true_function, acq_optimizer
    ) -> None:
        self.model = model
        self.acquisition_function = acquisition_function
        self.true_function = true_function
        self.acq_optimizer = acq_optimizer

    def step(self, observations):
        # optimize model using acq_optimizer
        # calculate acq
        # return query point
        return NotImplementedError

    def add_point_to_observations(self, new_x, new_y, observations):
        # add new points to observations
        return NotImplementedError

    def optimize(self, observations, n_iter):
        current_observations = observations
        for i in range(n_iter):
            new_x = self.step(current_observations)
            new_y = self.true_function(new_x)
            current_observations = self.add_point_to_observations(
                new_x, new_y, current_observations
            )
        return current_observations


class CMAESAcqOptimizer:
    """
    usage:
    pip install cmaes 
    https://pypi.org/project/cmaes/
    """
    def __init__(self, xdim, n_candidates) -> None:
        self.xdim = xdim
        self.n_candidates = n_candidates

    def optimize(self, acquisition_function, model):
        # Optimize with CMAES
        return NotImplementedError

    def _cmaes_minimizer(obj_f, start, bounds=None, n_generation=100):
        if len(start) == 1:
            start = np.append(start, [0])
            obj_f_acq = lambda x: obj_f([x[0]])
            bounds = np.stack([bounds[0], [0, 1]])
        else:
            obj_f_acq = obj_f

        optimizer = CMA(mean=start, sigma=0.1, bounds=bounds)
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

        return best_solution[0], best_solution[1]


class AugmentedEIAcqFunction:
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def initialize_acquisition_function(self, model):
        # run this everytime model is updated
        return NotImplementedError

    def __call__(self, x_cand) -> Any:
        return NotImplementedError
