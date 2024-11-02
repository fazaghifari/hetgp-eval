from typing import Any


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
    def __init__(self, xdim, n_candidates) -> None:
        self.xdim = xdim
        self.n_candidates = n_candidates

    def optimize(self, acquisition_function, model):
        # Optimize with CMAES
        return NotImplementedError


class AugmentedEIAcqFunction:
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def initialize_acquisition_function(self, model):
        # run this everytime model is updated
        return NotImplementedError

    def __call__(self, x_cand) -> Any:
        return NotImplementedError
