from .optimizer import Optimizer
from typing import List


class AdamOptimizer(Optimizer):
    """
    Adam optimizers for gradient-based optimization.

    Attributes:
        learning_rate (float): The learning rate for the optimizers.
        beta1 (float): Decay rate for the first moment estimate.
        beta2 (float): Decay rate for the second moment estimate.
        epsilon (float): Small constant to prevent division by zero.
        m (List[float] or None): First moment estimates.
        v (List[float] or None): Second moment estimates.
        t (int): Timestep counter.
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the Adam optimizers with specified hyperparameters.

        Args:
            learning_rate (float, optional): The learning rate
                for the optimizers. Defaults to 0.001.
            beta1 (float, optional): Decay rate for the first
                moment estimate. Defaults to 0.9.
            beta2 (float, optional): Decay rate for the second
                moment estimate. Defaults to 0.999.
            epsilon (float, optional): Small constant to
                revent division by zero. Defaults to 1e-8.
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # Initialize first moment vector
        self.v = None  # Initialize second moment vector
        self.t = 0     # Initialize timestep

    def update(self, params: List[float], grads: List[float]) -> List[float]:
        """
        Update the parameters based on the gradients
        using the Adam optimization algorithm.

        Args:
            params (List[float]): List of parameters
                (e.g., weights, biases).
            grads (List[float]): List of gradients
                corresponding to parameters.

        Returns:
            List[float]: Updated parameters after applying
                the optimization update rule.
        """
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1
        learning_rate_t = (self.learning_rate
                           * (pow(self.beta2, self.t) ** 0.5)
                           / (1 - pow(self.beta1, self.t)))

        updated_params = []
        for i in range(len(params)):
            self.m[i] = (self.beta1 * self.m[i]
                         + (1 - self.beta1) * grads[i])

            self.v[i] = (self.beta2 * self.v[i]
                         + (1 - self.beta2)
                         * grads[i] ** 2)

            updated_params.append(
                params[i] - learning_rate_t
                * self.m[i]
                / (self.epsilon + self.v[i] ** 0.5)
            )

        return updated_params

    def get_config(self) -> dict:
        """
        Get the configuration of the optimizers.

        Returns:
            dict: Configuration of the optimizers.
        """
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon
        }
