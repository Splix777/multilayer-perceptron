import numpy as np
from .optimizer import Optimizer


class RMSpropOptimizer(Optimizer):
    def __init__(self, rho=0.9, epsilon=1e-8, learning_rate=0.001):
        super().__init__()
        self.rho = rho
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.accumulated_weights = None
        self.accumulated_bias = None

    def initialize_accumulators(self, weights_shape, bias_shape):
        self.accumulated_weights = np.zeros(weights_shape)
        self.accumulated_bias = np.zeros(bias_shape)

    def update(self, weights, bias, weights_gradients, bias_gradients):
        if self.accumulated_weights is None:
            self.initialize_accumulators(weights.shape, bias.shape)

        self.accumulated_weights = (self.rho
                                    * self.accumulated_weights
                                    + (1 - self.rho)
                                    * (weights_gradients ** 2))
        self.accumulated_bias = (self.rho
                                 * self.accumulated_bias
                                 + (1 - self.rho)
                                 * (bias_gradients ** 2))

        updated_weights = (weights
                           - self.learning_rate
                           * weights_gradients
                           / (np.sqrt(self.accumulated_weights) + self.epsilon))
        updated_biases = (bias
                          - self.learning_rate
                          * bias_gradients
                          / (np.sqrt(self.accumulated_bias) + self.epsilon))

        return updated_weights, updated_biases

    def get_config(self) -> dict:
        return {
            'learning_rate': self.learning_rate,
            'rho': self.rho,
            'epsilon': self.epsilon
        }
