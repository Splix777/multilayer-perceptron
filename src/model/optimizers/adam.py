import numpy as np

from .optimizer import Optimizer

# Adaptive Moment Estimation


class AdamOptimizer(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 learning_rate=0.001):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.iterations = 0
        self.m_weights = None
        self.v_weights = None
        self.m_bias = None
        self.v_bias = None

    def initialize_moments(self, weights_shape, bias_shape):
        self.m_weights = np.zeros(weights_shape)
        self.v_weights = np.zeros(weights_shape)
        self.m_bias = np.zeros(bias_shape)
        self.v_bias = np.zeros(bias_shape)

    def update(self, weights, bias, weights_gradients, bias_gradients):
        if self.m_weights is None:
            self.initialize_moments(weights.shape, bias.shape)

        self.iterations += 1

        self.m_weights = (self.beta1
                          * self.m_weights
                          + (1 - self.beta1)
                          * weights_gradients)
        self.m_bias = (self.beta1
                       * self.m_bias
                       + (1 - self.beta1)
                       * bias_gradients)

        self.v_weights = (self.beta2
                          * self.v_weights
                          + (1 - self.beta2)
                          * (weights_gradients ** 2))
        self.v_bias = (self.beta2
                       * self.v_bias
                       + (1 - self.beta2)
                       * (bias_gradients ** 2))

        m_hat_weights = self.m_weights / (1 - self.beta1 ** self.iterations)
        v_hat_weights = self.v_weights / (1 - self.beta2 ** self.iterations)
        m_hat_bias = self.m_bias / (1 - self.beta1 ** self.iterations)
        v_hat_bias = self.v_bias / (1 - self.beta2 ** self.iterations)

        updated_weights = (weights
                           - self.learning_rate
                           * m_hat_weights
                           / (np.sqrt(v_hat_weights) + self.epsilon))
        updated_biases = (bias
                          - self.learning_rate
                          * m_hat_bias
                          / (np.sqrt(v_hat_bias) + self.epsilon))

        return updated_weights, updated_biases

    def get_config(self) -> dict:
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon
        }
