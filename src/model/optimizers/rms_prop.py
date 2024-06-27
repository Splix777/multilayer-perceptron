import numpy as np

from src.model.optimizers.optimizer import Optimizer


class RMSpropOptimizer(Optimizer):
    """
    RMSpropOptimizer (Root Mean Square Propagation) is an optimization
    algorithm commonly used in training deep learning models.

    It addresses some limitations of the Adagrad optimizer by
    modifying the gradient accumulation in the update rule.

    Key features and workings of RMSpropOptimizer:
    - **Adaptive Learning Rates**: RMSprop adapts the learning rate
        for each parameter based on the average of recent magnitudes
        of the gradients for that parameter.
    - **Decaying Average**: It maintains a decaying average of
        squared gradients, similar to momentum in SGD.
    - **Stability**: RMSprop helps stabilize learning rates during
        training by dividing the learning rate by a moving average
        of the recent magnitudes of the gradients.

    ### Algorithm Steps:
    1. Initialize parameters (learning rate, rho, epsilon).
    2. Compute gradient of the loss function with respect to parameters.
    3. Update the accumulated squared gradient:
       - Calculate the squared gradient as a moving average of
        its past gradients.
    4. Compute the root-mean-square (RMS) of the gradients.
    5. Update parameters using the RMS of the gradients
        and the learning rate.

    ### Purpose:
    - **Efficient Optimization**: RMSpropOptimizer aims to improve
        the efficiency of training deep neural networks by adapting
        learning rates per parameter.
    - **Mitigating Adagrad's Limitations**: It addresses the
        diminishing learning rate problem of Adagrad by introducing
        a decaying average of squared gradients.
    - **Better Handling of Sparse Gradients**: RMSprop is particularly
        useful when dealing with sparse gradients or non-stationary
        objectives.
    """

    def __init__(self, rho=0.9, epsilon=1e-8, learning_rate=0.001):
        super().__init__()
        self.rho = rho
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.accumulated_weights = None
        self.accumulated_bias = None

    def initialize_accumulators(self, weights_shape: tuple, bias_shape: tuple):
        """
        Initialize the accumulated squared gradients for
        weights and biases.

        Args:
            weights_shape (tuple): Shape of the weights.
            bias_shape (tuple): Shape of the biases.

        Returns:
            None
        """
        self.accumulated_weights = np.zeros(weights_shape)
        self.accumulated_bias = np.zeros(bias_shape)

    def update(self, weights: np.ndarray, bias: np.ndarray,
               weights_gradients: np.ndarray, bias_gradients: np.ndarray):
        """
        Update the weights and biases using RMSprop optimization.

        Args:
            weights (np.ndarray): Weights of the model.
            bias (np.ndarray): Biases of the model.
            weights_gradients (np.ndarray): Gradients of the weights.
            bias_gradients (np.ndarray): Gradients of the biases.

        Returns:
            tuple: Updated weights and biases.
        """
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

        updated_weight = (weights
                          - self.learning_rate
                          * weights_gradients
                          / (np.sqrt(self.accumulated_weights) + self.epsilon))
        updated_bias = (bias
                        - self.learning_rate
                        * bias_gradients
                        / (np.sqrt(self.accumulated_bias) + self.epsilon))

        return updated_weight, updated_bias

    def get_config(self) -> dict:
        """
        Get the configuration of the optimizer.
        """
        return {
            'learning_rate': self.learning_rate,
            'rho': self.rho,
            'epsilon': self.epsilon
        }
