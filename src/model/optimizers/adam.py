import numpy as np

from src.model.optimizers.optimizer import Optimizer


class AdamOptimizer(Optimizer):
    """
    It stands for Adaptive Moment Estimation (Adam) and combines the
    advantages of two other extensions of stochastic gradient descent:
    AdaGrad and RMSProp.

    Key features and workings of AdamOptimizer:
    - **Adaptive Learning Rates**: Adam dynamically adjusts the
        learning rates for each parameter based on the estimates
        of first and second moments of the gradients.
    - **Momentum**: It incorporates momentum by using moving
        averages of the gradient and the squared gradient.
    - **Bias Correction**: Adam performs bias correction to account
        for the initialization of moment estimates at zero,
        particularly effective in the early stages of training.

    ### Algorithm Steps:
    1. Initialize parameters (learning rate, beta1, beta2, epsilon).
    2. Compute gradient of loss function with respect to parameters.
    3. Update biased first and second moment estimates:
       - First moment (mean): Momentum term adjusted by beta1.
       - Second moment (uncentered variance): Squared gradient
            adjusted by beta2.
    4. Compute bias-corrected estimates of the first and second moments.
    5. Update parameters using the bias-corrected estimates
        and the learning rate.

    ### Purpose:
    - **Efficient Optimization**: AdamOptimizer aims to provide an
        efficient and effective method for training deep neural
        networks by adapting learning rates per parameter.
    - **Convergence Speed**: It often leads to faster convergence
        compared to standard stochastic gradient descent (SGD)
        and helps in handling sparse gradients and noisy data.
    """

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

    def initialize_moments(self, weights_shape: tuple, bias_shape: tuple):
        """
        Initialize the first and second moments for weights and biases.

        Args:
            weights_shape (tuple): Shape of the weights.
            bias_shape (tuple): Shape of the biases.

        Returns:
            None
        """
        self.m_weights = np.zeros(weights_shape)
        self.v_weights = np.zeros(weights_shape)
        self.m_bias = np.zeros(bias_shape)
        self.v_bias = np.zeros(bias_shape)

    def update(self, weights: np.ndarray, bias: np.ndarray,
               weights_gradients: np.ndarray, bias_gradients: np.ndarray):
        """
        Update the weights and biases using the Adam optimization.

        Args:
            weights (np.ndarray): Weights of the model.
            bias (np.ndarray): Biases of the model.
            weights_gradients (np.ndarray): Gradients of the weights.
            bias_gradients (np.ndarray): Gradients of the biases.

        Returns:
            tuple: Updated weights and biases.
        """
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
        """
        Get the configuration of the optimizer.
        """
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon
        }
