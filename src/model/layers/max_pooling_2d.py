from .layer import Layer


class MaxPooling2D(Layer):
    def __init__(self, pool_size, strides=None, padding='valid'):
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size
        self.padding = padding

    def build(self, input_shape):
        pass

    def call(self, inputs):
        # Perform max pooling
        pass

    def backward(self, output_gradient, learning_rate):
        # Perform backward propagation
        pass

    def output_shape(self):
        # Calculate output shape
        pass
