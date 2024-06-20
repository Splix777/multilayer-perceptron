from .layer import Layer


class Dropout(Layer):
    def __init__(self, rate):
        self.rate = rate

    def build(self, input_shape):
        pass

    def call(self, inputs, training=False):
        if training:
            # Apply dropout
            pass
        return inputs

    def backward(self, output_gradient, learning_rate):
        return output_gradient

    def output_shape(self):
        return self.input_shape
