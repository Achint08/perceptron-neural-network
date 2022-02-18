import numpy as np
import math


class Layer:

    def __init__(self) -> None:
        '''
        Initialization of stateful constructor for a layer
        '''
        self.weights = None
        self.input = None
        self.output = None
        self.learning_rate = None
        self.t0 = 0.1
        self.t1 = 5

        self.sigmoid_threshold = 0.5

    def forward_propagation(self, input: np.ndarray) -> np.ndarray:
        '''
        Performs forward progragation for stochastic GD

        '''
        self.input = input
        self.output = self.activation_stochastic()
        return self.output

    def back_propagation_stochastic(self, expected: np.ndarray, learning_rate: float) -> None:
        '''
        Performs backward progragation for stochastic GD

        '''
        self.learning_rate = learning_rate
        self.update_weights_stochastic(expected)

    def update_weights_stochastic(self, expected) -> None:
        '''
        Updates weight based on stochastic GD

        '''
        eta = self.learning_schedule(self.learning_rate)
        self.weights -= self.activation_derivative_stochastic(
            expected
        ) * eta

    def activation_stochastic(self) -> np.ndarray:
        '''
        Activation function for stochastic GD, based on sigmoid function
        '''
        output = np.dot(self.input, self.weights)
        sigmoid_val = self.sigmoid(output)
        if sigmoid_val < self.sigmoid_threshold:
            return 0
        else:
            return 1

    def activation_derivative_stochastic(self, expected: int) -> np.ndarray:
        '''
        Derivative of activation function for stochastic GD
        '''
        return -2 * (expected - self.output) * self.input

    def back_propagation_batch(self, learning_rate: float, expected_batch: np.ndarray) -> None:
        '''
        Performs backward progragation for batch GD
        '''
        self.learning_rate = learning_rate
        self.update_weights_batch(expected_batch)

    def update_weights_batch(self, expected_batch: np.ndarray) -> None:
        '''
        Updates weight based on batch GD
        '''
        self.weights -= self.activation_derivative_batch(
            expected_batch
        ) * self.learning_rate

    def batch_activation(self) -> np.ndarray:
        '''
        Activation function for batch GD, based on sigmoid function
        '''
        output = np.dot(self.weights, self.input.T)
        for i in range(len(output)):
            sigmoid_val = self.sigmoid(output[i])
            if sigmoid_val < self.sigmoid_threshold:
                output[i] = 0
            else:
                output[i] = 1
        return output

    def activation_derivative_batch(self, expected_batch: np.ndarray):
        '''
        Derivative of activation function for batch GD
        '''
        diff = (expected_batch - self.batch_activation())
        return -2/len(diff) * self.input.T.dot(diff)

    def feed_forward_batch(self, batch_x: np.ndarray) -> np.ndarray:
        '''
        Performs forward progragation for batch GD
        '''
        self.input = batch_x
        self.output = self.batch_activation()
        return self.output

    def feed_forward_mini_batch(self, batch_x: np.ndarray) -> np.ndarray:
        '''
        Performs forward progragation for mini batch GD
        '''
        self.input = batch_x
        self.output = self.batch_activation()
        return self.output

    def back_propagation_mini_batch(self, learning_rate: float, expected_batch: np.ndarray) -> None:
        '''
        Performs backward progragation for mini batch GD
        '''
        self.learning_rate = learning_rate
        self.update_weights_mini_batch(expected_batch)

    def update_weights_mini_batch(self, expected_batch: np.ndarray) -> None:
        '''
        Updates weight based on mini batch GD
        '''
        eta = self.learning_schedule(self.learning_rate)
        self.weights -= self.activation_derivative_batch(
            expected_batch
        ) * eta

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Performs forward progragation for prediction
        '''
        return self.forward_propagation(X)

    def learning_schedule(self, t):
        '''
        Learning schedule for stochastic and mini batch GD to prevent overfitting
        and get learning rate
        '''
        return self.t0/(t + self.t1)

    @staticmethod
    def sigmoid(x):
        '''
        Sigmoid function
        '''
        return 1 / (1 + math.exp(-x))
