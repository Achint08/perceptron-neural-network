import numpy as np

from layer import Layer

import matplotlib.pyplot as plt


class NeuralNetwork(object):

    def __init__(self) -> None:
        '''
        Initialization of stateful constructor for Neural network
        '''
        self.input = None
        self.layers = []
        self.output = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.minibatch_size = 12
        self.stochashtic_mse_per_iter_train = []
        self.batch_mse_per_iter_train = []
        self.mini_batch_mse_per_iter_train = []
        self.stochashtic_mse_per_iter_test = []
        self.batch_mse_per_iter_test = []
        self.mini_batch_mse_per_iter_test = []

    def add_layer(self, layer: Layer) -> None:
        '''
         To add layer to neural network
        '''
        self.layers.append(layer)

    def feed_forward_stochastic(self, input: np.ndarray) -> np.ndarray:
        '''
         Feed forward for stochashtic gradient descent
        '''
        self.input = input
        for layer in self.layers:
            self.input = layer.forward_propagation(self.input)
        return self.input

    def back_propagation_stochastic(self, expected: np.ndarray, learning_rate) -> None:
        '''
         Backward propogation for stochashtic gradient descent
        '''
        for layer in reversed(self.layers):
            layer.back_propagation_stochastic(
                expected, learning_rate)

    def feed_forward_batch(self, batch_input: np.ndarray):
        '''
         Feed forward for batch gradient descent
        '''
        self.input = batch_input
        for layer in self.layers:
            self.input = layer.feed_forward_batch(self.input)
        return self.input

    def back_propagation_batch(self, expected_batch: np.ndarray, learning_rate: float) -> None:
        '''
         Backward propogation for batch gradient descent
        '''
        for layer in reversed(self.layers):
            layer.back_propagation_batch(
                learning_rate, expected_batch)

    def feed_forward_mini_batch(self, batch_input: np.ndarray):
        '''
         Feed forward for mini-batch gradient descent
        '''
        self.input = batch_input
        for layer in self.layers:
            self.input = layer.feed_forward_mini_batch(self.input)
        return self.input

    def back_propagation_mini_batch(self, expected_batch: np.ndarray, learning_rate: float) -> None:
        '''
         Backward propogation for mini batch gradient descent
        '''
        for layer in reversed(self.layers):
            layer.back_propagation_mini_batch(
                learning_rate, expected_batch
            )

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int,
              learning_rate: float, train_type: str, X_test: np.ndarray, y_test: np.ndarray) -> None:
        '''
         Performs training of neural network:
         1. Stochastic
         2. Batch
         3. Mini batch

        Bias feature 1.0 is added to the input to compute the bias.
        '''
        train_sample_length = len(X)
        self.x_train = X
        self.y_train = y
        self.x_test = X_test
        self.y_test = y_test
        if train_type == "stochastic":
            for epoch in range(epochs):
                for train_idx in range(train_sample_length):
                    # Randomly select a sample from training set
                    random_point = np.random.randint(train_idx + 1)
                    x = X[random_point]
                    target = y[random_point]

                    # Bias added
                    bias = np.ones((1, 1))
                    x_new = np.append(x, bias)

                    # Training and backward propogation
                    self.feed_forward_stochastic(x_new)
                    learning_rate = epoch * train_sample_length + train_idx
                    self.back_propagation_stochastic(target, learning_rate)

                # Calculate and store MSE
                self.stochashtic_mse_per_iter_train.append(
                    self.get_mse_per_iter_train())

                self.stochashtic_mse_per_iter_test.append(
                    self.get_mse_per_iter_test())

            # Plot MSE per iteration
            self.plot_stochashtic_mse_per_iter_train()
            self.plot_stochashtic_mse_per_iter_test()

        elif train_type == "batch":

            # Add bias to input
            bias = np.ones((train_sample_length, 1))
            X_new = np.hstack((X, bias))

            for _ in range(epochs):
                # Batch training and propogation
                self.feed_forward_batch(X_new)
                self.back_propagation_batch(y, learning_rate)

                # Calculate and store MSE
                self.batch_mse_per_iter_train.append(
                    self.get_mse_per_iter_train())
                self.batch_mse_per_iter_test.append(
                    self.get_mse_per_iter_test())

            # Plot MSE per iteration
            self.plot_batch_mse_per_iter_train()
            self.plot_batch_mse_per_iter_test()
        elif train_type == "mini_batch":
            # Shuffling indices
            shuffled_indices = np.random.permutation(train_sample_length)

            # Add bias to input
            bias = np.ones((train_sample_length, 1))
            X_new = np.hstack((X, bias))

            # Shuffling the batch batch to create randomness
            X_shuffled = X_new[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for epoch in range(epochs):
                for i in range(0, train_sample_length, self.minibatch_size):
                    # Updating learning rate for each epoch
                    learning_rate += 1

                    # Mini batch training and propogation
                    x_mini_batch = X_shuffled[i:i+self.minibatch_size]
                    y_mini_batch = y_shuffled[i:i+self.minibatch_size]
                    self.feed_forward_mini_batch(x_mini_batch)
                    self.back_propagation_mini_batch(
                        y_mini_batch, learning_rate
                    )

                # Calculate and store MSE
                self.mini_batch_mse_per_iter_train.append(
                    self.get_mse_per_iter_train())
                self.mini_batch_mse_per_iter_test.append(
                    self.get_mse_per_iter_test())

            # Plot MSE per iteration
            self.plot_mini_batch_mse_per_iter_train()
            self.plot_mini_batch_mse_per_iter_test()

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Method to predict based on input
        '''
        X_new = np.append(X, [1.0])
        return self.feed_forward(X_new)

    def feed_forward(self, input: np.ndarray) -> np.ndarray:
        '''
        Performs normal feed forward for a single input
        '''
        self.input = input
        for layer in self.layers:
            self.input = layer.forward_propagation(self.input)
        return self.input

    def get_mse_per_iter_train(self) -> float:
        '''
        Calculates MSE for training set
        '''
        train_io = zip(self.x_train, self.y_train)
        n = len(self.x_train)
        squared_sum = 0
        for x, y in train_io:
            x_new = np.append(x, [1.0])
            squared_sum += (y - self.feed_forward(x_new)) ** 2

        return squared_sum/n

    def get_mse_per_iter_test(self) -> float:
        '''
        Calculates MSE for test set
        '''
        test_io = zip(self.x_test, self.y_test)
        n = len(self.x_test)
        squared_sum = 0
        for x, y in test_io:
            x_new = np.append(x, [1.0])
            squared_sum += (y - self.feed_forward(x_new)) ** 2

        return squared_sum/n

    def plot_stochashtic_mse_per_iter_train(self):
        '''
        Plots MSE per iteration for training set on Stochastic Gradient Descent
        '''
        plt.plot(self.stochashtic_mse_per_iter_train)
        plt.title('Stochastic Train MSE iteration per epoch')
        plt.savefig('plots/Stochastic Train.png')
        plt.clf()

    def plot_stochashtic_mse_per_iter_test(self):
        '''
        Plots MSE per iteration for test set on Stochastic Gradient Descent
        '''
        plt.plot(self.stochashtic_mse_per_iter_test)
        plt.title('Stochastic Test MSE iteration per epoch')
        plt.savefig('plots/Stochastic Test.png')
        plt.clf()

    def plot_batch_mse_per_iter_train(self):
        '''
        Plots MSE per iteration for training set on Batch Gradient Descent
        '''
        plt.plot(self.batch_mse_per_iter_train)
        plt.title('Batch Train MSE per iteration')
        plt.savefig('plots/Batch Train.png')
        plt.clf()

    def plot_batch_mse_per_iter_test(self):
        '''
        Plots MSE per iteration for test set on Batch Gradient Descent  
        '''
        plt.plot(self.batch_mse_per_iter_test)
        plt.title('Batch Test MSE per iteration')
        plt.savefig('plots/Batch Test.png')
        plt.clf()

    def plot_mini_batch_mse_per_iter_train(self):
        '''
        Plots MSE per iteration for training set on Mini Batch Gradient Descent  
        '''
        plt.plot(self.mini_batch_mse_per_iter_train)
        plt.title('Mini Batch Train MSE iteration per epoch')
        plt.savefig('plots/Mini Batch Train.png')
        plt.clf()

    def plot_mini_batch_mse_per_iter_test(self):
        '''
        Plots MSE per iteration for test set Mini Batch Gradient Descent  
        '''
        plt.plot(self.mini_batch_mse_per_iter_test)
        plt.title('Mini Batch Test MSE iteration per epoch')
        plt.savefig('plots/Mini Batch Test.png')
        plt.clf()
