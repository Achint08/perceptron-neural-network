import os
import numpy as np
from layer import Layer
from neural_network import NeuralNetwork
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Load iris dataset
    iris_data = load_iris()
    x = iris_data.data
    y = iris_data.target

    # Data preprocessing

    for idx, target_y in enumerate(y):
        if y[idx] == 0:
            y[idx] = 1
        else:
            y[idx] = 0

    # Split data into training and testing data

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=10)

    # Create a neural network, train, and test based on different training types

    if not os.path.exists('plots'):
        os.mkdir('plots')
    for train_type in ["stochastic", "batch", "mini_batch"]:
        print("Training type: {}{}".format(
            train_type[0].upper(), train_type.replace('_', ' ')[1:]))
        train_io = zip(x_train, y_train)

        # Create a neural network
        nn = NeuralNetwork()
        layer = Layer()

        # Updates random weight to the layer
        np.random.seed(100)
        # Total 5 weights, 4 inputs + 1 bias
        layer.weights = np.random.rand(1, 5)[0]
        nn.add_layer(layer)

        # Train the neural network
        nn.train(x_train, y_train, 1000, 0.001, train_type, x_test, y_test)

        # Perform the model on test data set and print results
        test_io = zip(x_test, y_test)

        print("Result on test set")
        for input, output in test_io:
            print(nn.predict(input), output)

    print("Plots for MSE per iteration stored in plots/ directory")
