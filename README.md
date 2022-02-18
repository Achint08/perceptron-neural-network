# Perceptron Neural Network with one layer

We are going to implement stochastic, batch and mini batch gradient descent using basic data science libraries.

## How to run?

Note: Please use Python3

Install the requirements:

```
pip install -r requirements.txt
```

Run the following command to run the code:

```
python main.py
```

For each gradient descent type, graphs for MSE per iteration will be stored in plots/ directory after running the code.

## Directory Structure

- main.py - Entry level for the project
- layer.py - Contains layer class for each layer
- neural_network.py - Contains neural network class with added layers
- plots/ - Contains plots for MSEs

## Assumptions

- We've divided data set in ration of train: test :: 8: 2.
- We're considering bias as a weight for a feature, instead of computing it individually. So, we've added 1.0 to the feature set along with the four feature. The weights for this data point will compensate for bias value.
- Mini batch size = 12
- Total epochs = 1000
- Using sigmoid as activation functon with threshold = 0.5
- Hyperparameters for mini-batch and stochastic:
  - t0 = 0.1
  - t1 = 5

# Thank you :)
