import numpy as np

class CustomLogisticRegression:
    def __init__(self, weights = None, bias = 0, alpha = 0.001, epochs = 1000):
        """
            Linear Regression Model from scratch using Gradient Descent
            This model tries to minimize the loss using Gradient Descent

            Parameters
            ----------
            weights: ndarray
            bias: float, default=0
            alpha: float, default=0.001
                - Learning Rate
            epochs: int, default=1000
                - Number of training iterations
            losses: list
                - History of the losses per training iteration
        """

        self.weights = weights
        self.bias = bias
        self.alpha = alpha
        self.epochs = epochs
        self.losses = []

    def __sigmoid(self, features):
        """
            It calculates the sigmoid value for the given features
            using current weights and bias

            Parameters
            ----------
            features: arraylike

            Returns
            -------
            sigmoid_value: float
                Value after calculating sigmoid
        """
        negative_z = -(np.dot(self.weights, features) + self.bias)
        sigmoid_value = 1 / (1 + np.exp(negative_z))
        return sigmoid_value

    
    def __gradient_descent(self, X, Y):
        pass
