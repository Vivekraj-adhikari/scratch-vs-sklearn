import numpy as np
import pandas as pd

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
            features: numpy_arraylike

            Returns
            -------
            sigmoid_value: float
                Value after calculating sigmoid
        """
        negative_z = -(np.dot(self.weights, features) + self.bias)
        sigmoid_value = 1 / (1 + np.exp(negative_z))
        return sigmoid_value

    def __ensure_numpy_array(self, X):
        """
            Ensures that the training data is numpy_array

            Parameters
            ----------
            X: matrixlike

            Returns
            -------
            X: numpy_array
        """
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        
        return X

    def __gradient_descent(self, X, Y):
        """
            Calculates gradients of the weights and bias
            And updates weights and bias

            Parameters
            ----------
            X: matrixlike
                - Input features
            Y: arraylike
                - Observed output
        """

        X = self.__ensure_numpy_array(X)
        Y = self.__ensure_numpy_array(Y)

        data_length = len(X)
        dw = np.zeros(len(X[1]))
        db = 0

        for i in range(data_length):
            features = X[i]
            error = self.__sigmoid(features) - Y[i]

            dw += features * error
            db += error
        
        self.weights -= self.alpha * (dw / data_length)
        self.bias -= self.alpha * (db / data_length)

    
    def fit(self, X, Y):
        """
            Trains the model using batch gradient descent

            Parameters
            ----------
            X: matrixlike
                - Input features
            Y: arraylike
                - Observed output
        """

        for _ in range(self.epochs):
            self.__gradient_descent(X, Y)