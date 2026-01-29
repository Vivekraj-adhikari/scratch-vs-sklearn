import numpy as np
class LinearRegression1:
    def __init__(self, weights = None, bias = 0, alpha = 0.001, epochs = 1000):
        """
            Linear Regression Model from scratch using Gradient Descent
            This model tries to minimize the MSE using Gradient Descent

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

    def _gradient_descent(self, X, Y):
        """
            Performs Gradient Descent

            Parameters
            ----------
            X: matrix like
                - Input Features
            Y: 1D arraylike
                - Outputs
            
            Returns
            -------
            tuple
                Updated weights and bias
        """
        data_length = len(X)
        if self.weights is None:
            self.weights = np.zeros(len(X[0]))
        weights_gradients = np.zeros(len(X[0]))
        bias_gradient = 0
        
        for i in range(data_length):
            features = np.array(X[i])
            prediction = np.dot(self.weights, features) + self.bias
            error = prediction - Y[i]
            weights_gradients += features * error
            bias_gradient += error
        
        self.weights = self.weights - self.alpha * (weights_gradients / data_length)
        self.bias = self.bias - self.alpha * bias_gradient / data_length

        return self.weights, self.bias


    def fit(self, X, Y):
        """
            Train the model using Batch Gradient Descent

            Parameters
            ----------
            X: matrix like
                - Input Features
            Y: 1D arraylike
                - Outputs
            
            Returns
            -------
            None
        """
        for i in range(self.epochs):
            self.weights, self.bias = self._gradient_descent(X, Y)
            y_pred = np.dot(X, self.weights) + self.bias
            loss = np.mean((y_pred - Y) ** 2)
            self.losses.append(loss)


    def predict(self, X):
        """
            Train the model using Batch Gradient Descent

            Parameters
            ----------
            X: matrix like
                - Input Features
            
            Returns
            -------
            list
                - Predicted Outputs
        """
        test_length = len(X)
        test_prediction = []
        for i in range(test_length):
            features = np.array(X[i])
            y = np.dot(self.weights, features) + self.bias
            test_prediction.append(round(float(y), 3))
        return test_prediction
    
    def mean_squared_error(self, test, prediction):
        """
            Train the model using Batch Gradient Descent

            Parameters
            ----------
            test: arraylike
                - Actual Outputs
            prediction: arraylike
                - Predicted Outputs
            
            Returns
            -------
            float
                - Mean Squared Error
        """
        test_length = len(test)
        test = np.array(test)
        prediction = np.array(prediction)
        squared_difference = ((test - prediction) ** 2)
        mse = np.sum(squared_difference)
        mse /= test_length
        return mse
    
    def model_score(self, test, prediction):
        """
            Train the model using Batch Gradient Descent

            Parameters
            ----------
            test: arraylike
                - Actual Outputs
            prediction: arraylike
                - Predicted Outputs
            
            Returns
            -------
            float
                - R Squared Score
        """
        test = np.array(test)
        prediction = np.array(prediction)
        mean = np.mean(test)
        tss = np.sum(((test - mean) ** 2))
        rss = np.sum((test - prediction) ** 2)
        score = 1 - (rss / tss)
        return score
    


        