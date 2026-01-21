import numpy as np
class LinearRegression:
    def __init__(self, weights = None, bias = 0, alpha = 0.0001, epochs = 1000):
        self.weights = weights
        self.bias = bias
        self.alpha = alpha
        self.epochs = epochs

    def _gradient_descent(self, X, Y):
        data_length = len(X)
        if self.weights == None:
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
        for _ in range(self.epochs):
            self.weights, self.bias = self._gradient_descent(X, Y)


    def predict(self, X):
        test_length = len(X)
        test_prediction = []
        for i in range(test_length):
            features = np.array(X[i])
            y = np.dot(self.weights, features) + self.bias
            test_prediction.append(round(float(y), 3))
        return test_prediction
    
    def mean_squared_error(self, test, prediction):
        test_length = len(test)
        test = np.array(test)
        prediction = np.array(prediction)
        squared_difference = ((test - prediction) ** 2)
        mse = np.sum(squared_difference)
        mse /= test_length
        return mse
    
    def model_score(self, test, prediction):
        test = np.array(test)
        prediction = np.array(prediction)
        mean = np.mean(test)
        tss = np.sum(((test - mean) ** 2))
        rss = np.sum((test - prediction) ** 2)
        score = 1 - (rss / tss)
        return score
    


        