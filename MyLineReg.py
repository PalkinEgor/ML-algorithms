import random
import numpy as np


class MyLineReg:

    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None,
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)
        if self.sgd_sample is not None:
            if 0 < self.sgd_sample <= 1:
                self.sgd_sample = int(round(X.shape[0] * self.sgd_sample, 0))
        new_X = X.copy()
        new_X.insert(0, "x0", 1)
        self.weights = np.ones(new_X.shape[1])
        predictions_sample = None
        sample_rows_idx = None

        for i in range(1, self.n_iter + 1):
            if self.sgd_sample is not None:
                sample_rows_idx = random.sample(range(new_X.shape[0]), self.sgd_sample)
            predictions = np.matmul(new_X, self.weights)  # make predictions
            if self.sgd_sample is not None:
                predictions_sample = np.matmul(new_X.iloc[sample_rows_idx], self.weights)

            loss = sum((predictions - y) ** 2) / new_X.shape[0]  # calculating loss function
            if self.reg == "l1":  # calculating loss with regularization
                loss += self.l1_coef * sum(abs(self.weights))
            if self.reg == "l2":
                loss += self.l2_coef * sum(self.weights ** 2)
            if self.reg == "elasticnet":
                loss += self.l1_coef * sum(abs(self.weights)) + self.l2_coef * sum(self.weights ** 2)

            if self.sgd_sample is not None:  # calculating gradient
                gradient = np.matmul((predictions_sample - y.iloc[sample_rows_idx]),
                                     new_X.iloc[sample_rows_idx]) * 2 / len(sample_rows_idx)
            else:
                gradient = np.matmul((predictions - y), new_X) * 2 / new_X.shape[0]
            if self.reg == "l1":  # calculating gradient with regularization
                gradient += self.l1_coef * np.sign(self.weights)
            if self.reg == "l2":
                gradient += self.l2_coef * 2 * self.weights
            if self.reg == "elasticnet":
                gradient += self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights

            if callable(self.learning_rate):  # update weights
                self.weights = self.weights - self.learning_rate(i) * gradient
            else:
                self.weights = self.weights - self.learning_rate * gradient

            if self.metric is not None:  # calculate metrics
                self.score = getattr(self, self.metric)(y, np.matmul(new_X, self.weights))

            if verbose:  # print logs
                if i % verbose == 0:
                    if self.metric is None:
                        print(f"iteration: {i}, loss: {loss}")
                    else:
                        print(f"iteration: {i}, loss: {loss}, {self.metric}: {self.score}")

    def predict(self, X):
        new_X = X.copy()
        new_X.insert(0, "x0", 1)
        return np.array(np.matmul(new_X, self.weights))

    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(abs(y_true - y_pred))

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def rmse(y_true, y_pred):
        return (np.mean((y_true - y_pred) ** 2)) ** 0.5

    @staticmethod
    def mape(y_true, y_pred):
        return np.mean(abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def r2(y_true, y_pred):
        mean = np.mean(y_true)
        return 1 - (sum((y_true - y_pred) ** 2) / sum((y_true - mean) ** 2))

    def get_coef(self):
        return np.array(self.weights[1:])

    def get_best_score(self):
        return self.score
