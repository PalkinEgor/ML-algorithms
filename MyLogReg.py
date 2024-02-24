import pandas as pd
import numpy as np


class MyLogReg:
    _eps = 1e-15

    def __init__(self, n_iter=10, learning_rate=0.1, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.score = None

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, x, y, verbose=False):
        new_x = x.copy()
        new_x.insert(0, "x0", 1)
        self.weights = np.ones(new_x.shape[1])
        for i in range(1, self.n_iter + 1):
            proba = 1 / (1 + np.exp(-np.matmul(new_x, self.weights)))
            loss = -(y * np.log(proba + self._eps) + (1 - y) * np.log(1 - proba + self._eps)) / new_x.shape[0]
            gradient = (np.matmul(proba - y, new_x)) / new_x.shape[0]
            self.weights = self.weights - self.learning_rate * gradient

            if self.metric is not None:
                self.score = getattr(self, self.metric)(y, 1 / (1 + np.exp(-np.matmul(new_x, self.weights))))

            if verbose and verbose % i == 0:
                print(f"iteration: {i}, loss: {loss}, metric: {self.score}")

    @staticmethod
    def func(y_true, y_pred):
        result = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        size = y_true.size
        for i in range(size):
            if y_pred[i] > 0.5:
                if y_true[i] == 1:
                    result["TP"] += 1
                else:
                    result["FP"] += 1
            else:
                if y_true[i] == 1:
                    result["FN"] += 1
                else:
                    result["TN"] += 1
        return result

    @staticmethod
    def accuracy(y_true, y_pred):
        t = MyLogReg.func(y_true, y_pred)
        return (t["TP"] + t["TN"]) / (t["TP"] + t["TN"] + t["FP"] + t["FN"])

    @staticmethod
    def precision(y_true, y_pred):
        t = MyLogReg.func(y_true, y_pred)
        return t["TP"] / (t["TP"] + t["FP"])

    @staticmethod
    def recall(y_true, y_pred):
        t = MyLogReg.func(y_true, y_pred)
        return t["TP"] / (t["TP"] + t["FN"])

    @staticmethod
    def f1(y_true, y_pred):
        return 2 * (MyLogReg.precision(y_true, y_pred) * MyLogReg.recall(y_true, y_pred)) / (
                MyLogReg.precision(y_true, y_pred) + MyLogReg.recall(y_true, y_pred))

    @staticmethod
    def roc_auc(y_true, y_pred):
        sorted_index = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[sorted_index]
        y_pred_sorted = y_pred[sorted_index]
        result = 0
        class_1_index = np.where(y_true_sorted == 1)[0]
        for i in class_1_index:
            result += np.sum((y_true_sorted[i:] == 0) & (y_pred_sorted[i:] != y_pred_sorted[i]))
            result += np.sum((y_true_sorted[i:] == 0) & (y_pred_sorted[i:] == y_pred_sorted[i])) / 2
        return result / (np.sum(y_true == 1) * np.sum(y_true == 0))

    def predict_proba(self, x):
        new_X = x.copy()
        new_X.insert(0, "x0", 1)
        return 1 / (1 + np.exp(-np.matmul(new_X, self.weights)))

    def predict(self, x):
        proba = self.predict_proba(x)
        return np.where(proba > 0.5, 1, 0)

    def get_coef(self):
        return np.array(self.weights[1:])

    def get_best_score(self):
        return self.score
