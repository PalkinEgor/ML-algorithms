import numpy as np
import pandas as pd


class MyKNNReg:

    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = None
        self.x_train = None
        self.y_train = None

    def __str__(self):
        return f"MyKNNReg class: k={self.k}"

    def fit(self, x, y):
        self.x_train = x.copy()
        self.y_train = y.copy()
        self.train_size = x.shape

    def _calculate_distances(self, x_test, i):
        distances = []
        if self.metric == 'euclidean':
            distances = list(map(lambda x: np.linalg.norm(x - np.array(x_test)[i]), np.array(self.x_train)))
        if self.metric == 'manhattan':
            distances = list(map(lambda x: np.sum(np.abs(x - np.array(x_test)[i])), np.array(self.x_train)))
        if self.metric == 'chebyshev':
            distances = list(map(lambda x: np.max(np.abs(x - np.array(x_test)[i])), np.array(self.x_train)))
        if self.metric == 'cosine':
            distances = list(map(lambda x: 1 - (np.sum(x * np.array(x_test)[i])) / (
                    np.linalg.norm(x) * np.linalg.norm(np.array(x_test)[i])), np.array(self.x_train)))
        return distances

    def _calculate_weights(self, distances, sort_indexes):
        if self.weight == 'uniform':
            return np.mean(self.y_train[sort_indexes])
        if self.weight == 'rank':
            weights = list(1 / i for i in range(1, self.k + 1))
            sub_sum = sum(1 / i for i in range(1, self.k + 1))
            weights = list(map(lambda x: x / sub_sum, weights))
            return np.sum(weights * self.y_train[sort_indexes])
        if self.weight == 'distance':
            weights = list(1 / distances[i] for i in sort_indexes)
            sub_sum = sum(1 / distances[i] for i in sort_indexes)
            weights = list(map(lambda x: x / sub_sum, weights))
            return np.sum(weights * self.y_train[sort_indexes])

    def predict(self, x_test):
        result = np.zeros(x_test.shape[0])
        size = x_test.shape[0]
        for i in range(size):
            distances = self._calculate_distances(x_test, i)
            sort_indexes = np.argsort(distances)[:self.k]
            result[i] = self._calculate_weights(distances, sort_indexes)
        return result
