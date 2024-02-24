import numpy as np
import pandas as pd


class MyKNNClf:

    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = None
        self.x_train = None
        self.y_train = None

    def __str__(self):
        return f"MyKNNClf class: k={self.k}"

    def fit(self, x, y):
        self.x_train = x.copy()
        self.y_train = y.copy()
        self.train_size = x.shape

    def _calculate_distances(self, x_test, i):
        distances = []
        if self.metric == 'euclidean':
            distances = list(map(lambda x: np.linalg.norm(np.array(x_test)[i] - x), np.array(self.x_train)))
        if self.metric == 'manhattan':
            distances = list(map(lambda x: np.sum(np.abs(np.array(x_test)[i] - x)), np.array(self.x_train)))
        if self.metric == 'chebyshev':
            distances = list(map(lambda x: np.max(np.abs(np.array(x_test)[i] - x)), np.array(self.x_train)))
        if self.metric == 'cosine':
            distances = list(map(lambda x: 1 - ((np.sum(np.array(x_test)[i] * x)) / (
                    np.linalg.norm(np.array(x_test)[i]) * np.linalg.norm(x))), np.array(self.x_train)))
        return distances

    def _calculate_class(self, distances, sort_indexes):
        result = []
        if self.weight == 'uniform':
            result.append(list(self.y_train[sort_indexes]).count(0) / self.k)
            result.append(list(self.y_train[sort_indexes]).count(1) / self.k)
        if self.weight == 'rank':
            class0_weight = 0
            class1_weight = 0
            for i in range(self.k):
                if self.y_train[sort_indexes[i]] == 0:
                    class0_weight += 1 / (i + 1)
                else:
                    class1_weight += 1 / (i + 1)
            result.append(class0_weight / (sum(1 / i for i in range(1, self.k + 1))))
            result.append(class1_weight / (sum(1 / i for i in range(1, self.k + 1))))
        if self.weight == 'distance':
            class0_weight = 0
            class1_weight = 0
            for i in sort_indexes:
                if self.y_train[i] == 0:
                    class0_weight += 1 / distances[i]
                else:
                    class1_weight += 1 / distances[i]
            result.append(class0_weight / (sum(1 / distances[i] for i in sort_indexes)))
            result.append(class1_weight / (sum(1 / distances[i] for i in sort_indexes)))
        return result

    def predict(self, x_test):
        result = np.zeros(x_test.shape[0])
        size = x_test.shape[0]
        for i in range(size):
            distances = self._calculate_distances(x_test, i)
            sort_indexes = np.argsort(distances)[:self.k]
            class_weight = self._calculate_class(distances, sort_indexes)
            if class_weight[1] >= class_weight[0]:
                result[i] = 1
        return result.astype(int)

    def predict_proba(self, x_test):
        result = np.zeros(x_test.shape[0])
        size = x_test.shape[0]
        for i in range(size):
            distances = self._calculate_distances(x_test, i)
            sort_indexes = np.argsort(distances)[:self.k]
            class_weight = self._calculate_class(distances, sort_indexes)
            result[i] = class_weight[1]
        return result
