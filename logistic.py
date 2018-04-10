# CISC 5800 Machine Learning Fall 2017
# Final Project -- SPECTF Heart Data Set
# Due: Dec.13
# Muye Zhao
# logistic.py

import numpy as np
import random
from math import exp


def sigmoid(x):
    if len([x]) > 1:
        raise ValueError('wrong value in sigmoid function, dimension of x is %s', x.shape)
    else:
        try:
            ret = 1 / (1 + exp(-x))
        except OverflowError:
            ret = 0
        return ret


class LogisticRegreesionClassifier:

    def __init__(self, penalty='l2', epsilon=0.1, maxiter=500, lmbda=10):
        self.penaly = penalty
        self.w = np.array([])
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.lmbda = lmbda
        self.get_params()

    def __str__(self):
        s = "LogisticRegreesion("
        dic = self.get_params()
        for key in dic:
            s = s + str(key) + "=" + str(dic[key]) + ", "
        s = s + ")"
        return s

    def fit(self, x, y):
        n_samples, n_features = x.shape
        # random pick initial weight in (0, 1)
        w = np.array([])
        for i in range(0, n_features+1):
            w = np.append(w, random.randint(1, 10000) / 10000)

        lmbda = self.lmbda
        for k in range(0, self.maxiter):
            for indexi, i in enumerate(x):
                xi = np.append(i, 1)    # x = [x, 1]
                mul = w.dot(xi.transpose())    # w^Tx + b
                error = y[indexi] - sigmoid(mul)    # y - g(w^Tx+b)
                if self.penaly == 'l2':
                    var = self.epsilon * (error * xi - w * (1 / lmbda))
                elif self.penaly == 'l1':
                    var = self.epsilon * (error * xi - np.ones((1, n_features+1)) * (1 / lmbda))
                else:
                    var = self.epsilon * (error * xi)    # w = w + e(x(y - g(w^Tx+b)))
                w = np.add(w, var)
        self.w = w

    def predict(self, x):
        ans = []
        for indexi, i in enumerate(x):
            mul = sigmoid(self.w.dot(np.append(i, 1)))
            if mul >= 0.5:
                ans.append(1)
            else:
                ans.append(0)
        return ans

    def get_params(self):
        ret = dict()
        ret["penalty"] = self.penaly
        ret["epsilon"] = self.epsilon
        ret["maxiter"] = self.maxiter
        ret["lambda"] = self.lmbda
        return ret
