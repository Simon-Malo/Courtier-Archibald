import numpy as np
import pandas as pd

class Model():
    def __init__(self, data, y=None, function=None, proportion: float = 0.0, res=None, shuffle = False):
        self.fonction = function
        self.shuf = shuffle
        self.data = data
        self.reseau = res
        self.data['True'] = y if y is not None else self.fonction(data)
        self.don = self.data.to_numpy()
        self.n_train = int(len(data.index)*(1.0-proportion))
        self.n_test = len(data.index) - self.n_train
        self.y_train = np.zeros(self.n_train)
        self.y_test = np.zeros(self.n_test)
        self.X_train = None
        self.X_test = None
        if self.shuffle : self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.don)
        self.X_train = self.don[:self.n_train, :-1]
        self.y_train = self.don[:self.n_train, -1:]
        self.X_test = self.don[self.n_train:, :-1]
        self.y_test = self.don[self.n_train:, -1:]
        if self.reseau is not None : self.reseau.train(self.X_train, self.y_train, self.X_test, self.y_test)

    def add(self, data, y=None):
        data['true'] = y if y is not None else self.fonction(data)
        self.data.append(data)
        if self.shuffle : self.shuffle()

    def delete(self, *args):
        self.data.drop(args)
        if self.shuffle : self.shuffle()

    def variation(self, liste):
        return np.diff(liste)


