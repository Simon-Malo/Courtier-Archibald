import numpy as np
import pandas as pd

class Model():
    def __init__(self, data, y=None, function=None, proportion: float = 0.0, res=None, shuffle = False, time = None):
        self.fonction = function
        self.shuf = shuffle
        self.data = data
        if time is not None:
            self.data = self.time_strat(self.data, t = time)
            self.data = self.df.apply (pd.to_numeric, errors='coerce')
            self.data = self.df.dropna()
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
        # np.random.shuffle(self.don)
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
        return np.diff([0]+liste)

    def time_strat(self, data, t = 60):
        i = 0
        d = []
        liste = []
        for index, row in data.iterrows():
            if i <= t:
                d.append(["NaN","NaN","NaN","NaN","NaN"])
            else:
                d.append(liste)
                for _ in range(5):
                    liste.pop()
            liste.insert(0, row['Open'])
            liste.insert(0, row['Volume'])
            liste.insert(0, row['RSI'])
            liste.insert(0, row['diff_M_G'])
            liste.insert(0, row['Vortex'])
            i += 1
        h = pd.DataFrame(d)
        h.index = data.index
        r = pd.concat([data, h], axis=1)
        return r
