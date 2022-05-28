from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class _Adaboost:

    def __init__(self):
        print("init _Adaboost")
        self.df = pd.read_csv("heart.csv")
        self.df.sex = pd.factorize(self.df.sex)[0]
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(self.df)

    def Q1(self):
        X = self.df[["sex", "age", "fbs"]]
        Y = self.df['output']
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            AdaBoost = AdaBoostClassifier()
            AdaBoost.fit(X_train, Y_train)
            err = AdaBoost.score(X_test, Y_test)
            sum += err

        print("Accuracy chance of heart attack : ", sum / rounds)
