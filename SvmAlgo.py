from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class _Svm:
    def __init__(self):
        print("init _Svm")
        self.df = pd.read_csv("heart.csv")
        self.df.sex = pd.factorize(self.df.sex)[0]

    def Q1(self):
        X = self.df[["sex", "age", "fbs"]]
        Y = self.df['output']
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            Svm = svm.SVC(kernel="linear")
            Svm.fit(X_train, Y_train)
            err = Svm.score(X_test, Y_test)
            sum += err

        print("Accuracy chance of heart attack : ", sum / rounds)
