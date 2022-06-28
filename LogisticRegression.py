from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import help_ as h


class _LogisticRegression:
    def __init__(self):
        self.df = pd.read_csv("stroke-data.csv")
        self.df = h.initialize(self.df)

    def Q1(self):
        X = self.df[["gender", "age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level",
                     "smoking_status"]]
        Y = self.df['stroke']

        sum, rounds = fit_algo(X, Y)
        return float("{0:.3f}".format(sum / rounds * 100))

    def Q2(self):
        X = self.df[["gender", "heart_disease", "ever_married", "smoking_status", "bmi"]]
        Y = self.df["hypertension"]

        sum, rounds = fit_algo(X, Y)
        return float("{0:.3f}".format(sum / rounds * 100))

    def Q3(self):
        X = self.df[["gender", "Residence_type", "work_type", "smoking_status", "age"]]
        Y = self.df['ever_married']

        sum, rounds = fit_algo(X, Y)
        return float("{0:.3f}".format(sum / rounds * 100))

    def Q4(self):
        self.df['age'] = np.where(self.df['age'] < 43.22, 0, 1)  # age_avg = 43.22
        X = self.df[["stroke", "heart_disease", "hypertension", "avg_glucose_level", "bmi"]]
        Y = self.df['age']

        sum, rounds = fit_algo(X, Y)
        return float("{0:.3f}".format(sum / rounds * 100))

def fit_algo(X, Y):
    rounds = 50
    sum = 0
    for round in range(rounds):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
        logistic = LogisticRegression(random_state=1, max_iter=250)
        logistic.fit(X_train, Y_train)
        sum += logistic.score(X_test, Y_test)
    return sum, rounds