from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import help_ as h


class _Adaboost:

    def __init__(self):
        self.df = pd.read_csv("stroke-data.csv")
        self.df = h.initialize(self.df)

    def Q1(self):
        X = self.df[["gender", "age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level",
                     "smoking_status"]]
        Y = self.df['stroke']
        sum, rounds = fit_algo(X, Y)

        # print("Accuracy chance of stroke : ", sum / rounds)
        return float("{0:.3f}".format(sum / rounds * 100))


    def Q2(self):
        self.df = h.change_bmi(self.df)
        X = self.df[
            ["gender", "heart_disease", "ever_married", "smoking_status", "bmi"]]
        Y = self.df['hypertension']
        sum, rounds = fit_algo(X, Y)

        return float("{0:.3f}".format(sum / rounds * 100))

    def Q3(self):
        X = self.df[["gender", "Residence_type", "work_type", "smoking_status"]]
        Y = self.df['ever_married']
        sum, rounds = fit_algo(X, Y)

        print("Accuracy chance of ever married : ", sum / rounds)

    def Q4(self):
        # self.df = h.change_avg_glucose_level(self.df)
        # self.df = h.change_bmi(self.df)
        h.print_change(self.df)
        # X = self.df[["stroke", "heart_disease", "smoking_status", "hypertension", "avg_glucose_level", "bmi"]]
        X = self.df[["stroke", "heart_disease", "hypertension", "avg_glucose_level", "bmi"]]  # 0.68
        Y = self.df['age']
        sum, rounds = fit_algo(X, Y)
        print("Accuracy chance of age>43.22: ", sum / rounds)


def fit_algo(X, Y):
    rounds = 50
    sum = 0

    for round in range(rounds):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
        AdaBoost = AdaBoostClassifier()
        AdaBoost.fit(X_train, Y_train)
        sum += AdaBoost.score(X_test, Y_test)
    return sum, rounds
