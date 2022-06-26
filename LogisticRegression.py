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
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            logistic = LogisticRegression(random_state=1, max_iter=250)
            logistic.fit(X_train, Y_train)
            sum += logistic.score(X_test, Y_test)

        return float("{0:.3f}".format(sum / rounds * 100))

    def Q2(self):
        self.df = h.change_bmi(self.df)
        X = self.df[
            ["gender", "ever_married", "smoking_status", "bmi"]]
        Y = self.df["hypertension"]
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            logistic = LogisticRegression(random_state=1)
            logistic.fit(X_train, Y_train)
            sum += logistic.score(X_test, Y_test)

        print("Accuracy chance of hypertension : ", sum / rounds)

    def Q3(self):
        X = self.df[["gender", "Residence_type", "work_type", "smoking_status", "age"]]
        Y = self.df['ever_married']
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            logistic = LogisticRegression(random_state=1)
            logistic.fit(X_train, Y_train)
            sum += logistic.score(X_test, Y_test)

        print("Accuracy chance of ever married : ", sum / rounds)

    def Q4(self):
        X = self.df[["stroke", "heart_disease", "heart_disease"]]
        Y = self.df['age']
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            logistic = LogisticRegression(random_state=1)
            logistic.fit(X_train, Y_train)
            sum += logistic.score(X_test, Y_test)

        print("Accuracy chance of age>43.22: ", sum / rounds)
