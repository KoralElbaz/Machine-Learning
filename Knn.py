from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class _KNN:
    def __init__(self):
        self.df = pd.read_csv("stroke-data.csv")
        self.df.gender = pd.factorize(self.df.gender)[0]
        self.df['age'] = np.where(self.df['age'] < 43.22, 0, 1)  # age_avg = 43.22
        self.df.ever_married = pd.factorize(self.df.ever_married)[0]
        self.df.Residence_type = pd.factorize(self.df.Residence_type)[0]
        self.df['bmi'] = self.df['bmi'].replace(to_replace=['None'], value=[0])
        self.df.smoking_status = pd.factorize(self.df.smoking_status)[0]

    def Q1(self):
        X = self.df[
            ["gender", "age", "hypertension", "heart_disease", "ever_married", "Residence_type", "avg_glucose_level",
             "smoking_status"]]
        Y = self.df['stroke']
        rounds = 50
        sum = 0

        for round in range(rounds):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.50, random_state=None)
            KNN = KNeighborsClassifier(n_neighbors=15)
            KNN.fit(X_train, y_train)
            sum += KNN.score(X_test, y_test)

        print("Accuracy chance of stroke : ", sum / rounds)
