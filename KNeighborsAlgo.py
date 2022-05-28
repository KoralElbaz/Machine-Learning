from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class _KNN:
    def __init__(self):
        print("init _KNN")
        self.df = pd.read_csv("heart.csv")
        self.df.sex = pd.factorize(self.df.sex)[0]


