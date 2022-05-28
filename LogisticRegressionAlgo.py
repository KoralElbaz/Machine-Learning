from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class _LogisticRegression:
    def __init__(self):
        print("init _LogisticRegression")
        self.df = pd.read_csv("heart.csv")
        self.df.sex = pd.factorize(self.df.sex)[0]
