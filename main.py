from Adaboost import _Adaboost
from SVM import _Svm
from LogisticRegression import _LogisticRegression
from Knn import _KNN
from termcolor import colored
import help_ as h
import pandas as pd
import numpy as np


df = h.initialize(pd.read_csv("stroke-data.csv"))

adaboost = _Adaboost(df)
svm = _Svm(df)
lg = _LogisticRegression(df)
knn = _KNN(df)
"""
  ========= Q1 =============
"""
a = adaboost.Q1()
k = knn.Q1()
s = svm.Q1()
l = lg.Q1()
h.result(a, k, s, l, "----- Q1 -----")
#
print(colored('\n\n----- Q1: Accuracy chance of stroke -----\n', 'red'))
print(colored('    Algo             |      Accuracies', 'yellow', ))
alg_str = 'Adaboost:            |       ' + str(a) + '%'
print(colored(alg_str, 'cyan'))

alg_str = 'Knn:                 |       ' + str(k) + '%'
print(colored(alg_str, 'green'))

alg_str = 'SVM:                 |       ' + str(s) + '%'
print(colored(alg_str, 'blue'))

alg_str = 'Logistic Regression: |       ' + str(l) + '%'
print(colored(alg_str, 'magenta'))

"""
  ========= Q2 =============
"""
a = adaboost.Q2()
k = knn.Q2()
s = svm.Q2()
l = lg.Q2()
h.result(a, k, s, l, "----- Q2 -----")

print(colored('\n\n----- Q2: Accuracy chance of hypertension -----\n', 'red'))
print(colored('    Algo             |      Accuracies', 'yellow'))
alg_str = 'Adaboost:            |       ' + str(a) + '%'
print(colored(alg_str, 'cyan'))

alg_str = 'Knn:                 |       ' + str(k) + '%'
print(colored(alg_str, 'green'))

alg_str = 'SVM:                 |       ' + str(s) + '%'
print(colored(alg_str, 'blue'))

alg_str = 'Logistic Regression: |       ' + str(l) + '%'
print(colored(alg_str, 'magenta'))

"""
#   ========= Q3 =============
# """
df = h.oneEncodeDF(df)
a = adaboost.Q3()
k = knn.Q3()
s = svm.Q3()
l = lg.Q3()
h.result(a, k, s, l, "----- Q3 -----")

print(colored('\n\n----- Q3: Accuracy chance of ever married -----\n', 'red'))
print(colored('    Algo             |      Accuracies', 'yellow'))
alg_str = 'Adaboost:            |       ' + str(a) + '%'
print(colored(alg_str, 'cyan'))

alg_str = 'Knn:                 |       ' + str(k) + '%'
print(colored(alg_str, 'green'))

alg_str = 'SVM:                 |       ' + str(s) + '%'
print(colored(alg_str, 'blue'))

alg_str = 'Logistic Regression: |       ' + str(l) + '%'
print(colored(alg_str, 'magenta'))

"""
  ========= Q4 =============
"""
df['age'] = np.where(df['age'] < 43.22, 0, 1)  # age_avg = 43.22
a = adaboost.Q4()
k = knn.Q4()
s = svm.Q4()
l = lg.Q4()
h.result(a, k, s, l, "----- Q4 -----")

print(colored('\n\n----- Q4: Accuracy chance of age > 43.22 -----\n', 'red'))
print(colored('    Algo             |      Accuracies', 'yellow'))
alg_str = 'Adaboost:            |       ' + str(a) + '%'
print(colored(alg_str, 'cyan'))

alg_str = 'Knn:                 |       ' + str(k) + '%'
print(colored(alg_str, 'green'))

alg_str = 'SVM:                 |       ' + str(s) + '%'
print(colored(alg_str, 'blue'))

alg_str = 'Logistic Regression: |       ' + str(l) + '%'
print(colored(alg_str, 'magenta'))


"""
  ========= Q5 =============
"""
h.Q5()
