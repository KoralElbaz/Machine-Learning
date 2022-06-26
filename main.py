from Adaboost import _Adaboost
from SVM import _Svm
from LogisticRegression import _LogisticRegression
from Knn import _KNN
from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import help_ as h

adaboost = _Adaboost()
svm = _Svm()
lg = _LogisticRegression()
knn = _KNN()
"""
  ========= Q1 =============
"""
# a = adaboost.Q1()
# k = knn.Q1()
# s = svm.Q1()
# l = lg.Q1()
# h.result(a, k, s, l, "----- Q1 -----")
#
# print(colored('\n\n----- Q1: Accuracy chance of stroke -----\n', 'red'))
# print(colored('    Algo             |      Accuracies', 'yellow', ))
# alg_str = 'Adaboost:            |       ' + str(a) + '%'
# print(colored(alg_str, 'cyan'))
#
# alg_str = 'Knn:                 |       ' + str(k) + '%'
# print(colored(alg_str, 'green'))
#
# alg_str = 'SVM:                 |       ' + str(s) + '%'
# print(colored(alg_str, 'blue'))
#
# alg_str = 'Logistic Regression: |       ' + str(l) + '%'
# print(colored(alg_str, 'magenta'))

"""
  ========= Q2 =============
"""
# a = adaboost.Q2()
# k = knn.Q2()
# s = svm.Q2()
# l = lg.Q2()
# h.result(a, k, s, l, "----- Q2 -----")
#
# print(colored('\n\n----- Q2: Accuracy chance of hypertension -----\n', 'red'))
# print(colored('    Algo             |      Accuracies', 'yellow'))
# alg_str = 'Adaboost:            |       ' + str(a) + '%'
# print(colored(alg_str, 'cyan'))
#
# alg_str = 'Knn:                 |       ' + str(k) + '%'
# print(colored(alg_str, 'green'))
#
# alg_str = 'SVM:                 |       ' + str(s) + '%'
# print(colored(alg_str, 'blue'))
#
# alg_str = 'Logistic Regression: |       ' + str(l) + '%'
# print(colored(alg_str, 'magenta'))

# print("\n\n~~~~~Q3~~~~~~~\n")
# print("Adaboost:")
# adaboost.Q3()
# print("\nKnn:")
# knn.Q3()
# print("\nSvm:")
# svm.Q3()
# print("\nLogistic Regression:")
# lg.Q3()

# print("\n\n~~~~~Q4~~~~~~~\n")
# print("Adaboost:")
# adaboost.Q4()
# print("\nKnn:")
# knn.Q4()
# print("\nSvm:")
# svm.Q4()
# print("\nLogistic Regression:")
# lg.Q4()


df = pd.read_csv("stroke-data.csv")
df = h.initialize(df)

# Show width
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Checking the importance of each column with ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x, y)
# Do a pie model of size 11 of the most to the least importance from the columns dataset
slices = model.feature_importances_
activities = model.feature_names_in_
cols = ['olive', 'cyan', 'purple', 'blue', 'pink', 'red', 'gold', 'yellowgreen', 'lightcoral', 'lightskyblue',
        'orangered']
# plotting the pie chart
plt.pie(slices, labels=activities, colors=cols,
        startangle=90, shadow=True, explode=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        radius=1.4, autopct='%1.1f%%')

plt.show()
