from Adaboost import _Adaboost
from SVM import _Svm
from LogisticRegression import _LogisticRegression
from Knn import _KNN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import help_ as h


adaboost = _Adaboost()
svm = _Svm()
lg = _LogisticRegression()
knn = _KNN()

# print("\n\n~~~~~Q1~~~~~~~\n")
# print("Adaboost:")
# adaboost.Q1()
# print("\nKnn:")
# knn.Q1()
# print("\nSvm:")
# svm.Q1()
# print("\nLogistic Regression:")
# lg.Q1()

# print("\n\n~~~~~Q2~~~~~~~\n")
# print("Adaboost:")
# adaboost.Q2()
# print("\nKnn:")
# knn.Q2()
# print("\nSvm:")
# svm.Q2()
# print("\nLogistic Regression:")
# lg.Q2()

# print("\n\n~~~~~Q3~~~~~~~\n")
# print("Adaboost:")
# adaboost.Q3()
# print("\nKnn:")
# knn.Q3()
# print("\nSvm:")
# svm.Q3()
# print("\nLogistic Regression:")
# lg.Q3()

print("\n\n~~~~~Q4~~~~~~~\n")
print("Adaboost:")
adaboost.Q4()
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
# Print the importance of each column by its order
print(model.feature_importances_)
# Do a pie model of size 10 of the most to the least importance from the columns dataset
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='pie')
plt.show()