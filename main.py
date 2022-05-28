from AdaboostAlgo import _Adaboost
from SvmAlgo import _Svm
from DecisionTreeAlgo import _DecisionTree
from LogisticRegressionAlgo import _LogisticRegression
from KNeighborsAlgo import _KNN


adaboost = _Adaboost()
svm = _Svm()
decisionTree = _DecisionTree()
logisticRegression = _LogisticRegression
knn = _KNN

print("\n\n~~~~~Q1~~~~~~~\n")
print("Adaboost:")
adaboost.Q1()
print("\nSvm:")
svm.Q1()
print("\nDecisionTree:")
decisionTree.Q1()
