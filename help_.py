import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


def initialize(df):
    df.gender = pd.factorize(df.gender)[0]
    df.ever_married = pd.factorize(df.ever_married)[0]
    df.Residence_type = pd.factorize(df.Residence_type)[0]
    df['bmi'] = df['bmi'].replace(to_replace=['None'], value=[-1])
    df.smoking_status = pd.factorize(df.smoking_status)[0]
    df.Residence_type = pd.factorize(df.Residence_type)[0]

    return df


def result(accuracy_Adaboost, accuracy_Knn, accuracy_SVM, accuracy_LogisticRegression, Q_str):
    x = ['Adaboost', 'Knn', 'SVM', 'LogisticRegression']

    y = [accuracy_Adaboost, accuracy_Knn, accuracy_SVM, accuracy_LogisticRegression]

    # setting figure size by using figure() function
    plt.figure(figsize=(10, 5))

    # making the bar chart on the data
    plt.bar(x, y, color=['red', 'green', 'blue', 'yellow'])

    # calling the function to add value labels
    addlabels(x, y)

    # giving title to the plot
    plt.title(Q_str)

    # giving X and Y labels
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracies (In percent)")
    ax = plt.gca()
    ax.set_ylim([0, 100])
    # visualizing the plot
    plt.show()


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i])


def oneEncodeDF(df):
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(df[['work_type']])
    df[ohe.categories_[0]] = transformed.toarray()
    del df['work_type']
    return df

def Q5():
    df = pd.read_csv("stroke-data.csv")
    df = initialize(df)
    df.work_type = pd.factorize(df.work_type)[0]

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
