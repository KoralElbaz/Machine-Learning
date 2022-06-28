import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


def print_change(df):
    bmi = [x for x in df['bmi'] if x == 1]
    glucose = [x for x in df['avg_glucose_level'] if x == 1]
    print("bmi counter: ", len(bmi))
    print("glucose counter: ", len(glucose))


def initialize(df):
    df.gender = pd.factorize(df.gender)[0]
    df['age'] = np.where(df['age'] < 43.22, 0, 1)  # age_avg = 43.22
    df.ever_married = pd.factorize(df.ever_married)[0]
    df.Residence_type = pd.factorize(df.Residence_type)[0]
    df['bmi'] = df['bmi'].replace(to_replace=['None'], value=[-1])
    df.smoking_status = pd.factorize(df.smoking_status)[0]
    df.Residence_type = pd.factorize(df.Residence_type)[0]
    df.work_type = pd.factorize(df.work_type)[0]
    return df


def change_bmi(df):
    r_num = 0
    for bmi_val in df['bmi']:
        bmi_f = float(bmi_val)
        if bmi_f == -1:
            df = df.drop(labels=r_num)
        elif 18.7 <= bmi_f <= 24.9:
            df['bmi'] = df['bmi'].replace(bmi_val, 1)
        else:
            df['bmi'] = df['bmi'].replace(bmi_val, 0)
        r_num += 1
    return df


def change_avg_glucose_level(df):
    r_num = 0
    for glucose_val in df['avg_glucose_level']:
        glucose_f = float(glucose_val)
        if 90 <= glucose_f <= 110:
            df['avg_glucose_level'] = df['avg_glucose_level'].replace(glucose_val, 1)
        else:
            df['avg_glucose_level'] = df['avg_glucose_level'].replace(glucose_val, 0)
        r_num += 1
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

    # visualizing the plot
    plt.show()


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i])


