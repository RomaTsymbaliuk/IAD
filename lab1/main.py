import csv
import pandas as pd
import numpy as np

def initialize_variables():
    df = pd.read_csv('lab1/lab1var5.csv', delimiter=';', decimal=",")
    X = df.iloc[: , 1:11]
    Y = df.iloc[: , 11]
    return X,Y

def normalize_X(X):
    X_normalized = X.copy(deep=True)
    for column in X_normalized:
        mean = X[column].mean()
        std = X[column].std()
        for i, x in enumerate(X_normalized[column]):
            X_normalized[column][i] = (x - mean) / std
    return X_normalized
X,Y = initialize_variables()
X_normalized = normalize_X(X)
print(X_normalized)
