import csv
import pandas as pd
import numpy as np

def initialize_variables():
    df = pd.read_csv('lab1/lab1var5.csv', delimiter=';', decimal=",")
    X = df.iloc[: , 1:11]
    Y = df.iloc[: , 11]
    return X,Y

def normalize_X(X):
    X_norm = X.copy(deep=True)
    for column in X_norm:
        mean = X[column].mean()
        std = X[column].std()
        for i, x in enumerate(X_norm[column]):
            X_norm[column][i] = (x - mean) / std
    return X_norm.to_numpy()

def correlation_matrix_calculate(X_norm):
    N =  X_norm.shape[1]
    X_transp = np.divide(X_norm.transpose(), N)
    X_result = np.dot(X_transp, X_norm)
    return X_result

X,Y = initialize_variables()
X_normalized = normalize_X(X)
X_normalized = correlation_matrix_calculate(X_normalized)
print(X_normalized)
