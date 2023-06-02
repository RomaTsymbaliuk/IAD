import pandas as pd
import numpy as np
import math
from scipy.stats import *

alpha = 0.05
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
    M = X_norm.shape[0]
    N = X_norm.shape[1]
    X_transp = np.divide(X_norm.transpose(), N)
    X_result = np.dot(X_transp, X_norm)
    return X_result, N, M

def calculate_Pirson(R, n, m):
    return -(m - 1 - 1/6 * (2 * n + 5)) * np.log(np.linalg.det(R))

def calculate_Phisher(m, n, X, D):
    F = []

    for i in range(X.shape[1]):
        F.append((m - n) / (n - 1) * abs((D[i][i] - 1)))

    return F
def compare_Pirson(calculated_pirson, probability, n, m, R, X):
    D = np.linalg.inv(R)
    table_pirson = chi2.ppf(q=probability, df=(n * (n -1) / 2))
    print('table pirson : ', table_pirson, 'calculated pirson ', calculated_pirson)
    if calculated_pirson > table_pirson:
        F = calculate_Phisher(m, n, X, D)
        fisher_table = f.ppf(q=probability, dfd=(m - n), dfn=(n - 1))
        for i, fisher_value in enumerate(F):
            if fisher_value > fisher_table:
                print('Variable x',(i + 1), ' multicolinear with others')
        tkj = np.zeros((X.shape[1], X.shape[1]))
        for i, x in enumerate(X):
            for j, y in enumerate(X):
                if (i != j):
                    pkj = D[i][j] / (math.sqrt(D[i][i] * D[j][j]))
                    tkj[i][j] = pkj * math.sqrt(m - n) / math.sqrt(1 - pkj * pkj)
        for i, tkj_rows in enumerate(tkj):
            for j, y in enumerate(tkj_rows):
                t_criteria = t.ppf(df=(m - n), q=probability)
                if (abs(y) > t_criteria):
                    print('Variable ',(i + 1), ' colinear ', (j + 1))
def Farrar_Glober():
    X,Y = initialize_variables()
    X_normalized = normalize_X(X)
    R, n, m = correlation_matrix_calculate(X_normalized)
    pirson = calculate_Pirson(R, n, m)
    compare_Pirson(pirson, 1 - alpha, n, m, R, X)

Farrar_Glober()