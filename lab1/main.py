import pandas as pd
import numpy as np
import math
from scipy.stats import *
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
import random

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
    X_correlated = np.zeros((X.shape[0], 1))
    print(X_correlated)
    X = X.to_numpy()
    D = np.linalg.inv(R)
    table_pirson = chi2.ppf(q=probability, df=(n * (n - 1) / 2))
    print('table pirson : ', table_pirson, 'calculated pirson ', calculated_pirson)
    if calculated_pirson > table_pirson:
        F = calculate_Phisher(m, n, X, D)
        fisher_table = f.ppf(q=probability, dfd=(m - n), dfn=(n - 1))
        for i, fisher_value in enumerate(F):
            if fisher_value > fisher_table:
                print('Variable x',(i + 1), ' multicolinear with others')
        tkj = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                if i != j:
                    pkj = D[i][j] / (math.sqrt(D[i][i] * D[j][j]))
                    tkj[i][j] = pkj * math.sqrt(m - n) / math.sqrt(1 - pkj * pkj)
        for i, tkj_rows in enumerate(tkj):
            for j, y in enumerate(tkj_rows):
                t_criteria = t.ppf(df=(m - n), q=probability)
                if (abs(y) > t_criteria):
                    print('Variable x', (i + 1), ' colinear ', 'x',(j + 1))
                    X_correlated = np.append(X_correlated, np.array(X[..., i]).reshape(-1, 1), axis=1)
        return X_correlated[:, 1:]
def ADD_DELL(X, y, raiting_function=LinearRegression):
    X_first = ADD(X, y, raiting_function=LinearRegression)
    X = DEL(X_first, y, raiting_function=LinearRegression)
    return X
def DEL_ADD(X, y, raiting_function=LinearRegression):
    X_first = DEL(X, y, raiting_function=LinearRegression)
    X = ADD(X_first, y, raiting_function=LinearRegression)
    return X

def ADD(X, y, raiting_function=LinearRegression, N=1):
    X_copy = X
    X_first = np.zeros((X.shape[0], 1))

    for index in range(0, N):
        Err_array = []
        deleted_indexes = []
        for i in range(X_copy.shape[1]):
            X_add = np.array([X[:, i]])
            X_add = X_add.reshape(-1, 1)
            X_first_copy = np.concatenate((X_first, X_add), axis=1)
            X_first_copy = X_first_copy[:, 1:]
            X_train, X_test, y_train, y_test = train_test_split(X_first_copy, y, test_size=0.2, random_state=42)
            lr = raiting_function().fit(X_train, y_train)
            Err = sum([x for x in abs(y_test - lr.predict(X_test))])
            Err_array.append(Err)
            deleted_indexes.append(i)
            X_copy = np.delete(X, deleted_indexes, 1)
        min_arg_X = np.argmin(Err_array)
        X_first = np.concatenate((X_first, np.array(X[:, min_arg_X]).reshape(-1, 1)), axis=1)
        X_copy = X

    return X_first[:, 1:]

def DEL(X, y, raiting_function=LinearRegression, N=3):

    for index in range(N):
        Err_array = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lr = raiting_function().fit(X_train, y_train)
        Err0 = sum([x for x in abs(y_test.to_numpy() - lr.predict(X_test))])
        for j in range(X.shape[1]):
            X_train, X_test, y_train, y_test = train_test_split(X_copy, y, test_size=0.2, random_state=42)
            lr = raiting_function().fit(X_train, y_train)
            Err = sum([x for x in abs(y_test.to_numpy() - lr.predict(X_test))])
            Err_array.append(Err - Err0)
        min_arg_X = Err_array.index(min(Err_array))
        X = np.delete(X, min_arg_X, 1)

    return X

def Farrar_Glober(X):
    X_normalized = normalize_X(X)
    R, n, m = correlation_matrix_calculate(X_normalized)
    pirson = calculate_Pirson(R, n, m)
    X_correlated = compare_Pirson(pirson, 1 - alpha, n, m, R, X)
    return X_correlated

X,Y = initialize_variables()
X_correlated = Farrar_Glober(X)
#print(pd.DataFrame(X_correlated))
X_valuable = ADD(X_correlated, Y)
print(pd.DataFrame(X_valuable))
#DEL_ADD(X.to_numpy(), Y)