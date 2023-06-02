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

def add_Del_Algorithm():
    pass

def get_index_matrix(A, vector):
    #for i in range(A.shape[1]):
        #for vector2 in A[..., i]:
            #if np.array_equal(vector2, vector):
                #return i
    print(vector)

def ADD(X, y, raiting_function=LinearRegression):
    N = 6
    X_copy = X
    X_first = np.empty((X.shape[0], 1), float)
    start = 1
    for index in range(N):
        Err_array = []
        for i in range(X_copy.shape[1]):
            X_add = np.array([X_copy[:, i]])
            X_add = X_add.reshape(-1, 1)
            if not start:
                X_first_copy = np.concatenate((X_first, X_add), axis=1)
            else:
                X_first_copy = X_add
            X_train, X_test, y_train, y_test = train_test_split(X_first_copy, y, test_size=0.2, random_state=42)
            lr = raiting_function().fit(X_train, y_train)
            Err = sum([x for x in abs(y_test - lr.predict(X_test))])
            Err_array.append(Err)
        min_arg_X = Err_array.index(min(Err_array))
        print('X', min_arg_X + 1, ' is informative')
        X_first = np.concatenate((X_first, np.array(X_copy[:, min_arg_X]).reshape(-1, 1)), axis=1)
#        print(X_first)
        X_copy = np.delete(X_copy, min_arg_X, 1)
        start = 0

def DEL(X, y, raiting_function=LinearRegression):
    N = 3

    for index in range(N):
        Err_array = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lr = raiting_function().fit(X_train, y_train)
        Err0 = sum([x for x in abs(y_test.to_numpy() - lr.predict(X_test))])

        for j in range(X.shape[1]):
            X_copy = np.delete(X, j, 1)
            X_train, X_test, y_train, y_test = train_test_split(X_copy, y, test_size=0.2, random_state=42)
            lr = raiting_function().fit(X_train, y_train)
            Err = sum([x for x in abs(y_test.to_numpy() - lr.predict(X_test))])
            Err_array.append(Err - Err0)
        min_arg_X = Err_array.index(min(Err_array))
        print('X',min_arg_X, ' is not significant')
        X = np.delete(X, min_arg_X, 1)

def Farrar_Glober(X):
    X_normalized = normalize_X(X)
    R, n, m = correlation_matrix_calculate(X_normalized)
    pirson = calculate_Pirson(R, n, m)
    compare_Pirson(pirson, 1 - alpha, n, m, R, X)

#Farrar_Glober()
X,Y = initialize_variables()
Farrar_Glober(X)
ADD(X.to_numpy(), Y.to_numpy())