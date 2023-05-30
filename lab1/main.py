import csv
import pandas as pd

x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x8 = []
x9 = []
x10 = []
Y = []

def initialize_variables(x1, x2, x3
                         x4, x5, x6
                         x7, x8, x9
                         x10, Y):
    with open('lab1var5.csv', 'r') as rf:
        reader = csv.reader(rf, delimiter=';')
        for row in reader:
            x1.append(row[1])
            x2.append(row[2])
            x3.append(row[3])
            x4.append(row[4])
            x5.append(row[5])
            x6.append(row[6])
            x7.append(row[7])
            x8.append(row[8])
            x9.append(row[9])
            x10.append(row[10])
            Y.append(row[11])

print(Y)
