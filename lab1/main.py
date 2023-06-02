<<<<<<< HEAD
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
=======
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
>>>>>>> e183f56 (Initial commit)
