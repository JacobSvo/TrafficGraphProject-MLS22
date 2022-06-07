import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

PANDAS_AXIS_COLUMN = 1
PANDAS_AXIS_ROWS = 0
NUMPY_AXIS_COLUMN = 1
NUMPY_AXIS_ROW = 0
DATA_ZERO = 0.0001


def format_csv(file):
    df = pd.read_csv('titanic/' + str(file))
    df = df.drop(columns=['Ticket', 'Name'])
    df["Embarked"].replace({"S": "1.0", "C": "2.0", "Q": "3.0"}, inplace=True)
    df["Sex"].replace({"male": 0, "female": 1}, inplace=True)
    mapping = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5", "F": "6", "G": "7", "H": "8", "I": "9",
               "J": "10", "K": "11", "L": "12", "M": "13", "N": "14", "O": "15", "P": "16", "Q": "17", "R": "18",
               "S": "19", "T": "20", "U": "21", "V": "22", "W": "23", "X": "24", "Y": "25", "Z": "26"}
    for k, v in mapping.items():
        df["Cabin"] = df["Cabin"].str.replace(k, v)

    # replace cabin value with the number of cabins
    cabins = df["Cabin"]
    temp = []
    for idx, itm in enumerate(cabins):
        print(idx, end=' ')
        print(itm, end=' ')
        print(str(itm).split(), end=' ')
        print(len(str(itm).split()))
        if str(itm) == 'nan':
            temp.append(DATA_ZERO)
        else:
            temp.append(str(len(str(itm).split())))

    df["Cabin"] = temp
    # remove nulls
    df = df.fillna(DATA_ZERO)
    df.to_csv('titanic/pp_' + str(file))


if __name__ == '__main__':
    format_csv('train.csv')
    format_csv('test.csv')
