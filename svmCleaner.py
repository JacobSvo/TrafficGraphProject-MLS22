import numpy as np
import pandas as pd

train_data = np.loadtxt("./train.csv", 
                        delimiter=",", skiprows=1, dtype = str)
test_data = np.loadtxt("./test.csv", 
                        delimiter=",", skiprows=1, dtype = str)

def cleanData(data):
    maleAmount = 0
    maleSum = 0
    femaleAmount = 0
    femaleSum = 0
    pos= 0
    if len(data[0]) == 12:
        pos = 1

    # DELETE NAMES FROM DATA
    data = deleteNames(data)

    for i in range(len(data)):
        val = data[i][3-pos]
        if (data[i][4-pos] != '' ):
            if (val == 'male'):
                maleAmount += 1
                maleSum += float(data[i][4-pos])
            else:
                femaleAmount += 1
                femaleSum += float(data[i][4-pos])

    # FIND MEANS FOR BOTH GENDERS
    maleMean = np.trunc(maleSum/maleAmount)
    femaleMean = np.trunc(femaleSum/femaleAmount)

    print("MALE ->", maleMean, "FEMALE ->", femaleMean)

    # Fill in missing ages
    for i in range(len(data)):
        val = data[i][3]
        if (data[i][4] == '' ):
            if (val == 'male'):
                data[i][4] = maleMean
            else:
                data[i][4] = femaleMean
    return data

def driver ():
    global train_data, test_data
    train_data = cleanData(train_data)
    test_data = cleanData(test_data)

def deleteNames(data):

    pos = 3
    if len(data[0]) == 12:
        pos -= 1
    data = np.delete(data, pos, 1)
    data = np.delete(data, pos, 1)
    return data

driver()