from sklearn.feature_selection import RFE
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.colors as mcolors
import math

# FUNCTION: returns new order given the transform ('rotate'/'mirror'), default: rotate. 
# also give dimensions of model EWxNS
def getNewOrder(numPopEW, numPopNS,transform='mirror'):
    newOrder = []
    # set new order for mirror - 5 4 3 2 1; 10 9 8 7 6; 15, 14 13...
    if(transform == 'mirror'):
        for i in range(numPopNS):
            for j in range(numPopEW, 0, -1):
                cur = (i * numPopEW) + j
                newOrder.append(cur)
    # set new order for rotation: 5 10 15 20 25; 4 9 14 19 24; 3 8 13 18 23...
    elif(transform == 'rotate'):      
        for i in range(numPopEW):
            for j in range(1,numPopNS+1):
                cur = (j*numPopEW) - i
                newOrder.append(cur)
    return newOrder
    

numPopEW = 5
numPopNS = 5
sampleSize = 10
populations = numPopEW * numPopNS
N = 1000

file = "output_jobarray/EW.5_NS.5_N.1000_n.10_asym_input.txt"
data = pd.read_csv(file, sep = "\t", header = None)
data = data.dropna(axis='columns')


fst_start = populations * sampleSize * 2 + 4
y = data.iloc[:, 0:4]

newOrder = getNewOrder(5,5, transform = 'rotate')

afX = data.iloc[:,4:fst_start]

afLabels = []
# set column labels
for i in range(1,populations+1):
    string_SFS = "SFS: " + str(i) + " - "
    for x in range(1,sampleSize *2+1):
        afLabels.append(string_SFS + str(x))

afX.columns = afLabels 

# new labels
afNewLabels = []
for i in range(populations):
    string_SFS = "SFS: " + str(newOrder[i]) + " - "
    for x in range(1,sampleSize *2+1):
        afNewLabels.append(string_SFS + str(x))


afX = afX[afNewLabels] # rearrange

fstX = data.iloc[:,fst_start:]

fstLabels = []
for i in range(1,populations+1):
    for j in range(i+1, populations+1):
        string_FST="FST: " + "[" + str(i) + "," + str(j) + "]"
        fstLabels.append(string_FST)

fstX.columns = fstLabels

fstNewLabels = []
for i in range(0,populations):
    for j in range(i+1, populations):
        if(newOrder[i] > newOrder[j]):
            string_FST="FST: " + "[" + str(newOrder[j]) + "," + str(newOrder[i]) + "]"
        else:
            string_FST="FST: " + "[" + str(newOrder[i]) + "," + str(newOrder[j]) + "]"
        fstNewLabels.append(string_FST)

fstNewLabels

fstX = fstX[fstNewLabels]

X = pd.concat([afX,fstX], axis = 1)

# print(X.columns.values)