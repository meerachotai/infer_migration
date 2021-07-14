# --------------- WITH L1 PENALTY ----------------------------

from sklearn.feature_selection import RFE
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn import linear_model
import pandas as pd
import matplotlib.colors as mcolors
import math

numPopEW = 5
numPopNS = 5
sampleSize = 10
populations = numPopEW * numPopNS
N = 1000
file = "EW.5_NS.5_N.1000_n.10_1_lowest_input.txt"
cv = 3
alpha = 0.1

data = pd.read_csv(file, sep = "\t", header = None)
data = data.dropna(axis='columns')

data= data.sample(frac=1).reset_index(drop=True)
data = (np.log(data)).replace(-np.inf, 0) 

X = data.iloc[:,1:]
y = data.iloc[:, 0]

test_size = math.floor(len(data) / cv)
fig, ax = plt.subplots(1,1,figsize = (8,8))

i=0
for j in range(cv):
    mask = list(range(i, i+test_size))
    not_mask = [m for m in range(0,len(data)) if m not in mask]
    
    print(mask)
    print(not_mask)
    i += test_size
    
    dfTrain = data.iloc[not_mask,:]
    dfTest = data.iloc[mask,:]

    trainy = dfTrain.iloc[:,0]
    trainX = dfTrain.iloc[:,1:]

    
    testy = dfTest.iloc[:,0]
    testX = dfTest.iloc[:,1:]
    
    clf_lin = linear_model.Lasso(alpha=alpha)
    clf_lin = clf_lin.fit(trainX, trainy)
    predictY_lin_test = np.array(clf_lin.predict(testX))
    actualY_lin_test = np.array(testy)

    predictY_lin_train = np.array(clf_lin.predict(trainX))
    actualY_lin_train = np.array(trainy)
        
    # convert back
    predictY_lin_test = np.exp(predictY_lin_test)
    actualY_lin_test = np.exp(actualY_lin_test)
    predictY_lin_train = np.exp(predictY_lin_train)
    actualY_lin_train = np.exp(actualY_lin_train)
    
    errorY_lin_train = (predictY_lin_train - actualY_lin_train) / actualY_lin_train
    errorY_lin_test = (predictY_lin_test - actualY_lin_test) / actualY_lin_test
    
    if(j == 0):
        ax.scatter(actualY_lin_train,errorY_lin_train, facecolors='dodgerblue', label="train") # 'bo'
        ax.scatter(actualY_lin_test,errorY_lin_test, facecolors='indianred',label="test") # 'ro'
    else:
        ax.scatter(actualY_lin_train,errorY_lin_train,facecolors='dodgerblue', label='_nolegend_') #'bo'
        ax.scatter(actualY_lin_test,errorY_lin_test,facecolors='indianred',label='_nolegend_')
    ax.set_ylabel("error in predicted migration rates")
    ax.set_xlabel("actual migration rate (m)")
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax.axhline(y=0, color='k', linestyle='--')
    
FS = "\nL1 penalty: " + str(alpha)
fig.suptitle("number of m tested every iteration: " + str(test_size) + " out of " + str(len(data)) + "\ntotal iterations: " + str(cv) + FS)
fig.tight_layout()
