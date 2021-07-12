# --------------- WITH (some) MANUAL FEATURE SELECTION ----------------------------

from sklearn.feature_selection import RFE
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.colors as mcolors
import math

numPopEW = 5
numPopNS = 5
sampleSize = 10
populations = numPopEW * numPopNS
N = 1000
file = "EW.5_NS.5_N.1000_n.10_1_lowest_input.txt"
cv = 5

# manual feature selection
data = pd.read_csv(file, sep = "\t", header = None)
data = data.dropna(axis='columns')

# only use columns that occur as max/min multiple times
SFS_minim = data.iloc[:,1:501].idxmin(axis = 1)
SFS_minim = SFS_minim[SFS_minim.duplicated(keep = False)].drop_duplicates()

SFS_maxim = data.iloc[:,1:501].idxmax(axis = 1)
SFS_maxim = SFS_maxim[SFS_maxim.duplicated(keep = False)].drop_duplicates()

FST_minim = data.iloc[:,501:].idxmin(axis = 1)
FST_minim = FST_minim[FST_minim.duplicated(keep = False)].drop_duplicates()

FST_maxim = data.iloc[:,501:].idxmax(axis = 1)
FST_maxim = FST_maxim[FST_maxim.duplicated(keep = False)].drop_duplicates()

y = pd.Series([0], index = [0])
extremes = pd.concat([y, FST_maxim, FST_minim, SFS_minim, SFS_maxim]).drop_duplicates()
# print(extremes)

# calculate means
SFS_mean = data.iloc[:,1:501].mean(axis = 1) 
FST_mean = data.iloc[:,501:].mean(axis = 1)

data = data.iloc[:,extremes]
# add means
data = data.assign(SFS_mean = SFS_mean)
data = data.assign(FST_mean = FST_mean)

print("data columns now:", len(data.columns))

data= data.sample(frac=1).reset_index(drop=True)
data = (np.log(data)).replace(-np.inf, 0) 

test_size = math.floor(len(data) / cv)
fig, ax = plt.subplots(1,1,figsize = (10,10))

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
    
    clf_lin = LinearRegression().fit(trainX, trainy)
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
    
FS = "\nmanual feature selection: using max/min columns that appear > 1 times and mean $F_{ST}$ and SFS values, columns: " + str(len(data.columns))
fig.suptitle("number of m tested every iteration: " + str(test_size) + " out of " + str(len(data)) + "\ntotal iterations: " + str(cv) + FS)
fig.tight_layout()

fig.savefig("predictMig_LR_logged_manual_FS_" + str(cv) +".png")