import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.colors as mcolors
import math
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

numPopEW = 5
numPopNS = 5
sampleSize = 10
N = 1000
# recursive feature elimination (RFE) options:
# at every iteration, _remove_ % of the features will be eliminated
remove = 0.4

# for every type of regression, k-fold cross-validation is used
k = 5

# for train-test split, how much of it is testing?
test = 1/(k+1)
# seed for train-test splitting
seed = 5

file = "output_jobarray/EW.5_NS.5_N.1000_n.10_logAsym_input.txt"


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

# get column labels
def getOrigLabels(populations, sampleSize, which='all'):
    labels = []
    if(which=='sfs'):
        for i in range(1,populations+1):
            string_SFS = "SFS: " + str(i) + " - "
            for x in range(1,sampleSize *2+1):
                labels.append(string_SFS + str(x)) 
    elif(which=='fst'):
        for i in range(1,populations+1):
            for j in range(i+1, populations+1):
                string_FST="FST: " + "[" + str(i) + "," + str(j) + "]"
                labels.append(string_FST)
    elif(which=='all'):
        for i in range(1,populations+1):
            string_SFS = "SFS: " + str(i) + " - "
            for x in range(1,sampleSize *2+1):
                labels.append(string_SFS + str(x)) 
        for i in range(1,populations+1):
            for j in range(i+1, populations+1):
                string_FST="FST: " + "[" + str(i) + "," + str(j) + "]"
                labels.append(string_FST)
    
    return labels
          
# FUNCTION: returns rearranged feature columns, with original labels
def rearrangeFeatures(numPopEW, numPopNS, sampleSize, X, transform='mirror'):
    
    populations = numPopEW * numPopNS
    fst_start = populations * sampleSize * 2

    newOrder = getNewOrder(numPopEW,numPopNS, transform = transform)
    
    # for SFS
    afX = X.iloc[:,0:fst_start]
    
    afLabels = getOrigLabels(populations, sampleSize, 'sfs')

    afX.columns = afLabels 

    # new labels
    afNewLabels = []
    for i in range(populations):
        string_SFS = "SFS: " + str(newOrder[i]) + " - "
        for x in range(1,sampleSize *2+1):
            afNewLabels.append(string_SFS + str(x))


    afX = afX[afNewLabels] # rearrange
    afX.columns = afLabels # so that another transformation can be done using original column names
    
    # for F_st
    fstX = X.iloc[:,fst_start:]
    
    fstLabels = getOrigLabels(populations, sampleSize, 'fst')

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
    fstX.columns = fstLabels
    
    X = pd.concat([afX,fstX], axis = 1) # concat by column
    
    return X

def RunLinearRegression(trainX, trainy, testX, testy):
    
    # 1. SIMPLE LINEAR REGRESSION ----------------------
    error_test = [[] for _ in range(len(testy))]
    simpleLin = cross_validate(LinearRegression(), trainX, trainy, cv=k,return_estimator = True)
    counter = 0
    for i in simpleLin['estimator']:
        simpleLin_y = i.predict(testX) # pick one of the folds' estimators to predict
        error = (simpleLin_y - testy) / testy
        for j in range(len(testy)):
            error_test[j].append(error.iloc[j])

    # taking the mean for each of the folds' predicted value
    simpleLin_error_test = []
    for i in error_test:
        simpleLin_error_test.append(sum(i)/len(i))

    simpleLin_train = cross_val_predict(LinearRegression(), trainX, trainy, cv=k)
    simpleLin_error_train = (simpleLin_train - trainy) / trainy
    
    # 2. RFE FEATURE SELECTION ---------------------------------

    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
    selector = RFECV(LinearRegression(), step=remove, cv=k).fit(trainX,trainy) # fit using training data only

    # gets *selected* X columns only
    selected_trainX = selector.transform(trainX) 
    selected_testX = selector.transform(testX)

    # conduct linear regression on training+testing using selected features ONLY
    RFELin = cross_validate(LinearRegression(), selected_trainX, trainy, cv=k,return_estimator = True)

    # testing data
    error_test = [[] for _ in range(len(testy))]
    counter = 0
    for i in RFELin['estimator']:
        RFELin_y = i.predict(selected_testX) # pick one of the folds' estimators to predict
        error = (RFELin_y - testy) / testy
        for j in range(len(testy)):
            error_test[j].append(error.iloc[j])

    # taking the mean for each of the folds' predicted value
    RFELin_error_test = []
    for i in error_test:
        RFELin_error_test.append(sum(i)/len(i))

    # training data
    RFELin_train = cross_val_predict(LinearRegression(), selected_trainX, trainy, cv=k) # use selected features to do simple regression
    RFELin_error_train = (RFELin_train - trainy) / trainy
    
    # 3. L1/Lasso FEATURE SELECTION ---------------------------

    regL1 = LassoLarsCV(cv=k).fit(trainX,trainy)
    L1Lin_train = regL1.predict(trainX)
    L1Lin_test = regL1.predict(testX)

    L1Lin_error_train = (L1Lin_train - trainy)/trainy
    L1Lin_error_test = (L1Lin_test - testy) / testy

    # 4. L2/Ridge FEATURE SELECTION ---------------------------

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
    regL2 = RidgeCV(cv=k).fit(trainX,trainy)
    L2Lin_train = regL2.predict(trainX)
    L2Lin_test = regL2.predict(testX)

    L2Lin_error_train = (L2Lin_train - trainy)/trainy
    L2Lin_error_test = (L2Lin_test - testy) / testy
    
    return simpleLin_error_train, simpleLin_error_test, RFELin_error_train, RFELin_error_test, L1Lin_error_train, \
        L1Lin_error_test, L2Lin_error_train, L2Lin_error_test

# main pipeline ----------------------------

data = pd.read_csv(file, sep = "\t", header = None)
data = data.dropna(axis='columns')

data = (np.log(data)).replace(-np.inf, 0) # FOR LOG-LOG LINEAR REGRESSION

populations = numPopEW * numPopNS

ewX = data.iloc[:,4:]
ewX.columns = getOrigLabels(populations, sampleSize, 'all')
ewy = data.iloc[:,0]

weX = rearrangeFeatures(numPopEW, numPopNS, sampleSize, ewX, transform='mirror')
wey = data.iloc[:,1]

nsX = rearrangeFeatures(numPopEW, numPopNS, sampleSize, ewX, transform='rotate')
nsy = data.iloc[:,2]

snX = rearrangeFeatures(numPopEW, numPopNS, sampleSize, nsX, transform='mirror')
sny = data.iloc[:,3]

X = pd.concat([ewX, weX, nsX, snX], axis = 0)
y = pd.concat([ewy,wey,nsy,sny], axis = 0) # concat by row

trainX, testX, trainy, testy = train_test_split(X, y, test_size=test, random_state=seed)

simpleLin_error_train, simpleLin_error_test, RFELin_error_train, RFELin_error_test, L1Lin_error_train, \
        L1Lin_error_test, L2Lin_error_train, L2Lin_error_test = RunLinearRegression(trainX, trainy, testX, testy)
        
        
# GRAPHS -------------------------

cross_val = "\n" + str(k) + "-fold " + "Cross-Validation with training samples"
test_train = "\nTesting samples:" + str(len(testy)) + ", Training samples:" + str(len(trainy))

fig, ax = plt.subplots(1,4,figsize = (80,30), facecolor = 'w')

maxy = max(max(simpleLin_error_train), max(simpleLin_error_test), max(RFELin_error_test), max(RFELin_error_train), max(L1Lin_error_train), max(L1Lin_error_test), max(L2Lin_error_train), max(L2Lin_error_test))
maxy += maxy/30 # so it doesn't end EXACTLY at the max point
miny = min(min(simpleLin_error_train), min(simpleLin_error_test), min(RFELin_error_test), min(RFELin_error_train), min(L1Lin_error_train), min(L1Lin_error_test), min(L2Lin_error_train), min(L2Lin_error_test))
miny += miny/30 # so it doesn't end EXACTLY at the min point

ax[0].scatter(trainy,simpleLin_error_train, facecolors='dodgerblue', label="train", s=80)
ax[0].scatter(testy,simpleLin_error_test, facecolors='indianred',label="test", s=80)
ax[0].set_title("Log-Log Simple Linear Regression", size=50)
ax[0].set_ylabel("error in predicted migration rates", fontsize=50)
ax[0].set_xlabel("actual migration rate ln(m)",fontsize=50)
ax[0].axhline(y=0, color='k', linestyle='--')
ax[0].set_ylim([miny,maxy])
# ax[0].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=50)
ax[0].tick_params(axis='x', labelsize=40)
ax[0].tick_params(axis='y', labelsize=40)

ax[1].scatter(trainy,RFELin_error_train, facecolors='dodgerblue', label="train", s=80) 
ax[1].scatter(testy,RFELin_error_test, facecolors='indianred',label="test", s=80)
ax[1].set_title("Log-Log Linear Regression with RFECV", size=50)
ax[1].set_ylabel("error in predicted migration rates", fontsize=50)
ax[1].set_xlabel("actual migration rate ln(m)",fontsize=50)
ax[1].axhline(y=0, color='k', linestyle='--')
ax[1].set_ylim([miny,maxy])
# ax[1].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=50)
ax[1].tick_params(axis='x', labelsize=40)
ax[1].tick_params(axis='y', labelsize=40)

ax[2].scatter(trainy,L1Lin_error_train, facecolors='dodgerblue', label="train", s=80) # 'bo'
ax[2].scatter(testy,L1Lin_error_test, facecolors='indianred',label="test", s=80) # 'ro'
ax[2].set_title("Log-Log Linear Regression with LassoLarsCV", size=50)
ax[2].set_ylabel("error in predicted migration rates", fontsize=50)
ax[2].set_xlabel("actual migration rate ln(m)",fontsize=50)
ax[2].axhline(y=0, color='k', linestyle='--')
ax[2].set_ylim([miny,maxy])
# ax[2].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=50)
ax[2].tick_params(axis='x', labelsize=40)
ax[2].tick_params(axis='y', labelsize=40)

ax[3].scatter(trainy,L2Lin_error_train, facecolors='dodgerblue', label="train", s=80) # 'bo'
ax[3].scatter(testy,L2Lin_error_test, facecolors='indianred',label="test", s=80) # 'ro'
ax[3].set_title("Log-Log Linear Regression with RidgeCV", size=50)
ax[3].set_ylabel("error in predicted migration rates", fontsize=50)
ax[3].set_xlabel("actual migration rate ln(m)",fontsize=50)
ax[3].axhline(y=0, color='k', linestyle='--')
ax[3].set_ylim([miny,maxy])
ax[3].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=50)
ax[3].tick_params(axis='x', labelsize=40)
ax[3].tick_params(axis='y', labelsize=40)

fig.suptitle("Predicting migration rates in all directions using projections" + test_train + cross_val + "\n", fontsize=60)

fig.tight_layout()
fig.savefig("predictMig_transform.png")
