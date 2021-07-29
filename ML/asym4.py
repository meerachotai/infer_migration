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

# ------------------ functions ---------------------------------

# FUNCTION: returns new order given the transform ('rotate'/'mirror'), default: mirror. 
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

def getPredicted(trainX, trainy, testX, testy):
    
    # 1. RANDOM FOREST REGRESSION ----------------------
    
    regr = RandomForestRegressor().fit(trainX, trainy)
    RFLin_train = regr.predict(trainX)
    RFLin_test = regr.predict(testX)
    
    RFLin_error_train = (RFLin_train - trainy)
    RFLin_error_test = (RFLin_test - testy)
    
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
    RFELin_predict = [[] for _ in range(len(testy))]
    counter = 0
    for i in RFELin['estimator']:
        RFELin_y = i.predict(selected_testX) # pick one of the folds' estimators to predict
        error = (RFELin_y - testy)
        for j in range(len(testy)):
            error_test[j].append(error.iloc[j])
            RFELin_predict[j].append(RFELin_y[j])
            
    # taking the mean for each of the folds' predicted value
    RFELin_error_test = []
    RFELin_test = []
    for i in error_test:
        RFELin_error_test.append(sum(i)/len(i))
    for i in RFELin_predict:
        RFELin_test.append(sum(i)/len(i))
        
    # training data
    RFELin_train = cross_val_predict(LinearRegression(), selected_trainX, trainy, cv=k) # use selected features to do simple regression
    RFELin_error_train = (RFELin_train - trainy)
    
    # 3. L1/Lasso FEATURE SELECTION ---------------------------

    regL1 = LassoLarsCV(cv=k).fit(trainX,trainy)
    L1Lin_train = regL1.predict(trainX)
    L1Lin_test = regL1.predict(testX)

    L1Lin_error_train = (L1Lin_train - trainy)
    L1Lin_error_test = (L1Lin_test - testy)

    # 4. L2/Ridge FEATURE SELECTION ---------------------------

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
    regL2 = RidgeCV(cv=k).fit(trainX,trainy)
    L2Lin_train = regL2.predict(trainX)
    L2Lin_test = regL2.predict(testX)

    L2Lin_error_train = (L2Lin_train - trainy)
    L2Lin_error_test = (L2Lin_test - testy)
    
    return RFLin_train, RFLin_test, RFELin_train, RFELin_test, L1Lin_train, \
        L1Lin_test, L2Lin_train, L2Lin_test
        

#------------- options ---------------------------

numPopEW = 5
numPopNS = 5
sampleSize = 10
N = 1000
# recursive feature elimination (RFE) options:
# at every iteration, _remove_ % of the features will be eliminated
remove = 0.4

# for every type of regression, k-fold cross-validation is used
k = 19

# for train-test split, how much of it is testing?
test = 1/20
# seed for train-test splitting
seed = 5

file = "EW.5_NS.5_N.1000_n.10_asym4_600_input.txt"
output = "asym4.png"

# -------------------------------------------------

populations = numPopEW * numPopNS

data = pd.read_csv(file, sep = "\t", header = None)
data = data.dropna(axis='columns')

data = (np.log10(data)).replace(-np.inf, 0) # FOR LOG-LOG LINEAR REGRESSION

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

colors = np.repeat(np.array(["dodgerblue","indianred","chartreuse","darkorchid"]),[len(ewy), len(wey),len(nsy), len(sny)])
trainX, testX, trainy, testy, trainC, testC = train_test_split(X, y, colors, test_size=test, random_state=seed)

RFLin_train, RFLin_test, RFELin_train, RFELin_test, L1Lin_train, L1Lin_test, L2Lin_train, L2Lin_test = getPredicted(trainX, trainy, testX, testy)

fig, ax = plt.subplots(1,4,figsize = (40,15), facecolor = 'w')

for i in ax:
    i.spines[:].set_linewidth(4)

    
maxy = max(max(RFLin_train), max(RFLin_test), max(RFELin_test), max(RFELin_train), max(L1Lin_train), max(L1Lin_test), max(L2Lin_train), max(L2Lin_test))
maxy -= maxy/30 # so it doesn't end EXACTLY at the max point
miny = min(min(RFLin_train), min(RFLin_test), min(RFELin_test), min(RFELin_train), min(L1Lin_train), min(L1Lin_test), min(L2Lin_train), min(L2Lin_test))
miny += miny/30 # so it doesn't end EXACTLY at the min point

lims = [miny, maxy]

ax[0].scatter(trainy,RFLin_train, facecolors='dodgerblue', label="train", s=40)
ax[0].scatter(testy,RFLin_test, facecolors='indianred',label="test", s=40)
ax[0].set_ylabel("predicted log10(m)", fontsize = 40)
ax[0].set_xlabel("simulated log10(m)", fontsize=40)
ax[0].set_title("Log-Log Random Forest Regression", size=35)
ax[0].tick_params(axis='x', labelsize=40)
ax[0].tick_params(axis='y', labelsize=40)
ax[0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax[0].set_aspect('equal')
ax[0].set_xlim(lims)
ax[0].set_ylim(lims)

ax[1].scatter(trainy,RFELin_train, facecolors='dodgerblue', label="train", s=40)
ax[1].scatter(testy,RFELin_test, facecolors='indianred',label="test", s=40)
ax[1].set_ylabel("predicted log10(m)", fontsize = 40)
ax[1].set_xlabel("simulated log10(m)", fontsize=40)
ax[1].set_title("Log-Log RFE Linear Regression", size=35)
ax[1].tick_params(axis='x', labelsize=40)
ax[1].tick_params(axis='y', labelsize=40)
ax[1].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax[1].set_aspect('equal')
ax[1].set_xlim(lims)
ax[1].set_ylim(lims)

ax[2].scatter(trainy,L1Lin_train, facecolors='dodgerblue', label="train", s=40)
ax[2].scatter(testy,L1Lin_test, facecolors='indianred',label="test", s=40)
ax[2].set_ylabel("predicted log10(m)", fontsize = 40)
ax[2].set_xlabel("simulated log10(m)", fontsize=40)
ax[2].set_title("Log-Log L1 Linear Regression", size=35)
ax[2].tick_params(axis='x', labelsize=40)
ax[2].tick_params(axis='y', labelsize=40)
ax[2].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax[2].set_aspect('equal')
ax[2].set_xlim(lims)
ax[2].set_ylim(lims)

ax[3].scatter(trainy,L2Lin_train, facecolors='dodgerblue', label="train", s=40)
ax[3].scatter(testy,L2Lin_test, facecolors='indianred',label="test", s=40)
ax[3].set_ylabel("predicted log10(m)", fontsize = 40)
ax[3].set_xlabel("simulated log10(m)", fontsize=40)
ax[3].set_title("Log-Log L2 Linear Regression", size=35)
ax[3].tick_params(axis='x', labelsize=40)
ax[3].tick_params(axis='y', labelsize=40)
ax[3].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax[3].set_aspect('equal')
ax[3].set_xlim(lims)
ax[3].set_ylim(lims)
ax[3].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)

fig.suptitle(test_train + "\n" + file, fontsize=40)

fig.tight_layout()

fig.savefig(output)