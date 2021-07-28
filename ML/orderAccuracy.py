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
test = 1/5
# seed for train-test splitting
seed = 5

file = "EW.5_NS.5_N.1000_n.10_asym4_input.txt"

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

def getOrderAccuracy(model_test, model_train, testy,trainy,testS, trainS, testD, trainD, data_len):
    d = {'predict': np.concatenate((model_test,model_train)), 'actual': np.concatenate((testy, trainy)), 'sample': np.concatenate((testS, trainS)), 'direction':np.concatenate((testD, trainD))}
    PA = pd.DataFrame(data = d).sort_values(by ='sample')

    NSorder=0
    EWorder=0
    allOrder=0
    for i in range(data_len):
        start = i * 4;  # print(start)
        end = start + 4; # print(end)

        # get descending order
        predictOrder = PA.iloc[start:end,3].values[np.argsort(-PA.iloc[start:end,0], axis = 1)] # get predicted order
        trueOrder = PA.iloc[start:end,3].values[np.argsort(-PA.iloc[start:end,1], axis = 1)] # get actual order
        
        if(np.array_equal(predictOrder, trueOrder)):
            allOrder += 1
            
        # check NS vs SN
        trueNSindex = np.where(trueOrder=='NS')[0][0]
        trueSNindex = np.where(trueOrder=='SN')[0][0]

        predictNSindex = np.where(predictOrder=='NS')[0][0]
        predictSNindex = np.where(predictOrder == 'SN')[0][0]

        if((trueNSindex > trueSNindex) == (predictNSindex > predictSNindex)):
            NSorder += 1

        # check EW vs WE
        trueEWindex = np.where(trueOrder=='EW')[0][0]
        trueWEindex = np.where(trueOrder=='WE')[0][0]

        predictEWindex = np.where(predictOrder=='EW')[0][0]
        predictWEindex = np.where(predictOrder == 'WE')[0][0]

        if((trueEWindex > trueWEindex) == (predictEWindex > predictWEindex)):
            EWorder += 1
            
    return NSorder/data_len, EWorder/data_len, allOrder/data_len

populations = numPopEW * numPopNS

data = pd.read_csv(file, sep = "\t", header = None)
data = data.dropna(axis='columns')

data = (np.log(data)).replace(-np.inf, 0) # FOR LOG-LOG LINEAR REGRESSION

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

data_len = len(ewy)

samples = np.tile(list(range(0,100)), 4)
direc = np.repeat(np.array(["EW","WE","NS","SN"]),[len(ewy), len(wey),len(nsy), len(sny)])

trainX, testX, trainy, testy, trainS, testS, trainD, testD = train_test_split(X, y, samples, direc, test_size=test, random_state=seed)

# 1. SIMPLE LINEAR REGRESSION ----------------------
simpleLin_all_test = [[] for _ in range(len(testy))]
simpleLin = cross_validate(LinearRegression(), trainX, trainy, cv=k,return_estimator = True)
counter = 0
for i in simpleLin['estimator']:
    simpleLin_y = i.predict(testX) # pick one of the folds' estimators to predict
    for j in range(len(testy)):
        simpleLin_all_test[j].append(simpleLin_y[j])

# taking the mean for each of the folds' predicted value
simpleLin_test = []
for i in simpleLin_all_test:
    simpleLin_test.append(sum(i)/len(i))

simpleLin_train = cross_val_predict(LinearRegression(), trainX, trainy, cv=k)

simpleOrder = getOrderAccuracy(simpleLin_test, simpleLin_train, testy,trainy,testS,trainS,testD,trainD, data_len)

# 2. RFE FEATURE SELECTION ---------------------------------

selector = RFECV(LinearRegression(), step=remove, cv=k).fit(trainX,trainy) # fit using training data only

# gets *selected* X columns only
selected_trainX = selector.transform(trainX) 
selected_testX = selector.transform(testX)

# conduct linear regression on training+testing using selected features ONLY
RFELin = cross_validate(LinearRegression(), selected_trainX, trainy, cv=k,return_estimator = True)

# testing data
RFELin_all_test = [[] for _ in range(len(testy))]
counter = 0
for i in RFELin['estimator']:
    RFELin_y = i.predict(selected_testX) # pick one of the folds' estimators to predict
    for j in range(len(testy)):
        RFELin_all_test[j].append(RFELin_y[j])

# taking the mean for each of the folds' predicted value
RFELin_test = []
for i in RFELin_all_test:
    RFELin_test.append(sum(i)/len(i))

    # training data
RFELin_train = cross_val_predict(LinearRegression(), selected_trainX, trainy, cv=k) # use selected features to do simple regression

RFEOrder = getOrderAccuracy(RFELin_test, RFELin_train, testy,trainy,testS,trainS,testD,trainD,data_len)

# 3. L1/Lasso FEATURE SELECTION ---------------------------

regL1 = LassoLarsCV(cv=k).fit(trainX,trainy)
L1Lin_train = regL1.predict(trainX)
L1Lin_test = regL1.predict(testX)

L1Order = getOrderAccuracy(L1Lin_test, L1Lin_train, testy,trainy,testS,trainS,testD,trainD,data_len)

# 4. L2/Ridge FEATURE SELECTION ---------------------------

regL2 = RidgeCV(cv=k).fit(trainX,trainy)
L2Lin_train = regL2.predict(trainX)
L2Lin_test = regL2.predict(testX)
      
L2Order = getOrderAccuracy(L2Lin_test, L2Lin_train, testy,trainy,testS,trainS,testD,trainD,data_len)

x_axis = ['NS-SN','EW-WE','EW-WE-NS-SN']

# models = [simpleOrder, RFEOrder, L1Order,L2Order]
# NSorder = [model[0] for model in models]
# SNorder = [model[1] for model in models]
# allOrder = [model[2] for model in models]

fig, ax = plt.subplots(1,1,figsize = (8,8), facecolor = 'w')

line1 = ax.plot(x_axis, simpleOrder,'ko-',label='simple')
line2 = ax.plot(x_axis, RFEOrder,'ro-',label='RFE') 
line3 = ax.plot(x_axis, L1Order,'mo-',label='L1')
line3 = ax.plot(x_axis, L2Order,'bo-',label='L2')

ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=12)
ax.set_title("Plotting ordering accuracy for Log-Log Linear Regression models", fontsize=13)
ax.set_xlabel("order", size=13)

fig.tight_layout()
fig.savefig("orderAccuracy.png")