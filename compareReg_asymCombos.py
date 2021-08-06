#!/usr/bin/env python3

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns
import sys

sns.set_theme()

# numPopEW = 5
# numPopNS = 5
# sampleSize = 10
# N = 1000
# seed = 5 # seed for train-test splitting
# file = "EW.5_NS.5_N.1000_n.10_sym693_input.txt"

# EW.5_NS.5_N.1000_n.10_asym4_1GB_input.txt reg_asym_1GB.png reg_r2_asym_1GB.png 5 5 10 1000 5 "Asymmetric 1GB - Log-Log Regression Models" "Asymmetric 1GB - Simulated vs. Predicted $R^{2}$" 'all'

file = str(sys.argv[1])
graph = str(sys.argv[2])
heatmap = str(sys.argv[3])
numPopEW = int(sys.argv[4]) 
numPopNS = int(sys.argv[5]) 
sampleSize = int(sys.argv[6])
N = int(sys.argv[7])
seed = int(sys.argv[8]) # seed for train-test splitting
combo = str(sys.argv[9])
# title1 = str(sys.argv[10])
# title2 = str(sys.argv[11])
starter = str(sys.argv[10])

title1 = starter + " - " + "Log-Log Regression Models - " + combo
title2 = starter + " - " + "Asymmetric 1GB - Simulated vs. Predicted $R^{2}$ - " + combo

graph = combo + graph
heatmap = combo + heatmap

# for every type of regression, k-fold cross-validation is used
k = 10
# at every iteration, _remove_ % of the features will be eliminated
remove = 0.4
# for train-test split, how much of it is testing?
test = 0.2
trees = 500
features = 0.5
depth = 5

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

def getPredicted(trainX, trainy, testX, testy):
    
     # SIMPLE LINEAR REGRESSION ----------------------
    
    simpleLin = cross_validate(LinearRegression(), trainX, trainy, cv=k,return_estimator = True)
    counter = 0
    error_test = [[] for _ in range(len(testy))]
    simpleLin_predict = [[] for _ in range(len(testy))]
    for i in simpleLin['estimator']:
        simpleLin_y = i.predict(testX) # pick one of the folds' estimators to predict
        error = (simpleLin_y - testy)
        for j in range(len(testy)):
            error_test[j].append(error.iloc[j])
            simpleLin_predict[j].append(simpleLin_y[j])
            
    # taking the mean for each of the folds' predicted value
    simpleLin_error_test = []
    simpleLin_test = []
    for i in error_test:
        simpleLin_error_test.append(sum(i)/len(i))
    for i in simpleLin_predict:
        simpleLin_test.append(sum(i)/len(i))
        
    simpleLin_train = cross_val_predict(LinearRegression(), trainX, trainy, cv=k)
    simpleLin_error_train = (simpleLin_train - trainy)
    
    # FEATURE SELECTION ---------------------------------

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
    
    # L1/Lasso FEATURE SELECTION ---------------------------

    regL1 = LassoLarsCV(cv=k).fit(trainX,trainy)
    L1Lin_train = regL1.predict(trainX)
    L1Lin_test = regL1.predict(testX)

    L1Lin_error_train = (L1Lin_train - trainy)
    L1Lin_error_test = (L1Lin_test - testy)

    # L2/Ridge FEATURE SELECTION ---------------------------

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
    regL2 = RidgeCV(cv=k).fit(trainX,trainy)
    L2Lin_train = regL2.predict(trainX)
    L2Lin_test = regL2.predict(testX)

    L2Lin_error_train = (L2Lin_train - trainy)
    L2Lin_error_test = (L2Lin_test - testy)
    
    # KERNEL RIDGE REGRESSION --------------------------
    
    KR = KernelRidge(alpha=4, gamma = 0.001).fit(trainX, trainy)
    
    KR_train = KR.predict(trainX)
    KR_test = KR.predict(testX)
    
    KR_error_test = (KR_test - testy)
    KR_error_train = (KR_train - trainy)
	
	# RANDOM FOREST REGRESSION ----------------------
    
    regr = RandomForestRegressor(random_state = seed, oob_score = True, n_estimators = trees, max_features = features, max_depth = depth).fit(trainX, trainy)
    RFLin_train = regr.predict(trainX)
    RFLin_test = regr.predict(testX)
    
    RFLin_error_train = (RFLin_train - trainy)
    RFLin_error_test = (RFLin_test - testy)
    
    return simpleLin_train, simpleLin_test, RFLin_train, RFLin_test, RFELin_train, RFELin_test, L1Lin_train, \
        L1Lin_test, L2Lin_train, L2Lin_test, KR_train, KR_test


populations = numPopEW * numPopNS

data = pd.read_csv(file, sep = "\t", header = None)
data = data.iloc[: , :-1] # last column is NaN
data = data.dropna(axis='index') # some rows might have NaNs
data.drop_duplicates()

data = (np.log10(data)).replace(-np.inf, 0) # FOR LOG-LOG LINEAR REGRESSION

X = pd.DataFrame([0])
y = pd.DataFrame([0])

if(combo == 'EW-WE'):
	print("chose EW-WE")
	ewX = data.iloc[:,4:]
	ewX.columns = getOrigLabels(populations, sampleSize, 'all')
	ewy = data.iloc[:,0]

	weX = rearrangeFeatures(numPopEW, numPopNS, sampleSize, ewX, transform='mirror')
	wey = data.iloc[:,1]	

	X = pd.concat([ewX, weX], axis = 0)
	y = pd.concat([ewy,wey], axis = 0) # concat by row
elif(combo == 'EW-NS-SN'):
	print("chose EW-NS-SN")
	ewX = data.iloc[:,4:]
	ewX.columns = getOrigLabels(populations, sampleSize, 'all')

	nsX = rearrangeFeatures(numPopEW, numPopNS, sampleSize, ewX, transform='rotate')
	nsy = data.iloc[:,2]

	snX = rearrangeFeatures(numPopEW, numPopNS, sampleSize, nsX, transform='mirror')
	sny = data.iloc[:,3]

	X = pd.concat([nsX, snX], axis = 0)
	y = pd.concat([nsy,sny], axis = 0) # concat by row
elif(combo == 'NS-SN'):
	print("chose NS-SN")
	nsX = data.iloc[:,4:]
	nsX.columns = getOrigLabels(populations, sampleSize, 'all')
	nsy = data.iloc[:,2]

	snX = rearrangeFeatures(numPopEW, numPopNS, sampleSize, nsX, transform='mirror')
	sny = data.iloc[:,3]

	X = pd.concat([nsX, snX], axis = 0)
	y = pd.concat([nsy,sny], axis = 0) # concat by row
elif(combo == 'EW-NS'):
	print("chose EW-NS")
	ewX = data.iloc[:,4:]
	ewX.columns = getOrigLabels(populations, sampleSize, 'all')
	ewy = data.iloc[:,0]

	nsX = rearrangeFeatures(numPopEW, numPopNS, sampleSize, ewX, transform='rotate')
	nsy = data.iloc[:,2]

	X = pd.concat([ewX,nsX], axis = 0)
	y = pd.concat([ewy,nsy], axis = 0) # concat by row
elif(combo == 'EW-SN'):
	print("chose EW-SN")
	ewX = data.iloc[:,4:]
	ewX.columns = getOrigLabels(populations, sampleSize, 'all')
	ewy = data.iloc[:,0]

	nsX = rearrangeFeatures(numPopEW, numPopNS, sampleSize, ewX, transform='rotate')
	nsy = data.iloc[:,2]

	X = pd.concat([ewX, snX], axis = 0)
	y = pd.concat([ewy,sny], axis = 0) # concat by row
elif(combo == 'all'):
	print("chose ALL")
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

simpleLin_train, simpleLin_test, RFLin_train, RFLin_test, RFELin_train, RFELin_test, \
    L1Lin_train, L1Lin_test, L2Lin_train, L2Lin_test, KR_train, KR_test = getPredicted(trainX, trainy, testX, testy)

RFELin_train_r2 = np.average(np.corrcoef(RFELin_train,trainy)**2)
RFELin_test_r2 = np.average(np.corrcoef(RFELin_test,testy)**2)
L1Lin_train_r2 = np.average(np.corrcoef(L1Lin_train,trainy)**2)
L1Lin_test_r2 = np.average(np.corrcoef(L1Lin_test,testy)**2)
simpleLin_train_r2 = np.average(np.corrcoef(simpleLin_train,trainy)**2)
simpleLin_test_r2 = np.average(np.corrcoef(simpleLin_test,testy)**2)
L2Lin_train_r2 = np.average(np.corrcoef(L2Lin_train,trainy)**2)
L2Lin_test_r2 = np.average(np.corrcoef(L2Lin_test,testy)**2)
RFLin_train_r2 = np.average(np.corrcoef(RFLin_train,trainy)**2)
RFLin_test_r2 = np.average(np.corrcoef(RFLin_test,testy)**2)
KR_train_r2 = np.average(np.corrcoef(KR_train,trainy)**2)
KR_test_r2 = np.average(np.corrcoef(KR_test,testy)**2)

test_train = "\nTesting samples:" + str(len(testy)) + ", Training samples:" + str(len(trainy))
cross_val = "\n" + str(k) + "-fold " + "Cross-Validation with training samples"

fig, ax = plt.subplots(2,3,figsize = (65,32), facecolor = 'w')
ax = ax.flatten()

for i in ax:
#     i.spines[:].set_linewidth(4)
    i.tick_params(axis='x', labelsize=35)
    i.tick_params(axis='y', labelsize=35)

lims = [-4 - 4/30, - 2 + 2/30]


ax[0].scatter(trainy,simpleLin_train, facecolors='dodgerblue', label="train, $R^{2}$=" + f'{simpleLin_train_r2:.3f}', s=40)
ax[0].scatter(testy,simpleLin_test, facecolors='indianred',label="test, $R^{2}$=" + f'{simpleLin_test_r2:.3f}', s=40)
ax[0].set_ylabel("predicted log10(m)", fontsize = 35)
ax[0].set_title("Log-Log Simple Linear Regression" + cross_val, size=35)
ax[0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax[0].set_aspect('equal')
ax[0].set_xlim(lims)
ax[0].set_ylim(lims)
ax[0].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)

ax[1].scatter(trainy,RFELin_train, facecolors='dodgerblue', label="train, $R^{2}$=" + f'{RFELin_train_r2:.3f}', s=40)
ax[1].scatter(testy,RFELin_test, facecolors='indianred',label="test, $R^{2}$=" + f'{RFELin_test_r2:.3f}', s=40)
ax[1].set_title("Log-Log RFE Linear Regression" + cross_val, size=35)
ax[1].set_xlabel("simulated log10(m)", fontsize=35)
ax[1].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax[1].set_aspect('equal')
ax[1].set_xlim(lims)
ax[1].set_ylim(lims)
ax[1].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)

ax[2].scatter(trainy,L1Lin_train, facecolors='dodgerblue', label="train, $R^{2}$=" + f'{L1Lin_train_r2:.3f}', s=40)
ax[2].scatter(testy,L1Lin_test, facecolors='indianred',label="test, $R^{2}$=" + f'{L1Lin_test_r2:.3f}', s=40)
ax[2].set_title("Log-Log L1 Linear Regression" + cross_val, size=35)
ax[2].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax[2].set_aspect('equal')
ax[2].set_xlim(lims)
ax[2].set_ylim(lims)
ax[2].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)

ax[3].scatter(trainy,L2Lin_train, facecolors='dodgerblue', label="train, $R^{2}$=" + f'{L2Lin_train_r2:.3f}', s=40)
ax[3].scatter(testy,L2Lin_test, facecolors='indianred',label="test, $R^{2}$=" + f'{L2Lin_test_r2:.3f}', s=40)
ax[3].set_title("Log-Log L2 Linear Regression" + cross_val, size=35)
ax[3].set_ylabel("predicted log10(m)", fontsize = 35)
ax[3].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax[3].set_aspect('equal')
ax[3].set_xlim(lims)
ax[3].set_ylim(lims)
ax[3].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)

ax[4].scatter(trainy,KR_train, facecolors='dodgerblue', label="train, $R^{2}$=" + f'{KR_train_r2:.3f}', s=40)
ax[4].scatter(testy,KR_test, facecolors='indianred',label="test, $R^{2}$=" + f'{KR_test_r2:.3f}', s=40)
ax[4].set_title("Log-Log Kernel Ridge Regression \nalpha = 4, gamma = 0.001", size=35)
ax[4].set_xlabel("simulated log10(m)", fontsize=35)
ax[4].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax[4].set_aspect('equal')
ax[4].set_xlim(lims)
ax[4].set_ylim(lims)
ax[4].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)

ax[5].scatter(trainy,RFLin_train, facecolors='dodgerblue', label="train, $R^{2}$=" + f'{RFLin_train_r2:.3f}', s=40)
ax[5].scatter(testy,RFLin_test, facecolors='indianred',label="test, $R^{2}$=" + f'{RFLin_test_r2:.3f}', s=40)
ax[5].set_title("Log-Log Random Forest Regression\n with oob, trees:"+ str(trees) + ", depth:" + str(depth) + ", features:" + str(features), size=35)
ax[5].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax[5].set_aspect('equal')
ax[5].set_xlim(lims)
ax[5].set_ylim(lims)
ax[5].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)

fig.suptitle(title1 + test_train + "\n" + file, fontsize=40)

fig.savefig(graph, bbox_inches="tight")

fig = plt.figure(figsize=(3, 5), facecolor='w', edgecolor='k')
plt.yticks(rotation=0)

cmap = sns.cubehelix_palette(light=0.9, as_cmap=True, start=2.8, rot=.1)
# cmap = sns.light_palette("seagreen", as_cmap=True)

R2 = np.array([[simpleLin_train_r2, simpleLin_test_r2],[RFELin_train_r2, RFELin_test_r2], [L1Lin_train_r2, L1Lin_test_r2], [L2Lin_train_r2, L2Lin_test_r2], [KR_train_r2, KR_test_r2],[RFLin_train_r2, RFLin_test_r2]])
np.transpose(R2)

yticks=['simple-LR', 'RFE-LR', 'Lasso-LR','Ridge-LR', 'KernelRidge', 'RandomForest']
xticks=['train','test']

sns.heatmap(R2, cmap=cmap, annot=True, yticklabels = yticks, xticklabels=xticks,fmt='.5f')
plt.yticks(rotation=0) 

# ax.invert_yaxis()
plt.title(title2 + test_train + cross_val,fontsize=15)

# plt.tight_layout()
plt.savefig(heatmap,bbox_inches="tight")

