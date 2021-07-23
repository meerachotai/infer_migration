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

numPopEW = 5
numPopNS = 5
sampleSize = 10
populations = numPopEW * numPopNS
N = 1000
file = "EW.5_NS.5_N.1000_n.10_logAsym_input.txt"

# recursive feature elimination (RFE) options:
# at every iteration, _remove_ % of the features will be eliminated
remove = 0.4

# for every type of regression, k-fold cross-validation is used
k = 4

# for train-test split, how much of it is testing?
test = 1/(k+1)

# seed for train-test splitting
seed = 5

data = pd.read_csv(file, sep = "\t", header = None)
data = data.dropna(axis='columns')
data = (np.log(data)).replace(-np.inf, 0)

X = data.iloc[:,4:]
EWy = data.iloc[:, 0]
WEy = data.iloc[:, 1]
NSy = data.iloc[:, 2]
SNy = data.iloc[:, 3]

EWtrainX, EWtestX, EWtrainy, EWtesty = train_test_split(X, EWy, test_size=test, random_state=seed)
WEtrainX, WEtestX, WEtrainy, WEtesty = train_test_split(X, WEy, test_size=test, random_state=seed)
NStrainX, NStestX, NStrainy, NStesty = train_test_split(X, NSy, test_size=test, random_state=seed)
SNtrainX, SNtestX, SNtrainy, SNtesty = train_test_split(X, SNy, test_size=test, random_state=seed)

# LINEAR REGRESSION ----------------------------------

EW_simpleLin_error_train, EW_simpleLin_error_test, EW_RFELin_error_train, EW_RFELin_error_test, EW_L1Lin_error_train, \
    EW_L1Lin_error_test, EW_L2Lin_error_train, EW_L2Lin_error_test = RunLinearRegression(EWtrainX, EWtrainy, EWtestX, EWtesty)

WE_simpleLin_error_train, WE_simpleLin_error_test, WE_RFELin_error_train, WE_RFELin_error_test, WE_L1Lin_error_train, \
    WE_L1Lin_error_test, WE_L2Lin_error_train, WE_L2Lin_error_test = RunLinearRegression(WEtrainX, WEtrainy, WEtestX, WEtesty)

NS_simpleLin_error_train, NS_simpleLin_error_test, NS_RFELin_error_train, NS_RFELin_error_test, NS_L1Lin_error_train, \
    NS_L1Lin_error_test, NS_L2Lin_error_train, NS_L2Lin_error_test = RunLinearRegression(NStrainX, NStrainy, NStestX, NStesty)

SN_simpleLin_error_train, SN_simpleLin_error_test, SN_RFELin_error_train, SN_RFELin_error_test, SN_L1Lin_error_train, \
    SN_L1Lin_error_test, SN_L2Lin_error_train, SN_L2Lin_error_test = RunLinearRegression(SNtrainX, SNtrainy, SNtestX, SNtesty)

# GRAPHS ---------------------------------------------

cross_val = "\n" + str(k) + "-fold " + "Cross-Validation with training data"
EW_test_train = "\ntesting samples:" + str(len(EWtesty)) + ", training samples:" + str(len(EWtrainy))
WE_test_train = "\ntesting samples:" + str(len(WEtesty)) + ", training samples:" + str(len(WEtrainy))
NS_test_train = "\ntesting samples:" + str(len(NStesty)) + ", training samples:" + str(len(NStrainy))
SN_test_train = "\ntesting samples:" + str(len(SNtesty)) + ", training samples:" + str(len(SNtrainy))

maxy = max(max(EW_simpleLin_error_train), max(EW_simpleLin_error_test), max(EW_RFELin_error_test), \
           max(EW_RFELin_error_train), max(EW_L1Lin_error_train), max(EW_L1Lin_error_test), max(EW_L2Lin_error_train),\
           max(EW_L2Lin_error_test), max(WE_simpleLin_error_train), max(WE_simpleLin_error_test), max(WE_RFELin_error_test), \
           max(WE_RFELin_error_train), max(WE_L1Lin_error_train), max(WE_L1Lin_error_test), max(WE_L2Lin_error_train),\
           max(WE_L2Lin_error_test), max(NS_simpleLin_error_train), max(NS_simpleLin_error_test), max(NS_RFELin_error_test), \
           max(NS_RFELin_error_train), max(NS_L1Lin_error_train), max(NS_L1Lin_error_test), max(NS_L2Lin_error_train),\
           max(NS_L2Lin_error_test), max(SN_simpleLin_error_train), max(SN_simpleLin_error_test), max(SN_RFELin_error_test), \
           max(SN_RFELin_error_train), max(SN_L1Lin_error_train), max(SN_L1Lin_error_test), max(SN_L2Lin_error_train),\
           max(SN_L2Lin_error_test))
maxy += maxy/30 # so it doesn't end EXACTLY at the max point
miny = min(min(EW_simpleLin_error_train), min(EW_simpleLin_error_test), min(EW_RFELin_error_test), \
           min(EW_RFELin_error_train), min(EW_L1Lin_error_train), min(EW_L1Lin_error_test), min(EW_L2Lin_error_train),\
           min(EW_L2Lin_error_test), min(WE_simpleLin_error_train), min(WE_simpleLin_error_test), min(WE_RFELin_error_test), \
           min(WE_RFELin_error_train), min(WE_L1Lin_error_train), min(WE_L1Lin_error_test), min(WE_L2Lin_error_train),\
           min(WE_L2Lin_error_test), min(NS_simpleLin_error_train), min(NS_simpleLin_error_test), min(NS_RFELin_error_test), \
           min(NS_RFELin_error_train), min(NS_L1Lin_error_train), min(NS_L1Lin_error_test), min(NS_L2Lin_error_train),\
           min(NS_L2Lin_error_test), min(SN_simpleLin_error_train), min(SN_simpleLin_error_test), min(SN_RFELin_error_test), \
           min(SN_RFELin_error_train), min(SN_L1Lin_error_train), min(SN_L1Lin_error_test), min(SN_L2Lin_error_train),\
           min(SN_L2Lin_error_test))
miny += miny/30 # so it doesn't end EXACTLY at the min point
fig1, ax1 = plt.subplots(1,4,figsize = (60,15))
fig2, ax2 = plt.subplots(1,4,figsize = (60,15))
fig3, ax3 = plt.subplots(1,4,figsize = (60,15))
fig4, ax4 = plt.subplots(1,4,figsize = (60,15))

ax1 = ax1.flatten()
ax2 = ax2.flatten()
ax3 = ax3.flatten()
ax4 = ax4.flatten()

ax1[0].scatter(EWtrainy,EW_simpleLin_error_train, facecolors='dodgerblue', label="train", s=100)
ax1[0].scatter(EWtesty,EW_simpleLin_error_test, facecolors='indianred',label="test", s=100)
ax1[0].set_title("Simple Log-Log Linear Regression EW" +EW_test_train+ cross_val, size=30)
ax1[0].set_ylabel("error in predicted migration rates", fontsize=30)
ax1[0].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax1[0].axhline(y=0, color='k', linestyle='--')
ax1[0].set_ylim([miny,maxy])
ax1[0].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax1[0].tick_params(axis='x', labelsize=20)
ax1[0].tick_params(axis='y', labelsize=20)


ax2[0].scatter(EWtrainy,EW_RFELin_error_train, facecolors='dodgerblue', label="train", s=100) 
ax2[0].scatter(EWtesty,EW_RFELin_error_test, facecolors='indianred',label="test", s=100)
ax2[0].set_title("Log-Log Linear Regression with RFECV EW" +EW_test_train + cross_val, size=30)
ax2[0].set_ylabel("error in predicted migration rates", fontsize=30)
ax2[0].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax2[0].axhline(y=0, color='k', linestyle='--')
ax2[0].set_ylim([miny,maxy])
ax2[0].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax2[0].tick_params(axis='x', labelsize=20)
ax2[0].tick_params(axis='y', labelsize=20)

# ax2 = inset_axes(ax[1], width="60%", height = "70%", bbox_to_anchor=(.2, .455, .75, .5), bbox_transform=ax[1].transAxes, loc=1)
# ax2.scatter(y,RFELin_error, facecolors='dodgerblue')
# ax2.axhline(y=0, color='k', linestyle='--')

ax3[0].scatter(EWtrainy,EW_L1Lin_error_train, facecolors='dodgerblue', label="train", s=100) # 'bo'
ax3[0].scatter(EWtesty,EW_L1Lin_error_test, facecolors='indianred',label="test", s=100) # 'ro'
ax3[0].set_title("Log-Log Linear Regression with LassoLarsCV EW" +EW_test_train + cross_val, size=30)
ax3[0].set_ylabel("error in predicted migration rates", fontsize=30)
ax3[0].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax3[0].axhline(y=0, color='k', linestyle='--')
ax3[0].set_ylim([miny,maxy])
ax3[0].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax3[0].tick_params(axis='x', labelsize=20)
ax3[0].tick_params(axis='y', labelsize=20)

# ax3 = inset_axes(ax[2], width="60%", height = "70%", bbox_to_anchor=(.2, .455, .75, .5), bbox_transform=ax[2].transAxes, loc=1)
# ax3.scatter(y,L1Lin_error, facecolors='dodgerblue')
# ax3.axhline(y=0, color='k', linestyle='--')

ax4[0].scatter(EWtrainy,EW_L2Lin_error_train, facecolors='dodgerblue', label="train", s=100) # 'bo'
ax4[0].scatter(EWtesty,EW_L2Lin_error_test, facecolors='indianred',label="test", s=100) # 'ro'
ax4[0].set_title("Log-Log Linear Regression with RidgeCV EW" +EW_test_train + cross_val, size=30)
ax4[0].set_ylabel("error in predicted migration rates", fontsize=30)
ax4[0].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax4[0].axhline(y=0, color='k', linestyle='--')
ax4[0].set_ylim([miny,maxy])
ax4[0].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax4[0].tick_params(axis='x', labelsize=20)
ax4[0].tick_params(axis='y', labelsize=20)

ax1[1].scatter(WEtrainy,WE_simpleLin_error_train, facecolors='dodgerblue', label="train", s=100)
ax1[1].scatter(WEtesty,WE_simpleLin_error_test, facecolors='indianred',label="test", s=100)
ax1[1].set_title("Simple Log-Log Linear Regression WE" +WE_test_train+ cross_val, size=30)
ax1[1].set_ylabel("error in predicted migration rates", fontsize=30)
ax1[1].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax1[1].axhline(y=0, color='k', linestyle='--')
ax1[1].set_ylim([miny,maxy])
ax1[1].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax1[1].tick_params(axis='x', labelsize=20)
ax1[1].tick_params(axis='y', labelsize=20)


ax2[1].scatter(WEtrainy,WE_RFELin_error_train, facecolors='dodgerblue', label="train", s=100) 
ax2[1].scatter(WEtesty,WE_RFELin_error_test, facecolors='indianred',label="test", s=100)
ax2[1].set_title("Log-Log Linear Regression with RFECV WE" +WE_test_train + cross_val, size=30)
ax2[1].set_ylabel("error in predicted migration rates", fontsize=30)
ax2[1].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax2[1].axhline(y=0, color='k', linestyle='--')
ax2[1].set_ylim([miny,maxy])
ax2[1].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax2[1].tick_params(axis='x', labelsize=20)
ax2[1].tick_params(axis='y', labelsize=20)

# ax2 = inset_axes(ax[1], width="60%", height = "70%", bbox_to_anchor=(.2, .455, .75, .5), bbox_transform=ax[1].transAxes, loc=1)
# ax2.scatter(y,RFELin_error, facecolors='dodgerblue')
# ax2.axhline(y=0, color='k', linestyle='--')

ax3[1].scatter(WEtrainy,WE_L1Lin_error_train, facecolors='dodgerblue', label="train", s=100) # 'bo'
ax3[1].scatter(WEtesty,WE_L1Lin_error_test, facecolors='indianred',label="test", s=100) # 'ro'
ax3[1].set_title("Log-Log Linear Regression with LassoLarsCV WE" +WE_test_train + cross_val, size=30)
ax3[1].set_ylabel("error in predicted migration rates", fontsize=30)
ax3[1].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax3[1].axhline(y=0, color='k', linestyle='--')
ax3[1].set_ylim([miny,maxy])
ax3[1].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax3[1].tick_params(axis='x', labelsize=20)
ax3[1].tick_params(axis='y', labelsize=20)

# ax3 = inset_axes(ax[2], width="60%", height = "70%", bbox_to_anchor=(.2, .455, .75, .5), bbox_transform=ax[2].transAxes, loc=1)
# ax3.scatter(y,L1Lin_error, facecolors='dodgerblue')
# ax3.axhline(y=0, color='k', linestyle='--')

ax4[1].scatter(WEtrainy,WE_L2Lin_error_train, facecolors='dodgerblue', label="train", s=100) # 'bo'
ax4[1].scatter(WEtesty,WE_L2Lin_error_test, facecolors='indianred',label="test", s=100) # 'ro'
ax4[1].set_title("Log-Log Linear Regression with RidgeCVWE" +WE_test_train + cross_val, size=30)
ax4[1].set_ylabel("error in predicted migration rates", fontsize=30)
ax4[1].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax4[1].axhline(y=0, color='k', linestyle='--')
ax4[1].set_ylim([miny,maxy])
ax4[1].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax4[1].tick_params(axis='x', labelsize=20)
ax4[1].tick_params(axis='y', labelsize=20)

ax1[2].scatter(NStrainy,NS_simpleLin_error_train, facecolors='dodgerblue', label="train", s=100)
ax1[2].scatter(NStesty,NS_simpleLin_error_test, facecolors='indianred',label="test", s=100)
ax1[2].set_title("Simple Log-Log Linear Regression NS" +NS_test_train+ cross_val, size=30)
ax1[2].set_ylabel("error in predicted migration rates", fontsize=30)
ax1[2].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax1[2].axhline(y=0, color='k', linestyle='--')
ax1[2].set_ylim([miny,maxy])
ax1[2].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax1[2].tick_params(axis='x', labelsize=20)
ax1[2].tick_params(axis='y', labelsize=20)

ax2[2].scatter(NStrainy,NS_RFELin_error_train, facecolors='dodgerblue', label="train", s=100) 
ax2[2].scatter(NStesty,NS_RFELin_error_test, facecolors='indianred',label="test", s=100)
ax2[2].set_title("Log-Log Linear Regression with RFECV NS" +NS_test_train + cross_val, size=30)
ax2[2].set_ylabel("error in predicted migration rates", fontsize=30)
ax2[2].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax2[2].axhline(y=0, color='k', linestyle='--')
ax2[2].set_ylim([miny,maxy])
ax2[2].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax2[2].tick_params(axis='x', labelsize=20)
ax2[2].tick_params(axis='y', labelsize=20)

# ax2 = inset_axes(ax[1], width="60%", height = "70%", bbox_to_anchor=(.2, .455, .75, .5), bbox_transform=ax[1].transAxes, loc=1)
# ax2.scatter(y,RFELin_error, facecolors='dodgerblue')
# ax2.axhline(y=0, color='k', linestyle='--')

ax3[2].scatter(NStrainy,NS_L1Lin_error_train, facecolors='dodgerblue', label="train", s=100) # 'bo'
ax3[2].scatter(NStesty,NS_L1Lin_error_test, facecolors='indianred',label="test", s=100) # 'ro'
ax3[2].set_title("Log-Log Linear Regression with LassoLarsCV NS" +NS_test_train + cross_val, size=30)
ax3[2].set_ylabel("error in predicted migration rates", fontsize=30)
ax3[2].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax3[2].axhline(y=0, color='k', linestyle='--')
ax3[2].set_ylim([miny,maxy])
ax3[2].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax3[2].tick_params(axis='x', labelsize=20)
ax3[2].tick_params(axis='y', labelsize=20)

# ax3 = inset_axes(ax[2], width="60%", height = "70%", bbox_to_anchor=(.2, .455, .75, .5), bbox_transform=ax[2].transAxes, loc=1)
# ax3.scatter(y,L1Lin_error, facecolors='dodgerblue')
# ax3.axhline(y=0, color='k', linestyle='--')

ax4[2].scatter(NStrainy,NS_L2Lin_error_train, facecolors='dodgerblue', label="train", s=100) # 'bo'
ax4[2].scatter(NStesty,NS_L2Lin_error_test, facecolors='indianred',label="test", s=100) # 'ro'
ax4[2].set_title("Log-Log Linear Regression with RidgeCV NS" +NS_test_train + cross_val, size=30)
ax4[2].set_ylabel("error in predicted migration rates", fontsize=30)
ax4[2].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax4[2].axhline(y=0, color='k', linestyle='--')
ax4[2].set_ylim([miny,maxy])
ax4[2].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax4[2].tick_params(axis='x', labelsize=20)
ax4[2].tick_params(axis='y', labelsize=20)

ax1[3].scatter(SNtrainy,SN_simpleLin_error_train, facecolors='dodgerblue', label="train", s=100)
ax1[3].scatter(SNtesty,SN_simpleLin_error_test, facecolors='indianred',label="test", s=100)
ax1[3].set_title("Simple Log-Log Linear Regression SN" +SN_test_train+ cross_val, size=30)
ax1[3].set_ylabel("error in predicted migration rates", fontsize=30)
ax1[3].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax1[3].axhline(y=0, color='k', linestyle='--')
ax1[3].set_ylim([miny,maxy])
ax1[3].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax1[3].tick_params(axis='x', labelsize=20)
ax1[3].tick_params(axis='y', labelsize=20)


ax2[3].scatter(SNtrainy,SN_RFELin_error_train, facecolors='dodgerblue', label="train", s=100) 
ax2[3].scatter(SNtesty,SN_RFELin_error_test, facecolors='indianred',label="test", s=100)
ax2[3].set_title("Log-Log Linear Regression with RFECV SN" +SN_test_train + cross_val, size=30)
ax2[3].set_ylabel("error in predicted migration rates", fontsize=30)
ax2[3].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax2[3].axhline(y=0, color='k', linestyle='--')
ax2[3].set_ylim([miny,maxy])
ax2[3].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax2[3].tick_params(axis='x', labelsize=20)
ax2[3].tick_params(axis='y', labelsize=20)

# ax2 = inset_axes(ax[1], width="60%", height = "70%", bbox_to_anchor=(.2, .455, .75, .5), bbox_transform=ax[1].transAxes, loc=1)
# ax2.scatter(y,RFELin_error, facecolors='dodgerblue')
# ax2.axhline(y=0, color='k', linestyle='--')

ax3[3].scatter(SNtrainy,SN_L1Lin_error_train, facecolors='dodgerblue', label="train", s=100) # 'bo'
ax3[3].scatter(SNtesty,SN_L1Lin_error_test, facecolors='indianred',label="test", s=100) # 'ro'
ax3[3].set_title("Log-Log Linear Regression with LassoLarsCV SN" +SN_test_train + cross_val, size=30)
ax3[3].set_ylabel("error in predicted migration rates", fontsize=30)
ax3[3].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax3[3].axhline(y=0, color='k', linestyle='--')
ax3[3].set_ylim([miny,maxy])
ax3[3].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax3[3].tick_params(axis='x', labelsize=20)
ax3[3].tick_params(axis='y', labelsize=20)

# ax3 = inset_axes(ax[2], width="60%", height = "70%", bbox_to_anchor=(.2, .455, .75, .5), bbox_transform=ax[2].transAxes, loc=1)
# ax3.scatter(y,L1Lin_error, facecolors='dodgerblue')
# ax3.axhline(y=0, color='k', linestyle='--')

ax4[3].scatter(SNtrainy,SN_L2Lin_error_train, facecolors='dodgerblue', label="train", s=100) # 'bo'
ax4[3].scatter(SNtesty,SN_L2Lin_error_test, facecolors='indianred',label="test", s=100) # 'ro'
ax4[3].set_title("Log-Log Linear Regression with RidgeCV SN" +SN_test_train + cross_val, size=30)
ax4[3].set_ylabel("error in predicted migration rates", fontsize=30)
ax4[3].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax4[3].axhline(y=0, color='k', linestyle='--')
ax4[3].set_ylim([miny,maxy])
ax4[3].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax4[3].tick_params(axis='x', labelsize=20)
ax4[3].tick_params(axis='y', labelsize=20)

fig1.tight_layout()
fig1.set_facecolor('w')
fig1.savefig("simple_LR_asym.png")
fig2.tight_layout()
fig2.set_facecolor('w')
fig2.savefig("RFECV_asym.png")
fig3.tight_layout()
fig3.set_facecolor('w')
fig3.savefig("L1_asym.png")
fig4.tight_layout()
fig4.set_facecolor('w')
fig4.savefig("L2_asym.png")
