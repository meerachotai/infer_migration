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
populations = numPopEW * numPopNS
N = 1000
file = "EW.5_NS.5_N.1000_n.10_1_lowest_input.txt"

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

X = data.iloc[:,1:]
y = data.iloc[:, 0]

trainX, testX, trainy, testy = train_test_split(X, y, test_size=test, random_state=seed)

# FOR LOG-LOG LINEAR REGRESSION
trainLX = (np.log(trainX)).replace(-np.inf, 0)
testLX = (np.log(testX)).replace(-np.inf, 0)
trainLy = (np.log(trainy)).replace(-np.inf, 0)
testLy = (np.log(testy)).replace(-np.inf, 0)

# data_log = (np.log(data)).replace(-np.inf, 0) 
# trainLX, testLX, trainLy, testLy = train_test_split(X_log, y_log, test_size=test, random_state=seed)
# X_log = data_log.iloc[:,1:]
# y_log = data_log.iloc[:, 0]

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

# with log-log
error_testL = [[] for _ in range(len(testLy))]
simpleLin = cross_validate(LinearRegression(), trainLX, trainLy, cv=k,return_estimator = True)
counter = 0
for i in simpleLin['estimator']:
    simpleLin_y = i.predict(testLX) # pick one of the folds' estimators to predict
    error = (simpleLin_y - testLy) / testLy
    for j in range(len(testLy)):
        error_testL[j].append(error.iloc[j])

# taking the mean for each of the folds' predicted value
simpleLin_error_testL = []
for i in error_testL:
    simpleLin_error_testL.append(sum(i)/len(i))
        
simpleLin_trainL = cross_val_predict(LinearRegression(), trainLX, trainLy, cv=k)
simpleLin_error_trainL = (simpleLin_trainL - trainLy) / trainLy

# 2. RFE FEATURE SELECTION ---------------------------------

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
selector = RFECV(LinearRegression(), step=remove, cv=k).fit(trainX,trainy) # only use training data for selecting features, and fitting

RFELin_train = selector.predict(trainX)
RFELin_test = selector.predict(testX)

RFELin_error_train = (RFELin_train - trainy)/ trainy
RFELin_error_test = (RFELin_test - testy) / testy

# with log-log
selector = RFECV(LinearRegression(), step=remove, cv=k).fit(trainLX,trainLy) # only use trainLing data for selecting features, and fitting

RFELin_trainL = selector.predict(trainLX)
RFELin_testL = selector.predict(testLX)

RFELin_error_trainL = (RFELin_trainL - trainLy)/ trainLy
RFELin_error_testL = (RFELin_testL - testLy) / testLy

# 3. L1/Lasso FEATURE SELECTION ---------------------------

# LassoLarsCV explores more relevant alpha values compared to LassoCV
regL1 = LassoLarsCV(cv=k).fit(trainX,trainy)
L1Lin_train = regL1.predict(trainX)
L1Lin_test = regL1.predict(testX)

L1Lin_error_train = (L1Lin_train - trainy)/trainy
L1Lin_error_test = (L1Lin_test - testy) / testy

# with log-log
regL1 = LassoLarsCV(cv=k).fit(trainLX,trainLy)
L1Lin_trainL = regL1.predict(trainLX)
L1Lin_testL = regL1.predict(testLX)

L1Lin_error_trainL = (L1Lin_trainL - trainLy)/trainLy
L1Lin_error_testL = (L1Lin_testL - testLy) / testLy

# 4. L2/Ridge FEATURE SELECTION ---------------------------

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
regL2 = RidgeCV(cv=k).fit(trainX,trainy)
L2Lin_train = regL2.predict(trainX)
L2Lin_test = regL2.predict(testX)

L2Lin_error_train = (L2Lin_train - trainy)/trainy
L2Lin_error_test = (L2Lin_test - testy) / testy

# with log-log
regL2 = RidgeCV(cv=k).fit(trainLX,trainLy)
L2Lin_trainL = regL2.predict(trainLX)
L2Lin_testL = regL2.predict(testLX)

L2Lin_error_trainL = (L2Lin_trainL - trainLy)/trainLy
L2Lin_error_testL = (L2Lin_testL - testLy) / testLy

# GRAPHS ---------------------------------------------


cross_val = "\n" + str(k) + "-fold " + "Cross-Validation with training data"
test_train = "\ntesting samples:" + str(len(testy)) + ", training samples:" + str(len(trainy))

# calculate number of selected of features
features = len(X.columns)
for i in range(k):
    features -= features * remove
selected = (features / len(X.columns)) * 100
removal_RFECV = "\n" + str(selected) + " % selected, with " + str(remove * 100) + " % removed every iteration"
seeder = "\nseed: " + str(seed)
removal_L1 = "\n" + str((len(regL1.coef_.nonzero()[0]) / len(X.columns)) * 100) + " % selected"

# to set axes up with same limits for all subplots
maxy = max(max(simpleLin_error_train), max(simpleLin_error_test), max(RFELin_error_test), max(RFELin_error_train), max(L1Lin_error_train), max(L1Lin_error_test), max(L2Lin_error_train), max(L2Lin_error_test))
maxy += maxy/30 # so it doesn't end EXACTLY at the max point
maxLy = max(max(simpleLin_error_trainL), max(simpleLin_error_testL), max(RFELin_error_testL), max(RFELin_error_trainL), max(L1Lin_error_trainL), max(L1Lin_error_testL), max(L2Lin_error_trainL), max(L2Lin_error_testL))
maxLy += maxLy/30

miny = min(min(simpleLin_error_train), min(simpleLin_error_test), min(RFELin_error_test), min(RFELin_error_train), min(L1Lin_error_train), min(L1Lin_error_test), min(L2Lin_error_train), min(L2Lin_error_test))
miny += miny/30 # so it doesn't end EXACTLY at the min point
minLy = min(min(simpleLin_error_trainL), min(simpleLin_error_testL), min(RFELin_error_testL), min(RFELin_error_trainL), min(L1Lin_error_trainL), min(L1Lin_error_testL), min(L2Lin_error_trainL), min(L2Lin_error_testL))
minLy += minLy/30

fig, ax = plt.subplots(2,4,figsize = (60,30))

ax = ax.flatten()

ax[0].scatter(trainy,simpleLin_error_train, facecolors='dodgerblue', label="train", s=100)
ax[0].scatter(testy,simpleLin_error_test, facecolors='indianred',label="test", s=100)
ax[0].set_title("Simple Linear Regression" +test_train + cross_val, size=30)
ax[0].set_ylabel("error in predicted migration rates", fontsize=30)
ax[0].set_xlabel("actual migration rate (m)",fontsize=30)
ax[0].axhline(y=0, color='k', linestyle='--')
ax[0].set_ylim([miny,maxy])
ax[0].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax[0].tick_params(axis='x', labelsize=20)
ax[0].tick_params(axis='y', labelsize=20)

ax[4].scatter(trainLy,simpleLin_error_trainL, facecolors='dodgerblue', label="train", s=100)
ax[4].scatter(testLy,simpleLin_error_testL, facecolors='indianred',label="test", s=100)
ax[4].set_title("Simple Log-Log Linear Regression" +test_train+ cross_val, size=30)
ax[4].set_ylabel("error in predicted migration rates", fontsize=30)
ax[4].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax[4].axhline(y=0, color='k', linestyle='--')
ax[4].set_ylim([minLy,maxLy])
ax[4].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax[4].tick_params(axis='x', labelsize=20)
ax[4].tick_params(axis='y', labelsize=20)

ax[1].scatter(trainy,RFELin_error_train, facecolors='dodgerblue', label="train", s=100) 
ax[1].scatter(testy,RFELin_error_test, facecolors='indianred',label="test", s=100)
ax[1].set_title("Linear Regression with RFECV"  +test_train+ cross_val + removal_RFECV, size=30)
ax[1].set_ylabel("error in predicted migration rates", fontsize=30)
ax[1].set_xlabel("actual migration rate (m)",fontsize=30)
ax[1].axhline(y=0, color='k', linestyle='--')
ax[1].set_ylim([miny,maxy])
ax[1].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax[1].tick_params(axis='x', labelsize=20)
ax[1].tick_params(axis='y', labelsize=20)

ax[5].scatter(trainLy,RFELin_error_trainL, facecolors='dodgerblue', label="train", s=100) 
ax[5].scatter(testLy,RFELin_error_testL, facecolors='indianred',label="test", s=100)
ax[5].set_title("Log-Log Linear Regression with RFECV" +test_train + cross_val + removal_RFECV, size=30)
ax[5].set_ylabel("error in predicted migration rates", fontsize=30)
ax[5].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax[5].axhline(y=0, color='k', linestyle='--')
ax[5].set_ylim([minLy,maxLy])
ax[5].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax[5].tick_params(axis='x', labelsize=20)
ax[5].tick_params(axis='y', labelsize=20)

# ax2 = inset_axes(ax[1], width="60%", height = "70%", bbox_to_anchor=(.2, .455, .75, .5), bbox_transform=ax[1].transAxes, loc=1)
# ax2.scatter(y,RFELin_error, facecolors='dodgerblue')
# ax2.axhline(y=0, color='k', linestyle='--')

ax[2].scatter(trainy,L1Lin_error_train, facecolors='dodgerblue', label="train", s=100) # 'bo'
ax[2].scatter(testy,L1Lin_error_test, facecolors='indianred',label="test", s=100) # 'ro'
ax[2].set_title("Linear Regression with LassoLarsCV" +test_train + cross_val + removal_L1, size=30)
ax[2].set_ylabel("error in predicted migration rates", fontsize=30)
ax[2].set_xlabel("actual migration rate (m)",fontsize=30)
ax[2].axhline(y=0, color='k', linestyle='--')
ax[2].set_ylim([miny,maxy])
ax[2].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax[2].tick_params(axis='x', labelsize=20)
ax[2].tick_params(axis='y', labelsize=20)

ax[6].scatter(trainLy,L1Lin_error_trainL, facecolors='dodgerblue', label="train", s=100) # 'bo'
ax[6].scatter(testLy,L1Lin_error_testL, facecolors='indianred',label="test", s=100) # 'ro'
ax[6].set_title("Log-Log Linear Regression with LassoLarsCV" +test_train + cross_val + removal_L1, size=30)
ax[6].set_ylabel("error in predicted migration rates", fontsize=30)
ax[6].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax[6].axhline(y=0, color='k', linestyle='--')
ax[6].set_ylim([minLy,maxLy])
ax[6].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax[6].tick_params(axis='x', labelsize=20)
ax[6].tick_params(axis='y', labelsize=20)

# ax3 = inset_axes(ax[2], width="60%", height = "70%", bbox_to_anchor=(.2, .455, .75, .5), bbox_transform=ax[2].transAxes, loc=1)
# ax3.scatter(y,L1Lin_error, facecolors='dodgerblue')
# ax3.axhline(y=0, color='k', linestyle='--')

ax[3].scatter(trainy,L2Lin_error_train, facecolors='dodgerblue', label="train", s=100) # 'bo'
ax[3].scatter(testy,L2Lin_error_test, facecolors='indianred',label="test", s=100) # 'ro'
ax[3].set_title("Linear Regression with RidgeCV" +test_train + cross_val, size=30)
ax[3].set_ylabel("error in predicted migration rates", fontsize=30)
ax[3].set_xlabel("actual migration rate (m)",fontsize=30)
ax[3].axhline(y=0, color='k', linestyle='--')
ax[3].set_ylim([miny,maxy])
ax[3].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax[3].tick_params(axis='x', labelsize=20)
ax[3].tick_params(axis='y', labelsize=20)

ax[7].scatter(trainLy,L2Lin_error_trainL, facecolors='dodgerblue', label="train", s=100) # 'bo'
ax[7].scatter(testLy,L2Lin_error_testL, facecolors='indianred',label="test", s=100) # 'ro'
ax[7].set_title("Log-Log Linear Regression with RidgeCV" +test_train + cross_val, size=30)
ax[7].set_ylabel("error in predicted migration rates", fontsize=30)
ax[7].set_xlabel("actual migration rate ln(m)",fontsize=30)
ax[7].axhline(y=0, color='k', linestyle='--')
ax[7].set_ylim([minLy,maxLy])
ax[7].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
ax[7].tick_params(axis='x', labelsize=20)
ax[7].tick_params(axis='y', labelsize=20)

fig.tight_layout()
fig.set_facecolor('w')
fig.savefig("compare_LR_all.png")
