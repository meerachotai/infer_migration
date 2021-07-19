#!/usr/bin/env python3

# predictMig_LinReg.py EW.5_NS.5_N.1000_n.10_1_lowest_input.txt 4 0.4 0 compare_LR.png

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.colors as mcolors
import math
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import RidgeCV
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys 

file = str(sys.argv[1])

# for every type of regression, k-fold cross-validation is used
k = int(sys.argv[2])

# recursive feature elimination (RFE) options:
# at every iteration, _remove_ % of the features will be eliminated
remove = float(sys.argv[3])

# add small zoomed-in graphs?
zoom = int(sys.argv[4])

out_graph = str(sys.argv[5])

# seed for LassoCV/RidgeCV - used as a starting point for selecting features
# https://stackoverflow.com/questions/48909927/why-is-random-state-required-for-ridge-lasso-regression-classifiers/48910233
# seed = 5

data = pd.read_csv(file, sep = "\t", header = None)
data = data.dropna(axis='columns')

data = (np.log(data)).replace(-np.inf, 0) # FOR LOG-LOG LINEAR REGRESSION

X = data.iloc[:,1:]
y = data.iloc[:, 0]

# 1. SIMPLE LOG-LOG LINEAR REGRESSION ----------------------

simpleLin_y = cross_val_predict(LinearRegression(), X, y, cv=k)
# error = (predict - actual) / actual
simpleLin_error = (simpleLin_y - y) / y

# 2. RFE FEATURE SELECTION ---------------------------------

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
selector = RFECV(LinearRegression(), step=remove, cv=k).fit(X,y)
RFELin_y = selector.predict(X)
RFELin_error = (RFELin_y - y) / y

# 3. L1/Lasso FEATURE SELECTION ---------------------------

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV
# regL1 = LassoCV(cv=k, random_state=seed).fit(X,y) # note: does not converge for some values, not ideal
# L1Lin_y = regL1.predict(X)
# L1Lin_error = (L1Lin_y - y)/y

# LassoLarsCV explores more relevant alpha values compared to LassoCV
regL1 = LassoLarsCV(cv=k).fit(X,y) # note: does not converge for some values, not ideal
L1Lin_y = regL1.predict(X)
L1Lin_error = (L1Lin_y - y)/y

# 4. L2/Ridge FEATURE SELECTION ---------------------------

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
regL2 = RidgeCV(cv=k).fit(X, y) 
L2Lin_y = regL2.predict(X)
L2Lin_error = (L2Lin_y - y) / y

# GRAPHS ---------------------------------------------

fig, ax = plt.subplots(1,4,figsize = (30,10))
cross_val = "\n" + str(k) + "-fold " + "Cross-Validation (#y = " + str(len(y)) + ")"

# calculate number of selected of features
features = len(X.columns)
for i in range(k):
    features -= features * remove
selected = (features / len(X.columns)) * 100
removal_RFECV = "\n" + str(selected) + " % selected, with " + str(remove * 100) + " % removed every iteration"
# seeder = "\nseed: " + str(seed)
removal_L1 = "\n" + str((len(regL1.coef_.nonzero()[0]) / len(X.columns)) * 100) + " % selected"

# to set axes up with same limits for all subplots
maxy = max(max(simpleLin_error), max(RFELin_error), max(L1Lin_error), max(L2Lin_error))
maxy += maxy/10 # so it doesn't end EXACTLY at the max point
miny = min(min(simpleLin_error), min(RFELin_error), min(L1Lin_error), min(L2Lin_error))
miny += miny/10

ax[0].scatter(y,simpleLin_error, facecolors='dodgerblue')
ax[0].set_ylim([miny,maxy])
ax[0].set_title("Simple Log-Log Linear Regression" + cross_val, size=15)
ax[0].set_ylabel("error in predicted migration rates", fontsize=15)
ax[0].set_xlabel("actual migration rate ln(m)",fontsize=15)
ax[0].axhline(y=0, color='k', linestyle='--')
    
ax[1].scatter(y,RFELin_error, facecolors='dodgerblue')
ax[1].set_ylim([miny,maxy])
ax[1].set_title("Log-Log Linear Regression with RFECV" + cross_val + removal_RFECV, size=15)
ax[1].set_ylabel("error in predicted migration rates", fontsize=15)
ax[1].set_xlabel("actual migration rate ln(m)",fontsize=15)
ax[1].axhline(y=0, color='k', linestyle='--')

if(zoom != 0):
	ax2 = inset_axes(ax[1], width="60%", height = "70%", bbox_to_anchor=(.2, .455, .75, .5), bbox_transform=ax[1].transAxes, loc=1)
	ax2.scatter(y,RFELin_error, facecolors='dodgerblue')
	ax2.axhline(y=0, color='k', linestyle='--')

ax[2].scatter(y,L1Lin_error, facecolors='dodgerblue')
ax[2].set_ylim([miny,maxy])
ax[2].set_title("Log-Log Linear Regression with LassoLarsCV" + cross_val + removal_L1, size=15)
ax[2].set_ylabel("error in predicted migration rates", fontsize=15)
ax[2].set_xlabel("actual migration rate ln(m)",fontsize=15)
ax[2].axhline(y=0, color='k', linestyle='--')

if(zoom != 0):
	ax3 = inset_axes(ax[2], width="60%", height = "70%", bbox_to_anchor=(.2, .455, .75, .5), bbox_transform=ax[2].transAxes, loc=1)
	ax3.scatter(y,L1Lin_error, facecolors='dodgerblue')
	ax3.axhline(y=0, color='k', linestyle='--')

ax[3].scatter(y,L2Lin_error, facecolors='dodgerblue')
ax[3].set_ylim([miny,maxy])
ax[3].set_title("Log-Log Linear Regression with RidgeCV" + cross_val, size=15)
ax[3].set_ylabel("error in predicted migration rates", fontsize=15)
ax[3].set_xlabel("actual migration rate ln(m)",fontsize=15)
ax[3].axhline(y=0, color='k', linestyle='--')

fig.set_facecolor('w')
fig.savefig(out_graph)
