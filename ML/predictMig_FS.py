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

numPopEW = 5
numPopNS = 5
sampleSize = 10
populations = numPopEW * numPopNS
N = 1000
file = "lowest_input.txt"

# recursive feature elimination (RFE) options:
# at every iteration, _remove_ % of the features will be eliminated
remove = 0.4

# for every type of regression, k-fold cross-validation is used
k = 5

# seed for LassoCV/RidgeCV - used as a starting point for selecting features
# https://stackoverflow.com/questions/48909927/why-is-random-state-required-for-ridge-lasso-regression-classifiers/48910233
seed = 3

data = pd.read_csv(file, sep = "\t", header = None)
data = data.dropna(axis='columns')

data = (np.log(data)).replace(-np.inf, 0) # FOR LOG-LOG LINEAR REGRESSION

X = data.iloc[:,1:]
y = data.iloc[:, 0]

labels = []
SFS_AF = [] # stores allele frequency of each column
FST_distance = [] # stores distance between the two populations being compared
fst_start = populations * 2 * sampleSize

# set column labels
for i in range(1,populations+1):
    string_SFS = "SFS: " + str(i) + " - "
    for x in range(1,sampleSize *2+1):
        labels.append(string_SFS + str(x))
        SFS_AF.append(x)
        
for i in range(1,populations+1):
    for j in range(i+1, populations+1):
        string_FST="FST: " + "[" + str(i) + "," + str(j) + "]"
        labels.append(string_FST)
        x_i = math.floor(i/numPopEW) # goes row-by-row
        y_i = i % numPopEW
        x_j = math.floor(j/numPopEW)
        y_j = j % numPopEW
        distance = math.sqrt(((x_i - x_j) ** 2) + ((y_i - y_j) ** 2))
        FST_distance.append(distance)


# for RFECV ------------------------------
selector = RFECV(LinearRegression(), step=remove, cv=k).fit(X,y)
selected_RFECV = np.array(labels)[selector.get_support()]

selected_FST = selector.get_support()[fst_start:]
distance_RFECV = np.array(FST_distance)[selected_FST]

selected_SFS = selector.get_support()[0:fst_start]
SFS_RFECV = np.array(SFS_AF)[selected_SFS]

selected_X = selector.transform(X) # gets *selected* X columns only
RFELR = LinearRegression().fit(selected_X, y)
RFE_coef = RFELR.coef_

fig1, ax1 = plt.subplots(2,2,figsize = (15,10))

cross_val = "\n" + str(k) + "-fold " + "Cross-Validation (#y = " + str(len(y)) + ")"
ax1 = ax1.flatten()

FST_coef_RFECV = np.array(RFE_coef[len(SFS_RFECV):])

dist_dict = {}
for i in range(len(distance_RFECV)):
    dist = distance_RFECV[i]
    if dist not in dist_dict:
        dist_dict[dist] = []
    dist_dict[dist].append(FST_coef_RFECV[i])

for key in dist_dict:
    weights = dist_dict[key]
    dist_dict[key] = sum(weights)/len(weights) # average for that distance
    
df = pd.DataFrame.from_dict(dist_dict, orient='index')

distances = np.array(df.index)
weights = np.array(df.iloc[:,0])

ax1[0].bar(distances, weights, alpha=0.7, width = 0.09)
ax1[0].set_xlim(0,7)
ax1[0].set_title("Average coefficients for $F_{ST}$ distances - RFECV", fontsize=15)
ax1[0].set_xlabel("Distance")
ax1[0].set_ylabel("Avg. Coefficients")

SFS_coef_RFECV = np.array(RFE_coef[0:len(SFS_RFECV)])

# average-out SFS by allele-frequencies
SFS_dict = {} # carries the coefficents for each SFS value
for i in range(len(SFS_RFECV)):
    af = SFS_RFECV[i]
    if af not in SFS_dict:
        SFS_dict[af] = []
    SFS_dict[af].append(SFS_coef_RFECV[i])

for key in SFS_dict:
    weights = SFS_dict[key]
    SFS_dict[key] = sum(weights)/len(weights) # average for that allele frequency
    
df = pd.DataFrame.from_dict(SFS_dict, orient='index')
# sort by allele_freq
af = np.array(df.index)
weights = np.array(df.iloc[:,0]) 

ax1[2].bar(af, weights, alpha=0.5, width = 0.6)
ax1[2].set_title("Average coefficients for allele freq. - RFECV", fontsize=15)
ax1[2].set_xlabel("Allele Frequencies")
ax1[2].set_ylabel("Avg. Coefficients")

# trainX, testX, trainy, testy = train_test_split(selected_X, y, test_size=1/k, random_state=seed)
# RFELR = LinearRegression().fit(trainX, trainy)

# for L1 --------------------------------

regL1 = LassoLarsCV(cv=k).fit(X,y)

selected = np.take(labels,np.array(regL1.coef_).nonzero())
distance_L1 = np.take(FST_distance, np.array(regL1.coef_[fst_start:]).nonzero())[0]
SFS_L1 = np.take(SFS_AF, np.array(regL1.coef_[0:fst_start]).nonzero())[0]
L1_coef = regL1.coef_

FST_coef_L1 = np.take(L1_coef[fst_start:], np.array(L1_coef[fst_start:].nonzero()))[0]

dist_dict = {} # carries the coefficents for each SFS value
for i in range(len(distance_L1)):
    dist = distance_L1[i]
    if dist not in dist_dict:
        dist_dict[dist] = []
    dist_dict[dist].append(FST_coef_L1[i])

for key in dist_dict:
    weights = dist_dict[key]
    dist_dict[key] = sum(weights)/len(weights) # average for that distance
    
dist_dict
df = pd.DataFrame.from_dict(dist_dict, orient='index')

distances = np.array(df.index)
weights = np.array(df.iloc[:,0])

ax1[1].bar(distances, weights, alpha=0.7, width = 0.09)
ax1[1].set_xlim(0,7)
ax1[1].set_title("Average coefficients for $F_{ST}$ distances - L1", fontsize=15)
ax1[1].set_xlabel("Distance")
ax1[1].set_ylabel("Avg. Coefficients")


SFS_coef_L1 = np.take(L1_coef[0:fst_start], np.array(L1_coef[0:fst_start].nonzero()))[0]

# average-out SFS by allele-frequencies
SFS_dict = {} # carries the coefficents for each SFS value
for i in range(len(SFS_L1)):
    af = SFS_L1[i]
    if af not in SFS_dict:
        SFS_dict[af] = []
    SFS_dict[af].append(SFS_coef_L1[i])

for key in SFS_dict:
    weights = SFS_dict[key]
    SFS_dict[key] = sum(weights)/len(weights) # average for that allele frequency
    

df = pd.DataFrame.from_dict(SFS_dict, orient='index')

# df
# sort by allele_freq
af = np.array(df.index)
weights = np.array(df.iloc[:,0]) 

ax1[3].bar(af, weights, alpha=0.5, width = 0.6)
ax1[3].set_title("Average coefficients for allele freq. - L1", fontsize=15)
ax1[3].set_xlabel("Allele Frequencies")
ax1[3].set_ylabel("Avg. Coefficients")

fig1.suptitle(cross_val, fontsize=20)
fig1.tight_layout()
fig1.savefig("examine_FS_coef.png")

fig, ax = plt.subplots(2,2,figsize = (15,10))

cross_val = "\n" + str(k) + "-fold " + "Cross-Validation (#y = " + str(len(y)) + ")"
ax = ax.flatten()

ax[0].hist(SFS_RFECV, bins = 20)
ax[0].set_title("chosen allele frequencies for RFECV",fontsize=20)

ax[1].hist(SFS_L1, bins = 20)
ax[1].set_title("chosen allele frequencies for L1",fontsize=20)

ax[2].hist(distance_RFECV)
ax[2].set_title("chosen $F_{ST}$ distances for RFECV", fontsize=20)

ax[3].hist(distance_L1)
ax[3].set_title("chosen $F_{ST}$ distances for L1", fontsize=20)

# for L2, there is no real 'feature elimination'
# np.array(regL2.coef_).nonzero() 

fig.suptitle(cross_val, fontsize=20)
fig.tight_layout()
fig.savefig("examine_FS.png")