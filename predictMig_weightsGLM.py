import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import matplotlib.colors as mcolors

numPopEW = 5
numPopNS = 5
sampleSize = 10
populations = numPopEW * numPopNS
N = 1000
file = "EW.5_NS.5_N.1000_n.10_normalised_input.txt"
cv = 4 # number of iterations of 'cross validation'

# to give weight values in 3 sections
weight_val = [0.002,0.004] # HAS to be in ascending order
weight_dist = [50,1,0.1] # according to the weights given to smallest->largest sections


data = pd.read_csv(file, sep = "\t", header = None)
data = data.dropna(axis='columns')

# fst_start = (populations*sampleSize * 2) + 1 # where do F_st columns start?
# data = pd.concat([data.iloc[:,0], data.iloc[:,fst_start:]], axis=1) # only 1, Fst columns

data= data.sample(frac=1).reset_index(drop=True)
test_size = math.floor(len(data) / cv) # subsectioning the data
fig, ax = plt.subplots(1,1,figsize = (8,8))

i=0 # for counting what section we're using
for j in range(cv):
    mask = list(range(i, i+test_size)) # TEST DATA
    not_mask = [m for m in range(0,len(data)) if m not in mask] # TRAINING DATA
    
    i += test_size # for next iteration
    
    dfTrain = data.iloc[not_mask,:]
    dfTest = data.iloc[mask,:]

    trainy = dfTrain.iloc[:,0]
    trainX = dfTrain.iloc[:,1:]

    testy = dfTest.iloc[:,0]
    testX = dfTest.iloc[:,1:]
    
    weight = []
    for elem in trainy:
        if(elem < weight_val[1]):
            if(elem < weight_val[0]):
                weight.append(weight_dist[0])
            else:
                weight.append(weight_dist[1])
        else:
            weight.append(weight_dist[2])
            
    clf_GLM = linear_model.PoissonRegressor()
    clf_GLM.fit(trainX, trainy, weight) # Poisson dist. GLM

    predictY_GLM_test = np.array(clf_GLM.predict(testX))
    actualY_GLM_test = np.array(testy)

#     test training data as well
    predictY_GLM_train = np.array(clf_GLM.predict(trainX))
    actualY_GLM_train = np.array(trainy)
        
    errorY_GLM_train = (predictY_GLM_train - actualY_GLM_train) / actualY_GLM_train
    errorY_GLM_test = (predictY_GLM_test - actualY_GLM_test) / actualY_GLM_test
    
    if(j == 0):
        ax.scatter(actualY_GLM_train,errorY_GLM_train, facecolors='dodgerblue', label="train") # 'bo'
        ax.scatter(actualY_GLM_test,errorY_GLM_test, facecolors='indianred',label="test") # 'ro'
    else:
        ax.scatter(actualY_GLM_train,errorY_GLM_train,facecolors='dodgerblue', label='_nolegend_') #'bo'
        ax.scatter(actualY_GLM_test,errorY_GLM_test,facecolors='indianred',label='_nolegend_')
    
    ax.set_ylabel("error in predicted migration rates")
    ax.set_xlabel("actual migration rate (m)")
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    
weightlab = "\nweights: m < " + str(weight_val[0]) + ": " + str(weight_dist[0])  + ", " + str(weight_val[0]) + " < m < " + str(weight_val[1]) + ": " + str(weight_dist[1]) + ", m > " + str(weight_val[1]) + ": " + str(weight_dist[2])
fig.suptitle("m tested every iteration = " + str(test_size) + " out of " + str(len(data)) + "\ntotal iterations = " + str(cv) + weightlab)
fig.tight_layout()

fig.savefig("calc_GLM_iter_cv" + str(cv) +"_weights.png")
