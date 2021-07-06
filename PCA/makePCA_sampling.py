#!/usr/bin/env python3

# seed=1
# size=5
# mig=0.01
# Ne=1000
# sampleSize=10
# cmd="./makePCA_sampling.py EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}.vcf pop_meta.txt EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}_PCAsampling.png ${size} ${size} ${sampleSize} 2 ${Ne} ${mig}"
# qsub -V -N PCA_${size}_${mig} -cwd -j y -o qsub_logs/PCA_${size}_${mig}.txt -m bae -b y -l h_rt=5:00:00,h_data=30G $cmd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import sys

input_vcf = str(sys.argv[1])
meta = str(sys.argv[2])
graph = str(sys.argv[3])
numPopEW = int(sys.argv[4]) 
numPopNS = int(sys.argv[5]) 
sampleSize = int(sys.argv[6])
increments = int(sys.argv[7])
ne = int(sys.argv[8])
mig = float(sys.argv[9])


# file = "test_20.vcf"
# numPopEW = 5
# numPopNS = 5
# sampleSize = 10
# increments = 2
# ne = 1000
# mig=0.005

# GRAPHING FUNCTION
def plot_pca(pca_df, ax, sample_population, populations, per_var, eachPop):
    x = pca_df.loc[:, 'PC1']
    y = pca_df.loc[:, 'PC2']
    for pop in populations:
        flt = (sample_population == pop)
        ax.plot(x[flt], y[flt], marker='o', linestyle=' ', 
                    label=pop, markersize=6, mec='k', mew=.5)
    ax.set_xlabel('PC1 (' + str(per_var[0]) + '%)')
    ax.set_ylabel('PC2 (' + str(per_var[1]) + '%)')
    ax.set_title("samples used = " + str(eachPop))
	
totalSamples = numPopEW * numPopNS * sampleSize   

# read in input vcf file
with open(input_vcf, 'r') as f:
    lines = [l for l in f if not l.startswith('##')] # remove commented section
    
df = pd.read_csv(io.StringIO(''.join(lines)), sep = "\t") # treat as in-memory stream

index = 9
df = df.replace("0|0", "0")
df = df.replace("0|1", "1")
df = df.replace("1|0", "1")
df = df.replace("1|1", "2")

df = df.T
df = df.iloc[index:,:]   

eachPopList = range(increments,sampleSize+increments,increments)
Len = len(eachPopList)
fig, axs = plt.subplots(Len, 1, figsize=(8, 10*Len))

# fig, axs = plt.subplots(1,Len, figsize=(5*Len, 5))

figIndex=0

for eachPop in eachPopList: 
    rows = []
    i = 0
    while i < totalSamples:
        for j in range(eachPop):
            rows.append(i + j)
        i += sampleSize

    pca = PCA(n_components = 2)
    pca_fit = pca.fit_transform(df.iloc[rows])
    pca_df = pd.DataFrame(data = pca_fit, columns = ['PC1', 'PC2'])
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)
    
    df_meta = pd.read_csv(meta, delimiter='\t', index_col='index')
    populations = df_meta.iloc[rows].population.unique()
    sample_population = df_meta.iloc[rows].population.values
    
    plot_pca(pca_df, axs[figIndex], sample_population, populations, per_var, eachPop)
    
    figIndex += 1


# axs[-1].legend(bbox_to_anchor=(1, 1), loc='upper left')
axs[0].legend(bbox_to_anchor=(1, 1), loc='upper left')

fig.suptitle("Principal Component Analysis" + "\n$N_e$: " + str(ne) + ", Sample size: " + str(sampleSize) + ", Homogenous Migration Rate: " + str(mig))
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

fig.savefig(graph)

