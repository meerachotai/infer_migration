#!/usr/bin/env python3

# make_PCA.py EW.5_NS.5_mig.0.005_N.1000_n.10.vcf pop_meta.txt EW.5_NS.5_mig.0.005_N.1000_n.10_PCA.png
import pandas as pd
import io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import numpy as np

# import seaborn as sns
# sns.set_style('white')
# sns.set_style('ticks')

file = str(sys.argv[1])
nfile = str(sys.argv[2])
img = str(sys.argv[3]) 
	
# read in input vcf file
with open(file, 'r') as f:
    lines = [l for l in f if not l.startswith('##')] # remove commented section
    
df = pd.read_csv(io.StringIO(''.join(lines)), sep = "\t") # treat as in-memory stream

index = 9
df = df.replace("0|0", "0")
df = df.replace("0|1", "1")
df = df.replace("1|0", "1")
df = df.replace("1|1", "2")

df = df.T
df = df.iloc[index:,:]

pca = PCA(n_components = 2)
pca_fit = pca.fit_transform(df)
pca_df = pd.DataFrame(data = pca_fit, columns = ['PC1', 'PC2'])

df_samples = pd.read_csv(nfile, delimiter='\t', index_col='index')
populations = df_samples.population.unique()
sample_population = df_samples.population.values

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('PC1 (' + str(per_var[0]) + '%)')
plt.ylabel('PC2 (' + str(per_var[1]) + '%)')
# sns.despine(offset=5)

x = pca_df.loc[:, 'PC1']
y = pca_df.loc[:, 'PC2']

for pop in populations:
    flt = (sample_population == pop)
    plt.plot(x[flt], y[flt], marker='o', linestyle=' ', 
                label=pop, markersize=6, mec='k', mew=.5)

plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

plt.savefig(img)    
    
