#!/usr/bin/env python3

# Dependencies: python3 (matplotlib, numpy)
# python3 calculate_fst.py test.vcf fst.png table.txt 5 5 10 1000 0.005
import math
import numpy as np
import matplotlib.pyplot as plt
import sys

file = str(sys.argv[1])
new_file = str(sys.argv[2])
table = str(sys.argv[3])
numPopEW = int(sys.argv[4]) 
numPopNS = int(sys.argv[5]) 
sampleSize = int(sys.argv[6])
ne = int(sys.argv[7])
mig = str(sys.argv[8])


populations = numPopEW * numPopNS

line = ""
index = 0

sumFstMatrix = np.zeros((populations, populations))
snpMatrix = np.zeros((populations, populations))
sqFstMatrix = np.zeros((populations, populations))

first_line = True

with open(file, 'r') as f:
    for line in f:
        if line.startswith("##"):
            continue
        else:
            if(first_line):
                line = line.strip().split("\t")
                for num,string in enumerate(line):
                    if(any(char.isdigit() for char in string)): # looking for index of first sample
                        index = num 
                        break
                first_line = False
                continue
            else:
                line = line.strip().split("\t") # each line is kth SNP
                line_sub = "".join(line[index:]).replace("|", "")

                # initialize a set for every SNP k
                hsMatrix = np.zeros((populations, populations))
                htMatrix = np.zeros((populations, populations))
                fstMatrix = np.zeros((populations, populations))
                
                for i in range(populations):
                    top_i = (i * sampleSize * 2) # start index for population i 
                    bottom_i = top_i + (sampleSize *2) # end index for population i
                    
                    p_1 = line_sub[top_i:bottom_i].count("1") # count freq of alternate allele in ith population
                    p_1_frac = p_1 / (sampleSize * 2) # as a fraction of the diploid subpop
                    
                    for j in range(i+1,populations):
                        
                        top_j = (j * sampleSize * 2)
                        bottom_j = top_j + (sampleSize *2)
                        p_2 = line_sub[top_j:bottom_j].count("1")
                        
                        if p_1 != p_2: # consider ONLY if they are segregating
                            snpMatrix[i,j] += 1
                        else:
                            continue
                            
                        p_2_frac = p_2 / (sampleSize * 2)
                        
                        # H_s = 2*p_i*q_i*N_i + 2*p_j*q_j*N_j / (N_i + N_j)
                        h = (2 * p_1_frac * (1-p_1_frac) * sampleSize) + (2 * p_2_frac * (1-p_2_frac) * sampleSize) 
                        sub_h = h / (2 * sampleSize)
                        
                        overall_p = p_1 + p_2 # p for total population
                        overall_p = overall_p / (2* sampleSize * 2) # overall_p / (2 * total_pop)
                        total_h = overall_p * (1 - overall_p) * 2 # 2pq for total population
                        
                        hsMatrix[i,j] = sub_h
                        htMatrix[i,j] = total_h
                        
                        if total_h == 0:
                            continue
                        fstMatrix[i,j] = (total_h - sub_h) / total_h
                            
                sqFst = np.multiply(fstMatrix, fstMatrix) # (F_st)^2
                
                sqFstMatrix = np.add(sqFstMatrix, sqFst) # sum((F_st)^2)
                sumFstMatrix = np.add(sumFstMatrix, fstMatrix) # sum(F_st)
                
meanFstMatrix = np.divide(sumFstMatrix, snpMatrix, out=np.zeros((populations, populations)), where=snpMatrix!=0)

# (E[x])^2 - square the meanFstMatrix
second_half = np.multiply(meanFstMatrix,meanFstMatrix)

# E[x^2] - take the mean of sqFstMatrix
first_half = np.divide(sqFstMatrix, snpMatrix, out=np.zeros((populations, populations)), where=snpMatrix!=0)

# variance = E[x^2] - (E[x])^2
variance = np.subtract(first_half, second_half)
variance = np.absolute(variance)

# standard deviation
sd = np.sqrt(variance)
# standard error = sd / square-root(snpMatrix)
sqrtSNPmatrix = np.sqrt(snpMatrix)
seFstMatrix = np.divide(sd, sqrtSNPmatrix, out=np.zeros((populations, populations)), where=sqrtSNPmatrix!=0)

#------------------------ calculate by distance ------------------------

ij = [] # populations
x = [] # distance
y = [] # mean 
e = [] # SE

for i in range(populations):
    for j in range(i+1,populations):
        x_i = math.floor(i/numPopEW) # goes row-by-row
        y_i = i % numPopEW
        x_j = math.floor(j/numPopEW)
        y_j = j % numPopEW
        distance = math.sqrt(((x_i - x_j) ** 2) + ((y_i - y_j) ** 2)) # calculate distance
        ij.append((i,j))
        x.append(distance)
        y.append(meanFstMatrix[i][j])
        e.append(seFstMatrix[i][j])

write_tab = open(table, "w")
# write_tab.write("Distance\tMean_FST\tSE_FST\n")

for i in range(len(x)):
    tb = [str(ij[i][0]), str(ij[i][1]), f'{y[i]:.20f}', f'{e[i]:.20f}']
    write_tab.write("\t".join(tb) + "\n")
write_tab.close()

title = "$N_e$: " + str(ne) + ", Sample size: " + str(sampleSize) + ", Migration Rates: " + mig

plt.scatter(x, y)
plt.errorbar(x, y, yerr = e, fmt="o")
plt.xlabel("distance between populations")
plt.ylabel("mean $F_{ST}$")
plt.suptitle("$F_{ST}$ vs. distance")
plt.title(title, fontsize = 10)

plt.savefig(new_file)
