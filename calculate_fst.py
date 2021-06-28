import pandas as pd
import io
import math
import matplotlib.pyplot as plt
import sys

input_vcf = str(sys.argv[1])
new_file = str(sys.argv[2])
numPopEW = int(sys.argv[3]) 
numPopNS = int(sys.argv[4]) 
sampleSize = int(sys.argv[5])

populations = numPopEW * numPopNS

line = ""
index = 0
all_fst = []
all_pos = []
 
first_line = True
with open(input_vcf, 'r') as f:
    for line in f:
        if line.startswith("##"):
            continue
        else:
            if(first_line):
                line = line.strip().split("\t")
                for num,string in enumerate(line):
                    if(any(char.isdigit() for char in string)):
                        index = num
                        break
                first_line = False
                continue
            else:
                line = line.strip().split("\t")
                pos = line[1]
                line_sub = "".join(line[index:]).replace("|", "")
                top = 0
                h = 0
                overall_p = 0
                for i in range(populations):
                    bottom = top + (sampleSize *2)  # for diploid, x2
                    p = line_sub[top:bottom].count("1") # count freq of alternate allele
                    top = bottom
                    
                    overall_p += p # incrementing alt allele freq for total population
                    p = p / (sampleSize * 2) # p / (2 * subpop size) 
                    h += (2 * p * (1-p) * sampleSize) # 2pq * N_i, weighted by subpop size
                
                overall_p = overall_p / (2* sampleSize * populations) # p / (2 * total_pop)
                
                sub_h = h / (populations * sampleSize) # h / total_pop, weighted avg. of h
                total_h = overall_p * (1 - overall_p) * 2 # 2pq for total population
                fst = (total_h - sub_h) / total_h
                
                all_fst.append(fst)
                all_pos.append(pos)

plt.plot(all_pos, all_fst)
plt.ylabel('F_st')
plt.xlabel('SNP position')
plt.xticks([])

plt.savefig(new_file)
                

#                 print(sub_h)
#                 print(total_h)
#                 print("F_st =",fst * 100, "%")

				