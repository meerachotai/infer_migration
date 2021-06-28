# python steppingStoneSimulation.py 1 test.vcf 5 5 1000 10 0.001 0.005 0.0005 0.0005
import msprime
import numpy as np
import sys
import math
import random
# a stepping stone model would looks like a grid where the intersection points are populations
# neighboring populations can exchange genetic materials through migration
#-|--|--|--|--|--|-
#-|--|--|--|--|--|-
#-|--|--|--|--|--|-
#-|--|--|--|--|--|-
#-|--|--|--|--|--|-
#-|--|--|--|--|--|-
def steppingStone(fVCF, numPopEW, numPopNS, Ne, sampleSize, migEW, migWE, migNS, migSN):
	demeNumber = numPopNS * numPopEW
	migration_matrix = np.zeros((demeNumber,demeNumber)) #[[int(num) for num in line.split('\t')] for line in f_mig]
	coalescent_matrix = np.zeros((demeNumber,demeNumber)) 

	population_configurations = []
	demographic_events = []
	for i in range(0, numPopEW): #rows -- East & West Direction
		for j in range(0, numPopNS): #columns -- North & South Direction
			population_configurations.append(msprime.PopulationConfiguration(sample_size= 2*sampleSize, initial_size= Ne)) #diploid population size, haploid sampleSize
			indDeme = i * numPopEW + j
			# homogeneous Ne
			coalescent_matrix[indDeme, indDeme] = 1/Ne/2
			# for each node (i,j), assign migration rate for all the incoming edges if they exist.
			# M[j, k] is the rate at which lineages move from population j to population k in the coalescent process, that is, backwards in time  
			if (i > 0):
				migration_matrix[indDeme, indDeme-numPopEW] = migSN # back migration (i,j) to (i-1,j)
			if (i < numPopEW - 1):
				migration_matrix[indDeme, indDeme+numPopEW] = migNS #back migration (i,j) to (i+1,j)
			if (j > 0):
				migration_matrix[indDeme, indDeme - 1] = migEW  # back migration (i,j) to (i,j-1)
			if (j < numPopNS - 1):
				migration_matrix[indDeme, indDeme + 1] = migWE # back migration (i,j) to (i, j+1)

	dd = msprime.DemographyDebugger(
		population_configurations=population_configurations,
		migration_matrix=migration_matrix,
		demographic_events=demographic_events)
	dd.print_history()

	sim = msprime.simulate(
		population_configurations = population_configurations,
		migration_matrix = migration_matrix, 
		demographic_events = demographic_events,
		num_replicates = 1, length= 1e8, 
		recombination_rate=2e-8, mutation_rate=2e-8)#edit if needed
	outVCF = open(fVCF, 'w')
	for rep, tree_sequence in enumerate(sim):
		tree_sequence.write_vcf(outVCF, ploidy = 2) #ploidy needs to be 2 in order to convert to Relate format
		outVCF.close()#only need to output once as when there is only one replicate


random.seed(sys.argv[1])
fVCF = str(sys.argv[2]) 
numPopEW = int(sys.argv[3]) 
numPopNS = int(sys.argv[4]) 
Ne = int(sys.argv[5]) 
sampleSize = int(sys.argv[6]) 
migEW = float(sys.argv[7])
migWE = float(sys.argv[8])
migNS = float(sys.argv[9]) 
migSN = float(sys.argv[10]) 

steppingStone(fVCF, numPopEW, numPopNS, Ne, sampleSize, migEW, migWE, migNS, migSN);

