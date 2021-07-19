# Estimating demographic parameters in stepping stone models

**Ben, Meera, April, Sriram**

**UCLA B.I.G. (2021) Project**

**Table of Contents**
- [Simulating stepping-stone model](#run-script-steppingstonesimulationpy)
- [Calculating site frequency spectrum](#run-script-allele_freqpy)
- [Calculating F<sub>ST</sub>](#run-script-calculate_fstpy)
- [Maching Learning Models](#Machine-Learning-Models)
  * [Input files](#run-script-make_MLinputsh)
  * [Predicting Migration Rates](#run-script-predictMig_LinRegpy)

#### Run script: `steppingStoneSimulation.py`
Uses the python msprime library to generate a vcf file for a stepping-stone population.
```
seed=1
size=5
mig=0.005
Ne=1000
sampleSize=10
cmd="python steppingStoneSimulation.py $seed EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}.vcf $size $size ${Ne} ${sampleSize} $mig $mig $mig $mig"
qsub -V -N job_${size}_${mig} -cwd -j y -o qsub_logs/${size}_${mig}.txt -m bae -b y -l h_rt=5:00:00,h_data=30G $cmd
```
Alternately, to run a number of migration values at once:
```
for i in "${mig[@]}"; do
cmd="python steppingStoneSimulation.py $seed EW.${size}_NS.${size}_mig.${i}_N.${Ne}_n.${sampleSize}_${seed}.vcf $size $size ${Ne} ${sampleSize} $i $i $i $i"
qsub -V -N job_${size}_${i}_${seed} -cwd -j y -o qsub_logs/${size}_${i}_${seed}.txt -m bae -b y -l h_rt=5:00:00,h_data=30G $cmd
done
```
#### Run script: `calculate_fst.py`
Calculates F<sub>ST</sub> for pairs of populations given a vcf file.
```
cmd="./calculate_fst.py EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}.vcf EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}_fst.png EW.${size}_NS.${size}_mig.${i}_N.${Ne}_n.${sampleSize}_${seed}_fst.txt $size $size $sampleSize $Ne $mig"
qsub -V -N job_${size}_${mig}_fst -cwd -j y -o qsub_logs/${size}_${mig}_fst.txt -m bae -b y -l h_rt=5:00:00,h_data=30G $cmd
```

#### Run script: `allele_freq.py`
Calculates the allele/site frequency spectrum for each population given a vcf file.
```
cmd="./allele_freq.py EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}.vcf EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}_freq.png EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}_freq.txt $size $size $sampleSize $Ne $mig"
qsub -V -N job_${size}_${mig}_freq -cwd -j y -o qsub_logs/${size}_${mig}_freq.txt -m bae -b y -l h_rt=5:00:00,h_data=20G $cmd
```

### Machine Learning Models

#### Run script: `make_MLinput.sh`
Runs both `calculate_fst.py` and `allele_freq.py` on vcf files for a given array of migration values and generates an output file with a summary of the data generated, which can be used to train and test ML algorithms to predict migration rates.
```
scriptsDir=$( pwd ) # current working directory
vcfDir=$( pwd )
outDir=$( pwd )/output
mig=(0.005 0.01 0.015 0.02 0.03 0.05) # define array of migration values
cmd="${scriptsDir}/make_MLinput.sh $scriptsDir $vcfDir $outDir $size $Ne $sampleSize $seed ${mig[@]}"
module load python/3.6.1
qsub -V -N job_ML_${seed} -cwd -j y -o qsub_logs/ML_${seed}.txt -m bae -b y -l h_rt=5:00:00,h_data=30G $cmd
```

**Output:** `${outDir}/EW.${size}_NS.${size}_N.${Ne}_n.${sampleSize}_input.txt`

**Format:** each row in the file is organized in the following format:

* m - migration rates
* SFS - site frequency spectrum
* N - number of total populations (size * size)
* A - maximum number of alternate alleles in a sampled diploid population (sampleSize * 2)
* s - number of SNPs with subscript number of alternate allele

| m | SFS<sub>1</sub>     | ... | SFS<sub>N</sub> | F<sub>ST[1,2]</sub> | ... | F<sub>ST[N-1,N]</sub>
|--- | -------- | ---- | ------------- |---------| -------- | -----------|
| | s<sub>1</sub> ... s<sub>A</sub> | ... | s<sub>1</sub> ... s<sub>A</sub> | | | |

**Dimensions:**

* The total number of SFS columns = N * A
* The total number of F<sub>ST</sub> columns (upper-triangular NxN matrix, without the diagonal) = N * (N - 1) / 2

#### Run script: `predictMig_LinReg.py`

Prediciting migration rates using variations of linear regression models with cross-validation
```
input=EW.5_NS.5_N.1000_n.10_1_lowest_input.txt
k=4 # number of folds for cross-validation
eliminate=0.4 # % of features to eliminate per iteration for RFE
zoom=0 # 0/1 boolean for adding zoomed-in graphs for Lasso and RFE
out=compare_LR.png
predictMig_LinReg.py $input $k $eliminate $zoom $out
```
**Cross Validation:** Avoids overfitting of the data without reducing the number of samples that can be used for learning the model. Using CV, the training set is split into k smaller sets. For each of the k “folds”:

* A model is trained using k-1 of the folds as training data
* The resulting model is tested using the remaining part of the data

**Log-log linear regression:** Uses the `cross_val_predict` method for k-fold cross-validated predicted estimates for each migration rate (for when it belonged to the testing dataset).

**Log-log linear regression with recursive feature elimination:** Uses the `RFECV` method to eliminate features within k-fold cross-validation iterations, and generates predictions for migration rates at the end with the remaining selected features.

**Log-log linear regression with L1/Lasso feature elimination:** Uses the `LassoLarsCV` method to carry out regularization, setting some features' coefficients to zero, effectively selecting the features deemed most important for prediction. The k-fold cross-validation helps find an appropriate regularization alpha parameter. 

**Log-log linear regression with L2/Ridge:** Uses the `RidgeCV` method to carry out regularization, setting un-important features very close to 0 but not removing them altogether. In our case, SFS and F<sub>ST</sub> columns can be redundant, which would mean that Lasso may be a more appropriate regularization method.

Note that error in the graphs is calculated as `(predicted_m - actual_m) / (actual_m)`.

#### Run scripts: `PCA/make_PCA.py`,`PCA/make_metadata.sh` and `PCA/makePCA_sampling.py`
Visualising the stepping-stone model using PCA. Also investigates the effect of downsampling on these PCA plots.
```
# first make a population metadata file
sampleSize=10
samples=249 # 0-indexed, so this is actually (size * size - 1)
./make_metadata.sh $samples $sampleSize > metadata.txt

cmd="./make_PCA.py EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}.vcf metadata.txt EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}_PCA.png"
qsub -V -N job_${size}_${mig}_PCA -cwd -j y -o qsub_logs/${size}_${mig}_PCA.txt -m bae -b y -l h_rt=2:00:00,h_data=30G $cmd

cmd="./makePCA_sampling.py EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}.vcf metadata.txt EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}_PCAsampling.png ${size} ${size} ${sampleSize} 2 ${Ne} $mig"
qsub -V -N PCA_${size}_${mig} -cwd -j y -o qsub_logs/PCA_${size}_${mig}.txt -m bae -b y -l h_rt=5:00:00,h_data=30G $cmd
```
