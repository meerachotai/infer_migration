# Estimating migration rates in stepping stone models

**Ben, Meera, April, Sriram**

**UCLA B.I.G. (2021) Project**

**Table of Contents**
- [Simulating stepping-stone model](#run-script-steppingstonesimulationpy)
- [Calculating F<sub>ST</sub>](#run-script-calculate_fstpy)
- [Calculating site frequency spectrum](#run-script-allele_freqpy)
- [Maching Learning Models](#Machine-Learning-Models)
  * [Input files](#run-script-make_MLinputsh)
  * [Available test data](#test-data)
  * [Predicting Migration Rates](#run-script-compare_regpy-or-comparereg_asymcombospy)
- [Visualising: PCA](#run-scripts-pcamake_pcapypcamake_metadatash-and-pcamakepca_samplingpy)

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
Alternately, run `ML/jobArray/generateMig.py` and `ML/jobArray/migJobArray.sh` to run a job array/ batch job of n uniformly-distributed values between given boundaries.
```
seed=1
size=5
Ne=1000
sampleSize=10
n=50 # number of rounds
lower=0.0001
upper=0.01
scripts_dir=$( pwd ) # current working directory
vcfDir=$( pwd )
outdir=$( pwd )/output_jobarray
migfile=mig_val.txt
generateMig.py $lower $upper 1 $seed $n $migfile
qsub -t 1:$n migJobArray.sh $scripts_dir $vcfDir $outdir $size $Ne $sampleSize $seed $migfile
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

#### Test Data
* EW.5_NS.5_N.1000_n.10_asym1_input.txt - asymmetric migration rates with **one** different migration rate (can be either EW, WE, NS or SN direction), not-normalized SFS columns
* EW.5_NS.5_N.1000_n.10_asym1_SN_input.txt - asymmetric migration rates with a different **SN** direction migration rate, normalized SFS columns
* EW.5_NS.5_N.1000_n.10_asym4_input.txt - asymmetric migration rates with different migration rates for all **four** directions, normalized SFS columns
* EW.5_NS.5_N.1000_n.10_sym693_input.txt - symmetric migration rates, normalized SFS columns
* EW.5_NS.5_N.1000_n.10_asym4_1GB_input.txt - asymmetric migration rates with different migration rates for all **four** directions, normalized SFS columns, genome size 1GB

#### Run script: `compare_reg.py` or `compareReg_asymCombos.py`

Predicting migration rates using variations of log-log regression models (with tuned hyperparameters), including:
* Simple Linear Regression (with cross-validation)
* Linear Regression with Recursive Feature Elimination (RFE) (with cross-validation)
* Linear Regression with L1/Lasso Penalty (with cross-validation)
* Linear Regression with L2/Ridge Penalty (with cross-validation)
* KernelRidge Regression
* RandomForest Regression (with out-of-bag error)
```
compare_reg.py EW.5_NS.5_N.1000_n.10_asym4_1GB_input.txt reg_asym_1GB.png reg_r2_asym_1GB.png 5 5 10 1000 5 'Symmetric 1GB - Log-Log Regression Models' 'Asymmetric 1GB - Simulated vs. Predicted \$R^{2}\$'"

compareReg_asymCombos.py EW.5_NS.5_N.1000_n.10_asym4_1GB_input.txt reg_asym_1GB.png reg_r2_asym_1GB.png 5 5 10 1000 5 NS-SN 'Asymmetric 1GB'"
```
**Output:** Predicted vs. Simulated log10(m) scatterplot, R<sup>2</sup> heatmap

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
