# Estimating demographic parameters in stepping stone models

### Ben, Meera, April, Sriram
#### UCLA B.I.G. (2021) Project

#### Run script: `steppingStoneSimulation.py`
```
seed=1
size=5
mig=0.005
Ne=1000
sampleSize=10
cmd="python steppingStoneSimulation.py $seed EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}.vcf $size $size ${Ne} ${sampleSize} $mig $mig $mig $mig"
qsub -V -N job_${size}_${mig} -cwd -j y -o qsub_logs/${size}_${mig}.txt -m bae -b y -l h_rt=5:00:00,h_data=30G $cmd
```
#### Run script: `calculate_fst.py`
```
cmd="./calculate_fst.py EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}.vcf EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}_fst.png EW.${size}_NS.${size}_mig.${i}_N.${Ne}_n.${sampleSize}_${seed}_fst.txt $size $size $sampleSize $Ne $mig"
qsub -V -N job_${size}_${mig}_fst -cwd -j y -o qsub_logs/${size}_${mig}_fst.txt -m bae -b y -l h_rt=5:00:00,h_data=30G $cmd
```

#### Run script: `allele_freq.py`
```
cmd="./allele_freq.py EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}.vcf EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}_freq.png EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}_freq.txt $size $size $sampleSize $Ne $mig"
qsub -V -N job_${size}_${mig}_freq -cwd -j y -o qsub_logs/${size}_${mig}_freq.txt -m bae -b y -l h_rt=5:00:00,h_data=20G $cmd
```

#### Run scripts: `PCA/make_PCA.py`,`PCA/make_metadata.sh` and `PCA/makePCA_sampling.py`
```
# first make a population metadata file
sampleSize=10
samples=249 # 0-indexed, so this is actually (size * size - 1)
./make_metadata.sh $samples $sampleSize > metadata.txt

cmd="./make_PCA.py EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}.vcf metadata.txt EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}_PCA.png"
qsub -V -N job_${size}_${mig}_PCA -cwd -j y -o qsub_logs/${size}_${mig}_PCA.txt -m bae -b y -l h_rt=2:00:00,h_data=30G $cmd

cmd="./makePCA_sampling.py EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}.vcf metadata.txt EW.${size}_NS.${size}_mig.${mig}_N.${Ne}_n.${sampleSize}_${seed}_PCAsampling.png ${size} ${size} ${sampleSize} 2 ${Ne} ${mig}"
qsub -V -N PCA_${size}_${mig} -cwd -j y -o qsub_logs/PCA_${size}_${mig}.txt -m bae -b y -l h_rt=5:00:00,h_data=30G $cmd
```

#### Run scripts: `make_MLinput.sh`
```
scriptsDir=$( pwd ) # current working directory
vcfDir=$( pwd )
outDir=$( pwd )/output
mig=(0.005 0.01 0.015 0.02 0.03 0.05) # define array of migration values
cmd="${scriptsDir}/make_MLinput.sh $scriptsDir $vcfDir $outDir $size $Ne $sampleSize $seed $mig"
qsub -V -N job_ML -cwd -j y -o qsub_logs/ML.txt -m bae -b y -l h_rt=5:00:00,h_data=30G $cmd
```

**Output:** `${outDir}/EW.${size}_NS.${size}_N.${Ne}_n.${sampleSize}_input.txt` - file with information that can be used to train and test ML algorithms.

**Format:** each row in the file is organized in the following format:

* m - migration rates
* SFS - site frequency spectrum
* N - number of total populations (size * size)
* A - maximum number of alternate alleles in a sampled diploid population (sampleSize * 2)
* s - number of SNPs with subscript number of alternate allele

| m | SFS<sub>1</sub>     | ... | SFS<sub>N</sub> | F<sub>ST,[0,1]</sub> | ... | F<sub>ST,[N-1,N]</sub>
|--- | -------- | ---- | ------------- |---------| -------- | -----------|
| | s<sub>1</sub> ... s<sub>A</sub> | ... | s<sub>1</sub> ... s<sub>A</sub> | | | |

**Dimensions:**

* The total number of SFS columns = N * A
* The total number of F<sub>ST</sub> columns (upper-triangular NxN matrix, without the diagonal) = N * (N - 1) / 2
