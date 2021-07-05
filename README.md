# Estimating demographic parameters in stepping stone models

### Ben, Meera, April, Sriram
#### UCLA B.I.G. (2021) Project

#### Run script: `steppingStoneSimulation.py`
```
seed=1
size=8
mig=0.005
cmd="python steppingStoneSimulation.py $seed EW.${size}_NS.${size}_mig.${mig}_N.1000_n.10.vcf $size $size 1000 10 $mig $mig $mig $mig"
qsub -V -N job_${size}_${mig} -cwd -j y -o qsub_logs/${size}_${mig}.txt -m bae -b y -l h_rt=10:00:00,h_data=30G $cmd
```
#### Run script: `calculate_fst.py`
```
cmd="calculate_fst.py EW.${size}_NS.${size}_mig.${mig}_N.1000_n.10.vcf EW.${size}_NS.${size}_mig.${mig}_N.1000_n.10_fst.png EW.${size}_NS.${size}_mig.${mig}_N.1000_n.10_fst.txt $size $size 10 1000 $mig"
qsub -V -N job_${size}_${mig}_fst -cwd -j y -o qsub_logs/${size}_${mig}_fst.txt -m bae -b y -l h_rt=5:00:00,h_data=30G $cmd
```

#### Run script: `allele_freq.py`
```
cmd="allele_freq.py EW.${size}_NS.${size}_mig.${mig}_N.1000_n.10.vcf EW.${size}_NS.${size}_mig.${mig}_N.1000_n.10_freq.png EW.${size}_NS.${size}_mig.${mig}_N.1000_n.10_freq.txt $size $size 10 1000 $mig"
qsub -V -N job_${size}_${mig}_freq -cwd -j y -o qsub_logs/${size}_${mig}_freq.txt -m bae -b y -l h_rt=5:00:00,h_data=20G $cmd
```

#### Run scripts: `make_PCA.py` and `make_metadata.sh`
```
# first make a population metadata file
sampleSize=10
samples=249 # 0-indexed, so this is actually (samples - 1)
make_metadata.sh $samples $sampleSize > metadata.txt

cmd="make_PCA.py EW.${size}_NS.${size}_mig.${mig}_N.1000_n.10.vcf metadata.txt EW.${size}_NS.${size}_mig.${mig}_N.1000_n.10_PCA.png"
qsub -V -N job_${size}_${mig}_PCA -cwd -j y -o qsub_logs/${size}_${mig}_PCA.txt -m bae -b y -l h_rt=2:00:00,h_data=30G $cmd
```

#### Run scripts: `make_MLinput.sh`

**Output:** .txt file with information that can be used to train and test ML algorithms.

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
