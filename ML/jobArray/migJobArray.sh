#!/bin/bash

#$ -cwd
#$ -o qsub_logs/MLjobarray.txt
#$ -j y
#$ -l h_rt=5:00:00,h_data=30G,highp
#$ -m a
#$ -V

# refer https://www.hoffman2.idre.ucla.edu/Using-H2/Computing/Computing.html#running-array-jobs for more info on job arrays like this one

# FIRST run: generateMig.py to get file "mig_val.txt"
# for symmetric values: generateMig.py 0.0001 0.01 0 $seed $n
# for asymmetric values: generateMig.py 0.0001 0.01 1 $seed $n

# THEN run this file:
# qsub -t 1-$n migJobArray.sh $( pwd ) $( pwd ) $( pwd )/output_jobarray 5 1000 10 5

scripts_dir=$1
vcfDir=$2
outdir=$3
size=$4
Ne=$5
sampleSize=$6
seed=$7
infile=$8

if [ -f $infile ]; then
    line=`sed -n ${SGE_TASK_ID}p ${infile}`
else
	exit 1
fi
 
migEW=$(echo $line | cut -d " " -f 1)
migWE=$(echo $line | cut -d " " -f 2)
migNS=$(echo $line | cut -d " " -f 3)
migSN=$(echo $line | cut -d " " -f 4)


. /u/local/Modules/default/init/modules.sh

module load python/3.6.1
# export PATH="/u/home/m/mchotai/anaconda3/bin:$PATH"

# run the simulation (around 3 hours)
python steppingStoneSimulation.py $seed ${vcfDir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}.vcf $size $size ${Ne} ${sampleSize} $migEW $migWE $migNS $migSN
# add a line to the input file (around 1-2 hours)
/bin/bash make_MLinput_asymmetric.sh $scripts_dir $vcfDir $outdir $size $Ne $sampleSize $seed $migEW $migWE $migNS $migSN
