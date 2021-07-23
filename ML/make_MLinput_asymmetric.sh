#!/usr/bin/env bash

# ./make_MLinput_asymmetric.sh $( pwd ) $( pwd ) $( pwd )/output_png 5 1000 10 1 0.005 0.005 0.005 0.0075
# qsub -V -N job_ML -cwd -j y -o qsub_logs/ML.txt -m bae -b y -l h_rt=10:00:00,h_data=30G $cmd

scripts_dir=$1
shift
vcfDir=$1
shift
outdir=$1
shift
size=$1
shift
Ne=$1
shift
sampleSize=$1
shift
seed=$1
shift
migEW=$1
shift
migWE=$1
shift
migNS=$1
migSN=$2

echo Creating output directory ${outdir}
mkdir $outdir
# > ${outdir}/EW.${size}_NS.${size}_N.${Ne}_n.${sampleSize}_input.txt # refresh old activity, if existent

echo Reading ${vcfDir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}.vcf... 	

# run scripts to get output table files
# echo Running allele_freq.py
# ${scripts_dir}/allele_freq.py ${vcfDir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}.vcf ${outdir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}_freq.png ${outdir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}_freq.txt $size $size $sampleSize $Ne "${migEW},${migWE},${migNS},${migSN}"

echo Running allele_freq_norm.py
${scripts_dir}/allele_freq_norm.py ${vcfDir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}.vcf ${outdir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}_freq.png ${outdir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}_freq.txt $size $size $sampleSize $Ne "${migEW},${migWE},${migNS},${migSN}"


# echo Running calculate_fst.py
${scripts_dir}/calculate_fst.py ${vcfDir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}.vcf ${outdir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}_fst.png ${outdir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}_fst.txt $size $size $sampleSize $Ne "${migEW},${migWE},${migNS},${migSN}"

# declare output table files variables
freq="${outdir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}_freq.txt"
fst="${outdir}/EW.${size}_NS.${size}_migEW.${migEW}_migWE.${migWE}_migNS.${migNS}_migSN.${migSN}_N.${Ne}_n.${sampleSize}_${seed}_fst.txt"

printf "${migEW}\t${migWE}\t${migNS}\t${migSN}\t" >> ${outdir}/EW.${size}_NS.${size}_N.${Ne}_n.${sampleSize}_${seed}_input.txt
cat $freq | cut -f 4- | awk 'BEGIN{FS="";ORS="\t"} {print}' >> ${outdir}/EW.${size}_NS.${size}_N.${Ne}_n.${sampleSize}_${seed}_input.txt
cat $fst | cut -f 3 | awk 'BEGIN{FS="";ORS="\t"} {print}' >> ${outdir}/EW.${size}_NS.${size}_N.${Ne}_n.${sampleSize}_${seed}_input.txt
printf "\n" >> ${outdir}/EW.${size}_NS.${size}_N.${Ne}_n.${sampleSize}_${seed}_input.txt
