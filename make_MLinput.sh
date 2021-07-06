#!/usr/bin/env bash

# ./make_MLinput.sh $( pwd ) $( pwd ) $( pwd )/output_3 5 1000 10 3 0.001 0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.05
# qsub -V -N job_ML -cwd -j y -o qsub_logs/ML.txt -m bae -b y -l h_rt=10:00:00,h_data=30G $cmd
# cut -f 27 -d ',' output/EW.5_NS.5_N.1000_n.10_input.csv

scripts_dir=$1
shift # shift options to left
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
mig=("$@") # enter array of migration values, separated by spaces

echo Creating output directory ${outdir}
mkdir $outdir
# > ${outdir}/EW.${size}_NS.${size}_N.${Ne}_n.${sampleSize}_input.txt # refresh old activity, if existent

for i in "${mig[@]}";
do
	echo Reading ${scripts_dir}/EW.${size}_NS.${size}_mig.${i}_N.${Ne}_n.${sampleSize}_${seed}.vcf... 	

	# run scripts to get output table files
	echo Running allele_freq.py
	${scripts_dir}/allele_freq.py ${scripts_dir}/EW.${size}_NS.${size}_mig.${i}_N.${Ne}_n.${sampleSize}_${seed}.vcf ${outdir}/EW.${size}_NS.${size}_mig.${i}_N.${Ne}_n.${sampleSize}_${seed}_freq.png ${outdir}/EW.${size}_NS.${size}_mig.${i}_N.${Ne}_n.${sampleSize}_${seed}_freq.txt $size $size $sampleSize $Ne $i
	
	echo Running calculate_fst.py
	${scripts_dir}/calculate_fst.py ${scripts_dir}/EW.${size}_NS.${size}_mig.${i}_N.${Ne}_n.${sampleSize}_${seed}.vcf ${outdir}/EW.${size}_NS.${size}_mig.${i}_N.${Ne}_n.${sampleSize}_${seed}_fst.png ${outdir}/EW.${size}_NS.${size}_mig.${i}_N.${Ne}_n.${sampleSize}_${seed}_fst.txt $size $size $sampleSize $Ne $i
	
	# declare output table files variables
	freq="${outdir}/EW.${size}_NS.${size}_mig.${i}_N.${Ne}_n.${sampleSize}_${seed}_freq.txt"
	fst="${outdir}/EW.${size}_NS.${size}_mig.${i}_N.${Ne}_n.${sampleSize}_${seed}_fst.txt"
	
	printf "${i}\t" >> ${outdir}/EW.${size}_NS.${size}_N.${Ne}_n.${sampleSize}_${seed}_input.txt
	cat $freq | cut -f 4- | awk 'BEGIN{FS="";ORS="\t"} {print}' >> ${outdir}/EW.${size}_NS.${size}_N.${Ne}_n.${sampleSize}_${seed}_input.txt
	cat $fst | cut -f 3 | awk 'BEGIN{FS="";ORS="\t"} {print}' >> ${outdir}/EW.${size}_NS.${size}_N.${Ne}_n.${sampleSize}_${seed}_input.txt
	printf "\n" >> ${outdir}/EW.${size}_NS.${size}_N.${Ne}_n.${sampleSize}_${seed}_input.txt
done
