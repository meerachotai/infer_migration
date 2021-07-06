#!/usr/bin/env bash

# make_metadata.sh 249 10 > pop_meta.txt
samples=$1
sampleSize=$2
printf "index\tpopulation\n"

for i in $(seq 0 1 $samples); do
	n=$(($i / $sampleSize))
	printf "$i\t$n\n"
done
