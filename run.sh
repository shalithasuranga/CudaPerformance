#!/bin/bash


# ----------------------------------------------
# Author - Shalitha Suranga
# -----------------------------------------------

# commands for each program compilation and its output files
programs=( 'c99 MatrixCPU.c -o output/out' 'nvcc MatrixGPUGlobal.cu -o output/out -Wno-deprecated-gpu-targets' 'nvcc MatrixGPUShared.cu -o output/out -Wno-deprecated-gpu-targets')
outputfiles=('output/datafile0.dat' 'output/datafile1.dat' 'output/datafile2.dat')

echo ""
echo "This script will create ${#outputfiles[@]} datafiles"
echo ""
	
# No of interations
SIZES=( 32 64 128 256 512 1024)
#SIZES=( 2 4 6 )
AVG_TIMES=4

GRAPHS=($(seq 1 ${#programs[@]}))
for g in "${GRAPHS[@]}"
do
	:
	outputfile="${outputfiles[$g-1]}"
	program="${programs[$g-1]}"

	# Remove if outputfile exists otherwise create
	if [ -e $outputfile ] 
	then
		rm -rf $outputfile
		touch $outputfile
	else
		touch $outputfile
	fi
	
	echo ""
	echo "---------- generating data for $outputfile ----------"
	echo ""

	for i in "${SIZES[@]}"
	do
		:

		TIMES=($(seq 0 $AVG_TIMES))
		total="0"
		echo "N=$i"
		for j in "${TIMES[@]}"
		do 
			:
			eval $program
			resp=$(./output/out $i)
			total=$(bc <<< "scale=10; $total+$resp")
			echo "# iteration=$j T=$resp"
		done
		avg=$(bc <<< "scale=10; $total/${#TIMES[@]}")
		printf "( $i, $avg )\n" >> $outputfile
		echo "T avg. = $avg"
	done
	echo "Written data to $outputfile"
done

echo ""
echo "Processing completed. Now executing report.sh"
echo ""

./report.sh



