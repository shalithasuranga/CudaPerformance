#!/bin/bash


# ----------------------------------------------
# Author - Shalitha Suranga
# ----------------------------------------------

# ---------- Configuration ---------------------
CUDA=1
SIZES=( 32 64 128 256 512 1024 )
THREADS_PER_BLOCK=( 16 64 256 1024 )
AVG_TIMES=10
FIXED_MATRIX=1024
FIXED_BLOCK_SIZE=256
MATRIX_FOR_TABLE=1024

let AVG_TIMES=AVG_TIMES-1
# commands for each program compilation and its output files
if [ $CUDA -eq 0 ] 
	then
	programs=( 'c99 MatrixCPU.c -o output/out' 'c99 MatrixGPUGlobal.c -o output/out' 'c99 MatrixGPUShared.c -o output/out' 'c99 MatrixGPUGlobal.c -o output/out' 'c99 MatrixGPUShared.c -o output/out')
else
	programs=( 'c99 MatrixCPU.c -o output/out' 'nvcc MatrixGPUGlobal.cu -o output/out -Wno-deprecated-gpu-targets' 'nvcc MatrixGPUShared.cu -o output/out -Wno-deprecated-gpu-targets' 'nvcc MatrixGPUGlobal.cu -o output/out -Wno-deprecated-gpu-targets' 'nvcc MatrixGPUShared.cu -o output/out -Wno-deprecated-gpu-targets')
fi

outputfiles=('output/datafile0.dat' 'output/datafile1.dat' 'output/datafile2.dat' 'output/datafile3.dat' 'output/datafile4.dat')

echo ""
echo "This script will create ${#outputfiles[@]} datafiles"
echo ""
	

# Remove if outputfile exists otherwise create
function createOrEmpty {
	if [ -e $1 ] 
	then
		rm -rf $1
		touch $1
	else
		touch $1
	fi
}

GRAPHS=($(seq 1 ${#programs[@]}))

for g in "${GRAPHS[@]}"
do
	:
	if [ $g -lt 4 ]
	then

		outputfile="${outputfiles[$g-1]}"
		program="${programs[$g-1]}"

		createOrEmpty $outputfile
	
		echo ""
		echo "---------- [STAGE 1] generating data for $outputfile ----------"
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
				resp=$(./output/out $i $FIXED_BLOCK_SIZE)
				total=$(bc <<< "scale=10; $total+$resp")
				echo "# iteration=$j T=$resp"
			done
			avg=$(bc <<< "scale=10; $total/${#TIMES[@]}")
			printf "( $i, $avg )\n" >> $outputfile
			echo "T avg. = $avg"
			if [ $i -eq $MATRIX_FOR_TABLE ]
				then
				printf "%s" $avg > output/meta_tablematrix_$g.dat 
			fi
		done
		echo "Written data to $outputfile"
	fi
done


# threads per block -----------------

GRAPHS=($(seq 4 ${#programs[@]}))
for g in "${GRAPHS[@]}"
do
	:
	outputfile="${outputfiles[$g-1]}"
	program="${programs[$g-1]}"

	createOrEmpty $outputfile
	
	echo ""
	echo "---------- [STAGE 2] generating data for $outputfile ----------"
	echo ""

	for i in "${THREADS_PER_BLOCK[@]}"
	do
		:

		TIMES=($(seq 0 $AVG_TIMES))
		total="0"
		echo "TPB=$i"
		for j in "${TIMES[@]}"
		do 
			:
			eval $program
			resp=$(./output/out $FIXED_MATRIX $i)
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

# Save metadata

printf "%s " "${SIZES[@]}" > output/meta_sizes.dat
printf "%s " "${THREADS_PER_BLOCK[@]}" > output/meta_block.dat
printf "%s " "${AVG_TIMES}" > output/meta_avg.dat
printf "%s " "${FIXED_MATRIX}" > output/meta_fixedmatrix.dat
printf "%s " "${FIXED_BLOCK_SIZE}" > output/meta_fixedblock.dat
printf "%s " "${MATRIX_FOR_TABLE}" > output/meta_tablematrixsize.dat

bash report.sh
bash report.sh



