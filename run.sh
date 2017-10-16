#!/bin/bash


# ----------------------------------------------
# Author - Shalitha Suranga
# -----------------------------------------------

outputfile="output/datafile.dat"

echo ""
echo "This script will create the *$outputfile*"
echo ""

if [ -e $outputfile ] 
then
	rm -rf $outputfile
	touch $outputfile
else
	touch $outputfile
fi	

SIZES=( 32 64 128 256 512 1024)

for i in "${SIZES[@]}"
do
	:
	TIMES=( 1 2 3 4 5 )
	total="0"
	echo "N=$i"
	for j in "${TIMES[@]}"
	do 
		:
		c99 Matrix.c -o output/out
		resp=$(./output/out $i)
		total=$(bc <<< "scale=10; $total+$resp")
		echo "# iteration=$j T=$resp"
	done
	avg=$(bc <<< "scale=10; $total/${#TIMES[@]}")
	printf "( $i, $avg )\n" >> $outputfile
	echo "T avg. = $avg"
done

echo ""
echo "Written to file *$outputfile*"
echo ""

./report.sh



