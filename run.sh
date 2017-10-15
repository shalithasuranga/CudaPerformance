#!/bin/bash


# ----------------------------------------------
# Author - Shalitha Suranga
# -----------------------------------------------


echo ""
echo "This script will create the *datafile.txt*"
echo ""

if [ -e datafile.txt ] 
then
	rm -rf datafile.txt
	touch datafile.txt
else
	touch datafile.txt
fi	

SIZES=( 64 128 256 )

for i in "${SIZES[@]}"
do
	:
	TIMES=( 1 2 )
	total="0"
	for j in "${TIMES[@]}"
	do 
		:
		echo "Running for N=$i x $i"
		c99 Matrix.c -o out
		resp=$(./out $i)
		total=$(bc <<< "scale=10; $total+$resp")
		echo "$resp"
	done
	echo "total = $total"
	printf "$i $total\n" >> datafile.txt
done

