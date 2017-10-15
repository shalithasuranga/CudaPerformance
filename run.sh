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

SIZES=( 64 128 256 512 )

for i in "${SIZES[@]}"
do
	:
	TIMES=( 1 2 3 4 5 )
	total="0"
	echo "N=$i"
	for j in "${TIMES[@]}"
	do 
		:
		c99 Matrix.c -o out
		resp=$(./out $i)
		total=$(bc <<< "scale=10; $total+$resp")
		echo "# iteration=$j T=$resp"
	done
	avg=$(bc <<< "scale=10; $total/${#TIMES[@]}")
	printf "$i $avg\n" >> datafile.txt
	echo "T avg. = $avg"
done

echo ""
echo "Written to file *datafile.txt*"

