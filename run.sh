#!/bin/bash

RPERC=0.1

for i in `seq 1 9`
do
	for j in 200 300 600 
	do
		./exec $RPERC $j  
	done > outputs/$i-$j
done
