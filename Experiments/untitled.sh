#!/bin/sh
echo "#################"
echo "Start of program"
echo "################"
mkdir mapreduce_results

python3 aprioriMapReduce.py --min_supp 2 data/itemsets5k.txt > mapreduce_results/rules.txt

echo "################################"
echo "Rules can be found in rules.txt"