#!/bin/bash

declare -a lines=("MgII2796" "CII1334" "SiIII1206" "CIV1548" "OVI1031")
declare -a predictors=("delta_rho" "T" "Z")

for l in "${lines[@]}"
do
	for p in "${predictors[@]}"
	do
		echo Starting line $l predictor $p
		python random_forest.py m100n1024 s50 151 $l $p
	done
done
