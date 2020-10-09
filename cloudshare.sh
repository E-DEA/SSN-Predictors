#!/usr/bin/env bash

savefolder="$PWD/saved_outputs/FFNN(with normalized input) - Upto Cycle 24"

log="$PWD/logs/*"
model="$PWD/models/*10000.pth"
lgraph="$PWD/graphs/loss/*"
sgraph="$PWD/graphs/ssn/*"

currcount=`ls "$savefolder" | wc -l`
currcount=$(($currcount + 1))

newfolder="$savefolder/$currcount"

mkdir "$newfolder"
mv $log $model $lgraph $sgraph "$newfolder"

rm $PWD/models/*
