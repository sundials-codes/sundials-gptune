#!/bin/bash
# coding=utf-8

nnodes=$1
ncorespernode=$2
arguments="${@:3}"

echo ${nnodes}
echo ${ncorespernode}
echo ${arguments}

echo mpirun -n $nnodes -npernode $ncorespernode $arguments
#mpirun -n "$(nnodes)" -npernode "$(ncorespernode)" python3 "$(arguments)"
