#!/bin/bash

nnodes=$1
ncorespernode=$2
arguments="${@:3}"

mpirun -n $nnodes -npernode $ncorespernode python3 $arguments
