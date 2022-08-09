#!/bin/bash
# Run this from in the /gpfs/alpine/csc326/proj-shared/afish/sundials-gptune/src directory

cd ./diffusion-paper/standard
bsub diffusion-cvode-10-128-all-additional.lsf
bsub diffusion-cvode-10-128-newton_iter.lsf
bsub diffusion-cvode-10-128-fixedpoint-additional.lsf

cd -

cd ./pele-Flamesheet
bsub pele-cvode-dodecane_lu-newton-gmres-additional.lsf
bsub pele-cvode-dodecane_lu_qss-newton-gmres-additional.lsf
#bsub pele-cvode-drm19-newton-gmres-additional.lsf #already finished

cd -
