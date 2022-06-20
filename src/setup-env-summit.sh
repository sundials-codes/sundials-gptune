export GPTUNEROOT=/ccs/home/afish/workspace/GPTune
export SUNDIALSBUILDROOT=/ccs/home/afish/workspace/feature-gptune-optimization-nonintegrated-mpi

export SCIKITOPTIMIZE=$GPTUNEROOT/scikit-optimize
export AUTOTUNE=$GPTUNEROOT/autotune

# By experimentation, found that $GPTUNROOT/GPTune is required in the $PYTHONPATH to use gptune outside of the GPTune repo.
export PYTHONPATH=$PYTHONPATH:$SCIKITOPTIMIZE:$AUTOTUNE:$GPTUNEROOT:$GPTUNEROOT/GPTune

# Declare mpi execution command 
export MPIRUN=jsrun

# GPTune requires OpenMPI built with the same compiler as the Lapack/Blas/Scalapack libraries
# module load gcc/8.3.1 # default on summit
# module load openmpi-4.1.4-gcc-8.3.1-a2mevfu # ensure installed locally through spack

# By experimentation, found that the default python/3.7.2 is too old. Importing newer python.
# Issue: filelock library is not found when running with python/3.7.2
module load python # module load python/3.8.2  # 3.8.2 is default on summit
