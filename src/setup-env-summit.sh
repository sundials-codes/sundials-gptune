export GPTUNEROOT=/ccs/home/afish/workspace/GPTune
export SUNDIALSBUILDROOT=/ccs/home/afish/workspace/feature-gptune-optimization-nonintegrated-mpi
export SUNDIALSGPTUNEROOT=/gpfs/alpine/csc326/proj-shared/afish/sundials-gptune

export SCIKITOPTIMIZE=$GPTUNEROOT/scikit-optimize
export AUTOTUNE=$GPTUNEROOT/autotune

# By experimentation, found that $GPTUNROOT/GPTune is required in the $PYTHONPATH to use gptune outside of the GPTune repo.
export PYTHONPATH=$PYTHONPATH:$SCIKITOPTIMIZE:$AUTOTUNE:$GPTUNEROOT:$GPTUNEROOT/GPTune:$SUNDIALSGPTUNEROOT/src/common

# Declare mpi execution command 
export MPIRUN=jsrun

# Load Pele-required modules
module load cmake gcc/9.1.0 cuda/11.0.3 openblas

# By experimentation, found that the default python/3.7.2 is too old. Importing newer python.
# Issue: filelock library is not found when running with python/3.7.2
module load python # module load python/3.8.2  # 3.8.2 is default on summit
