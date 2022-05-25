export GPTUNEROOT=/g/g20/fish7/workspace/GPTune
export SUNDIALSBUILDROOT=/g/g20/fish7/workspace/feature-sundialsbuild

export MPI4PY=$GPTUNEROOT/mpi4py
export SCIKITOPTIMIZE=$GPTUNEROOT/scikit-optimize
export AUTOTUNE=$GPTUNEROOT/autotune

# By experimentation, found that $GPTUNROOT/GPTune is required in the $PYTHONPATH to use gptune outside of the GPTune repo.
export PYTHONPATH=$PYTHONPATH:$MPI4PY:$SCIKITOPTIMIZE:$AUTOTUNE:$GPTUNEROOT:$GPTUNEROOT/GPTune

# By experimentation, found that the default python/3.7.2 is too old. Importing newer python.
# Issue: filelock library is not found when running with python/3.7.2
module load python/3.8.2
