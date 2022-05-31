import sys
import os
import subprocess
# import mpi4py
import logging
import numpy as np
import mpi4py
from mpi4py import MPI

params = { "order": 3, "controller_id": 0, "atol": 0.011, "rtol": 0.01, "targetlog10err": 0.01 }
sundialsfeaturenontinetegratedroot = "/g/g20/fish7/workspace/feature-gptune-optimization-nonintegrated-mpi"

diffusion2Dfolder = sundialsfeaturenontinetegratedroot + "/benchmarks/diffusion_2D/mpi_serial/"
diffusion2Dexe = "arkode_diffusion_2D_mpi"
diffusion2Dfullpath = diffusion2Dfolder + diffusion2Dexe

order = params["order"]
controller_id = params["controller_id"]
atol = params["atol"]
rtol = params["rtol"]

nodes=1
cores=36

#argslist = ['mpiexec', '-n', str(nodes), '-npernode', str(cores), diffusion2Dfullpath, '--order', str(order), '--controller', str(controller_id), '--atol', str(atol), '--rtol', str(rtol)]
#argslist = ['mpiexec', '-n', str(nodes*cores), diffusion2Dfullpath, '--order', str(order), '--controller', str(controller_id), '--atol', str(atol), '--rtol', str(rtol)]
argslist = ['mpiexec', '-n', str(nodes*cores), diffusion2Dfullpath, '--order', str(order), '--controller', str(controller_id), '--atol', str(atol), '--rtol', str(rtol), '--nx', '128', '--ny', '128']

print(argslist)
print(diffusion2Dfullpath)
print(" ".join(argslist) + " with tol: " + str(params["targetlog10err"]))
print("nodes: " + str(nodes) + ", cores: " + str(cores))

print("in execute, done with initialization. running mpi now")
print("running shell command")
p = subprocess.run(argslist,capture_output=True)
result = p.stdout
print(result)
print(result.decode('ascii'))
print(result.decode('utf-8'))
resultlist = result.decode('ascii').split(",")
print(resultlist)
runtime = float(resultlist[0])
error = float(resultlist[1])
print("runtime:")
print(runtime)
print("error:")
print(error)
