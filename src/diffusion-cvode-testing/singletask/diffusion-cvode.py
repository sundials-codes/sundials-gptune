#! /usr/bin/env python3
"""
Example of invocation of this script:
mpirun -n 1 python ./demo.py -nrun 20 -ntask 5 -perfmodel 0 -optimization GPTune
where:
	-ntask is the number of different matrix sizes that will be tuned
	-nrun is the number of calls per task
	-perfmodel is whether a coarse performance model is used
	-optimization is the optimization algorithm: GPTune,opentuner,hpbandster
"""
import sys
import os
import logging
import numpy as np
import math
import mpi4py
from mpi4py import MPI
#logging.getLogger('matplotlib.font_manager').disabled = True
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all
import argparse
import numpy as np
import time

import pygmo as pg

from callopentuner import OpenTuner
from callhpbandster import HpBandSter



# from GPTune import *

################################################################################

# Define Problem

# YL: for the spaces, the following datatypes are supported:
# Real(lower, upper, transform="normalize", name="yourname")
# Integer(lower, upper, transform="normalize", name="yourname")
# Categoricalnorm(categories, transform="onehot", name="yourname")


# Argmin{x} objectives(t,x), for x in [0., 1.]


def parse_args():

	parser = argparse.ArgumentParser()

	parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
	parser.add_argument('-cores', type=int, default=2,help='Number of cores per machine node')
	parser.add_argument('-machine', type=str,default='-1', help='Name of the computer (not hostname)')
	parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
	parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
	parser.add_argument('-order', type=int, default=3, help='Order of accuracy of the methods used')

	args = parser.parse_args()

	return args

def execute(params):
	diffusion2Dfolder = os.getenv("SUNDIALSBUILDROOT") + "/benchmarks/diffusion_2D/mpi_serial/"
	diffusion2Dexe = "cvode_diffusion_2D_mpi"
	diffusion2Dfullpath = diffusion2Dfolder + diffusion2Dexe
	
	# Build up command with command-line options from current set of parameters
	argslist = ['mpirun', '-n', str(nodes*cores), diffusion2Dfullpath, '--nx', '128', '--ny', '128',
		'--maxord', str(params["maxord"]),
		'--nlscoef', str(params["nonlin_conv_coef"]),
		'--maxncf', str(params["max_conv_fails"]),
		'--dgmax', str(params["delta_gamma_max"]),
		'--msbp', str(params["msbp"]),
		'--msbj', str(params["msbj"]),
		'--liniters', str(params['maxl']),
		'--epslin', str(params['epslin']),
	]
	if params["deduce_implicit_rhs"] == "true":
		argslist.append('--deduce')

	if params["nonlinear_solver"] == "fixedpoint":
		argslist.append('--fixedpoint')
		argslist.append(str(params['fixedpointvecs']))
		
	# Run the command and grab the output
	print("Running: " + " ".join(argslist))
	p = subprocess.run(argslist,capture_output=True)
	# Decode the stdout and stderr as they are in "bytes" format
	stdout = p.stdout.decode('ascii')
	stderr = p.stderr.decode('ascii')

	runtime = 0
	error = 0
	# If no errors occurred in the run, and the output was printed as expected, proceed
	# else, declare a failed point.
	if not stderr and stdout and "," in stdout:
		results = stdout.split(",")
		runtime = float(results[0])
		error = float(results[1])
	else:
		runtime = 1e8
		error = 1e8

	if error < 1e-15:
		runtime = 1e8
		error = 1e8

	if error > 1e-2:
		runtime = 1e8

	print(f"Finished. runtime: {runtime}, error: {error}")
	#print("done running shell command")

	return [runtime,error]

def objectives(point):
	execute_result = execute(point)
	runtime = execute_result[0]
	#error = execute_result[1]
	return runtime

def main():

	import matplotlib.pyplot as plt
	global nodes
	global cores

	# Parse command line arguments
	args = parse_args()
	nrun = args.nrun
	TUNER_NAME = args.optimization
	order = args.order

	(machine, processor, nodes, cores) = GetMachineConfiguration()
	print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
	os.environ['MACHINE_NAME'] = machine
	os.environ['TUNER_NAME'] = TUNER_NAME

	input_space = Space([Categoricalnorm(["diffusion"], transform="onehot", name="problemname")])
	
	parameter_space = Space([
		Integer(1, 5, 1.0, transform="normalize", name="maxord"),
		Real(0.0001, 1.0, transform="normalize", name="nonlin_conv_coef"),
		Integer(1, 10, transform="normalize", name="max_conv_fails"),
		Categoricalnorm(['false','true'], transform="onehot", name="deduce_implicit_rhs"),
		Real(0.0001, 1.0, transform="normalize", name="delta_gamma_max"),
		Integer(1, 100, transform="normalize", name="msbp"),
		Integer(1, 200, transform="normalize", name="msbj"),
		Integer(1, 200, transform="normalize", name="maxl"),
		Real(0.001, 1.0, transform="normalize", name="epslin"),
		Categoricalnorm(['newton', 'fixedpoint'], transform="onehot", name="nonlinear_solver"),
		Integer(1, 50, transform="normalize", name="fixedpointvecs")
	])
	constraints = {"cst1": "msbj >= msbp" }
	constants = {"nodes": nodes, "cores": cores}

	output_space = Space([
		Real(0.0,1e7, name="runtime") 
		#Real(float('-Inf'), 0.05, name="error",optimize=False)
	])
	
	problem = TuningProblem(input_space, parameter_space, output_space, objectives, constraints, None, constants=constants)

	computer = Computer(nodes=nodes, cores=cores, hosts=None)
	options = Options()
	
	options['model_restarts'] = 1

	options['distributed_memory_parallelism'] = False
	options['shared_memory_parallelism'] = False

	options['objective_evaluation_parallelism'] = False
	# options['objective_multisample_threads'] = 1
	# options['objective_multisample_processes'] = 4
	# options['objective_nprocmax'] = 1

	options['model_processes'] = 1
	# options['model_threads'] = 1
	# options['model_restart_processes'] = 1

	options['search_multitask_processes'] = 1
	options['search_multitask_threads'] = 1
	options['search_threads'] = 16

	# options['sample_algo'] = 'MCS'

	# Use the following two lines if you want to specify a certain random seed for the random pilot sampling
	options['sample_class'] = 'SampleOpenTURNS'
	options['sample_random_seed'] = 0
	# Use the following two lines if you want to specify a certain random seed for surrogate modeling
	options['model_class'] = 'Model_GPy_LCM' #'Model_LCM'
	options['model_random_seed'] = 0
	# Use the following two lines if you want to specify a certain random seed for the search phase
	options['search_class'] = 'SearchPyMoo'
	options['search_random_seed'] = 0

	# If using multiple objectives, uncomment following line 
	# options['search_algo'] = 'nsga2'

	options['verbose'] = False
	options.validate(computer=computer)

	giventask = [['diffusion']]
	NI=len(giventask) 
	NS=nrun

	TUNER_NAME = os.environ['TUNER_NAME']

	if(TUNER_NAME=='GPTune'):
		data = Data(problem)
		gt = GPTune(problem, computer=computer, data=data, options=options,driverabspath=os.path.abspath(__file__))
		(data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=int(NS/2), T_sampleflag=[True]*NI)
		# (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=NS-1)
		print("stats: ", stats)
		""" Print all input and parameter samples """
		for tid in range(NI):
			print(tid)
			print("tid: %d" % (tid))
			print("    t: " + (data.I[tid][0]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid].tolist())
			ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(data.O[tid])
			front = ndf[0]
			fopts = data.O[tid][front]
			xopts = [data.P[tid][i] for i in front]
			print('    Popt ', xopts)
			print('    Oopt ', fopts.tolist())

if __name__ == "__main__":
	main()
