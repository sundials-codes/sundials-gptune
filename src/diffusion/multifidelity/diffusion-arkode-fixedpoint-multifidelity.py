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
import matplotlib.pyplot as plt

#import pygmo as pg

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
    parser.add_argument('-plot_runtime', action='store_true', dest='plot_runtime')
    parser.add_argument('-plot_params', action='store_true', dest='plot_params')
    parser.set_defaults(plot_runtime=False)
    parser.set_defaults(plot_params=False)

    args = parser.parse_args()

    return args

def get_rtol_from_budget(budget):
    if budget == 1:
        return 1e-2
    elif budget == 3:
        return 1e-3
    elif budget == 9:
        return 1e-4

def execute(params):
    diffusion2Dfolder = os.getenv("SUNDIALSBUILDROOT") + "/benchmarks/diffusion_2D/mpi_serial/"
    diffusion2Dexe = "arkode_diffusion_2D_mpi"
    diffusion2Dfullpath = diffusion2Dfolder + diffusion2Dexe

    mpirun_command = os.getenv("MPIRUN")
    
    # Build up command with command-line options from current set of parameters
    rtol = get_rtol_from_budget(params['budget'])
    argslist = [mpirun_command, '-n', str(nodes*cores), diffusion2Dfullpath, '--nx', '128', '--ny', '128',
        '--rtol', str(rtol),
        '--controller', str(params["controller_id"]),
        '--method', str(params["method"]),
        '--nlscoef', str(params["nonlin_conv_coef"]),
        '--maxncf', str(params["max_conv_fails"]),
        '--fixedpoint', str(params["fixedpointvecs"])
    ]
    if params["deduce_implicit_rhs"] == "true":
        argslist.append('--deduce')

    # Run the command and grab the output
    print("Running: " + " ".join(argslist),flush=True)
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

    if error > 5e2*rtol:
        runtime = 1e8

    print(f"Finished. runtime: {runtime}, error: {error}",flush=True)
    #print("done running shell command")

    return [runtime,error]

def objectives(point):
    execute_result = execute(point)
    runtime = execute_result[0]
    #error = execute_result[1]
    return [runtime]

def get_methods(order):
    # https://github.com/LLNL/sundials/blob/develop/include/arkode/arkode_butcher_dirk.h
    if order == 2:
        return ["ARKODE_SDIRK_2_1_2"]
    elif order == 3:
        return ["ARKODE_BILLINGTON_3_3_2", "ARKODE_TRBDF2_3_3_2", "ARKODE_KVAERNO_4_2_3", "ARKODE_ARK324L2SA_DIRK_4_2_3", "ARKODE_ESDIRK324L2SA_4_2_3", "ARKODE_ESDIRK325L2SA_5_2_3", "ARKODE_ESDIRK32I5L2SA_5_2_3"]
    elif order == 4:
        return ["ARKODE_CASH_5_2_4", "ARKODE_CASH_5_3_4", "ARKODE_SDIRK_5_3_4", "ARKODE_KVAERNO_5_3_4", "ARKODE_ARK436L2SA_DIRK_6_3_4", "ARKODE_ARK437L2SA_DIRK_7_3_4", "ARKODE_ESDIRK436L2SA_6_3_4", "ARKODE_ESDIRK43I6L2SA_6_3_4", "ARKODE_QESDIRK436L2SA_6_3_4", "ARKODE_ESDIRK437L2SA_7_3_4"]
    elif order == 5:
        return ["ARKODE_KVAERNO_7_4_5", "ARKODE_ARK548L2SA_DIRK_8_4_5", "ARKODE_ARK548L2SA_DIRK_8_4_5", "ARKODE_ESDIRK547L2SA_7_4_5", "ARKODE_ESDIRK547L2SA2_7_4_5"]
    elif order == -1:
        return ["ARKODE_SDIRK_2_1_2", "ARKODE_BILLINGTON_3_3_2", "ARKODE_TRBDF2_3_3_2", "ARKODE_KVAERNO_4_2_3", "ARKODE_ARK324L2SA_DIRK_4_2_3", "ARKODE_ESDIRK324L2SA_4_2_3", "ARKODE_ESDIRK325L2SA_5_2_3", "ARKODE_ESDIRK32I5L2SA_5_2_3", "ARKODE_CASH_5_2_4", "ARKODE_CASH_5_3_4", "ARKODE_SDIRK_5_3_4", "ARKODE_KVAERNO_5_3_4", "ARKODE_ARK436L2SA_DIRK_6_3_4", "ARKODE_ARK437L2SA_DIRK_7_3_4", "ARKODE_ESDIRK436L2SA_6_3_4", "ARKODE_ESDIRK43I6L2SA_6_3_4", "ARK    ODE_QESDIRK436L2SA_6_3_4", "ARKODE_ESDIRK437L2SA_7_3_4", "ARKODE_KVAERNO_7_4_5", "ARKODE_ARK548L2SA_DIRK_8_4_5", "ARKODE_ARK548L2SA_DIRK_8_4_5", "ARKODE_ESDIRK547L2SA_7_4_5", "ARKODE_ESDIRK547L2SA2_7_4_5"] 
    else:
        return []

def main():

    global nodes
    global cores

    # Parse command line arguments
    args = parse_args()
    nrun = args.nrun
    plot_runtime = args.plot_runtime
    plot_params = args.plot_params
    TUNER_NAME = 'GPTune'
    order = args.order

    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    input_space = Space([Categoricalnorm(["diffusion-arkode-fixedpoint-multifidelity"], transform="onehot", name="problemname")])
    
    methods = get_methods(order)
    parameter_space = Space([
        Categoricalnorm(['0','1','2','3','4','5'], transform="onehot", name="controller_id"),
        Categoricalnorm(methods, transform="onehot", name="method"),
        Real(1e-5, 0.9, transform="normalize", name="nonlin_conv_coef"),
        Integer(3, 50, transform="normalize", name="max_conv_fails"),
        Categoricalnorm(['false','true'], transform="onehot", name="deduce_implicit_rhs"),
        Integer(1, 20, transform="normalize", name="fixedpointvecs")
    ])
    constraints = {}
    constants = {"nodes": nodes, "cores": cores}

    output_space = Space([
        Real(0.0,1e7, name="runtime") 
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
    options['sample_class'] = 'SampleLHSMDU'
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

    # Multifidelity options
    options['budget_min'] = 1
    options['budget_max'] = 9
    options['budget_base'] = 3

    giventask = [['diffusion-arkode-fixedpoint-multifidelity']]
    NI=len(giventask) 
    NS=nrun


    if(TUNER_NAME=='GPTune'):
        gt = GPTune_MB(problem, computer=computer, options=options)
        (data, modeler, stats) = gt.MB_LCM(NS=NS, Igiven=giventask, NI=NI, NS1=int(NS/2), Pdefault=None)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print(tid)
            print("tid: %d" % (tid))
            print("    t: " + (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
            #ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(data.O[tid])
            #front = ndf[0]
            #fopts = data.O[tid][front]
            #xopts = [data.P[tid][i] for i in front]
            #print('    Popt ', xopts)
            #print('    Oopt ', fopts.tolist())

            if plot_runtime:
                runtimes = [ elem[0] for elem in data.O[tid].tolist() ]
                runtimes = list(filter(lambda x: x != 1e8, runtimes))
                plt.plot(runtimes)
                plt.title('Runtime vs Sample Number, with failed Samples removed')
                plt.xlabel('Filtered Sample Number')
                plt.ylabel('Runtime (s)')
                plt.savefig('diffusion-arkode-fixedpoint-runtime.png')
                plt.close()
            
            if plot_params:
                plot_datas = [
                    { 'name': 'nonlin_conv_coef', 'values': [ elem[2] for elem in data.P[tid] ] },
                    { 'name': 'max_conv_fails', 'values': [ elem[3] for elem in data.P[tid] ] },
                    { 'name': 'deduce_implicit_rhs', 'values': [ int(elem[4] == 'true') for elem in data.P[tid] ] },
                    { 'name': 'fixedpointvecs', 'values': [ elem[5] for elem in data.P[tid] ] }
                ]
                for plot_data in plot_datas:
                    plt.plot(plot_data['values'])
                    plt.title(plot_data['name'] + ' vs Sample Number')
                    plt.xlabel('Sample Number')
                    plt.ylabel(plot_data['name'])
                    plt.savefig('diffusion-arkode-fixedpoint-' + plot_data['name'] + '.png')
                    plt.close()

if __name__ == "__main__":
    main()
