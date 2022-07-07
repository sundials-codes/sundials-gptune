#! /usr/bin/env python3
import sys
import os
import logging
import numpy as np
import math
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all
import argparse
import postprocess

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=2,help='Number of cores per machine node')
    parser.add_argument('-machine', type=str,default='-1', help='Name of the computer (not hostname)')
    parser.add_argument('-nloop', type=int, default=4, help='Number of loops of the budgets')
    parser.add_argument('-gen_plots', action='store_true', dest='gen_plots')
    parser.add_argument('-additional_params', action='store_true', dest='additional_params')
    parser.add_argument('-newton_gmres', action='store_true', dest='newton_gmres')
    #parser.add_argument('-multifidelity', type=str, default='-1', help='Turn on multifidelity. Value template: low,high,multiplicativefactor')
    parser.set_defaults(gen_plots=False)
    parser.set_defaults(additional_params=False)
    parser.set_defaults(newton_gmres=False)

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
    diffusion2Dexe = "cvode_diffusion_2D_mpi"
    diffusion2Dfullpath = diffusion2Dfolder + diffusion2Dexe
    mpirun_command = os.getenv("MPIRUN")
    
    rtol = get_rtol_from_budget(params['budget'])
    # Build up command with command-line options from current set of parameters
    argslist = [mpirun_command, '-n', str(nodes*cores), diffusion2Dfullpath, '--nx', '128', '--ny', '128',
            '--rtol', str(rtol),
            '--maxord', str(params["maxord"]),
            '--nlscoef', str(params["nonlin_conv_coef"]),
            '--maxncf', str(params["max_conv_fails"])
    ]

    if newton_gmres:
        newton_gmres_args = [
        '--gmres',
        '--liniters', str(params['maxl']),
        '--epslin', str(params['epslin']),
        ]
        argslist += newton_gmres_args
    else:
        fixedpoint_args = [
        '--fixedpoint', str(params['fixedpointvecs'])
        ]
        argslist += fixedpoint_args

    if additional_params:
        additional_params_args = [
        '--eta_cf', str(params['eta_cf']),
        '--eta_max_fx', str(params['eta_max_fx']),
        '--eta_min_fx', str(params['eta_min_fx']),
        '--eta_max_gs', str(params['eta_max_gs']),
        '--eta_min', str(params['eta_min']),
        '--eta_min_ef', str(params['eta_min_ef'])
        ]
        argslist += additional_params

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

def main():
    global nodes
    global cores
    global newton_gmres
    global additional_params

    # Parse command line arguments
    args = parse_args()
    nloop = args.nloop
    newton_gmres = args.newton_gmres
    additional_params = args.additional_params
    problem_name = 'diffusion-cvode-multifidelity'
    TUNER_NAME = 'GPTune'

    if newton_gmres:
        problem_name += '-newton-gmres'
    else:
        problem_name += '-fixedpoint'
    
    if additional_params:
        problem_name += '-additional'

    print("problem_name: " + problem_name)
    meta_config_dict = { 
        'tuning_problem_name': problem_name,
        'machine_configuration': {
            'machine_name': 'mymachine',
            'myprocessor': {
                'nodes': 1,
                'cores': 40
            }
        },
        'software_configuration': {},
        'loadable_machine_configurations': {
            'mymachine': {
                'myprocessor': {
                    'nodes': 1,
                    'cores': 40
                }
            }
        },
        'loadable_software_configurations': {}
    }

    (machine, processor, nodes, cores) = GetMachineConfiguration(meta_dict = meta_config_dict)
    historydb = HistoryDB(meta_dict = meta_config_dict)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    input_space = Space([Categoricalnorm([problem_name], transform="onehot", name="problemname")])

    parameter_space_list = [
        Integer(1, 5, transform="normalize", name="maxord"),
        Real(1e-5, 0.9, transform="normalize", name="nonlin_conv_coef"),
        Integer(3, 50, transform="normalize", name="max_conv_fails")
    ]
    constraints = {}
    if newton_gmres:
        parameter_space_list += [
            Integer(3, 500, transform="normalize", name="maxl"),
            Real(1e-5, 0.9, transform="normalize", name="epslin")
        ]
    else:
        parameter_space_list += [ 
            Integer(1, 20, transform="normalize", name="fixedpointvecs")
        ]

    if additional_params:
        parameter_space_list += [ 
            Real(1e-5, 0.9, transform="normalize", name="eta_cf"),
            Real(1e-5, 20, transform="normalize", name="eta_max_fx"),
            Real(1e-5, 20, transform="normalize", name="eta_min_fx"),
            Real(1e-2, 20, transform="normalize", name="eta_max_gs"),
            Real(1e-2, 1, transform="normalize", name="eta_min"),
            Real(1e-5, 0.9, transform="normalize", name="eta_min_ef")
        ]
        constraints['cst1'] = 'eta_max_fx > eta_min_fx'

    parameter_space = Space(parameter_space_list)
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
    options['sample_class'] = 'SampleLHSMDU'  # 'SampleOpenTURNS'
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

    giventask = [[problem_name]]
    NI=len(giventask) 
    NS=nloop

    data = Data(problem)
    gt = GPTune_MB(problem, computer=computer, historydb=historydb, options=options)
    (data, modeler, stats) = gt.MB_LCM(NS=NS, Igiven=giventask)
    print("stats: ", stats)
    """ Print all input and parameter samples """
    for tid in range(NI):
        print("tid: %d" % (tid))
        print(f"   t = {data.I[tid][0]}")
        print("    Ps ", data.P[tid])
        print("    Os ", data.O[tid].tolist())
        nth = np.argmin(data.O[tid])
        Popt = data.P[tid][nth]

        if args.gen_plots:
            print("GENERATING PLOTS")
            runtimes = [ elem[0] for elem in data.O[tid].tolist() ]
            postprocess.plot_runtime(runtimes,problem_name,1e8)
            param_datas = [
                { 'name': 'max_ord', 'type': 'integer', 'values': [ elem[0] for elem in data.P[tid] ] },
                { 'name': 'nonlin_conv_coef', 'type': 'real', 'values': [ elem[1] for elem in data.P[tid] ] },
                { 'name': 'max_conv_fails', 'type': 'integer', 'values': [ elem[2] for elem in data.P[tid] ] }
            ]
            if newton_gmres:
                param_datas += [
                    { 'name': 'maxl', 'type': 'integer', 'values': [ elem[3] for elem in data.P[tid] ] },
                    { 'name': 'epslin', 'type': 'real', 'values': [ elem[4] for elem in data.P[tid] ] },
                ]
            else:
                param_datas += [
                    { 'name': 'fixedpointvecs', 'type': 'integer', 'values': [ elem[3] for elem in data.P[tid] ] },
                ]

            if additional_params:
                start_index = 4
                if newton_gmres:
                    start_index += 1
                param_datas += [
                    { 'name': 'eta_cf', 'type': 'real', 'values': [ elem[start_index] for elem in data.P[tid] ] },
                    { 'name': 'eta_max_fx', 'type': 'real', 'values': [ elem[start_index+1] for elem in data.P[tid] ] },
                    { 'name': 'eta_min_fx', 'type': 'real', 'values': [ elem[start_index+2] for elem in data.P[tid] ] },
                    { 'name': 'eta_max_gs', 'type': 'real', 'values': [ elem[start_index+3] for elem in data.P[tid] ] },
                    { 'name': 'eta_min', 'type': 'real', 'values': [ elem[start_index+4] for elem in data.P[tid] ] },
                    { 'name': 'eta_min_ef', 'type': 'real', 'values': [ elem[start_index+5] for elem in data.P[tid] ] }
                ]
            postprocess.plot_params(param_datas,problem_name)
            postprocess.plot_params_with_fails(runtimes,param_datas,problem_name,1e8)
            postprocess.plot_params_vs_runtime(runtimes,param_datas,problem_name,1e8)
            postprocess.plot_cat_bool_param_freq_period(param_datas,problem_name,4)
            #postprocess.plot_real_int_param_std_period(param_datas,problem_name,4)
            postprocess.plot_real_int_param_std_window(param_datas,problem_name,10) 

        # find which arm and which sample the optimal param is from
        for arm in range(len(data_hist.P)):
            try:
                idx = (data_hist.P[arm]).index(Popt)
                arm_opt = arm
            except ValueError:
                pass
        print('    Popt ', Popt, 'Oopt ', min(data.O[tid])[0], 'nth ', nth, 'nth-bandit (s, nth) = ', (arm_opt, idx))


if __name__ == "__main__":
    main()
