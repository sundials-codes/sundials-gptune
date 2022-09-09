#! /usr/bin/env python3
import argparse
from curses import meta
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
from autotune.problem import *
from autotune.search import *
from autotune.space import *
from gptune import *  # import all

sys.path.insert(1, '../common/')
import postprocess
from decision_tree import *


def parse_args():
    parser = argparse.ArgumentParser(description="Tune PeleLMeX SUNDIALS configuration with GPTune")

    parser.add_argument('tuning_spec', type=str, help='Path to tuning problem sepcification file')
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes that to run the executable with')
    parser.add_argument('-gpus', type=int, default=1, help='Number of ')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-ninitial', type=int, default=-1, help='Number of samples in the initial search phase')
    parser.add_argument('-gen_plots', action='store_true', dest='gen_plots')
    parser.add_argument('-print_csv', action='store_true', dest='print_csv')
    parser.add_argument('-max_steps', type=int, default=10, help='Max number of steps per objective function evaluation')
    parser.set_defaults(gen_plots=False)
    parser.set_defaults(print_csv=False)

    args = parser.parse_args()

    return args

def parse_error(fcompare_out):
    error = 1e8
    found_first_err = False
    if len(fcompare_out) > 0:
        finest_level_out = fcompare_out.split("level")[-1]
        finest_level_lines = finest_level_out.split("\n")
        for line in finest_level_lines:
            if len(line) > 0:
                line_list = line.split()
                if line_list[0] == "temp" or "Y(" in line_list[0]:
                    if found_first_err:
                        error = max(error,float(line_list[2]))
                    else:
                        error = float(line_list[2])
                        found_first_err = True
    return error

def get_varying_argslist(solve_type_,params):
    args_list = []
    logfilelist = []
    if solve_type_ == 'all':
        args_list += [
            'cvode.solve_type=' + str(params['solver']),
            'ode.maxl='+str(params['maxl']),
            'ode.epslin='+str(params['epslin']),
            'ode.max_fp_accel='+str(params['fixedpointvecs']),
            'ode.msbp='+str(params['msbp']),
            'ode.msbj='+str(params['msbj']),
            'ode.dgmax='+str(params['dgmax'])
        ]
        logfilelist += [
            str(params['solver']),
            str(params['maxl']),
            str(params['epslin']),
            str(params['fixedpointvecs']),
            str(params['msbp']),
            str(params['msbj']),
            str(params['dgmax'])
        ]
    elif solve_type == 'newton_iter':
        args_list += [
            'cvode.solve_type=' + str(params['solver']),
            'ode.maxl='+str(params['maxl']),
            'ode.epslin='+str(params['epslin']),
        ]
        logfilelist += [
            str(params['solver']),
            str(params['maxl']),
            str(params['epslin']),
        ]
    elif solve_type == 'newton_gmres':
        args_list += [
            'cvode.solve_type=GMRES',
            'ode.maxl='+str(params['maxl']),
            'ode.epslin='+str(params['epslin']),
        ]
        logfilelist += [
            'GMRES',
            str(params['maxl']),
            str(params['epslin']),
        ]
    elif solve_type == 'newton_bcgs':
        args_list += [
            'cvode.solve_type=BCGS',
            'ode.maxl='+str(params['maxl']),
            'ode.epslin='+str(params['epslin']),
        ]
        logfilelist += [
            'BCGS',
            str(params['maxl']),
            str(params['epslin']),
        ]
    elif solve_type == 'newton_direct':
        args_list += [
            'cvode.solve_type=' + str(params['solver']),
            'ode.msbp='+str(params['msbp']),
            'ode.msbj='+str(params['msbj']),
            'ode.dgmax='+str(params['dgmax'])
        ]
        logfilelist += [
            str(params['solver']),
            str(params['msbp']),
            str(params['msbj']),
            str(params['dgmax'])
        ]
    elif solve_type == 'newton_magma':
        args_list += [
            'cvode.solve_type=magma_direct',
            'ode.msbp='+str(params['msbp']),
            'ode.msbj='+str(params['msbj']),
            'ode.dgmax='+str(params['dgmax'])
        ]
        logfilelist += [
            'magma_direct',
            str(params['msbp']),
            str(params['msbj']),
            str(params['dgmax'])
        ]
    elif solve_type == 'newton_sparse':
        args_list += [
            'cvode.solve_type=sparse_direct',
            'ode.msbp='+str(params['msbp']),
            'ode.msbj='+str(params['msbj']),
            'ode.dgmax='+str(params['dgmax'])
        ]
        logfilelist += [
            'sparse_direct',
            str(params['msbp']),
            str(params['msbj']),
            str(params['dgmax'])
        ]
    elif solve_type == 'fixedpoint':
        args_list += [
            'cvode.solve_type=fixed_point',
            'ode.max_fp_accel='+str(params['fixedpointvecs']),
        ]
        logfilelist += [
            'fixed_point',
            str(params['fixedpointvecs']),
        ]
    else:
        print('WARNING: Did not recognize solve_type: ' + str(solve_type))
    return (args_list,logfilelist)

def execute(point):
    metadata = point["metadata"]
    tuning_problem_name = metadata["tuning_problem_name"]
    work_dir = metadata["working_directory"]

    pltfile = f'plt.{tuning_problem_name}'

    # Build up command with command-line options from current set of parameters
    argslist = []
    argslist.extend(metadata["run_command"])
    argslist.append(metadata["executable"])
    argslist.extend(metadata["inputs"])
    argslist.extend([
        f'amr.plot_file={pltfile}'
    ])
    # argslist.extend(get_varying_argslist())

    # Run the command and grab the output
    print("Running: " + " ".join(argslist), flush=True)
    p = subprocess.run(argslist, capture_output=True, cwd=work_dir)
    # Decode bytes to string
    stdout = p.stdout.decode('ascii')

    # Set default value to fallback failure value
    runtime = 1e8
    # Parse runtime from output if everything went smoothly
    if p.returncode == 0:
        # Get a list of lines
        stdoutlines = stdout.split('\n')
        # Find the line with data about the main() function
        r = re.compile("PeleLM::main()*")
        regexstringlist = list(filter(r.match, stdoutlines))
        if len(regexstringlist) == 2:
            runtimeline = regexstringlist[1] # First time this text appears is not the correct runtime
            runtime = float(runtimeline.split()[2])
    print("Runtime: " + str(runtime))

    # Ensure the error is not too big
    reffile = 'ref.' + tuning_problem_name
    fcompareexe = metadata["additional_stuff"]["fcompare_executable"]
    argslist_fcompare = [fcompareexe, reffile + '00010', pltfile + '00010']
    p = subprocess.run(argslist_fcompare, capture_output=True,cwd=work_dir)
    fcompare_out = p.stdout.decode('ascii')
    error = parse_error(fcompare_out)
    if error > 5e-2:
        runtime = 1e8

    # Log the runtime and error
    logtext = stdout + "\nruntime: " + str(runtime) + "\nerror: " + str(error)
    logdir = metadata["log_directory"]
    logfilename = f"{tuning_problem_name}.log"
    logfile = open(logdir + "/" + logfilename, 'w')
    logfile.write(logtext)
    logfile.close()

    return [runtime]

def objectives(point):
    execute_result = execute(point)
    runtime = execute_result[0]
    #error = execute_result[1]
    return [runtime]

def main():
    # Parse command line arguments
    args = parse_args()
    nodes = args.nodes
    nrun = args.nrun
    max_steps = args.max_steps
    TUNER_NAME = 'GPTune'

    meta_config_dict = json.load(open(args.tuning_spec, "r"))

    decision_tree = DecisionTree()
    decision_tree_path = [ getattr(decision_tree, node) for node in meta_config_dict["decision_tree_path"] ]
    tuning_params_list = path_params(decision_tree.graph(), decision_tree_path)
    tuning_params_hash = hashlib.sha256(str(tuning_params_list).encode('utf-8'))

    problem_name = f'{meta_config_dict["tuning_problem_base_name"]}_{tuning_params_hash.hexdigest()}'
    meta_config_dict["tuning_problem_name"] = problem_name

    print("tuning_problem_name: " + meta_config_dict["tuning_problem_name"])

    machine, processor, nodes, cores = GetMachineConfiguration(meta_dict = meta_config_dict)
    historydb = HistoryDB(meta_dict = meta_config_dict)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    input_tasks = meta_config_dict["inputs"]
    input_space = Space([
        # Integer(1, max_steps, transform='normalize', name='max_steps'),
        Categoricalnorm(meta_config_dict["inputs"] , transform="onehot", name='pelelmex-inputs')
    ])

    parameter_space = Space(tuning_params_list)
    constraints = path_constraints(decision_tree.graph(), decision_tree_path)

    output_space = Space([
      Real(0.0, 1e7, name="runtime")
    ])

    constants = { "metadata": meta_config_dict }

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

    giventask = [input_tasks]
    NI = len(giventask)
    NS = nrun
    NS1 = int(NS/2)
    if args.ninitial != -1:
        NS1 = args.ninitial

    data = Data(problem)
    gt = GPTune(problem, computer=computer, data=data, historydb=historydb, options=options,driverabspath=os.path.abspath(__file__))
    (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=NS1, T_sampleflag=[True]*NI)
    print("stats: ", stats)
    """ Print all input and parameter samples """
    for tid in range(NI):
        print(tid)
        print("tid: %d" % (tid))
        print("    t: " + (data.I[tid][0]))
        print("    Ps ", data.P[tid])
        print("    Os ", data.O[tid].tolist())
        print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

        if args.print_csv:
            outfile = open(problem_name + ".csv", "w")
            for i in range(len(data.P[tid])):
                outlinelist = list(data.P[tid][i]) + list(data.O[tid][i])
                outlineliststr = [str(x) for x in outlinelist]
                outline = ",".join(outlineliststr) + "\n"
                outfile.write(outline)

            outfile.close()

        if args.sensitivity_analysis:
            json_filename = './gptune.db/' + problem_name + '.json'
            json_file = open(json_filename)
            json_data = json.load(json_file)

            function_evaluations = json_data['func_eval']
            problem_space = {
                "parameter_space": json_data['surrogate_model'][-1]['parameter_space'],
                "input_space": json_data['surrogate_model'][-1]['input_space'],
                "output_space": json_data['surrogate_model'][-1]['output_space']
            }

            sensitivity_data = SensitivityAnalysis(problem_space=problem_space, input_task=input_tasks, function_evaluations=function_evaluations, num_samples=1024)
            print(sensitivity_data)
            print("S1")
            print(sensitivity_data["S1"])

            json_file.close()

        """
        if args.gen_plots:
            problem_task_name = problem_name + '-' + data.I[tid][0]
            runtimes = [ elem[0] for elem in data.O[tid].tolist() ]
            postprocess.plot_runtime(runtimes,problem_task_name,1e8)
            param_datas = [
                { 'name': 'max_ord', 'type': 'integer', 'values': [ elem[0] for elem in data.P[tid] ] },
                { 'name': 'nonlin_conv_coef', 'type': 'real', 'values': [ elem[1] for elem in data.P[tid] ] },
                { 'name': 'max_conv_fails', 'type': 'integer', 'values': [ elem[2] for elem in data.P[tid] ] }
            ]
            if solve_type == 'newton_gmres' or solve_type == 'newton_bcgs' or solve_type == 'newton_iter':
                param_datas += [
                    { 'name': 'maxl', 'type': 'integer', 'values': [ elem[3] for elem in data.P[tid] ] },
                    { 'name': 'epslin', 'type': 'real', 'values': [ elem[4] for elem in data.P[tid] ] },
                ]
            elif solve_type == 'fixedpoint':
                param_datas += [
                    { 'name': 'fixedpointvecs', 'type': 'integer', 'values': [ elem[3] for elem in data.P[tid] ] },
                ]
            elif solve_type == 'newton_direct':
                param_datas += [
                    { 'name': 'msbp', 'type': 'integer', 'values': [ elem[3] for elem in data.P[tid] ] },
                    { 'name': 'msbj', 'type': 'integer', 'values': [ elem[4] for elem in data.P[tid] ] },
                    { 'name': 'dgmax', 'type': 'real', 'values': [ elem[5] for elem in data.P[tid] ] }
                ]
            elif solve_type == 'newton_all':
                param_datas += [
                    { 'name': 'maxl', 'type': 'integer', 'values': [ elem[3] for elem in data.P[tid] ] },
                    { 'name': 'epslin', 'type': 'real', 'values': [ elem[4] for elem in data.P[tid] ] },
                    { 'name': 'msbp', 'type': 'integer', 'values': [ elem[5] for elem in data.P[tid] ] },
                    { 'name': 'msbj', 'type': 'integer', 'values': [ elem[6] for elem in data.P[tid] ] },
                    { 'name': 'dgmax', 'type': 'real', 'values': [ elem[7] for elem in data.P[tid] ] }
                ]

            if additional_params:
                start_index = 4
                if solve_type == 'newton_gmres' or solve_type == 'newton_bcgs' or solve_type == 'newton_iter':
                    start_index += 1
                if solve_type == 'newton_direct':
                    start_index += 2
                if solve_type == 'newton_all':
                    start_index += 4
                param_datas += [
                    { 'name': 'eta_cf', 'type': 'real', 'values': [ elem[start_index] for elem in data.P[tid] ] },
                    { 'name': 'eta_max_fx', 'type': 'real', 'values': [ elem[start_index+1] for elem in data.P[tid] ] },
                    { 'name': 'eta_min_fx', 'type': 'real', 'values': [ elem[start_index+2] for elem in data.P[tid] ] },
                    { 'name': 'eta_max_gs', 'type': 'real', 'values': [ elem[start_index+3] for elem in data.P[tid] ] },
                    { 'name': 'eta_min', 'type': 'real', 'values': [ elem[start_index+4] for elem in data.P[tid] ] },
                    { 'name': 'eta_min_ef', 'type': 'real', 'values': [ elem[start_index+5] for elem in data.P[tid] ] }
                ]
            postprocess.plot_params(param_datas,problem_name)
            postprocess.plot_params_with_fails(runtimes,param_datas,problem_task_name,1e8)
            postprocess.plot_params_vs_runtime(runtimes,param_datas,problem_task_name,1e8)
            postprocess.plot_cat_bool_param_freq_period(param_datas,problem_task_name,4)
            #postprocess.plot_real_int_param_std_period(param_datas,problem_task_name,4)
            postprocess.plot_real_int_param_std_window(param_datas,problem_task_name,10)
        """
if __name__ == "__main__":
    main()
