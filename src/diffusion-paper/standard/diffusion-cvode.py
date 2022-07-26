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
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-kxy', type=int, default=10, help='Diffusion coefficient')
    parser.add_argument('-nxy', type=int, default=128, help='Number of points in each direction')
    parser.add_argument('-gen_plots', action='store_true', dest='gen_plots')
    parser.add_argument('-additional_params', action='store_true', dest='additional_params')
    parser.add_argument('-solve_type', type=str, default='newton_gmres', help='Solver type(s) to optimize. Ex: all/fixedpoint/newton/newton_iter/newton_gmres/newton_pcg')
    parser.add_argument('-print_csv', action='store_true', dest='print_csv')
    parser.add_argument('-sensitivity_analysis', action='store_true', dest='sensitivity_analysis')
    parser.set_defaults(gen_plots=False)
    parser.set_defaults(additional_params=False)
    parser.set_defaults(print_csv=False)

    args = parser.parse_args()

    return args

def get_args(solve_type_, params):
    argslist = []
    logfilelist = []
    if solve_type_ == 'newton_gmres':
        newton_gmres_args = [
        '--gmres',
        '--liniters', str(params['maxl']),
        '--epslin', str(params['epslin']),
        ]
        argslist += newton_gmres_args
        logfilelist += ['gmres',str(params['maxl']),str(params['epslin'])]
    elif solve_type_ == 'newton_pcg':
        newton_pcg_args = [
        '--liniters', str(params['maxl']),
        '--epslin', str(params['epslin']),
        ]
        argslist += newton_pcg_args
        logfilelist += [str(params['maxl']),str(params['epslin'])]
    elif solve_type_ == 'fixedpoint':
        fixedpoint_args = [
        '--fixedpoint', str(params['fixedpointvecs'])
        ]
        argslist += fixedpoint_args
        logfilelist += [str(params['fixedpointvecs'])]
    return (argslist, logfilelist)

def execute(params):
    diffusion2Dfolder = os.getenv("SUNDIALSBUILDROOT") + "/benchmarks/diffusion_2D/mpi_serial/"
    diffusion2Dexe = "cvode_diffusion_2D_mpi"
    diffusion2Dfullpath = diffusion2Dfolder + diffusion2Dexe
    mpirun_command = os.getenv("MPIRUN")
    logfolder = "log"
    logfilelist = [problem_name,solve_type,str(params['maxord']),str(params['nonlin_conv_coef']),str(params['max_conv_fails'])]
    
    # Build up command with command-line options from current set of parameters
    argslist = [mpirun_command, '-n', str(nodes*cores), diffusion2Dfullpath, '--stats', '--kx', str(params['kxy']), '--ky', str(params['kxy']), '--nx', str(params['nxy']), '--ny', str(params['nxy']),
            '--maxord', str(params["maxord"]),
            '--nlscoef', str(params["nonlin_conv_coef"]),
            '--maxncf', str(params["max_conv_fails"])
    ]

    if solve_type == 'newton_iter' or solve_type == 'all':
        (argslist_,logfilelist_) = get_args(params['solver'], params)
        argslist += argslist_
        logfilelist += logfilelist_
    else:
        (argslist_,logfilelist_) = get_args(solve_type, params)
        argslist += argslist_
        logfilelist += logfilelist_

    if additional_params:
        additional_params_args = [
        '--eta_cf', str(params['eta_cf']),
        '--eta_max_fx', str(params['eta_max_fx']),
        '--eta_min_fx', str(params['eta_min_fx']),
        '--eta_max_gs', str(params['eta_max_gs']),
        '--eta_min', str(params['eta_min']),
        '--eta_min_ef', str(params['eta_min_ef'])
        ]
        argslist += additional_params_args
        logfilelist += ['additional',str(params['eta_cf']),str(params['eta_max_fx']),str(params['eta_min_fx']),str(params['eta_max_gs']),str(params['eta_min']),str(params['eta_min_ef'])]

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
        resultline = stdout.split("\n")[-2]
        results = resultline.split(",")
        runtime = float(results[0])
        error = float(results[1])
    else:
        runtime = 1e8
        error = 1e8
    
    if error < 1e-15:
        runtime = 1e8
        error = 1e8
    
    if error > 5e-3:
        runtime = 1e8
    
    print(f"Finished. runtime: {runtime}, error: {error}",flush=True)
    #print("done running shell command")
    
    logtext = stdout + "\nruntime: " + str(runtime) + "\nerror: " + str(error)
    logfilename = "_".join(logfilelist) + ".log"
    logfullpath = logfolder + "/" + logfilename
    logfile = open(logfullpath, 'w')
    logfile.write(logtext)
    logfile.close()
     
    return [runtime,error]

def objectives(point):
    execute_result = execute(point)
    runtime = execute_result[0]
    #error = execute_result[1]
    return [runtime]

def main():
    global nodes
    global cores
    global additional_params
    global problem_name
    global solve_type

    # Parse command line arguments
    args = parse_args()
    nrun = args.nrun
    solve_type = args.solve_type
    additional_params = args.additional_params
    kxy = args.kxy
    nxy = args.nxy
   
    problem_name = 'diffusion-cvode-' + solve_type
    TUNER_NAME = 'GPTune'

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

    input_space = Space([
        Integer(1, 5000, transform="normalize", name="kxy"),
        Integer(1, 5000, transform="normalize", name="nxy") 
    ])

    parameter_space_list = [
        Integer(1, 5, transform="normalize", name="maxord"),
        Real(1e-5, 0.9, transform="normalize", name="nonlin_conv_coef"),
        Integer(3, 50, transform="normalize", name="max_conv_fails")
    ]
    constraints = {}

    if solve_type == 'all':
        parameter_space_list += [
            Categoricalnorm(['newton_gmres','newton_pcg','fixedpoint'],transform='onehot',name='solver'),
            Integer(3, 500, transform="normalize", name="maxl"),
            Real(1e-5, 0.9, transform="normalize", name="epslin"),
            Integer(1, 20, transform="normalize", name="fixedpointvecs")
        ]
    elif solve_type == 'newton_iter' or solve_type == 'newton_gmres' or solve_type == 'newton_pcg':
        parameter_space_list += [
            Categoricalnorm(['newton_gmres','newton_pcg'],transform='onehot',name='solver'),
            Integer(3, 500, transform="normalize", name="maxl"),
            Real(1e-5, 0.9, transform="normalize", name="epslin"),
        ]
    elif solve_type == 'fixedpoint':
        parameter_space_list += [ 
            Integer(1, 20, transform="normalize", name="fixedpointvecs")
        ]
    else:
        print('WARNING: Did not recognize solve_type: ' + str(solve_type))

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

    giventask = [[kxy,nxy]]
    NI=len(giventask) 
    NS=nrun

    data = Data(problem)
    gt = GPTune(problem, computer=computer, data=data, historydb=historydb, options=options,driverabspath=os.path.abspath(__file__))
    (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=int(NS/2), T_sampleflag=[True]*NI)
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
            outfile = open(problem_name + ".csv","w")
            headerlinelist = []
            for param in problem.parameter_space:
                headerlinelist.append(param.name)
            headerlinelist.append('runtime')
            headerline = ",".join(headerlinelist) + "\n"
            outfile.write(headerline)
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
            
            sensitivity_data = SensitivityAnalysis(problem_space=problem_space,input_task=[problem_name],function_evaluations=function_evaluations,num_samples=256)
            print(sensitivity_data)
            print("S1")
            print(sensitivity_data["S1"])
            
            json_file.close()
        """
        if args.gen_plots:
            filename_prefix = 
            runtimes = [ elem[0] for elem in data.O[tid].tolist() ]
            postprocess.plot_runtime(runtimes,problem_name,1e8)
            param_datas = [
                { 'name': 'max_ord', 'type': 'integer', 'values': [ elem[0] for elem in data.P[tid] ] },
                { 'name': 'nonlin_conv_coef', 'type': 'real', 'values': [ elem[1] for elem in data.P[tid] ] },
                { 'name': 'max_conv_fails', 'type': 'integer', 'values': [ elem[2] for elem in data.P[tid] ] }
            ]
            if solve_type == 'all' or solve_type == 'newton_iter' or solve_type == 'newton_gmres' or solve_type == 'newton_pcg':
                param_datas += [
                    { 'name': 'maxl', 'type': 'integer', 'values': [ elem[3] for elem in data.P[tid] ] },
                    { 'name': 'epslin', 'type': 'real', 'values': [ elem[4] for elem in data.P[tid] ] },
                ]
            if solve_type == 'all' or solve_type == 'fixedpoint':
                param_datas += [
                    { 'name': 'fixedpointvecs', 'type': 'integer', 'values': [ elem[3] for elem in data.P[tid] ] },
                ]

            if additional_params:
                start_index = len(param_datas)
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
        """

if __name__ == "__main__":
    main()
