"""
This is a script which prints the sample number, runtime, and param values of the successive most-optimal samples from a GPTune optimization

It reads in a csv which was generated from the GPTune process with the command line arugument -print_csv
It assumes the csv is stored in ./csv/ relative to the directory you call the script, so make sure the data exists where it expects.
    Probably easiest to just move the data to this folder's ./csv/

To call this script: python plot_params.py <prefix name of csv>
Ex: python print_processed_params diffusion-cvode-1-128-newton_gmres 
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

problem_name = sys.argv[1]

raw_file = open('./csv/' + problem_name + '.csv', 'r')
headerline = raw_file.readline().strip()
raw_file.close()
cols = headerline.split(',')
print(cols)

results = np.genfromtxt('./csv/' + problem_name + '.csv',delimiter=',')
var_vals = results[1:,:]
runtime_vals = results[1:, cols.index('runtime') ]

runtime_min = 1e8
for i in range(len(runtime_vals)):
    if runtime_vals[i] < runtime_min:
        print('Sample: ',i,', Runtime: ',runtime_vals[i],', Params: ',var_vals[i])
        runtime_min = runtime_vals[i]
