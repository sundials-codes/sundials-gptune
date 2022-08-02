"""
This is a script to make some simple plots
It generates 2 plots, one with just param values vs sample number, and one similar but with marked red dots for failed sample values
It then generates a runtime vs sample number plot, with failed samples removed 

It reads in a csv which was generated from the GPTune process with the command line arugument -print_csv
It assumes the csv is stored in ./csv/ relative to the directory you call the script, so make sure the data exists where it expects.
    Probably easiest to just move the data to this folder's ./csv/

To call this script: python plot_params.py <prefix name of csv>
Ex: python plot_params diffusion-cvode-1-128-newton_gmres 
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

for i in range(len(cols)):
    var_name = cols[i]
    vals = var_vals[:,i]
    
    if var_name != 'runtime':
        plt.figure()
        plt.plot(vals)
        plt.xlabel('Sample')
        plt.ylabel(var_name)
        plt.savefig('./png/' + problem_name + '-' + var_name + '.png')
    
        failure_indices = []
        failure_vals = []
        for i in range(len(vals)):
            if runtime_vals[i] > 1e7:
                failure_indices.append(i)
                failure_vals.append(vals[i])

        plt.scatter(failure_indices, failure_vals, color='r')
        plt.savefig('./png/' + problem_name + '-' + var_name + '-markfails.png')
        plt.close()

filtered_runtime_vals = list(filter(lambda x: x <= 1e7, runtime_vals))
plt.figure()
plt.plot(filtered_runtime_vals)
plt.xlabel('Sample')
plt.ylabel('runtime')
plt.savefig('./png/' + problem_name + '-runtime.png')
