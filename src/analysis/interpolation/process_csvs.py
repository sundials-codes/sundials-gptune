"""
This is a script to take the output csv's from the GPTune optimization using the command line parameter -print_csv
and simply stack them all into one csv

It assumes the csvs are stored in ./csv/ relative to the directory you call the script, so make sure the data exists where it expects.
    Probably easiest to just move the data to this folder's ./csv/

To call this script: python process_csvs.py
"""
import numpy as np

kns = [[1,128], [5,128], [20,128], [1,64], [1,256]]
tables = []
for kn in kns:
    k = kn[0]
    n = kn[1]
    table = np.genfromtxt('./csv/diffusion-cvode-' + str(k) + '-' + str(n) + '-newton-gmres.csv', delimiter=',')
    table = table[1:,:]
    table = table[table[:,-1]<1e8]
    kcolumn = k*np.ones((len(table),1))
    ncolumn = n*np.ones((len(table),1))
    table = np.hstack((kcolumn,ncolumn,table))
    tables.append(table)

combined_table = np.vstack(tuple(tables))

np.savetxt('./csv/combined-task-params-runtime.csv', combined_table, delimiter=',')
 
    
