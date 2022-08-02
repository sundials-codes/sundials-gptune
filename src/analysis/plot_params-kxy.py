"""
This is a script which plots the norm of the difference in optimal parameter values for all combinations of k1, k2, ...
This script assumes a csv file which will need to be hand-made (no current code exists to create it) with the following columns WITH a header line
    k,param1,param2,...,paramN

It assumes the csv exists in ./csv/ relative to whatever was the working directory that called this script

How to call script: python plot_params-kxy.csv <csvfileprefix>
Ex: python plot_params-kxy.csv diffusion-cvode-k-128-newton_gmres
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

raw_data = np.genfromtxt('./csv/' + sys.argv[1] + '.csv',delimiter=',')
data = { "x": raw_data[:,0], "y": raw_data[:,1:] }

for i in range(len(data["x"])):
    yplot = []
    for j in range(len(data["x"])):
        yplot.append((data["y"][j,1]-data["y"][i,1])**2 + (data["y"][j,4]-data["y"][i,4])**2)
    plt.plot(data["x"]-data["x"][i],yplot,marker='o')
plt.axvline(x=0,linewidth=1,color='k')
plt.ylabel("$||P_{k_{eval}}-P_{k_{opt}}||$")
plt.xlabel("$k_{eval}-k_{opt}$")
plt.legend([str(int(i)) for i in data["x"]],title="$k_{eval}$")
plt.title("Runtime for $k_{eval}$ using $P_{k_{opt}}$")
plt.savefig("./png/param-kxy-diff-norm-shift.png")

plt.close()


