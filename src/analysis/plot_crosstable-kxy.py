"""
This is a script which plots the norm of the difference in runtime values when evaluating with ki using parameters which were optimized for kj for all combinations of k1, k2, ...
This script assumes a csv file which will need to be hand-made (no current code exists to create it) with the following columns WITH a header line
    k-eval,k1-opt,k2-opt,k3-opt,...,kN-opt
where the first column is equal to the transpose of the header line, and the "cells" are the runtimes evaluating with ki-eval using parameters optimized for kj-opt
I call this the "crosstable" (term came from chess tournaments)

It assumes the csv exists in ./csv/ relative to whatever was the working directory that called this script

How to call script: python plot_crosstable-kxy.csv
"""

import numpy as np
import matplotlib.pyplot as plt

raw_data = np.genfromtxt('./csv/crosstable-kxy.csv',delimiter=',',skip_header=1)
data = { "x": raw_data[:,0], "y": raw_data[:,1:] }

plt.plot(data["x"],data["y"],marker='o')
plt.ylabel("Runtime (s)")
plt.xlabel("$k_{eval}$")
plt.legend([str(int(i)) for i in data["x"]],title="$k_{opt}$")
plt.title("Runtime for $k_{eval}$ using $P_{k_{opt}}$")
plt.savefig("./png/crosstable-kxy.png")

plt.close()

raw_dataT = np.transpose(raw_data)
dataT = { "x": raw_dataT[1:,0], "y": raw_dataT[1:,1:] }

plt.plot(dataT["x"],dataT["y"],marker='o')
plt.ylabel("Runtime (s)")
plt.xlabel("$k_{opt}$")
plt.legend([str(int(i)) for i in data["x"]],title="$k_{eval}$")
plt.title("Runtime for $k_{eval}$ using $P_{k_{opt}}$")
plt.savefig("./png/crosstable-kxy-t.png")

plt.close()

for i in range(len(data["x"])):
    plt.plot(data["x"]-data["x"][i],data["y"][:,i],marker='o')
plt.axvline(x=0,linewidth=1,color='k')
plt.ylabel("Runtime (s)")
plt.xlabel("$k_{eval}-k_{opt}$")
plt.legend([str(int(i)) for i in data["x"]],title="$k_{eval}$")
plt.title("Runtime for $k_{eval}$ using $P_{k_{opt}}$")
plt.savefig("./png/crosstable-kxy-shift.png")

plt.close()


