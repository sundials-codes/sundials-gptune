
"""
This is a script which plots the norm of the difference in runtime values when evaluating with ni using parameters which were optimized for nj for all combinations of n1, n2, ...
This script assumes a csv file which will need to be hand-made (no current code exists to create it) with the following columns WITH a header line
    n-eval,n1-opt,n2-opt,n3-opt,...,nN-opt
where the first column is equal to the transpose of the header line, and the "cells" are the runtimes evaluating with ni-eval using parameters optimized for nj-opt
I call this the "crosstable" (term came from chess tournaments)

It assumes the csv exists in ./csv/ relative to whatever was the working directory that called this script

How to call script: python plot_crosstable-nxy.csv
"""

import numpy as np
import matplotlib.pyplot as plt

raw_data = np.genfromtxt('./csv/crosstable-nxy.csv',delimiter=',')
data = { "x": raw_data[1:,0], "y": raw_data[1:,1:] }

plt.plot(data["x"],data["y"],marker='o')
plt.ylabel("Runtime (s)")
plt.xlabel("$n_{eval}$")
plt.legend([str(int(i)) for i in data["x"]],title="$n_{opt}")
plt.title("Runtime for $n_{eval}$ using $P_{n_{opt}}$")
plt.savefig("./png/crosstable-nxy.png")

plt.close()

raw_dataT = np.transpose(raw_data)
dataT = { "x": raw_dataT[1:,0], "y": raw_dataT[1:,1:] }

plt.plot(dataT["x"],dataT["y"],marker='o')
plt.ylabel("Runtime (s)")
plt.xlabel("$n_{opt}$")
plt.legend([str(int(i)) for i in data["x"]],title="$n_{eval}$")
plt.title("Runtime for $n_{eval}$ using $P_{n_{opt}}$")
plt.savefig("./png/crosstable-nxy-t.png")

plt.close()
