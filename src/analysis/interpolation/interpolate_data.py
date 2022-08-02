"""
This is a script to build a polynomial interpolator for the data from potentially multiple GPTune optimizations (using the result from process_csvs.py) 
and test the interpolator on a number of un-optimized points (points_to_optimize)

To find the "optimal" points for the test points, it locks in values for the task parameters k,n, and runs a minimizer over the rest of the variables.

It assumes the csvs are stored in ./csv/ relative to the directory you call the script, so make sure the data exists where it expects.
    Probably easiest to just move the data to this folder's ./csv/

To call this script: python interpolate_data.py
"""

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import NonlinearConstraint,LinearConstraint,minimize,differential_evolution,shgo,dual_annealing
import time

start = time.time()
table = np.genfromtxt('./csv/combined-task-params-runtime.csv',delimiter=',')
x = table[:,:-1]
y = table[:,-1]
end = time.time()
print("Reading data took: " + str(end-start) + " seconds")

start = time.time()
interpolator = RBFInterpolator(x,y,kernel='cubic',degree=3)
interpolator_wrapper = lambda x: interpolator([x])
end = time.time()
print("Generating interpolator took: " + str(end-start) + " seconds")

start = time.time()
points_to_optimize = [[3,128], [10,128], [1,200], [1,100]]
optimal_params = []
optimal_values = []

bounds = [(1,5000), (1,5000), (1,5), (1e-5,0.9), (3,50), (3,500), (1e-5,0.9)]

for point in points_to_optimize:
    print("Optimizing point ", point)
    
    constraints = (
        NonlinearConstraint(lambda x: np.floor(x[0])-point[0],0,0),
        NonlinearConstraint(lambda x: np.floor(x[1])-point[1],0,0),
        NonlinearConstraint(lambda x: np.floor(x[2])-x[2],0,0),
        NonlinearConstraint(lambda x: np.floor(x[4])-x[4],0,0),
        NonlinearConstraint(lambda x: np.floor(x[5])-x[5],0,0),
    )
    res = differential_evolution(interpolator_wrapper,bounds,constraints=constraints)
    
    """
    constraints = (
        { 'type': 'eq', 'fun': lambda x: x[0] - point[0] },
        { 'type': 'eq', 'fun': lambda x: x[1] - point[1] },
        { 'type': 'eq', 'fun': lambda x: np.floor(x[2])-x[2] },
        { 'type': 'eq', 'fun': lambda x: np.floor(x[4])-x[4] },
        { 'type': 'eq', 'fun': lambda x: np.floor(x[5])-x[5] },
    )
    res = shgo(interpolator_wrapper,bounds,constraints=constraints)
    """
    optimal_params.append(res.x)
    optimal_values.append(res.fun)
    

end = time.time()
print("Optimizing interpolated values took: " + str(end-start) + " seconds")

for i in range(len(points_to_optimize)): 
    #print("For task [k,n]=",points_to_optimize[i]," the predicted optimal parameters are: ", optimal_params[i][2:], " with predicted runtime: ", optimal_values[i])
    print("For task [k,n]=",points_to_optimize[i]," the predicted optimal parameters are: ", optimal_params[i][2:], " with predicted runtime: ", interpolator_wrapper(optimal_params[i]))
