import chaospy as cp
import os
import pickle
import numpy as np
import json
from sys import path
path.append('./sparseSpACE/src/')
path.append('./Larsim_Utility_Set/LarsimUtilityFunctions/')
from Function import *
import larsimPaths as paths
import larsimConfigurationSettings
import larsimDataPostProcessing
import larsimDataPreProcessing
import larsimInputOutputUtilities
import larsimTimeUtility
import larsimModel
import larsimPlottingUtility
import chaospy
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#NECESSARY PATH - directory where the output will be saved
working_dir= os.getcwd() + f"/Larsim_runs{rank}"

#NECESSARY PATH - path to the configuration file
configurationFile = os.getcwd() + "/Larsim_Utility_Set/configurations/configuration_larsim_updated_lai.json" 
# just a 2 parameters for speed in the demo and print of grid

# directory where source code is - can be left out as well
current_dir = os.getcwd()

# NECESSARY PATH - root directory for Larsim data
inputModelDir= os.getcwd() + "/Larsim-data"


class LarsimFunction(Function):
    def __init__(self, param_names):
        super().__init__()
        self.larsim_model = larsimModel.LarsimModel(configurationObject=configurationFile,create_master_dir=True,\
                                     current_dir=current_dir, inputModelDir=inputModelDir, working_dir=working_dir)
        self.param_names = param_names
        self.larsim_model.setUp()

    def output_length(self):
        return 115

    def eval(self, coords):
        params = {param_name: coord for coord, param_name in zip(coords, self.param_names)}
        larsim_res = self.larsim_model.run(parameters=[params], i_s = [0], take_direct_value=True, createNewFolder=True,
                                           deleteFolderAfterwards=False, calculate_GoF=True, run_and_save_simulations=True)
        df = larsimDataPostProcessing.filterResultForStationAndTypeOfOutpu\
                (resultsDataFrame=larsim_res[-1]['result_time_series'],\
                 station="MARI",\
                 type_of_output=['Abfluss Messung'])
        return np.array(df["Value"])


larsimConfig = json.load(open(configurationFile))
params = larsimConfig["parameters"]

dim = len(params)
distributions = [(param["distribution"], param["lower"], param["upper_limit"]) for param in params]
param_names = [(param["type"], param["name"]) for param in params]
timeframe = larsimConfig["Timeframe"]

# a and b are the weighted integration domain boundaries.
# They should be set according to the distribution.
a = np.array([param["lower_limit"]for param in params])
b = np.array([param["upper_limit"]for param in params])

problem_function = LarsimFunction(param_names)
results = []
for j in range(1, 500):   
    n_nodes = 100
    if rank == 0:
        
        def make_dist(name, lower, upper):
            if name == "Uniform":
                return chaospy.Uniform(lower, upper)
        distribution = chaospy.J(*[make_dist(name, a, b) for name, a, b in distributions])

        nodes = distribution.sample(size=n_nodes)
        cache = problem_function.f_dict

        cached = [vec for vec in nodes.T if tuple(vec) in cache]
        uncached = [vec for vec in nodes.T if tuple(vec) not in cache]
        chunks = [cached[i::size] + uncached[i::size] for i in range(size)]

    else:
        chunks = None
    nodes = comm.scatter(chunks, root=0)
    for node in nodes:
        results.append(problem_function(node))
    
    res = comm.gather(results)
    if rank == 0:
        total_result = []
        for r in res:
            total_result += [series.tolist() for series in r] 
        total_result = np.array(total_result)
        mean = np.mean(total_result, axis=0)
        var = np.var(total_result, axis=0)
        fp = open(f'/home/larsim/Larsim/mc/mc_results_{j}', 'w')
        json.dump({'E': mean.tolist(), 'Var': var.tolist(), 'n_nodes': len(total_result)}, fp)
        
