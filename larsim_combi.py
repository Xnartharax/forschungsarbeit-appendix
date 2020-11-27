import numpy as np
import json
import os
import matplotlib 
matplotlib.use("TkAgg")
from sys import path
path.append('./sparseSpACE/src/')
path.append('./Larsim_Utility_Set/LarsimUtilityFunctions/')
from Function import *
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
from GridOperation import *
from Integrator import *
import larsimPaths as paths
import larsimConfigurationSettings
import larsimDataPostProcessing
import larsimDataPreProcessing
import larsimInputOutputUtilities
import larsimTimeUtility
import larsimModel
import larsimPlottingUtility
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#NECESSARY PATH - directory where the output will be saved
working_dir= os.getcwd() + f"/Larsim_combi_runs{rank}"

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
        self.param_names = param_names # LARSIM requires named parameters
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
distributions = [(param["distribution"], param["lower_limit"], param["upper_limit"]) for param in params]
param_names = [(param["type"], param["name"]) for param in params]
timeframe = larsimConfig["Timeframe"]

# a and b are the weighted integration domain boundaries.
# They should be set according to the distribution.
a = np.array([param["lower_limit"]for param in params])
b = np.array([param["upper_limit"]for param in params])

problem_function = LarsimFunction(param_names)
# Create the grid operation and the weighted grid
op = UncertaintyQuantification(problem_function, distributions, a, b)
grid = GaussLegendreGrid(a, b, op)
# set the Integrator to use the parallel implementation
grid.integrator = IntegratorParallelArbitraryGridOptimized(grid)
# The grid initialization requires the weight functions from the
# operation; since currently the adaptive refinement takes the grid from
# the operation, it has to be passed here
op.set_grid(grid)

# Select the function for which the grid is refined;
# here it is the expectation and variance calculation via the moments
op.set_expectation_variance_Function()

# Initialize the adaptive refinement instance and refine the grid until
# it has at least 200 points

error_operator = ErrorCalculatorSingleDimVolumeGuided()

combiinstance = StandardCombi(a, b, operation=op, norm=2)

# Calculate the expectation and variance with the adaptive sparse grid
# weighted integral result
for i in range(20):
    combiinstance.perform_operation(1, i+1)
    print(f'cache_afterall: {len(op.f.f_dict)}')
    
    if rank == 0:
        E, Var = op.calculate_expectation_and_variance(combiinstance)
        print(f"E: {E}, Var: {Var}")

        fp = open(f"/home/larsim/Larsim/combi{i+1}_results.txt", "w")
        json.dump({'E':E.tolist(), 'Var':Var, 'n_nodes': len(op.f.f_dict),
            'nodes':list(op.f.f_dict.keys())}, fp)
        problem_function.reset_dictionary()
        fp.close()
