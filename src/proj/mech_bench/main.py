# Import Libraries
import argparse
import os
import sys
from pathlib import Path
proj_path = Path(os.path.abspath(os.getcwd()))
lib_path = proj_path.parents[1]
# adding  to the system path
sys.path.insert(1, str(lib_path))

# External packages
import numpy as np
from numpy.random import SeedSequence

# Opt Library
from lib.opt.so.mcs import MCS
from lib.opt.so.mcs_v1 import MCSV1
from lib.opt.so.mcs_v2 import MCSV2
from lib.opt.so.pymoo_pso import PymooPSO
from lib.opt.so.pymoo_pso_v1 import PymooPSOV1
from lib.opt.so.pymoo_pso_v2 import PymooPSOV2
from lib.opt.so.pymoo_ga import PymooGA
from lib.opt.so.pymoo_cmaes import PymooCMAES
from lib.opt.so.pymoo_de import PymooDE
from lib.opt.wrapper import Wrapper
from lib.opt.init import init

# Other Library
from lib.other.files_man import FilesMan
from lib.other.args_file import init_args
from lib.other.tools import Timer

# Project Library
from plib.so_mech_bench import SOMechBench
from plib.report import Report
from plib.proj_args_file import init_proj_args

# --------------------------------------------------------------- #
#          MAIN FILE FOR CONTROLLED PARALLEL OPTIMISATION         #
# --------------------------------------------------------------- #

# Version 1:

# - Single-Objective Optimisation Only

# Objective Functions:

# - Path Finding Optimisation

# Optimisation Algorithms:

# - Particle Swarm Optimisation (PSO)
# - Genetic Algorithm (GA)
# - Differential Evolution (DE)
# - Covariance Matrix Adaption - Evolution Strategy (CMA-ES)
# - Modified Cuckoo Search (MCS)

# TERMINAL:
# 'mpiexec -n n_devices python main.py'
# or
# 'mpirun -n n_devices python main.py'

# ----------------------- User Parameters ----------------------- #

# argparse object
optim_parser = init_args()
proj_parser = init_proj_args()

[args_optim, extras_optim] = optim_parser.parse_known_args()
[args_proj, extras_proj] = proj_parser.parse_known_args()

# Combine parser arguments
args = proj_parser.parse_args(extras_optim, namespace=args_optim)

# ---------------------------- Project ----------------------------- #

# Project object
proj = SOMechBench(args)

# Select function to optimise
proj.optim_func = proj.three_bars

# ------------------------ Files Management ------------------------ #

if args.run_id == 'A':

    prob_id = 'mech_bench/three_bars/'
    args.run_id = prob_id + 'Run' + str(args.n_param) + '_' + str(args.exp_id)

files_man = FilesMan(args)

# Create result directories when using only one optimiser
if args.n_devices == 1:
    # Create run folder
    files_man.paths_init()
    files_man.run_folder()
# -----------------------------  Device ---------------------------- #

# Number of devices
n_devices = args.n_devices

# Random seed for multiprocessors
if n_devices >= 4:
    seed_sequence = SeedSequence(pool_size=n_devices)
    seed = seed_sequence.generate_state(args.max_runs + 1)
else:
    seed = [2431]

if args.report == 0:
    # ------------------------------------------------------------ #
    # ---------------------------- OPTIM  ------------------------ #
    # ------------------------------------------------------------ #

    # ---------------------- Initialisation ---------------------- #

    init = init(args, files_man, seed)

    # Pop initialisation function
    # 'LHS' - Latin Hypercube
    # 'RU' - Random Uniform
    init.func = 'RU'

    # Param range: An array of size (2, n_param) where the first row is the
    # lower limit and the second row is the upper limit for each parameter.

    # Three bars --------------------------------
    # lower limit
    init.param_range[0, :] = args.lower_limit_1
    # upper limit
    init.param_range[1, :] = args.upper_limit_1

    # Vessel design -----------------------------
    # lower limit
    # init.param_range[0, :2] = args.lower_limit_1
    # init.param_range[0, 2:] = args.lower_limit_2
    # # upper limit
    # init.param_range[1, :2] = args.upper_limit_1
    # init.param_range[1, 2:] = args.upper_limit_2

    # Thrust design -----------------------------
    # lower limit
    # init.param_range[0, :3] = args.lower_limit_1
    # init.param_range[0, 3:] = args.lower_limit_2
    # # upper limit
    # init.param_range[1, :3] = args.upper_limit_1
    # init.param_range[1, 3:] = args.upper_limit_2

    # Wind Farm -----------------------------
    # # lower limit
    # init.param_range[0, :] = args.lower_limit_1
    # # upper limit
    # init.param_range[1, :] = args.upper_limit_1

    # ------------------------- Optimiser ------------------------ #
    optim = Wrapper(args, init, proj, files_man, seed)

    # Define optimisers in a list

    optim.optim_list = [MCS, MCSV1, MCSV2,
                        PymooGA,
                        PymooPSO, PymooPSOV1, PymooPSOV2,
                        PymooCMAES, PymooDE]

    # Run optimiser
    optim.run()

elif args.report == 1:
    # ------------------------------------------------------------ #
    # ------------------------ REPORT: Plot ---------------------- #
    # ------------------------------------------------------------ #

    # Report object
    rep = Report(args, files_man)

    # Plot results for selected train (args.plot)
    # The following call plots a simple version of convergence plots and
    # a fast version utilising colours per each optimiser.
    # A version of the plots used in the paper can be activated in
    # the report function.
    rep.report_plot(proj)


    # Optional: Plot comparison results for selected train (args.plot)
    # Statistical information to give to comparison plots ----------------
    # opt_flag = True: standalone optimiser comparison
    # opt_flag = False: Wrapper comparison
    # run_name = 'PSO200'
    # n_runs = 10
    # rep.report_avg(run_name, prob_id, n_runs, opt_flag=True, n_workers=15)


    # Comparison Plots ---------------------------------------------------
    # Subfolder name
    # n_comp = 10
    # # Runs to compare
    # runs_file = ['PSO200_5', 'GA200_7', 'CMAES200_1','MCS200_1','DE200_1']
    # # Best run id inside each folder of runs_file
    # best_run_id = ['4', '13', '4', '11', '6']
    # # Optimiser name
    # opt_id = ['PymooPSO', 'PymooGA', 'PymooCMAES', 'MCS', 'PymooDE']
    #
    # # Average fmin and std_dev for each run obtained in terminal using
    # # the report_avg function and introduced manually here
    # avg_fmin= [98.2314, 65.2243, 284.9940, 66.8578, 123.0248, 31.9412]
    # std_dev = [30.2439, 23.3537, 52.7251, 14.7948, 3.5586, 0.911]

    # Call report comparison function
    # rep.report_comparison(proj, prob_id, runs_file, best_run_id, opt_id,
    #                       avg_fmin, std_dev, n_comp)






