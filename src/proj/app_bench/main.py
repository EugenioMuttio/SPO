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
from plib.so_power_bench import SOPowerBench
from plib.so_chem_bench import SOChemBench
from plib.report import Report
from plib.proj_args_file import init_proj_args

# --------------------------------------------------------------- #
#          MAIN FILE FOR CONTROLLED PARALLEL OPTIMISATION         #
# --------------------------------------------------------------- #

# Version 1:

# - Single-Objective Optimisation Only

# Objective Functions:

# Chemical Benchmark Problems:
# - Propane, isobutane, n-butane nonsharp separation

# Mechanical Benchmark Problems:
# - Three Bars
# - Vessel Design
# - Thrust Design

# Power Benchmark Problems:
# - Wind Farm

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
# proj = SOChemBench(args)
proj = SOMechBench(args)

# Select function to optimise
proj.optim_func = proj.three_bars

# ------------------------ Files Management ------------------------ #

if args.run_id == 'A':

    prob_id = 'app_bench/three_bars/'
    args.run_id = prob_id + 'RunTest' + str(args.n_param) + '_' + str(args.exp_id)

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
    # init.param_range[0, :] = args.lower_limit_1
    # init.param_range[0, 3] = args.lower_limit_2
    # # upper limit
    # init.param_range[1, :] = args.upper_limit_1
    # init.param_range[1, 3] = args.upper_limit_2

    # Himmelblau's function ---------------------
    # # lower limit
    # init.param_range[0, 0] = 78
    # init.param_range[0, 1] = 33
    # init.param_range[0, [2, 3, 4]] = 27
    # # upper limit
    # init.param_range[1, 0] = 102
    # init.param_range[1, [1, 2, 3, 4]] = 45

    # Wind Farm -----------------------------
    # lower limit
    # init.param_range[0, :] = args.lower_limit_1
    # # upper limit
    # init.param_range[1, :] = args.upper_limit_1

    # SOPWM --------------------------------
    # # lower limit
    # init.param_range[0, :] = args.lower_limit_1
    # # upper limit
    # init.param_range[1, :] = args.upper_limit_1

    # Nonsharp Separation ------------------------
    # args.lower_limit_1 = 0.0
    # args.upper_limit_1 = 150.0
    # args.lower_limit_2 = 0.0
    # args.upper_limit_2 = 30.0
    # args.lower_limit_3 = 0.0
    # args.upper_limit_3 = 1.0
    # args.lower_limit_4 = 0.85
    # args.upper_limit_4 = 1.0
    # # lower limit
    # init.param_range[0, :] = 0.0
    # init.param_range[0, 0:20] = args.lower_limit_1
    # init.param_range[0, [24, 26, 31, 34, 36, 28]] = args.lower_limit_2
    # init.param_range[0, [20, 21, 22, 29, 32, 33, 35, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47]] = args.lower_limit_3
    # init.param_range[0, [23, 25, 27, 30]] = args.lower_limit_4
    # # upper limit
    # init.param_range[1, :] = 1.0
    # init.param_range[1, 0:20] = args.upper_limit_1
    # init.param_range[1, [24, 26, 31, 34, 36, 28]] = args.upper_limit_2
    # init.param_range[1, [20, 21, 22, 29, 32, 33, 35, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47]] = args.upper_limit_3
    # init.param_range[1, [23, 25, 27, 30]] = args.upper_limit_4

    # Heat exchanger network design (Case 1) ------------------------
    # lower limit
    # init.param_range[0, [0, 1, 2, 3, 5]] = 0
    # init.param_range[0, 4] = 1000
    # init.param_range[0, [6, 7, 8]] = 100
    # # upper limit
    # init.param_range[1, 0] = 10
    # init.param_range[1, [1, 3]] = 200
    # init.param_range[1, 2] = 100
    # init.param_range[1, 4] = 2000000
    # init.param_range[1, [5, 6, 7]] = 600
    # init.param_range[1, 8] = 900

    # Heat exchanger network design (Case 2) ------------------------
    # lower limit
    # init.param_range[0, [0, 1, 2]] = 1e4
    # init.param_range[0, [3, 4, 5]] = 0
    # init.param_range[0, [6, 7, 8, 9, 10]] = 100
    # # upper limit
    # init.param_range[1, 0] = 81.9e4
    # init.param_range[1, 1] = 113.1e4
    # init.param_range[1, 2] = 205e4
    # init.param_range[1, [3, 4, 5]] = 5.074e-2
    # init.param_range[1, 6] = 200
    # init.param_range[1, [7, 8, 9]] = 300
    # init.param_range[1, 10] = 400

    # ------------------------- Optimiser ------------------------ #
    optim = Wrapper(args, init, proj, files_man, seed)

    # Define optimisers in a list

    optim.optim_list = [MCS, MCSV1, MCSV2,
                        PymooGA, PymooPSO, PymooPSOV1, PymooPSOV2,
                        PymooCMAES, PymooDE]

    # optim = MCS(args, init, proj, files_man, seed)

    # Run optimiser
    optim.run()

elif args.report == 1:
    # ------------------------------------------------------------ #
    # ------------------------ REPORT: Plot ---------------------- #
    # ------------------------------------------------------------ #

    # Report object
    rep = Report(args, files_man)

    # Plot results for selected train (args.plot)
    # The following call plots a fast version utilising colours
    # per each optimiser.
    # A version of the plots used in the paper can be activated in
    # the report function.
    rep.report_plot(proj)

    # Statistical information to give to comparison plots --------------------
    # Optional: Plot comparison results for selected train (args.plot)
    # opt_flag = True: standalone optimiser comparison
    # opt_flag = False: Wrapper comparison
    # run_name = 'Run9'
    # n_runs = 25
    # rep.report_avg(run_name, prob_id, n_runs, opt_flag=False, n_workers=10)

    # Comparison Plots -------------------------------------------------------
    # Subfolder name
    # n_comp = 20
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

    # Optimisers Analysis ----------------------------------------------------
    # rep.report_optim_analysis(run_name, prob_id, n_runs, n_comp)




