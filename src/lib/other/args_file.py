import argparse


def init_args():
    # ----------------------- User Parameters ----------------------- #
    parser = argparse.ArgumentParser(description='User Parameters')

    # ---------------------- Optimiser - General --------------------- #
    parser.add_argument('--nrun', default=0,
                        type=int, metavar='N',
                        help='train number (default: 0)')
    parser.add_argument('--max_runs', default=1000,
                        type=int, metavar='N',
                        help='Max run number (default: 0)')
    parser.add_argument('--checkpoint', default=10,
                        type=int, metavar='N',
                        help='Number of Gen to save state (default: 10)')
    parser.add_argument('--n_obj', default=1,
                        type=int, metavar='N',
                        help='Number of objectives (default: 1)')
    parser.add_argument('--n_partitions', default=10,
                        type=int, metavar='N',
                        help='N partitions for reference directions'
                             ' (default: 10)')
    parser.add_argument('--obj_w', nargs='+', type=float,
                        help='Objective weighting')
    parser.add_argument('--live_plot', default=False,
                        type=bool, metavar='N',
                        help='Convergence live plot'
                             '(default: False)')
    parser.add_argument('--exp_id', default=1,
                        type=int, metavar='N',
                        help='Experiment ID'
                             ' (default: 1)')
    parser.add_argument('--work_out', default=False,
                        type=bool, metavar='N',
                        help='Worker out files'
                             '(default: False)')

    # --------------------- MCS / MOMCS Optimiser -------------------- #
    parser.add_argument('--mcs_pop_size', default=100,
                        type=int, metavar='N',
                        help='MCS population size (default: 100)')
    parser.add_argument('--mcs_max_gen', default=1000000,
                        type=int, metavar='N',
                        help='MCS number of generations (default: 1000000)')
    parser.add_argument('--frac_dis', default=0.7,
                        type=float, metavar='N',
                        help='Fraction discard (default: 0.7)')
    parser.add_argument('--constrain', default=1,
                        type=int, metavar='N',
                        help='(1) search constrained (default: 1)')
    parser.add_argument('--max_step', default=100,
                        type=int, metavar='N',
                        help='Step size factor (default: 100)')
    parser.add_argument('--pwr', default=0.5,
                        type=float, metavar='N',
                        help='step size power (default: 0.5)')
    parser.add_argument('--flight', default=1,
                        type=int, metavar='N',
                        help='Type of random walk (default: 1)')
    parser.add_argument('--nests_del', default=1,
                        type=int, metavar='N',
                        help='Number of eggs deleted each '
                             'generation (default: 1)')
    parser.add_argument('--min_nests', default=10,
                        type=int, metavar='N',
                        help='minimum number of nests (default: 10)')
    parser.add_argument('--sort_met', default=2,
                        type=int, metavar='N',
                        help='Sorting Method (1) Non-Dominated Sorting,'
                             ' (2) Pymoo Fast Sorting, (3) Weighted Sum'
                             ' (default: 2)')
    parser.add_argument('--wa_flag', default=0,
                        type=int, metavar='N',
                        help='Weighted aggregation flag: (0) Off (1) On'
                             ' (default: 0)')

    # --------------------- MCS V1 (Exploration) Optimiser ------------------ #
    parser.add_argument('--mcs_pop_size_v1', default=100,
                        type=int, metavar='N',
                        help='MCS population size (default: 50)')
    parser.add_argument('--mcs_max_gen_v1', default=1000000,
                        type=int, metavar='N',
                        help='MCS number of generations (default: 1000000)')
    parser.add_argument('--frac_dis_v1', default=0.9,
                        type=float, metavar='N',
                        help='Fraction discard (default: 0.9)')
    parser.add_argument('--constrain_v1', default=1,
                        type=int, metavar='N',
                        help='(1) search constrained (default: 1)')
    parser.add_argument('--max_step_v1', default=10,
                        type=int, metavar='N',
                        help='Step size factor (default: 10)')
    parser.add_argument('--pwr_v1', default=0.25,
                        type=float, metavar='N',
                        help='step size power (default: 0.5)')
    parser.add_argument('--flight_v1', default=1,
                        type=int, metavar='N',
                        help='Type of random walk (default: 1)')
    parser.add_argument('--nests_del_v1', default=1,
                        type=int, metavar='N',
                        help='Number of eggs deleted each '
                             'generation (default: 1)')
    parser.add_argument('--min_nests_v1', default=10,
                        type=int, metavar='N',
                        help='minimum number of nests (default: 10)')
    parser.add_argument('--sort_met_v1', default=2,
                        type=int, metavar='N',
                        help='Sorting Method (1) Non-Dominated Sorting,'
                             ' (2) Pymoo Fast Sorting, (3) Weighted Sum'
                             ' (default: 2)')
    parser.add_argument('--wa_flag_v1', default=0,
                        type=int, metavar='N',
                        help='Weighted aggregation flag: (0) Off (1) On'
                             ' (default: 0)')

    # --------------------- MCS V2 (Exploitation) Optimiser ----------------- #
    parser.add_argument('--mcs_pop_size_v2', default=100,
                        type=int, metavar='N',
                        help='MCS population size (default: 100)')
    parser.add_argument('--mcs_max_gen_v2', default=1000000,
                        type=int, metavar='N',
                        help='MCS number of generations (default: 1000000)')
    parser.add_argument('--frac_dis_v2', default=0.2,
                        type=float, metavar='N',
                        help='Fraction discard (default: 0.9)')
    parser.add_argument('--constrain_v2', default=1,
                        type=int, metavar='N',
                        help='(1) search constrained (default: 1)')
    parser.add_argument('--max_step_v2', default=100000,
                        type=int, metavar='N',
                        help='Step size factor (default: 10e5)')
    parser.add_argument('--pwr_v2', default=0.9,
                        type=float, metavar='N',
                        help='step size power (default: 0.5)')
    parser.add_argument('--flight_v2', default=1,
                        type=int, metavar='N',
                        help='Type of random walk (default: 1)')
    parser.add_argument('--nests_del_v2', default=1,
                        type=int, metavar='N',
                        help='Number of eggs deleted each '
                             'generation (default: 1)')
    parser.add_argument('--min_nests_v2', default=10,
                        type=int, metavar='N',
                        help='minimum number of nests (default: 10)')
    parser.add_argument('--sort_met_v2', default=2,
                        type=int, metavar='N',
                        help='Sorting Method (1) Non-Dominated Sorting,'
                             ' (2) Pymoo Fast Sorting, (3) Weighted Sum'
                             ' (default: 2)')
    parser.add_argument('--wa_flag_v2', default=0,
                        type=int, metavar='N',
                        help='Weighted aggregation flag: (0) Off (1) On'
                             ' (default: 0)')

    # ------------------------- PSO Optimiser ----------------------- #
    parser.add_argument('--pso_pop_size', default=25,
                        type=int, metavar='N',
                        help='PSO population size (default: 25)')
    parser.add_argument('--pso_max_gen', default=1000000,
                        type=int, metavar='N',
                        help='PSO number of generations (default: 1000000)')
    parser.add_argument('--w_inertia', default=0.9,
                        type=float, metavar='N',
                        help='Inertia w (default: 0.9)')
    parser.add_argument('--c1', default=2.0,
                        type=float, metavar='N',
                        help='Cognitive Impact (default: 2.0)')
    parser.add_argument('--c2', default=2.0,
                        type=float, metavar='N',
                        help='Social Impact (default: 2.0)')
    parser.add_argument('--max_vel_rate', default=0.2,
                        type=float, metavar='N',
                        help='Max Velocity Rate (default: 0.2)')
    parser.add_argument('--init_vel', default="random",
                        type=str, metavar='N',
                        help='Initial Velocity "random" or "zero" '
                             '(default: random)')
    parser.add_argument('--adapt', default=True,
                        type=bool, metavar='N',
                        help='Wheter w, c1 and c2 are adaptive '
                             '(default: random)')

    # ------------------- PSO V1 (Exploration) Optimiser -------------------- #
    parser.add_argument('--pso_pop_size_v1', default=25,
                        type=int, metavar='N',
                        help='PSO V1 population size (default: 25)')
    parser.add_argument('--pso_max_gen_v1', default=1000000,
                        type=int, metavar='N',
                        help='PSO V1 number of generations (default: 1000000)')
    parser.add_argument('--w_inertia_v1', default=0.9,
                        type=float, metavar='N',
                        help='Inertia w (default: 0.9)')
    parser.add_argument('--c1_v1', default=3.9,
                        type=float, metavar='N',
                        help='Cognitive Impact (default: 3.9)')
    parser.add_argument('--c2_v1', default=0.1,
                        type=float, metavar='N',
                        help='Social Impact (default: 0.1)')
    parser.add_argument('--max_vel_rate_v1', default=0.2,
                        type=float, metavar='N',
                        help='Max Velocity Rate (default: 0.2)')
    parser.add_argument('--init_vel_v1', default="random",
                        type=str, metavar='N',
                        help='Initial Velocity "random" or "zero" '
                             '(default: random)')
    parser.add_argument('--adapt_v1', default=False,
                        type=bool, metavar='N',
                        help='Wheter w, c1 and c2 are adaptive '
                             '(default: random)')

    # ------------------- PSO V2 (Exploitation) Optimiser ------------------ #
    parser.add_argument('--pso_pop_size_v2', default=25,
                        type=int, metavar='N',
                        help='PSO V2 population size (default: 25)')
    parser.add_argument('--pso_max_gen_v2', default=1000000,
                        type=int, metavar='N',
                        help='PSO V2 number of generations (default: 1000000)')
    parser.add_argument('--w_inertia_v2', default=0.2,
                        type=float, metavar='N',
                        help='Inertia w (default: 0.2)')
    parser.add_argument('--c1_v2', default=0.2,
                        type=float, metavar='N',
                        help='Cognitive Impact (default: 0.2)')
    parser.add_argument('--c2_v2', default=3.9,
                        type=float, metavar='N',
                        help='Social Impact (default: 3.8)')
    parser.add_argument('--max_vel_rate_v2', default=0.2,
                        type=float, metavar='N',
                        help='Max Velocity Rate (default: 0.2)')
    parser.add_argument('--init_vel_v2', default="random",
                        type=str, metavar='N',
                        help='Initial Velocity "random" or "zero" '
                             '(default: random)')
    parser.add_argument('--adapt_v2', default=False,
                        type=bool, metavar='N',
                        help='Wheter w, c1 and c2 are adaptive '
                             '(default: random)')

    # ----------------------- GA/NSGA Optimiser ---------------------- #
    parser.add_argument('--ga_pop_size', default=100,
                        type=int, metavar='N',
                        help='GA population size (default: 100)')
    parser.add_argument('--ga_max_gen', default=1000000,
                        type=int, metavar='N',
                        help='GA number of generations (default: 1000000)')
    parser.add_argument('--n_offsprings', default=25,
                        type=int, metavar='N',
                        help='Number of Offsprings (default: 25)')
    parser.add_argument('--elim_dup', default=True,
                        type=bool, metavar='N',
                        help='Wheter w, c1 and c2 are adaptive '
                             '(default: random)')

    # ----------------------- C-TAEA Optimiser ---------------------- #
    parser.add_argument('--ctaea_pop_size', default=50,
                        type=int, metavar='N',
                        help='C-TAEA population size (default: 50)')
    parser.add_argument('--ctaea_max_gen', default=1000000,
                        type=int, metavar='N',
                        help='number of generations (default: 1000000)')

    # ------------------------- DE Optimiser ------------------------- #
    parser.add_argument('--de_pop_size', default=50,
                        type=int, metavar='N',
                        help='de population size (default: 50)')
    parser.add_argument('--de_max_gen', default=1000000,
                        type=int, metavar='N',
                        help='number of generations (default: 1000000)')
    parser.add_argument('--variant', default="DE/rand/1/bin",
                        type=str, metavar='N',
                        help='selected method (default: "DE/rand/1/bin"')
    parser.add_argument('--CR', default=0.9,
                        type=int, metavar='N',
                        help='crossover constant (default: 0.9)')
    parser.add_argument('--NP', default=40,
                        type=int, metavar='N',
                        help='number of parents (default: 40)')
    parser.add_argument('--F', default=0.8,
                        type=int, metavar='N',
                        help='weighting factor (default: 0.8)')
    parser.add_argument('--dither', default="vector",
                        type=str, metavar='N',
                        help='a technique that improves performance '
                             '(default: "vector")')

    # ----------------------- CMA-ES Optimiser ----------------------- #
    parser.add_argument('--cmaes_pop_size', default=100,
                        type=int, metavar='N',
                        help='cma-es population size (default: 100)')
    parser.add_argument('--cmaes_max_gen', default=1000000,
                        type=int, metavar='N',
                        help='number of generations (default: 1000000)')
    parser.add_argument('--sigma', default=0.5,
                        type=float, metavar='N',
                        help='initial standard deviation in each '
                             'coordinate (default: 0.5)')
    parser.add_argument('--restarts', default=0,
                        type=int, metavar='N',
                        help='number of restarts with increasing '
                             'population size (default: 0)')

    # ---------------------------- Report ---------------------------- #
    parser.add_argument('--report', default=0,
                        type=int, metavar='N',
                        help='(0) Train (2) Report Results (default: 0)')
    parser.add_argument('--run_id', type=str,
                        default='A',
                        metavar='N',
                        help='Run ID to create folder'
                             ' (default: A)')

    # ------------------------------ SPO WRAPPER ---------------------------- #
    parser.add_argument('--n_devices', type=int, default=1, metavar='N',
                        help='Number of devices (default: 1)')
    parser.add_argument('--n_rep', type=int, default=100, metavar='N',
                        help='Max number of repository rows '
                             '(default: 100)')
    parser.add_argument('--pop_from_rep', default=5,
                        type=int, metavar='N',
                        help='N population from best repository'
                             ' (default: 5)')
    parser.add_argument('--max_pop_from_rep', default=1.0,
                        type=float, metavar='N',
                        help='# [0,1] max porcentage of population from rep '
                             '(default: 1.0)')
    parser.add_argument('--init_prob', default=0.5,
                        type=float, metavar='N',
                        help='# [0,1] Probability of initialise from rep '
                             '(default: 0.5)')
    parser.add_argument('--n_0', type=int, default=2, metavar='N',
                        help='Max number of checkpoints reached '
                             'without improving fmin (default: 2)')
    parser.add_argument('--n_stall', type=int, default=3, metavar='N',
                        help='Number of stall error tracking for each optim '
                             '(default: 3)')
    parser.add_argument('--p_n', type=int, default=3, metavar='N',
                        help='Parameter to define number of generations allowed '
                             '(default: 3)')
    parser.add_argument('--n_best_runs', type=int, default=2, metavar='N',
                        help='Number of best runs allowed to run max generations'
                             '(default: 2)')
    parser.add_argument('--stall_tol', type=float, default=0.01, metavar='N',
                        help='Stall reference level (default: 0.01)')
    parser.add_argument('--kill_flag', default=True, type=bool,
                        metavar='N',
                        help='Flag that turns off stall decision '
                             'making mechanism (default: True)')
    return parser
