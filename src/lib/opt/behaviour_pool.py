import numpy as np
def init_behaviour(args):

    # Exploration
    # w_inertia_v1, c1_v1, c2_v1, frac_dis_v1, max_step_v1, pwr_v1
    exploration_mat = np.array(
        [[0.9, 3.9, 0.1, 0.9, 10, 0.25],
         [0.85, 3.5, 0.5, 0.85, 50, 0.35],
         [0.75, 3.0, 1.0, 0.75, 100, 0.45],
         [0.65, 2.5, 2.5, 0.65, 500, 0.55],
         [0.60, 2.0, 2.0, 0.6, 1000, 0.6],
         [5.060e-01, 3.191, 1.096, 6.01e-01, 3.376e+02, 2.68e-01],
         [7.89e-01, 3.659, 8.090e-01, 5.180e-01, 7.306e+02, 5.220e-01],
         [7.91e-01, 3.097, 1.29e-01, 5.78e-01, 8.879e+02, 4.89e-01],
         [7.64e-01, 2.763, 6.31e-01, 5.66e-01, 9.44e+01, 5.67e-01],
         [5.9e-01, 3.528, 1.902, 8.74e-01, 6.595e+02, 4.10000e-01]])

    # Exploitation
    # w_inertia_v2, c1_v2, c2_v2, frac_dis_v2, max_step_v2, pwr_v2
    exploitation_mat = np.array(
        [[0.2, 0.2, 3.9, 0.2, 1000000, 0.9],
         [0.3, 0.5, 3.5, 0.3, 100000, 0.8],
         [0.4, 1.0, 3.0, 0.4, 10000, 0.7],
         [0.5, 1.5, 2.5, 0.5, 5000, 0.6],
         [0.60, 2.0, 2.0, 0.6, 1000, 0.5],
         [0.191, 0.889, 2.483, 0.545, 428263, 0.873],
         [0.401, 0.618, 3.252, 0.492, 895794, 0.533],
         [0.255, 1.259, 3.848, 0.538, 227037, 0.765],
         [0.232, 0.398, 2.451, 0.359, 393392, 0.536],
         [0.395, 1.031, 3.461, 0.264, 16759, 0.62]])

    random_behaviour = np.random.randint(low=0, high=9)

    # Exploration -----------------------------------------------------------
    # PSO V1
    args.w_inertia_v1 = exploration_mat[random_behaviour, 0]
    args.c1_v1 = exploration_mat[random_behaviour, 1]
    args.c2_v1 = exploration_mat[random_behaviour, 2]

    # MCS V1
    args.frac_dis_v1 = exploration_mat[random_behaviour, 3]
    args.max_step_v1 = exploration_mat[random_behaviour, 4]
    args.pwr_v1 = exploration_mat[random_behaviour, 5]

    # Exploitation ----------------------------------------------------------
    # PSO V1
    args.w_inertia_v2 = exploitation_mat[random_behaviour, 0]
    args.c1_v2 = exploitation_mat[random_behaviour, 1]
    args.c2_v2 = exploitation_mat[random_behaviour, 2]

    # MCS V1
    args.frac_dis_v2 = exploitation_mat[random_behaviour, 3]
    args.max_step_v2 = exploitation_mat[random_behaviour, 4]
    args.pwr_v2 = exploitation_mat[random_behaviour, 5]

    return args, random_behaviour