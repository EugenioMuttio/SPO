import numpy as np
import math
from numba import njit


class SOChemBench(object):
    def __init__(self, args):
        """
        Industrial chemical process benchmark problems class. The functions are based on the
        following paper:

        "A test-suite of non-convex constrained optimization problems
        from the real-world and some baseline results"
        Abhishek Kumar, et al. (2020)
        DOI: 10.1016/j.swevo.2020.100693

        The functions included in this class are:
        - Propane, isobutane, n-butane nonsharp separation
        Args:
            args: argparse object with parameters
        """

        self.n_param = args.n_param

        # Function to optimise
        self.optim_func = None

    @staticmethod
    @njit(cache=True)
    def nonsharp_sep(x):
        """
    RC07: Propane, isobutane, n-butane nonsharp separation

        This optimization problem is taken from Industrial chemical process
        optimization.

            Args:
                x: Design variable with bounds [0, 1]
                self: Object containing related parameters

            Returns:
                obj: Weighted summation of function and constraint values
        """

        # parameters
        c11 = 0.23947
        c12 = 0.75835
        c21 = -0.0139904
        c22 = -0.0661588
        c31 = 0.0093514
        c32 = 0.0338147
        c41 = 0.0077308
        c42 = 0.0373349
        c51 = -0.0005719
        c52 = 0.0016371
        c61 = 0.0042656
        c62 = 0.0288996

        # objective function
        f = c11 + (c21 + c31 * x[23] + c41 * x[27] + c51 * x[32] + c61 * x[33]) * x[4] + c12 + (
                c22 + c32 * x[25] + c42 * x[30] + c52 * x[37] + c62 * x[38]) * x[12]

        # constraints
        h1 = x[3] + x[2] + x[1] + x[0] - 300
        h2 = x[5] - x[7] - x[6]
        h3 = x[8] - x[11] - x[9] - x[10]
        h4 = x[13] - x[16] - x[14] - x[15]
        h5 = x[17] - x[19] - x[18]
        h6 = x[5] * x[20] - x[23] * x[24]
        h7 = x[13] * x[21] - x[25] * x[26]
        h8 = x[7] * x[22] - x[27] * x[28]
        h9 = x[17] * x[29] - x[30] * x[31]
        h10 = x[24] - x[4] * x[32]
        h11 = x[28] - x[4] * x[33]
        h12 = x[34] - x[4] * x[35]
        h13 = x[36] - x[12] * x[37]
        h14 = x[26] - x[12] * x[38]
        h15 = x[31] - x[12] * x[39]
        h16 = x[24] - x[5] * x[20] - x[8] * x[40]
        h17 = x[28] - x[5] * x[41] - x[8] * x[22]
        h18 = x[34] - x[5] * x[42] - x[8] * x[43]
        h19 = x[36] - x[13] * x[44] - x[17] * x[45]
        h20 = x[26] - x[13] * x[21] - x[17] * x[46]
        h21 = x[31] - x[13] * x[47] - x[17] * x[29]
        h22 = 0.333 * x[0] + x[14] * x[44] - x[24]
        h23 = 0.333 * x[0] + x[14] * x[21] - x[28]
        h24 = 0.333 * x[0] + x[14] * x[47] - x[34]
        h25 = 0.333 * x[1] + x[9] * x[40] - x[36]
        h26 = 0.333 * x[1] + x[9] * x[22] - x[26]
        h27 = 0.333 * x[1] + x[9] * x[43] - x[31]
        h28 = 0.333 * x[2] + x[6] * x[20] + x[10] * x[40] + x[15] * x[44] + x[18] * x[45] - 30
        h29 = 0.333 * x[2] + x[6] * x[41] + x[10] * x[22] + x[15] * x[21] + x[18] * x[46] - 50
        h30 = 0.333 * x[2] + x[6] * x[42] + x[10] * x[43] + x[15] * x[47] + x[18] * x[29] - 30
        h31 = x[32] + x[33] + x[35] - 1
        h32 = x[20] + x[41] + x[42] - 1
        h33 = x[40] + x[22] + x[43] - 1
        h34 = x[37] + x[38] + x[39] - 1
        h35 = x[44] + x[21] + x[47] - 1
        h36 = x[45] + x[46] + x[29] - 1
        h37 = x[42]
        h38 = x[45]

        # objective evaluation
        k_1 = 1e6
        obj = f + k_1 * (h1 ** 2 + h2 ** 2 + h3 ** 2 + h4 ** 2 + h5 ** 2 + h6 ** 2 + h7 ** 2 + h8 ** 2 + h9 ** 2 + h10 ** 2 \
              + h11 ** 2 + h12 ** 2 + h13 ** 2 + h14 ** 2 + h15 ** 2 + h16 ** 2 + h17 ** 2 + h18 ** 2 + h19 ** 2 \
              + h20 ** 2 + h21 ** 2 + h22 ** 2 + h23 ** 2 + h24 ** 2 + h25 ** 2 + h26 ** 2 + h27 ** 2 + h28 ** 2 \
              + h29 ** 2 + h30 ** 2 + h31 ** 2 + h32 ** 2 + h33 ** 2 + h34 ** 2 + h35 ** 2 + h36 ** 2 + h37 ** 2 \
              + h38 ** 2)

        return obj

    @staticmethod
    @njit(cache=True)
    def heat_exchanger_1(x):
        """
    RC01: Heat exchanger network design (Case 1)

        This optimization problem is taken from Industrial chemical process
         optimization.

            Args:
                x: Design variable with bounds [0, 1]
                self: Object containing related parameters

            Returns:
                obj: Weighted summation of function and constraint values
        """

        # objective function
        f = 35 * (x[0] ** 0.6) + 35 * (x[1] ** 0.6)

        # constraints
        h1 = 200 * x[0] * x[3] - x[2]
        h2 = 200 * x[1] * x[5] - x[4]
        h3 = x[2] - 10000 * (x[6] - 100)
        h4 = x[4] - 10000 * (300 - x[6])
        h5 = x[2] - 10000 * (600 - x[7])
        h6 = x[4] - 10000 * (900 - x[8])
        h7 = x[3] * np.log(x[7] - 100) - x[3] * np.log(600 - x[6]) - x[7] + \
             x[6] + 500
        h8 = x[5] * np.log(x[8] - x[6]) - x[5] * np.log(600) - x[8] + \
             x[6] + 600

        # objective evaluation
        k_1 = 1e1
        obj = f + k_1 * (h1 ** 2 + h2 ** 2 + h3 ** 2 + h4 ** 2 + h5 ** 2 +
                          h6 ** 2 + h7 ** 2 + h8 ** 2)

        return obj

    @staticmethod
    #@njit(cache=True)
    def heat_exchanger_2(x):
        """
    RC02: Heat exchanger network design (Case 2)

        This optimization problem is taken from Industrial chemical process
        optimization.

            Args:
                x: Design variable with bounds [0, 1]
                self: Object containing related parameters

            Returns:
                obj: Weighted summation of function and constraint values
        """

        # objective function
        f = (x[0] / (120 * x[3])) ** 0.6 + (x[1] / (80 * x[4])) ** 0.6 + (
                    x[2] / (40 * x[5])) ** 0.6

        # constraints
        h1 = x[0] - 1e4 * (x[6] - 100)
        h2 = x[1] - 1e4 * (x[7] - x[6])
        h3 = x[2] - 1e4 * (500 - x[7])
        h4 = x[0] - 1e4 * (300 - x[8])
        h5 = x[1] - 1e4 * (400 - x[9])
        h6 = x[2] - 1e4 * (600 - x[10])
        h7 = x[3] * np.log(x[8] - 100) - x[3] * np.log(300 - x[6]) - x[8] - \
             x[6] + 400
        h8 = x[4] * np.log(x[9] - x[6]) - x[4] * np.log(400 - x[7]) - x[9] + \
             x[6] - x[7] + 400
        h9 = x[5] * np.log(x[10] - x[7]) - x[5] * np.log(100) - x[10] - \
             x[7] + 100

        # objective evaluation
        k_1 = 1
        obj = f + k_1 * (h1 ** 2 + h2 ** 2 + h3 ** 2 + h4 ** 2 + h5 ** 2 +
                         h6 ** 2 + h7 ** 2 + h8 ** 2 + h9 ** 2)

        return obj