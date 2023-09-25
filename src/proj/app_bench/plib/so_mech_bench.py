import numpy as np
import math
from numba import njit

class SOMechBench(object):
    def __init__(self, args):
        """
        Mechanical benchmark problems class. The functions are based on the
        following paper:

        "A test-suite of non-convex constrained optimization problems
        from the real-world and some baseline results"
        Abhishek Kumar, et al. (2020)
        DOI: 10.1016/j.swevo.2020.100693

        The functions included in this class are:
        - Three bars problem
        - Pressure vessel design
        - Thrust design

        Args:
            args: argparse object with parameters
        """

        self.n_param = args.n_param

        # Function to optimise
        self.optim_func = None

    @staticmethod
    @njit(cache=True)
    def three_bars(x):
        """
        RC20: Three-bar truss design problem
        This optimization problem is taken from civil engineering, and has
        an accidented constrained space. The main objective of this problem
        is to minimize the weight of the bar structures. The constraints of this
        problem are based on the stress constraints of each bar. The resultant
        problem has linear objective function with three non-linear constraints.

            Args:
                x: Design variable with bounds [0, 1]
                self: Object containing related parameters

            Returns:
                obj: Weighted summation of function and constraint values
        """

        # parameters
        l = 100
        P = 2
        sigma = 2

        # Objective Function
        f = l * (x[1] + 2 * np.sqrt(2) * x[0])

        # Constraints
        #g_1 = x[1] * P / (2 * x[1] * x[0] + 2 * np.sqrt(2) * x[0] ** 2) - sigma
        g_1 = (np.sqrt(2) * x[0] + x[1]) * P / (np.sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1]) - sigma
        #g_2 = (x[1] + np.sqrt(2) * x[0]) * P / (2 * x[1] * x[0] + 2 * np.sqrt(2) * x[0] ** 2) - sigma
        g_2 = x[1] * P/ (np.sqrt(2) * x[1] ** 2 + 2 * x[0] * x[1]) - sigma
        #g_3 = P / (x[0] + np.sqrt(2) * x[1]) - sigma
        g_3 = P / (np.sqrt(2) * x[1] + x[0]) - sigma

        # Objective evaluation
        k_1 = 1_000_000

        obj = f + k_1 * ((max(0, g_1)) ** 2 + (max(0, g_2)) ** 2 + (max(0, g_3)) ** 2)

        return obj

    @staticmethod
    @njit(cache=True)
    def vessel_design(x):
        """
        RC18: Pressure vessel design
        The main objective of this problem is to optimize the welding cost,
        material, and forming of a vessel. This problem contains four constraints
         which are needed to be satisﬁed, and four variables are used to
         calculate the objective function: shell thickness (z 1 ),
         head thickness(z 2 ), inner radius (x 3 ), and length of the vessel
         without including the head (x 4 ). This problem can be stated as.

            Args:
                x: Design variable with bounds [0, 1]
                self: Object containing related parameters

            Returns:
                obj: Weighted summation of function and constraint values
        """

        # parameters
        # Shell thickness
        z_1 = int(x[0]) * 0.0625
        # Head thickness
        z_2 = int(x[1]) * 0.0625
        # Inner radius
        x_3 = x[2]
        # Length of the vessel without including the head
        x_4 = x[3]

        # Objective Function
        f = 1.7781 * z_2 * x_3 ** 2 + 0.6224 * z_1 * x_3 * x_4 + \
            3.1661 * z_1 ** 2 * x_4 + 19.84 * z_1 ** 2 * x_3

        # Inequality Constraints
        g_1 = 0.00954 * x_3 - z_2

        g_2 = 0.0193 * x_3 - z_1

        g_3 = x_4 - 240

        g_4 = -np.pi * x_3 ** 2 * x_4 - 4 / 3 * np.pi * x_3 ** 3 + 1296000

        # Objective evaluation
        k_1 = 1e5
        k_2 = 1e5
        k_3 = 1e5
        k_4 = 1e5

        obj = f + k_1 * (max(0, g_1)) ** 2 + k_2 * (max(0, g_2)) ** 2 \
              + k_3 * (max(0, g_3)) ** 2 + k_4 * (max(0, g_4)) ** 2

        return obj

    @staticmethod
    @njit(cache=True)
    def thrust_design(x):
        """
        RC25: Hydro-static Thrust bearing design problem
       The main objective of this design problem is to optimize bearing
       power loss using four design variables. These design variables are oil
       viscosity 𝜇 , bearing radius R, ﬂow rate Q, and recess radius R o . This
       problem contains seven non-linear constraints associated with inlet oil
       pressure, load-carrying capacity, oil ﬁlm thickness, and inlet oil pres-
       sure. The problem is deﬁned as follows..

            Args:
                x: Design variable with bounds [0, 1]
                self: Object containing related parameters

            Returns:
                obj: Weighted summation of function and constraint values
        """

        # parameters
        R = x[0]
        R_0 = x[1]
        Q = x[2]
        mu = x[3]
        gamma = 0.0307
        C = 0.5
        n = -3.55
        C1 = 10.04
        Ws = 101000
        Pmax = 1000
        delTmax = 50
        hmin = 0.001
        gg = 386.4
        N = 750

        #P = (np.log10(np.log10(8.122 * 1e6 * mu + 0.8)) + 3.55) / 10.04
        P = (np.log10(np.log10(8.122 * 1e6 * mu + 0.8)) - C1) / n
        DT = 2 * (10 ** P - 560)
        E_f = 9336 * Q * gamma * C * DT
        h = (2 * np.pi * N / 60) ** 2 * 2 * np.pi * mu / E_f * ((R ** 4) / 4 - (R_0 ** 4) / 4) - 1e-5
        P_0 = (6 * mu * Q / (np.pi * h ** 3)) * np.log(R / R_0)
        W = np.pi * P_0 * 0.5 * (R ** 2 - R_0 ** 2) / (np.log(R / R_0) - 1e-5)

        # Objective Function
        f = (Q * P_0 / 0.7 + E_f)

        # Inequality Constraints
        # g_1 = 1000 - P_0
        # g_2 = W - 101000
        # g_3 = 5000 - W / (np.pi * (R ** 2 - R_0 ** 2))
        # g_4 = 50 - P_0
        # g_5 = 0.001 - 0.0307 / (386.4 * P_0) * (Q / (2 * np.pi * R * h))
        # g_6 = R - R_0
        # g_7 = h - 0.001

        g_1 = Ws - W
        g_2 = P_0 - Pmax
        g_3 = DT - delTmax
        g_4 = hmin - h
        g_5 = R_0 - R
        g_6 = gamma / (gg * P_0) * (Q / (2 * np.pi * R * h)) - 0.001
        g_7 = W / (np.pi * (R ** 2 - R_0 ** 2) + 1e-5) - 5000

        # Objective evaluation
        k_1 = 1e8

        obj = f + k_1 * ((max(0, g_1)) ** 2 + (max(0, g_2)) ** 2 \
              + (max(0, g_3)) ** 2 + (max(0, g_4)) ** 2 \
              + (max(0, g_5)) ** 2 + (max(0, g_6)) ** 2 \
              + (max(0, g_7)) ** 2)

        return obj

    @staticmethod
    @njit(cache=True)
    def himmelblau(x):
        """
    RC32: Himmelblau's Function

        This optimization problem is taken from Mechanical Engineering optimization.

            Args:
                x: Design variable with bounds [0, 1]
                self: Object containing related parameters

            Returns:
                obj: Weighted summation of function and constraint values
        """

        # parameters
        G1 = 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] -\
             0.0022053 * x[2] * x[4]
        G2 = 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + \
             0.0021813 * x[2] * x[2]
        G3 = 9.300961 + 0.0047026 * x[2] * x[4] + 0.00125447 * x[0] * x[2] +\
             0.0019085 * x[2] * x[3]

        # objective function
        f = 5.3578547 * x[2] * x[2] + 0.8356891 * x[0] * x[4] + \
            37.293239 * x[0] - 40792.141

        # constraints
        g1 = -G1
        g2 = G1 - 92
        g3 = 90 - G2
        g4 = G2 - 110
        g5 = 20 - G3
        g6 = G3 - 25

        # Objective evaluation
        k_1 = 1e1

        # objective evaluation (Maximise)
        obj = f + k_1 * ((max(0, g1)) ** 2 + (max(0, g2)) ** 2 \
              + (max(0, g3)) ** 2 + (max(0, g4)) ** 2 \
              + (max(0, g5)) ** 2 + (max(0, g6)) ** 2)

        return obj
