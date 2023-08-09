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
        g_1 = x[1] * P / (2 * x[1] * x[0] + 2 * np.sqrt(2) * x[0] ** 2) - sigma

        g_2 = (x[1] + np.sqrt(2) * x[0]) * P / (2 * x[1] * x[0] + 2 * np.sqrt(2) * x[0] ** 2) - sigma

        g_3 = P / (x[0] + np.sqrt(2) * x[1]) - sigma

        # Objective evaluation
        k_1 = 1e6
        k_2 = 1e6
        k_3 = 1e6

        obj = f + k_1 * (max(0, g_1)) ** 2 + k_2 * (max(0, g_2)) ** 2 + k_3 * (max(0, g_3)) ** 2

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
        k_1 = 1e6
        k_2 = 1e6
        k_3 = 1e6
        k_4 = 1e6

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

        #P = (np.log10(np.log10(8.122 * 1e6 * mu + 0.8)) + 3.55) / 10.04
        P = (np.log10(np.log10(8.122 * 1e6 * mu + 0.8)) - 10.04) / -3.55
        DT = 2 * (10 ** P - 560)
        E_f = 9336 * Q * 0.0307 * 0.5 * DT
        h = (2 * np.pi * 750 / 60) ** 2 * 2 * np.pi * mu / E_f * ((R ** 4) / 4 - (R_0 ** 4) / 4) - 1e-5
        P_0 = 6 * mu * Q / (np.pi * h ** 3) * np.log(R / R_0)
        W = np.pi * P_0 * 0.5 * (R ** 2 - R_0 ** 2) / (np.log(R / R_0) - 1e-5)

        # Objective Function
        f = (Q * P_0 / 0.7 + E_f)/12

        # Inequality Constraints
        # g_1 = 1000 - P_0
        # g_2 = W - 101000
        # g_3 = 5000 - W / (np.pi * (R ** 2 - R_0 ** 2))
        # g_4 = 50 - P_0
        # g_5 = 0.001 - 0.0307 / (386.4 * P_0) * (Q / (2 * np.pi * R * h))
        # g_6 = R - R_0
        # g_7 = h - 0.001

        g_1 = 101000 - W
        g_2 = P_0 - 1000
        g_3 = DT - 50
        g_4 = 0.001 - h
        g_5 = R_0 - R
        g_6 = 0.0307 / (386.4 * P_0) * (Q / (2 * np.pi * R * h)) - 0.001
        g_7 = W / (np.pi * (R ** 2 - R_0 ** 2) + 1e-5) - 5000

        # Objective evaluation
        k_1 = 1e1
        k_2 = 1e1
        k_3 = 1e1
        k_4 = 1e1
        k_5 = 1e1
        k_6 = 1e1
        k_7 = 1e1

        obj = f + k_1 * (max(0, g_1)) ** 2 + k_2 * (max(0, g_2)) ** 2 \
              + k_3 * (max(0, g_3)) ** 2 + k_4 * (max(0, g_4)) ** 2 \
              + k_5 * (max(0, g_5)) ** 2 + k_6 * (max(0, g_6)) ** 2 \
              + k_7 * (max(0, g_7)) ** 2

        return obj

    def wind_farm(self, x):
        """
        RC44: Wind farm layout
        The wind farm layout is a key factor which determines the power
        output of a wind farm during its life cycle. A general target of wind
        farm layout is to maximize the total power output through optimizing
        the locations of wind turbines. The objective function can be described
        as.
            Args:
                x: Design variable with bounds [0, 1]
                self: Object containing related parameters

            Returns:
                obj: Weighted summation of function and constraint values
        """

        # parameters
        interval = 15
        interval_num = np.int(np.floor(360 / interval))
        cut_in_speed = 3.5
        rated_speed = 14
        cut_out_speed = 25
        R = 40
        H = 80
        CT = 0.8
        a = 1 - np.sqrt(1 - CT)
        kappa = 0.01
        minDistance = 5 * R
        N = 15
        X = 2000
        Y = 2000
        k = np.ones(interval_num) * 2
        c = [7, 5, 5, 5, 5, 4, 5, 6, 7, 7, 8, 9.5, 10, 8.5, 8.5, 6.5, 4.6,
             2.6, 8, 5, 6.4, 5.2, 4.5, 3.9]
        fre = [0.0003, 0.0072, 0.0237, 0.0242, 0.0222, 0.0301, 0.0397, 0.0268,
               0.0626, 0.0801, 0.1025, 0.1445, 0.1909, 0.1162, 0.0793,
               0.0082, 0.0041, 0.0008, 0.0010, 0.0005, 0.0013, 0.0031, 0.0085,
               0.0222]

        # Objective Function
        # x are coordinates arrangeed as:
        # [x_1, y_1, x_2, y_2, ..., x_n, y_n]
        f = 0
        for i in range(self.n_param):
            f -= self.wind_fitness(interval_num, interval, fre, N, x, a,
                                   kappa, R, k, c, cut_in_speed, rated_speed,
                                   cut_out_speed)

        # Inequality Constraints
        n_coord = int(self.n_param / 2)
        XX = np.zeros(n_coord)
        YY = np.zeros(n_coord)
        for i in range(n_coord):
            XX[i] = x[2 * i]
            YY[i] = x[2 * i + 1]

        ki = 0
        g_i = []
        for i in range(n_coord):
            for j in range(i + 1, n_coord):

                aux = 5 * R - np.sqrt((XX[i] - XX[j]) ** 2 + (YY[i] - YY[j]) ** 2)
                g_i.append(aux)
                ki += 1

        g_i = np.array(g_i)

        # Objective evaluation
        k_1 = np.ones(g_i.shape) * 1e2

        g_i_p = 0
        for i in range(g_i.shape[0]):
            g_i_p += k_1 * (max(0, g_i[i])) ** 2

        obj = f + g_i_p

        return obj

    def wind_fitness(self, interval_num, interval, fre, N, coordinate,
                   a, kappa, R, k, c, cut_in_speed, rated_speed,
                   cut_out_speed):

        all_power = 0
        for i in range(1, interval_num + 1):
            interval_dir = (i - 0.5) * interval
            power_eva = self.eva_power(i, interval_dir, N, coordinate, a, kappa, R,
                                  k[i-1], c[i-1], cut_in_speed, rated_speed,
                                  cut_out_speed)
            all_power = all_power + power_eva * fre[i-1]

        return all_power

    def eva_power(self, interval_dir_num, interval_dir, N, coordinate,
                  a, kappa, R, k, c, cut_in_speed, rated_speed,
                  cut_out_speed):

        vel_def = self.eva_func_deficit(interval_dir_num, N, coordinate,
                                   interval_dir, a, kappa, R)

        interval_c = np.zeros(N)

        for i in range(N):
            interval_c[i] = c * (1 - vel_def[i])

        n_ws = (rated_speed - cut_in_speed) / 0.3

        power_eva = 0.0

        for i in range(N):
            for j in range(1, n_ws + 1):
                v_j_1 = cut_in_speed + (j-1) * 0.3
                v_j = cut_out_speed + j * 0.3
                power_eva +=  1500 * np.exp((v_j_1 + v_j) / 2 -7.5) / \
                               (5 + np.exp((v_j_1 + v_j) / 2 - 7.5)) * \
                               (np.exp(-(v_j_1 / (interval_c[i]**k))) -
                                np.exp(-(v_j / (interval_c[i]**k))))

            power_eva += 1500 * (np.exp(-(rated_speed / (interval_c[i]**k)))
                               - np.exp(-(cut_out_speed / (interval_c[i]**k))))

        return power_eva

    def eva_func_deficit(self, interval_dir_num, N, coordinate, theta, a, kappa, R):

        vel_def = np.zeros(N)
        for i in range(N):
            vel_def_i = 0
            for j in range(N):
                affected, dij = self.downstream_wind_turbine_is_affected(coordinate, j, i, theta, kappa, R)
                if affected:
                    def_i = a / (1 + kappa * dij / R) ** 2
                    vel_def_i += def_i ** 2
            vel_def[i] = np.sqrt(vel_def_i)

        return vel_def

    @staticmethod
    @njit(cache=True)
    def downstream_wind_turbine_is_affected(coordinate, upstream_wind_turbine,
                                            downstream_wind_turbine, theta,
                                            kappa, R):
        affected = False
        Tijx = (coordinate[2 * downstream_wind_turbine] - coordinate[2 * upstream_wind_turbine])
        Tijy = (coordinate[2 * downstream_wind_turbine + 1] - coordinate[2 * upstream_wind_turbine + 1])
        dij = np.cosd(theta) * Tijx + np.sind(theta) * Tijy
        lij = np.sqrt(Tijx ** 2 + Tijy ** 2 - dij ** 2)
        l = dij * kappa + R

        if (upstream_wind_turbine != downstream_wind_turbine) and (dij > 0) and (l > lij - R):
            affected = True

        return affected, dij