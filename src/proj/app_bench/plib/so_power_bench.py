import numpy as np
import math
from numba import njit

class SOPowerBench(object):
    def __init__(self, args):
        """
        Mechanical benchmark problems class. The functions are based on the
        following paper:

        "A test-suite of non-convex constrained optimization problems
        from the real-world and some baseline results"
        Abhishek Kumar, et al. (2020)
        DOI: 10.1016/j.swevo.2020.100693

        The functions included in this class are:

        - Wind farm layout design
        Args:
            args: argparse object with parameters
        """

        self.n_param = args.n_param

        # Function to optimise
        self.optim_func = None

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

        f = -self.wind_fitness(interval_num, interval, fre, N, x, a,
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
            for j in range(i + 1, n_coord - 1):
                aux = 5 * R - np.sqrt((XX[i] - XX[j]) ** 2 + (YY[i] - YY[j]) ** 2)
                g_i.append(aux)
                ki += 1

        g_i = np.array(g_i)

        # Objective evaluation
        k_1 = np.ones(g_i.shape)

        g_i_p = 0
        for i in range(g_i.shape[0]):
            g_i_p += k_1[i] * (max(0, g_i[i])) ** 2

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

        n_ws = int((rated_speed - cut_in_speed) / 0.3)

        power_eva = 0.0

        for i in range(N):
            for j in range(n_ws):
                v_j_1 = cut_in_speed + (j) * 0.3
                v_j = cut_in_speed + j * 0.3
                power_eva += 1500 * np.exp((v_j_1 + v_j) / 2 -7.5) / \
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
                    def_i = a / ((1 + kappa * dij / R) ** 2)
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
        dij = np.cos(np.deg2rad(theta)) * Tijx + np.sin(np.deg2rad(theta)) * Tijy
        lij = np.sqrt(Tijx ** 2 + Tijy ** 2 - dij ** 2)
        l = dij * kappa + R

        if (upstream_wind_turbine != downstream_wind_turbine) and (dij > 0) and (l > lij - R):
            affected = True

        return affected, dij

    def sopwm_3(self, x):
        """
        RC45: SOPWM for 3-level inverter
        Synchronous Optimal Pulse-Width Modulation (SOPWM) is a rising
        approach to regulate Medium-Voltage (MV) drives. It provides a signif-
        icant reduction of switching frequency without raising the distortion.
        Consequently, it reduces the switching losses which enhances the per-
        formance of the inverter. Over a single fundamental period, switching
        angles are calculated while reducing the distortion of current. SOPWM
        can be cast as a scalable COP. For diﬀerent level inverters, the SOPWM
        problem can be stated in the following way.

            Args:
                x: Design variable with bounds [0, 1]
                self: Object containing related parameters

            Returns:
                obj: Weighted summation of function and constraint values
        """

        # parameters
        m = 0.32
        s = (-1 * np.ones(self.n_param)) ** [i for i in range(self.n_param)]
        k = [5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47, 49, 53,
             55, 59, 61, 65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97]
        k = np.array(k)
        # Objective Function
        su = 0
        for j in range(30):
            su2 = 0
            for l in range(self.n_param):
                su2 += s[l] * np.cos(k[j] * x[l] * np.pi/180)
            su += su2 ** 2 / (k[j] ** 4)
        f = su ** 0.5 / (np.sum(1 / k ** 2) ** 0.5)

        # Inequality and Equality Constraints

        # Penalty factors
        k_1 = 1e2 # Inequality
        k_2 = 1e2 # Equality

        g = 0
        h = 0
        for i in range(self.n_param):
            g_i = x[i] - x[i+1] + 1e-6
            g += k_1 * (max(0, g_i)) ** 2

            h_i = s[i] * np.cos(x[i] * np.pi / 180) - m
            h += k_2 * (h_i) ** 2

        # Objective evaluation
        obj = f + g + h

        return obj