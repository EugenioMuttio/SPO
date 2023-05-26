import numpy as np
import math
from numba import njit

class SOPathPlanning(object):
    def __init__(self, args):
        """
        Args:
            args: argparse object for obtaining path planning parameters
        """

        # Start Position
        self.ys = args.ys
        self.xs = args.xs

        # Target Position
        self.yt = args.yt
        self.xt = args.xt

        # Total distance
        self.total_dist = self.xt - self.xs

        # Subdomain
        self.n_sub_dom = 4
        self.sub_dom_range = self.total_dist / self.n_sub_dom

        # Number of Design Variables
        self.n_param = args.n_param

        # Circle Obstacles
        self.xobs = np.array(
            [2.5, 3.5, 2.5, 6, 6.5, 7, 8, 12, 14, 14.5, 15, 21, 22.5, 23, 27,
             19, 20, 27, 25, 17, 12, 11, 20, 11, 13, 18, 23, 10, 20,
             22, 28, 17, 18, 9, 11, 7.5, 12, 10.5, 25, 18, 16, 27, 28,
             5, 2.5, 3.5, 5, 4])  # circle centre x-coords
        self.yobs = np.array(
            [-5, 7.5, -0.5, 3, -8, 6.5, -2, 1, 4, -4, 10, 0, -3.5, 3, -1,
             5, -5, 7.5, -6.0, 2.5, 8, 5.5, -7.5, -8.5, -9.0, -8.0, 10, 3.5, 10,
             7.5, 2.5, 0.0, -2.5, -5, -6.5, 10, 12, 10, -9, 7.5, -9, -6,
             -8, 11, 2.5, 4.0, -3.5, -2])  # circle centre y-coords
        self.robs = np.array(
            [2.0, 2.3, 1.5, 2, 3, 1.5, 2.5, 1.5, 2, 3, 2.5, 2, 1.5, 2, 2,
             1.5, 1, 3.0, 1.5, 0.5, 1.5, 0.7, 2.0, 1.2, 1.5, 0.75, 1.5, 0.8, 1.2,
             0.8, 0.8, 1.1, 0.3, 0.4, 0.5, 1.5, 0.75, 0.45, 1.1, 0.5, 0.6,
             0.8, 0.9, 0.9, 0.4, 0.4, 0.4, 0.4])  # radii

        # Sorted circled obstacles w.r.t X coordinate
        index_sort = self.xobs.argsort()
        self.xobs_sort = self.xobs[index_sort]
        self.yobs_sort = self.yobs[index_sort]
        self.robs_sort = self.robs[index_sort]

        # maximum radius
        self.max_robs = np.max(self.robs)

        # Fix spacing in x direction
        self.x = np.linspace(self.xs, self.xt, num=self.n_param)

        # intermediate points
        self.inter_points = 5

        # Get x intermediate points
        self.xi = self.intermediate_points(self.x, self.inter_points)

        # subdomain computation
        self.sub_dom, self.sub_dom_x = \
            self.subdomain(self.xs, self.xi, self.xobs_sort, self.max_robs,
                           self.sub_dom_range, self.n_sub_dom)

        # Function to optimise
        self.optim_func = None

    def path_2d(self, y):
        """
            Implementation of 2D Path Planning Optimisation.

            Args:
                y: Design variable
                self: Object containing related parameters

            Returns:
                obj: Weighted summation of violation and path length
        """

        # Fix start and end points
        y[0] = self.ys
        y[-1] = self.yt

        # Calculate total length with fix x and y
        length = np.sum(np.sqrt((self.x[:-1] - self.x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2))

        # Get y intermediate points
        yi = self.intermediate_points(y, self.inter_points)

        # Circle violation
        violation = \
            self.points_in_circles(self.xi, yi, self.xobs_sort, self.yobs_sort,
                                   self.robs_sort, self.n_sub_dom,
                                   self.sub_dom, self.sub_dom_x)

        # Difference between total length and straight line from P_i and P_f

        # Straight line
        line_lenght = np.sqrt((self.xt - self.xs) ** 2 + (self.yt - self.ys) ** 2)

        diff_lengths = np.abs(length - line_lenght)

        # Objective evaluation
        k = 1
        obj = length + k * violation

        return obj

    @staticmethod
    @njit(cache=True)
    def points_in_circles(x, y, xobs, yobs, robs, n_sub_dom, sub_dom_obs, sub_dom_point):
        """
        Computes a violation based on the penetration of points into circle
        obstacles. This is a faster version that make use of subdomains for
        the points' location (sub_dom_x) and obstacles (sub_dom).

        Args:
            x: X coordinate
            y: y coordinate
            xobs: Obstacle centre X coordinate
            yobs: Obstacle centre Y coordinate
            robs: Obstacle radius
            n_sub_dom: number of subdomains
            sub_dom_obs: obstacles subdomain matrix
            sub_dom_point: points subdomain matrix

        Returns:
            violation: sum of all points violations
        """

        # Obstacle violation
        violation = 0

        # Check if points lie within obstacles
        xdi = 0  # Point index stored in sub_dom_point
        cdi = 0  # Obstacle index stored in sub_dom_obs

        # Loop through each subdomain
        for di in range(n_sub_dom):

            # Loop through each point in sub_dom_point di
            while sub_dom_point[di, xdi] > -1:

                # For each point, a verification is done by a
                # loop through each obstacle in sub_dom_obs di
                while sub_dom_obs[di, cdi] > -1:

                    # Recover point index from sub_dom_point
                    xi = sub_dom_point[di, xdi]
                    # Recover obstacle index from sub_dom_obs
                    ci = sub_dom_obs[di, cdi]

                    # Point x coordinate within an obstacle
                    if (xobs[ci] - robs[ci]) < x[xi] < (xobs[ci] + robs[ci]):

                        # Distance from point to center of circle
                        dist_center = math.sqrt((x[xi] - xobs[ci]) ** 2 +
                                                (y[xi] - yobs[ci]) ** 2)

                        # Distance considering y coordinate being inside
                        if dist_center < robs[ci]:
                            # Distance from radius
                            dist_rad = abs(robs[ci] - dist_center)

                            # Penalty due to penetration
                            # Normalised w.r.t obstacle radius
                            violation += dist_rad / robs[ci]
                    # Next obstacle counter in same sub_domain_obs
                    cdi += 1
                # Re initialise obstacle counter for next sub_domain_obs
                cdi = 0

                # Next point counter in same sub_domain_point
                xdi += 1
            # Re initialise point counter for next sub_domain_point
            xdi = 0

        return violation

    @staticmethod
    def subdomain(xs, x, xobs, max_rob, sub_dom_range, n_sub_dom):
        """
        Defines subdomains from n_sub_dom to make faster the objective
        verification.
        It lets a max radius safety to check circles that can have their
        radius in neighbour subdomain but affects current domain
        Args:
            xs: x coordinate start position
            x: x coordinates
            xobs: obstacle position
            max_rob: maximum radius
            sub_dom_range: length in x of subdomain
            n_sub_dom: number of divisions

        Returns:
            An array with the index of the circle in each subdomain
        """

        # Initialise sub_dom left boundary with initial position in x
        sub_dom_left = xs

        # Subdomain arrays set to -1
        sub_dom_obs = np.ones((n_sub_dom, len(xobs)), dtype=int) * -1
        sub_dom_point = np.ones((n_sub_dom, len(x)), dtype=int) * -1

        # Init first point
        sub_dom_point[0, 0] = 0

        cdi = 0  # Obstacle index stored in sub_dom_obs
        xdi = 1  # Point index stored in sub_dom_point

        # Loop through each subdomain
        for di in range(n_sub_dom):

            # Loop through obstacles
            for ci in range(len(xobs)):
                # Veryfing subdomain left and right boundary (including max_rob)
                if (sub_dom_left - max_rob) < xobs[ci] <= (sub_dom_left + sub_dom_range + max_rob):
                    sub_dom_obs[di, cdi] = ci
                    cdi += 1
            cdi = 0

            for xi in range(len(x)):
                # Veryfing subdomain left and right boundary (including max_rob)
                if (sub_dom_left) < x[xi] <= (sub_dom_left + sub_dom_range):
                    sub_dom_point[di, xdi] = xi
                    xdi += 1
            xdi = 0

            # Update subdomain left boundary
            sub_dom_left += sub_dom_range

        return sub_dom_obs, sub_dom_point

    @staticmethod
    def intermediate_points(x, n):
        """
        Compute intermediate points between original point locations
        Args:
            x: array of original locations
            n: number of intermediate points

        Returns:
            xi: array of original points and intermediate points
        """

        xi = [0] * (n * (len(x) - 1))
        c = 0

        for i in range(len(x) - 1):
            for j in range(n):  # store intermediate points
                xi[c] = x[i] * (n - j) / (n + 1) + x[i + 1] * (1 + j) / (n + 1)
                c += 1

        xi = np.concatenate((x, xi), axis=0)

        return xi
