import numpy as np
import math
from numba import njit

class SOPackOpt(object):
    def __init__(self, args):
        """
        Args:
            args: argparse object for obtaining pack optimisation parameters
        """

        # Number of Design Variables
        self.n_param = args.n_param
        self.n_coord = int(self.n_param / 2)

        # radii
        self.robs =np.array(
            [2.0, 2.3, 1.5, 2, 3, 1.5, 2.5, 1.5, 2, 3, 2.5, 2, 1.5, 2, 2,
             1.5, 1, 3.0, 1.5, 0.5, 1.5, 0.7, 2.0, 1.2, 1.5, 0.75, 1.5, 0.8, 1.2,
             0.8, 0.8, 1.1, 0.3, 0.4, 0.5, 1.5, 0.75, 0.45, 1.1, 0.5, 0.6,
             0.8, 0.9, 0.9, 0.4, 0.4, 0.4, 0.4, 0.1, 0.1])

        self.robs = np.concatenate((self.robs, self.robs)) # 100
        self.robs = np.concatenate((self.robs, self.robs)) # 200
        self.robs = np.concatenate((self.robs, self.robs))  # 400

        #self.robs = np.ones(200)

        # Target Position
        self.yt = args.yt
        self.xt = args.xt

        # compute area of each circle
        self.area_circles = np.pi * self.robs ** 2

        # Function to optimise
        self.optim_func = None

    def pack_obj(self, x):
        """
            Implementation of 2D Packing Problem

            Args:
                x: Design variable including x and y coordinates
                self: Object containing related parameters

            Returns:
                obj: Weighted summation of violation and path length
        """

        # Get xi and yi coordinates from design variable x

        yi = x[self.n_coord:]
        xi = x[:self.n_coord]

        # Calculate distance of each circle centre to target point
        objective = 0

        for oi in range(self.n_coord):
            dis_to_target = \
                np.sqrt((xi[oi] - self.xt) ** 2 + (yi[oi] - self.yt) ** 2)

            objective += dis_to_target * self.area_circles[oi]

        violation = self.circles_overlap(xi, yi, self.robs)

        # Objective evaluation
        obj = objective + violation * 100

        return obj

    @staticmethod
    @njit(cache=True)
    def circles_overlap(x, y, robs):
        """
        Computes a violation based on the penetration of points into circle
        obstacles.

        Args:
            x: X coordinate
            y: y coordinate
            robs: Obstacle radius
            c: boundary points

        Returns:
            violation: sum of all points violations
        """

        # Obstacle violation
        violation = 0

        # Loop through each obstacle
        for xi in range(len(x)):

            # Loop through each obstacle
            for xj in range(len(x)):

                # Check if points are the same
                if xi != xj:
                    # Distance from center to center of each circle
                    dist_center = math.sqrt((x[xi] - x[xj]) ** 2 +
                                            (y[xi] - y[xj]) ** 2)

                    # Check if circles are overlapping
                    if dist_center < (robs[xi] + robs[xj]):
                        # Distance from radius
                        dist_rad = abs(dist_center - (robs[xi] + robs[xj]))

                        # Penalty due to overlap
                        violation += dist_rad

        return violation
