import argparse


def init_proj_args():
    # ----------------------- User Parameters ----------------------- #
    parser = argparse.ArgumentParser()

    # ----------------------- Problem Definition --------------------- #
    parser.add_argument('--n_param', default=30,
                        type=int, metavar='N',
                        help='number of parameters (default: 30)')

    parser.add_argument('--upper_limit', default=15,
                        type=float, metavar='N',
                        help='parameters limit (default: 15)')

    parser.add_argument('--lower_limit', default=-15,
                        type=float, metavar='N',
                        help='parameters limit (default: -15)')

    # ------------- Path Planning Optimisation Definition ------------ #
    parser.add_argument('--xs', default=0,
                        type=float, metavar='N',
                        help='Starting Position x co-ordinate (default: 0)')
    parser.add_argument('--ys', default=0,
                        type=float, metavar='N',
                        help='Starting Position y co-ordinate (default: 0)')
    parser.add_argument('--xt', default=30,
                        type=float, metavar='N',
                        help='Target Position x co-ordinate (default: 30)')
    parser.add_argument('--yt', default=0,
                        type=float, metavar='N',
                        help='Target Position y co-ordinate (default: 0)')

    return parser
