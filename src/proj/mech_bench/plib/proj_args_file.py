import argparse


def init_proj_args():
    # ----------------------- User Parameters ----------------------- #
    parser = argparse.ArgumentParser()

    # ----------------------- Problem Definition --------------------- #
    parser.add_argument('--n_param', default=30,
                        type=int, metavar='N',
                        help='number of parameters (default: 30)')

    parser.add_argument('--upper_limit_1', default=15,
                        type=float, metavar='N',
                        help='parameters limit (default: 15)')

    parser.add_argument('--lower_limit_1', default=-15,
                        type=float, metavar='N',
                        help='parameters limit (default: -15)')

    parser.add_argument('--upper_limit_2', default=15,
                        type=float, metavar='N',
                        help='parameters limit (default: 15)')

    parser.add_argument('--lower_limit_2', default=-15,
                        type=float, metavar='N',
                        help='parameters limit (default: -15)')

    return parser
