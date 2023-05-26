import time
from mpi4py import MPI


class Timer(object):
    """
    Compute the time of computation.

    Usage:
        Use the following sentence before computation to measure time
        'with Timer('String'):' where string is a key name
    """

    def __init__(self, name=None, flag=0):
        self.name = name
        self.flag = flag

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        rank = MPI.COMM_WORLD.Get_rank()
        if self.flag == 0:
            if rank == 0:
                if self.name:
                    print('[%s]' % self.name, )

                print('Elapsed: %s' % (time.time() - self.tstart))