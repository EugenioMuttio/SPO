import math
import time
import numpy as np
from mpi4py import MPI
from numba import njit
from lib.other.tools import Timer

class MCSV1(object):
    def __init__(self, args, init, eval_obj, files_man, seed):
        """
        Modified Cuckoo Search Optimiser object based on the work by
        Sean Walton et. al in Swansea University (2011):

        S.Walton, O.Hassan, K.Morgan and M.R.Brown
        "Modified cuckoo search: A new gradient free optimisation
        algorithm" Chaos, Solitons & Fractals Vol 44 Issue 9, Sept 2011
        pp. 710-718 DOI:10.1016/j.chaos.2011.06.004

        General problem parameters:

        n_param: Number of trainable parameters per agent/individual

        max_gen: Maximum generations

        checkpoint: At "n" checkpoints, a state is saved and a MPI
        communication is done.

        MCS Parameters:
        mcs_pop_size: Population size
        frac_dis: Fraction discard
        constrain: Set to 1 if you want the search constrained within
                   param_range, zero otherwise
        max_step: Maximum distance a cuckoo can travel in one step as
                   fraction of search space diagonal
        pwr: Power that step size is reduced by each generation
        flight: Type of random walk (1) Levy flight
        nests_del: Number of eggs deleated each generation
        min_nests: Minimum nests

        Args:
            args: Arguments object created by argparse or main
            init: Population initialisation object
            eval_obj: Function to optimise
            files_man: Path manager object
            seed: Random seed

        """
        # Name
        self.name = 'MCSV1'

        # Problem parameters
        self.n_param = args.n_param

        # Optimiser Parameters
        self.pop_size = args.mcs_pop_size_v1
        self.max_gen = args.mcs_max_gen_v1

        self.checkpoint = args.checkpoint  # saving frequency
        self.n_states = int(self.max_gen / self.checkpoint)  # n updates

        # MCS Parameters
        self.frac_dis = args.frac_dis_v1
        self.constrain = args.constrain_v1
        self.max_step = args.max_step_v1
        self.pwr = args.pwr_v1
        self.flight = args.flight_v1
        self.nests_del = args.nests_del_v1
        self.min_nests = args.min_nests_v1

        # Seed and Device
        self.seed = seed

        # Function to optimise
        self.eval_obj = eval_obj

        # Evaluation function
        try:
            self.optim_func = self.eval_obj.optim_func
        except ValueError:
            print("Not selected function to optimise")
            exit()

        # Save state
        self.nrun = files_man.nrun
        self.run_path = files_man.run_path
        self.his_path = files_man.his_path

        # Population Initialisation
        self.param_range = init.param_range
        init.pop_size = self.pop_size
        # Max porcentage of population from repository
        init.max_pop_from_rep = args.max_pop_from_rep
        # How many samples from rep will be taken
        # Currently a percentage of the pop size
        init.pop_from_rep = int(np.ceil(args.max_pop_from_rep * self.pop_size))
        # Send optim seed to init
        init.seed = self.seed
        # Run init func
        init.init_func()
        # Init samples from rep count
        self.init_count = init.init_count

        self.pop_init = init.pop_init
        # print('MCS init - Pop Shape: ', self.pop_init.shape)

        # Devices
        self.n_devices = args.n_devices

        if self.n_devices > 1:
            # MPI Init
            self.comm = MPI.COMM_WORLD
            # Get my rank
            self.rank = MPI.COMM_WORLD.Get_rank()
            # Number of available ranks
            self.n_ranks = MPI.COMM_WORLD.Get_size()

        # Worker files (True / False)
        self.work_out = args.work_out

        if self.work_out:
            # Log file to return info about optimiser
            # Name
            filename = self.run_path + '/' + 'log.dat'
            log_file = open(filename, "a")
            log_file.write('MCS V1 init - Pop Shape: ')
            log_file.write(str(self.pop_init.shape))
            log_file.write("\n")
            log_file.close()


    @staticmethod
    def func_eval(func, *args):
        """
        Pre stage to evaluate the function, in case arguments or results
        are handled better here.

        Args:
            func: evaluadtion function
            *args: required trainable parameters and additional args

        Returns: loss value

        """
        loss = func(*args)

        return loss

    def run(self):
        """
        Function to activate MCS optimiser. Can be used in a single CPU
        or multiple CPUs by using MPI. It saves states with the
        convergence of results and the best model achieved in each
        state, including time.

        """
        # Training time
        self.start_time = time.time()

        np.random.seed(self.seed)

        # Number of evaluations counter
        self.num_eval = 0

        max_step_vec = \
            np.divide((self.param_range[1, :] - self.param_range[0, :]),
                      self.max_step)

        # Find number of dimensions and number of nests
        n_nests, n_param = self.pop_init.shape

        # Min function eval fi history
        self.fi_history = np.zeros((self.n_states + 1, 1))

        # (0) Generations (1) Evaluations History
        self.gen_history = np.zeros((self.n_states + 1, 2))

        # Allocate matrices for current nest position and fitness
        pi = np.zeros((n_nests, self.n_param))
        Fi = np.zeros(n_nests)

        ptemp = np.zeros(self.n_param)

        for i in range(n_nests):
            pi[i, :] = self.pop_init[i, :]
            Ftemp = self.func_eval(self.optim_func, pi[i, :])
            self.num_eval = self.num_eval + 1

            Fi[i] = Ftemp

        # Plot positions
        ind = np.argmin(Fi)
        # Reshape Fi
        Fi = Fi.reshape(Fi.shape[0], 1)
        # Vectors into matrix form
        Fipi = np.concatenate((Fi, pi), axis=1)
        # Sort by Fi in ascending order
        FipiS = Fipi[Fipi[:, 0].argsort()]

        local_fmin = FipiS[0, 0]
        local_sol = FipiS[0, 1:]

        # Generations counter init
        self.igen = 0
        state_i = 0

        self.fi_history[state_i, 0] = Fi[ind]
        self.gen_history[state_i, 0] = self.igen
        self.gen_history[state_i, 1] = self.num_eval

        ptop = 1 - self.frac_dis

        # Iterate over all generations
        while self.igen < self.max_gen:

            # time elapsed
            self.igen = self.igen + 1

            # a) Sort the current nets in order of fitness
            # Reshape Fi
            Fi = Fi.reshape(Fi.shape[0], 1)
            # Vectors into matrix form
            Fipi = np.concatenate((Fi, pi), axis=1)
            # Sort by Fi in ascending order
            FipiS = Fipi[Fipi[:, 0].argsort()]

            # Decrease number of nests.
            # Many nests just for initial sampling
            n_nests = np.max((self.min_nests, n_nests - self.nests_del))
            n_nests = n_nests.astype(int)
            pi = FipiS[:n_nests, 1:]
            Fi = FipiS[:n_nests, 0]

            n_top = np.max((3, np.round(n_nests * ptop)))
            n_top = n_top.astype(np.int)
            n_discard = np.subtract(n_nests, n_top).astype(np.int)

            # with Timer('(2) Loop over each Cuckoo which has been discarded'):
            pi, Fi, ptemp = \
                self.discarded_loop(n_nests, n_discard, max_step_vec,
                                    ptemp, pi, Fi)

            # with Timer('(3) Loop over each Cuckoo not to be discarded'):
            pi, Fi, ptemp = \
                self.top_loop(n_nests, n_top, n_discard, max_step_vec,
                              ptemp, pi, Fi)

            # with Timer('(2a) Emptying routine from yang and deb'):
            new_nest = self.empty_nests(pi, self.frac_dis)

            for i in range(pi.shape[0]):
                ptemp = new_nest[i, :]

                # Check if position is within bounds
                upper = np.greater(ptemp, self.param_range[1, :])
                lower = np.less(ptemp, self.param_range[0, :])

                if np.any(upper) or np.any(lower):
                    if self.constrain == 1:
                        # Constrain values outside the range, to be at the limit
                        ptemp = self.simple_bounds(np.copy(new_nest[i, :]), self.param_range)
                    else:
                        # The point is outside the bound, so don't continue
                        pass

                else:
                    if not self.ismember(ptemp, pi):

                        # Point valid so update fitness
                        # b) Calculate fitness of egg at ptemp
    
                        fold = Fi[i]
                        Ftemp = self.func_eval(self.optim_func, ptemp)
                        self.num_eval = self.num_eval + 1
    
                        if np.isreal(Ftemp) and np.less(Ftemp, fold):
                            pi[i, :] = ptemp
                            Fi[i] = Ftemp
                        else:
                            pass

            # Record values -------------------------------------------
            if self.igen % self.checkpoint == 0 or \
                    self.igen == self.max_gen:
                state_i += 1

                # Local best selection
                ind = np.argmin(Fi)

                local_best_pi = pi[ind, :]
                local_best_Fi = Fi[ind]

                self.gen_history[state_i, 0] = self.igen
                self.gen_history[state_i, 1] = self.num_eval

                if local_best_Fi < local_fmin:
                    local_fmin = local_best_Fi
                    local_sol = local_best_pi

                # Save Current State
                kill = self.save_state(local_sol, local_fmin)
                if kill is True:
                    break

    def save_state(self, best_sol, fmin):
        """
        Save current state of the optimisation in text file named
        'conv.dat'. The file is organised in each row as follows:

        (0) generations (1) fmin (2) time

        Args:
            n_eval: Number of evaluations
            best_sol: best solution (x) found
            fmin: optimiser function evaluation (fitness)

        Returns:
            A state file named 'conv.dat'
        """

        # Save state -------------------------------------------------
        if self.work_out:
            filename = self.run_path + '/' + 'model.dat'
            np.savetxt(filename, best_sol, fmt='%.10f', newline=" ")

            # State time
            state_time = time.time()

            measured_time = state_time - self.start_time

            hours, rem = divmod(measured_time, 3600)
            minutes, seconds = divmod(rem, 60)

            filename = self.run_path + '/' + 'conv.dat'
            conv_file = open(filename, "a")
            conv_file.write(str(self.igen))
            conv_file.write("\t")
            conv_file.write(str(self.num_eval))
            conv_file.write("\t")
            conv_file.write(str(fmin))
            conv_file.write("\t")
            conv_file.write(
                str("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),
                                                    int(minutes),
                                                    seconds)))
            conv_file.write("\n")
            conv_file.close()

            filename = self.his_path + '/' + 'model_his.dat'
            sol_file = open(filename, "a")
            for i in best_sol:
                sol_file.write(str(i))
                sol_file.write("\t")
            sol_file.write("\n")
            sol_file.close()

        # Communicate with controller ---------------------------------
        # send data to controller
        if self.n_devices > 1:
            data = [self.nrun, self.rank, fmin, best_sol, 
                    self.name, self.init_count]
            self.comm.send(data, dest=0, tag=2000)

            # receive controller response
            kill_run = self.comm.recv(source=0, tag=4000)
            if kill_run is True:
                #print("MCS Run Killed")
                return True
            else:
                return False

    @staticmethod
    @njit(cache=True)
    def levy_walk(n_param, flight):
        # Function to produce a levy random walk of
        # NoSteps in self.n_param dimensions

        # Allocate solution matrix

        dx = np.zeros(n_param)

        # Yang-Deb Levy
        if flight == 1:
            # Yang-Deb Levy
            beta = 3 / 2
            beta_2 = np.multiply(beta, 0.5)

            sigma_aux_1 = np.multiply(np.pi, beta_2)
            sigma_aux_2 = np.multiply((beta - 1), 0.5)

            gamma_1 = math.gamma(1 + beta)
            gamma_2 = np.multiply(math.gamma((1 + beta) * 0.5), beta)

            sigma_aux_3 = np.divide(
                np.multiply(gamma_1, np.sin(sigma_aux_1)),
                np.multiply(gamma_2, np.power(2, sigma_aux_2)))

            sigma = np.power(sigma_aux_3, np.divide(1, beta))

            u = np.random.randn(n_param) * sigma
            v = np.random.randn(n_param)
            dx_aux = np.ones(n_param) * np.divide(1, beta)
            dx = np.divide(u, np.multiply(np.sign(v),
                                          np.power(np.abs(v), dx_aux)))

        return dx

    @staticmethod
    @njit(cache=True)
    def simple_bounds(s, vardef):
        # Apply lower bound
        ns_tmp = s
        for si in range(s.shape[0]):
            if ns_tmp[si] < vardef[0, si]:
                ns_tmp[si] = vardef[0, si]

        # Upper bounds
        for si in range(s.shape[0]):
            if ns_tmp[si] > vardef[1, si]:
                ns_tmp[si] = vardef[1, si]

        # Update this new move
        s = ns_tmp
        return s

    @staticmethod
    @njit(cache=True)
    def ismember(vec, mat):

        bool = False
        for i in range(mat.shape[0]):
            if np.all(vec == mat[i, :]):
                bool = True
                break

        return bool

    @staticmethod
    @njit(cache=True)
    def empty_nests(nest, frac_dis):
        # A fraction of worse nests are discovered with a probability pa
        n = nest.shape[0]
        # Discovered or not -- a status vector
        K = np.random.rand(nest.shape[0], nest.shape[1]) > frac_dis

        # In the real world, if a cuckoo's egg is very similar to a host's egg
        # then, this cuckoo's egg is less likely to be discovered, thus the fitness
        # should be related to the difference in solutions. Therefore, it is a good
        # idea to do a random walk in a biased way with some random step sizes

        # New solution by biased/selective random walks
        ind_1 = np.random.permutation(n)
        ind_2 = np.random.permutation(n)

        stepsize = np.random.rand(1)[0] * (
                    nest[ind_1, :] - nest[ind_2, :])
        new_nest = np.add(nest, stepsize * K)

        return new_nest

    def discarded_loop(self, n_nests, n_discard, max_step_vec, ptemp,
                       pi, Fi):

        # (2) Loop over each Cuckoo which has been discarded
        for i in range(n_discard):
            a = np.divide(max_step_vec, np.power(self.igen + 1, self.pwr))

            # a) Random Walk
            dx = self.levy_walk(self.n_param, self.flight)

            for j in range(self.n_param):
                # Random number to determine direction
                rand_sign = \
                    np.add(np.multiply(-1.0, np.random.rand(1)), 0.5)

                ptemp[j] = np.add(np.multiply(a[j], dx[j]),
                                  np.multiply(np.sign(rand_sign),
                                              pi[n_nests - i - 1, j]))

            # Check if position is within bounds
            upper = np.greater(ptemp, self.param_range[1, :])
            lower = np.less(ptemp, self.param_range[0, :])

            if np.any(upper) or np.any(lower):
                if self.constrain == 1:
                    # Constrain values outside the range, to be at the limit
                    ptemp = self.simple_bounds(ptemp, self.param_range)
                else:
                    # The point is outside the bound, so don't continue
                    pass

            else:
                # Valid point -> fitness update

                # b) Calculate fitness of egg at ptemp
                Ftemp = self.func_eval(self.optim_func, ptemp)
                self.num_eval = self.num_eval + 1

                if np.isreal(Ftemp):
                    pi[n_nests - i - 1, :] = ptemp
                    Fi[n_nests - i - 1] = Ftemp
                else:
                    pass

        return pi, Fi, ptemp

    def top_loop(self, n_nests, n_top, n_discard, max_step_vec, ptemp,
                 pi, Fi):

        # (3) Loop over each Cuckoo not to be discarded
        for c in range(n_top):

            # Pick one of the top eggs to cross with
            rand_nest = \
                np.round(np.multiply((n_top - 1),
                                     np.random.rand(1)[0])).astype(int)

            if rand_nest == c:

                # Cross with egg outside elite
                rand_nest = n_nests - np.round(np.multiply(
                    (n_discard - 1), np.random.rand(1)[0])).astype(
                    int) - 2  # - 2 guarantees having more flights down

                dist = np.subtract(pi[rand_nest, :], pi[c, :])

                # Multiply distance by a random number
                dist = np.multiply(dist, np.random.rand(self.n_param))
                ptemp = np.add(pi[c, :], dist[:])

                if self.ismember(ptemp, pi):

                    # *Algorithm likes to enter here having same nest to
                    # fine tuning using a random walk.

                    # Perform random walk
                    a = np.divide(max_step_vec, np.power(
                        self.igen + 1, (2 * self.pwr)))

                    dx = self.levy_walk(self.n_param, self.flight)

                    for j in range(self.n_param):
                        # Random number to determine direction
                        rand_sign = \
                            np.add(np.multiply(
                                -1.0, np.random.rand(1)), 0.5)

                        ptemp[j] = np.add(np.multiply(a[j], dx[j]),
                                          np.multiply(
                                              np.sign(rand_sign),
                                              pi[rand_nest, j]))

            else:
                # Top egg "c" is different than random top egg "rand_nest"
                if Fi[rand_nest] > Fi[c]:

                    # Calculate distance
                    dist = np.subtract(pi[c, :], pi[rand_nest, :])

                    # Multiply distance by a random number
                    dist = np.multiply(dist, np.random.rand(self.n_param))
                    ptemp = np.add(pi[c, :], dist[:])

                    if self.ismember(ptemp, pi):

                        # Cross with egg outside elite

                        CI = n_nests - np.round(np.multiply(
                            (n_discard - 1),
                            np.random.rand(1)[0])).astype(int) - 2

                        dist = np.subtract(pi[rand_nest, :], pi[CI, :])

                        # Multiply distance by a random number
                        dist = np.multiply(dist, np.random.rand(self.n_param))
                        ptemp = np.add(pi[c, :], dist[:])

                        if self.ismember(ptemp, pi):

                            # Perform random walk instead
                            a = np.divide(max_step_vec, np.power(
                                self.igen + 1, (2 * self.pwr)))

                            dx = self.levy_walk(self.n_param, self.flight)

                            for j in range(self.n_param):
                                # Random number to determine direction
                                rand_sign = np.add(
                                    np.multiply(-1.0,
                                                np.random.rand(1)), 0.5)

                                ptemp[j] = np.add(
                                    np.multiply(a[j], dx[j]),
                                    np.multiply(np.sign(rand_sign),
                                                pi[rand_nest, j]))

                elif Fi[c] > Fi[rand_nest]:

                    # Calculate distance
                    dist = np.subtract(pi[rand_nest, :], pi[c, :])

                    # Multiply distance by a random number
                    dist = np.multiply(dist, np.random.rand(self.n_param))
                    ptemp = np.add(pi[c, :], dist[:])

                    if self.ismember(ptemp, pi):
                        # Cross with egg outside elite
                        rand_nest = n_nests - np.round(np.multiply(
                            (n_discard - 1),
                            np.random.rand(1)[0])).astype(np.int) - 2

                        dist = np.subtract(pi[rand_nest, :], pi[c, :])

                        # Multiply distance by a random number
                        dist = np.multiply(dist, np.random.rand(self.n_param))
                        ptemp = np.add(pi[c, :], dist[:])

                        if self.ismember(ptemp, pi):

                            # Perform random walk instead
                            a = np.divide(max_step_vec, np.power(self.igen + 1,
                                                      (2 * self.pwr)))

                            dx = self.levy_walk(self.n_param, self.flight)

                            for j in range(self.n_param):
                                # Random number to determine direction

                                rand_sign = np.add(
                                    np.multiply(-1.0,
                                                np.random.rand(1)), 0.5)

                                ptemp[j] = np.add(
                                    np.multiply(a[j], dx[j]),
                                    np.multiply(np.sign(rand_sign),
                                                pi[rand_nest, j]))

                else:
                    # Fitness of both top eggs are equal Fi[c] == Fi[rand_nest]
                    # Calculate distance
                    dist = np.subtract(pi[rand_nest, :], pi[c, :])

                    # Multiply distance by a random number
                    dist = np.multiply(dist, np.random.rand(self.n_param))
                    ptemp = np.add(pi[c, :], dist[:])

                    if self.ismember(ptemp, pi):
                        # Cross with egg outside elite
                        rand_nest = n_nests - np.round(np.multiply(
                            (n_discard - 1),
                            np.random.rand(1)[0])).astype(np.int) - 2

                        dist = np.subtract(pi[rand_nest, :], pi[c, :])

                        # Multiply distance by a random number
                        dist = np.multiply(dist, np.random.rand(self.n_param))
                        ptemp = np.add(pi[c, :], dist[:])

                        if self.ismember(ptemp, pi):

                            # Perform random walk instead
                            a = np.divide(max_step_vec, np.power(
                                self.igen + 1, (2 * self.pwr)))

                            dx = self.levy_walk(self.n_param, self.flight)

                            for j in range(self.n_param):
                                # Random number to determine direction

                                rand_sign = np.add(
                                    np.multiply(-1.0,
                                                np.random.rand(1)), 0.5)

                                ptemp[j] = np.add(
                                    np.multiply(a[j], dx[j]),
                                    np.multiply(np.sign(rand_sign),
                                                pi[rand_nest, j]))

            # Check if position is within bounds
            upper = np.greater(ptemp, self.param_range[1, :])
            lower = np.less(ptemp, self.param_range[0, :])

            if np.any(upper) or np.any(lower):
                if self.constrain == 1:
                    # Constrain values outside the range, to be at the limit
                    ptemp = self.simple_bounds(ptemp, self.param_range)
                else:
                    # The point is outside the bound, so don't continue
                    pass

            else:
                # Valid point -> fitness update

                # b) Calculate fitness of egg at ptemp
                Ftemp = self.func_eval(self.optim_func, ptemp)
                self.num_eval = self.num_eval + 1

                # c) Select random nest and replace/update
                # position if fitness is better

                # Select random index
                rand_nest = np.round(
                    np.multiply((n_nests - 1),
                                np.random.rand(1)[0])).astype(np.int)

                if np.greater(Fi[rand_nest], Ftemp) and \
                        np.isreal(Ftemp):

                    pi[rand_nest, :] = ptemp
                    Fi[rand_nest] = Ftemp
                else:
                    pass

        return pi, Fi, ptemp
