# import math
import time
import numpy as np
from mpi4py import MPI
# from numba import njit
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import ElementwiseProblem
from pymoo.util.normalization import denormalize


class PymooCMAES(object):
    def __init__(self, args, init, eval_obj, files_man, seed):
        """
        Class object to run a CMA-ES optimiser from Pymoo library

        General problem parameters:

        n_param: Number of trainable parameters per agent/individual

        max_gen: Maximum generations

        checkpoint: At "n" checkpoints, a state is saved and an MPI
        communication is done.

        Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES) Parameters:
        cmaes_pop_size: Population size
        sigma: initial standard deviation in each coordinate
        restarts: number of restarts with increasing population size


        https://pymoo.org/algorithms/soo/cmaes.html

        Args:
            args: Arguments object created by argparse or main
            init: Population initialisation object
            eval_obj: Function to optimise
            files_man: Path manager object
            seed: Random seed

        """
        # Name
        self.name = 'PymooCMAES'

        # Problem parameters
        self.n_param = args.n_param

        # Optimiser Parameters
        self.pop_size = args.cmaes_pop_size
        self.max_gen = args.cmaes_max_gen

        self.checkpoint = args.checkpoint  # saving frequency
        self.n_states = int(self.max_gen / self.checkpoint)  # n updates

        self.n_obj = args.n_obj

        # CMAES Parameters
        self.sigma = args.sigma
        self.restarts = args.restarts

        # Seed and Device
        self.seed = seed[0]

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
        #print('CMAES init - Pop Shape: ', self.pop_init.shape)

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
            log_file.write('CMAES init - Pop Shape: ')
            log_file.write(str(self.pop_init.shape))
            log_file.write("\n")
            log_file.write('Seed: ')
            log_file.write(str(self.seed))
            log_file.write("\n")
            log_file.write(str(self.pop_size))
            log_file.write("\n")
            log_file.write(str(self.max_gen))
            log_file.close()

        # Pymoo problem for specific arguments and initialisation
        self.pymoo_prob = self.PymooProb(optim_func=self.optim_func,
                                         n_var=self.n_param,
                                         n_obj=self.n_obj,
                                         xl=self.param_range[0, :],
                                         xu=self.param_range[1, :])

        x0 = denormalize(np.random.random(self.n_param), self.param_range[0, :], xu=self.param_range[1, :])
        self.algorithm = CMAES(x0=x0,
                               sigma=self.sigma,
                               restarts=self.restarts,
                               maxfevals=np.inf,
                               tolfun=1e-16,
                               tolx=1e-16,
                               restart_from_best=True,
                               bipop=True)

        # Change algorithm setup to never terminates
        self.algorithm.setup(self.pymoo_prob,
                             termination=('n_gen', self.max_gen),
                             seed=self.seed, verbose=False)

    class PymooProb(ElementwiseProblem):
        """
        Pymoo inner class to define the problem to solve using the
        ask and tell / functional framework.
        The eval_obj is structured to obtain a single output per
        agent, then this class is derived from ElementWiseProblem
        """
        def __init__(self, optim_func, n_var, n_obj, xl, xu):
            super().__init__(n_var=1,
                             n_obj=1,
                             xl=0,
                             xu=1)

            self.eval_obj = optim_func
            self.n_var = n_var
            self.n_obj = n_obj
            self.xl = xl
            self.xu = xu

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = self.eval_obj(x)

    def run(self):
        """
        Function to activate pymoo optimiser.
        """

        # Training time
        self.start_time = time.time()

        np.random.seed(self.seed)

        # Number of evaluations counter
        self.num_eval = 0

        # Min function eval fi history
        self.fi_history = np.zeros((self.n_states + 2, 1))

        # (0) Generations (1) Evaluations History
        self.gen_history = np.zeros((self.n_states + 2, 2))

        # Generations counter init
        self.igen = 0
        state_i = 0
        local_fmin = 0

        # Taking first sample from pop to initialise f min
        local_sol = self.pop_init[0, :]
        local_fmin = np.nan_to_num(self.pymoo_prob.eval_obj(local_sol),
                                   nan=10000)

        while self.algorithm.has_next():

            # ask the algorithm for the next solution to be evaluated
            pop = self.algorithm.ask()

            # evaluate the individuals using the algorithm's evaluator
            # (necessary to count evaluations for termination)
            self.algorithm.evaluator.eval(self.pymoo_prob, pop)

            # returned the evaluated individuals which have been
            # evaluated or even modified
            self.algorithm.tell(infills=pop)

            # Counting generations from 0
            self.igen = self.algorithm.n_gen - 1

            # Record values -------------------------------------------
            if self.igen % self.checkpoint == 0 or \
                    self.igen == self.max_gen:
                state_i += 1

                res = self.algorithm.result()

                local_best_pi = res.X
                local_best_Fi = res.F[0]

                self.num_eval = self.algorithm.evaluator.n_eval

                self.gen_history[state_i, 0] = self.igen
                self.gen_history[state_i, 1] = self.num_eval

                if local_best_Fi < local_fmin or self.igen == 0:
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
            best_sol: the best solution (x) found
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
        if self.n_devices > 1:
            data = [self.nrun, self.rank, fmin, best_sol,
                    self.name, self.init_count]
            self.comm.send(data, dest=0, tag=2000)  # send data to controller

            kill_run = self.comm.recv(source=0, tag=4000)  # receive controller response
            if kill_run is True:
                #print("CMAES Run Killed")
                return True
            else:
                return False
