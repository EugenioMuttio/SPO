import lhsmdu
import numpy as np
import os
from numba import njit

class init(object):
    def __init__(self, args, files_man, seed):

        # Problem parameters
        self.n_param = args.n_param
        self.lower_limit = args.lower_limit
        self.upper_limit = args.upper_limit
        self.pop_size = 0
        self.n_obj = args.n_obj

        self.seed = seed

        # Population Initialisation
        self.param_range = np.zeros((2, self.n_param))
        self.param_range[0, :] = self.lower_limit * np.ones(self.n_param)
        self.param_range[1, :] = self.upper_limit * np.ones(self.n_param)

        self.pop_init = []

        self.func = 'LHS'

        # Max porcentage of population from rep
        self.max_pop_from_rep = args.max_pop_from_rep
        
        # How many samples were taken from repository
        self.init_count = 0

        # Number of population from the repository
        self.pop_from_rep = args.pop_from_rep

        # Probability of initialise from rep
        self.init_prob = args.init_prob

        # Probability of mutate from rep -- Not currently used
        self.init_mut_prob = 0.85
        # Fraction of mutated samples
        self.mut_frac = 0.5
        # Fraction of mutated parameters (per sample)
        self.mut_frac_param = 0.7

        # Files path
        self.files_man = files_man
        self.run_path = files_man.run_path
        self.run_id_path = files_man.run_id_path
        self.his_path = files_man.his_path
        self.plot_path = files_man.plot_path

        # Repository path located at zero (controller)
        self.rep_path = self.run_id_path + '/0/rep.dat'

        # Repository exists flag
        self.rep_exists = False

    def init_func(self):

        # Init from functions -----------------------------------------
        if self.func == 'LHS':
            self.latin_hypercube()
        elif self.func == 'RU':
            self.random_uniform()

        # Init using repository ---------------------------------------
        if self.max_pop_from_rep > 0.0:

            # Verify existence of repository
            self.rep_exists = os.path.exists(self.rep_path)

            if self.rep_exists:

                # Determine number of lines
                rep_file = open(self.rep_path, 'r')
                rep_rows = len(rep_file.readlines())
                rep_file.close()

                if rep_rows > 0:
                    # Load repository
                    rep = np.genfromtxt(self.rep_path)
                    # Check dims
                    if rep.ndim == 1:
                        rep = np.expand_dims(rep, axis=0)

                    # Skipping objectives
                    rep = rep[:, self.n_obj:]  # Taking only solutions

                    # INIT from Rep ------------------------------------------
                    self.init_rep(rep)

            else:
                print('*Repository does not exists yet\n')

    def latin_hypercube(self):
        """
        Returns initial population by using a latin hypercube scheme, scaled
        to the parameter's range
        Args:
            param_range (ndarray): Array containing the range of each
                parameter the lower bound is in the first row [0, :],
                and upper bound is in the second row [1, :]
            pop_size (int): Population size
            seed (int): Random seed variable

        Returns:
            pop_init (ndarray): Array of size (pop_size, n_param)
        """
        # Seed definition
        np.random.seed(self.seed)
        # print("init seed: ", self.seed)

        self.pop_init = np.zeros((self.pop_size, self.n_param))

        # Difference for scaling
        diff = self.param_range[1, :] - self.param_range[0, :]

        norm_pop = lhsmdu.sample(self.pop_size, self.n_param, randomSeed=self.seed)

        # Scale
        for i in range(self.pop_size):
            for j in range(self.n_param):
                self.pop_init[i, j] = self.param_range[0, j] + \
                                 diff[j] * norm_pop[i, j]

    def random_uniform(self):
        """
        Returns initial population by using a uniform distribution scheme,
        scaled to the parameter's range
        Args:
            param_range (ndarray): Array containing the range of each
                parameter the lower bound is in the first row [0, :],
                and upper bound is in the second row [1, :]
            pop_size (int): Population size
            seed (int): Random seed variable

        Returns:
            pop_init (ndarray): Array of size (pop_size, n_param)
        """
        # Seed definition
        np.random.seed(self.seed)
        # print("init seed: ", self.seed)

        self.pop_init = np.zeros((self.pop_size, self.n_param))

        # Difference for scaling
        diff = self.param_range[1, :] - self.param_range[0, :]

        norm_pop = np.random.random([self.pop_size, self.n_param])

        # Scale
        for i in range(self.pop_size):
            for j in range(self.n_param):
                self.pop_init[i, j] = self.param_range[0, j] + \
                                 diff[j] * norm_pop[i, j]


    def init_rep(self, rep):
        """
        Initialise population using samples from repository
        Args:
            rep: Current repository

        Returns:
            Population with random samples from repository
        """
        # Number of population introduced, should not be
        # larger than repository
        rep_size = rep.shape[0]
        if self.pop_from_rep > rep_size:
            self.pop_from_rep = rep_size

        # Number of introduced samples will be random
        rand_pop_from_rep = \
            np.random.randint(low=0, high=self.pop_from_rep)

        # Probability to ensure some optimisers start with
        # rep and others keep current initialisation.
        if np.random.rand() < self.init_prob:
            # print("*Init from rep\n")

            # Loop to introduce samples from repository
            self.init_count = 0
            for pi in range(rand_pop_from_rep):
                # Choose a model from rep
                rep_rand_model = \
                    np.random.randint(0, rep_size)

                # Choose a random model from current pop_init
                init_rand_model = \
                    np.random.randint(0, self.pop_size)

                # Replace initialisation
                self.pop_init[init_rand_model, :] = \
                    rep[rep_rand_model, :]

                self.init_count += 1

    @staticmethod
    def mutate_rep(pop, rep, mut_frac, mut_frac_param):

        rep_size = rep.shape[0]
        pop_size = pop.shape[0]
        n_param = pop.shape[1]

        perm_from_rep = np.random.permutation(rep_size)

        if rep_size > pop_size:
            perm_from_rep = perm_from_rep[:pop_size]
            # Assigning all repository to mutated population
            pop[:, :] = rep[perm_from_rep, :]
        else:
            pop[perm_from_rep, :] = rep[perm_from_rep, :]

        # A fraction of population parameters is considered
        pop_bool_frac = \
            np.random.rand(pop_size, n_param) < mut_frac_param

        # Mutate just part of the population
        n_frac = int(mut_frac * pop_size)

        #pop_frac = np.random.randint(low=0, high=pop_size, size=[n_frac])

        pop_frac = np.random.permutation(pop_size)
        pop_frac = pop_frac[n_frac:]

        pop_bool_frac[pop_frac, :] = False

        # Biased solution by a random walk distance
        ind_1 = np.random.permutation(pop_size)
        ind_2 = np.random.permutation(pop_size)

        # Distance between random solutions
        stepsize = \
            np.random.rand(1)[0] * \
            (pop[ind_1, :] - pop[ind_2, :])

        # Random walk
        mut_pop = np.add(pop, stepsize * pop_bool_frac)

        return mut_pop