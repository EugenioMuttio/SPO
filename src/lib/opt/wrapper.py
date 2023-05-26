import math
import sys
import time
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE, ANY_TAG, Status
from numba import njit
import matplotlib.pyplot as plt
import os
from lib.opt.behaviour_pool import init_behaviour

class Wrapper(object):
    def __init__(self, args, init, proj, files_man, seed):
        """
        Wrapper V1

            A controller (rank==0) sends different optimisers to the
            rest of the active ranks. Each rank will notify to the
            controller how good is doing, and when finished or when
            is no longer optimising after some generations, it should
            be stopped and start a new optimiser in that rank, until
            reaching max_runs.

            Run max_runs.
                max_runs >= number of CPUs (ranks) (Required)
                First it will start n_cpus runs in parallel
                Case 1)
                    When an optimiser reach max_gen, it should notify
                    the controller that has finished and a new optimiser
                    should start in that rank.

                Case 2)
                    If an optimiser gets stuck in the same local for
                    more than n_stops = num * checkpoints,
                    it should be stopped and notify
                    to the controller, to start a new optimiser in its
                    rank

            MPI Communications:

                At each checkpoint, each rank sends its best solution to
                the controller.
                Because the ranks will send solutions at different
                times, the controller should be able to receive them all
                but should not wait for the last rank to determine
                best solution.

                The controller (rank==0) should collect all the
                solutions, determine current best solution and store it
                in global repository. The global repository should know
                which rank send each solution, to be able to plot a
                convergence graph with all the runs.

                Ranks should notify to the controller when can't
                optimise more (run stalled), i.e. when reaching
                n_stops = num * checkpoints.
                Then, the rank should be stopped and the controller
                should start a new optimiser there.


        Args:
            args: Arguments object created by argparse or main
            init: Population initialisation object
            proj: Function to optimise
            files_man: Path manager object
            seed: Random seed

        """

        # Copy arguments
        self.args = args
        self.init = init
        self.proj = proj
        self.files_man = files_man
        self.rank_seed = seed

        self.nrun = args.nrun
        self.max_runs = args.max_runs
        self.checkpoint = args.checkpoint

        self.optim_list = []
        self.optim_list_ind = []

        self.optim_ind = 0
        self.run_counter = 0

        # MPI
        self.comm = MPI.COMM_WORLD
        # Get my rank
        self.rank = MPI.COMM_WORLD.Get_rank()
        # Number of available ranks
        self.n_ranks = MPI.COMM_WORLD.Get_size()
        # Ranks minus controller
        self.ranks_alive = self.n_ranks - 1

        # Repository size
        self.n_rep = args.n_rep
        self.n_params = args.n_param
        self.max_pop_from_rep = args.max_pop_from_rep

        # Cost reference
        self.eps_0 = np.inf
        # (0) Not initialised (1) eps_0 initialised
        self.cost_ref_flag = 0

        # Number of stalled runs defined
        self.n_stall = args.n_stall

        # Opt name stalled
        self.optim_name_stall = []

        # Activate wrapper kill mechanism
        self.kill_flag = args.kill_flag
        # Max number of checkpoints reached without improving fmin
        self.n_0 = args.n_0
        # Stall reference level
        self.stall_tol = args.stall_tol

        # fmin history list of every rank
        self.eps_his = [[] * 1 for i in range(self.n_ranks)]

        # Parameter to define number of generations allowed
        self.p_n = args.p_n

        # Number of best runs allowed to run max generations
        self.n_best_runs = args.n_best_runs
        # Best runs array [fmin, rank, nrun]
        self.best_runs = np.ones((self.n_best_runs, 3)) * np.inf

        # Live plotting convergence (True / False)
        self.live_plot = args.live_plot

        # Worker files (True / False)
        self.work_out = args.work_out

    def run(self):
        """
        Starts wrapper function depending MPI rank

        Returns:

        """

        # Controller
        if self.rank == 0:
            self.control()

        # Workers
        else:
            self.work()

    def control(self):
        """
        Controller / Supervisor assigns optimisers to workers

        status.tag == 1000: Initialise new run to free rank
        status.tag == 2000: Receive state from worker
        status.tag == 3000: Notify a worker has finished

        Returns:

        """

        # Select optimiser from list
        if not self.optim_list:
            print("Error: List of optimisers is empty!")
            exit()

        # Index list with size of nruns
        self.optim_list_ind = np.zeros(self.max_runs, dtype='i')
        self.n_optim_list = len(self.optim_list)

        # Number of optimisers stall counter
        self.optim_stall_counter = [0] * self.n_optim_list

        self.optim_eps_stall = [[] * 1 for i in range(self.n_optim_list)]

        # Allocate optimiser index for each run in list
        oj = 0
        for oi in range(self.max_runs):
            self.optim_list_ind[oi] = oj
            oj += 1
            if oj >= self.n_optim_list:
                oj = 0

        # print("Optim List: ", self.optim_list_ind)

        # Global directory is number 0
        # Update counter in files management
        self.files_man.nrun = 0


        # Create run folder
        self.files_man.paths_init()
        self.files_man.run_folder()

        # Log problem definition to 'log.dat'
        self.log()

        # Counter for receiving from workers
        count = [0] * self.max_runs

        # Global counter
        msg_count = 0

        # Initialise global best fmin
        best_fmin = math.inf
        best_nrun = 0

        plot = False

        # Controller time
        self.start_control_time = time.time()

        while self.ranks_alive > 0:

            status = MPI.Status()

            self.comm.probe(status=status)

            # print("Source: ", status.source, " - Tag: ", status.tag)

            # Send new run
            if status.tag == 1000:
                # Receive ready msg from rank
                self.comm.recv(source=MPI.ANY_SOURCE, tag=status.tag)
                # print("Optim_list: ", self.optim_list_ind, '\n')

                # Send optimiser and run ID to free worker
                if self.optim_list_ind.size > 0:

                    # Send optimiser index to worker
                    self.comm.send(self.optim_list_ind[0],
                                   dest=status.source, tag=0)

                    # Send run id number to worker
                    self.run_counter += 1
                    self.comm.send(self.run_counter,
                                   dest=status.source,
                                   tag=1)

                    # Remove sent run from list
                    self.optim_list_ind = self.optim_list_ind[1:]

                else:

                    # If optimiser list is empty,
                    # tell free worker to exit rank
                    stop_flag = -1
                    self.comm.send(stop_flag, dest=status.source, tag=0)

            # Receive and save data
            elif status.tag == 2000:

                # receive data from workers
                data = self.comm.recv(source=MPI.ANY_SOURCE,
                                      tag=status.tag)

                # Data into variables
                nrun_w = data[0]
                rank_w = data[1]
                fmin_w = data[2]
                best_sol_w = data[3]
                name_w = data[4]
                init_count_w = data[5]

                # count number of times received from each worker
                count[nrun_w - 1] = count[nrun_w - 1] + 1
                msg_count += 1
                # print("Update Count:", count, '\n')

                # Update global best from worker data
                if fmin_w < best_fmin:
                    best_fmin = fmin_w
                    best_nrun = nrun_w
                    best_name = name_w
                    best_init = int(init_count_w > 0)

                    # Update seed repository '0\rep.dat'
                    self.save_repository(best_fmin, best_sol_w)

                # Update best array from worker data
                # Find worst run in best runs array
                worst_index = np.argmax(self.best_runs[:, 0])
                worst_best_fmin = self.best_runs[worst_index, 0]

                # Flag to indicate if current worker is one of the best
                best_worker = False

                # Check if worker was not one of the best to avoid duplicates
                if fmin_w < worst_best_fmin \
                        and np.any(self.best_runs[:, 2] != nrun_w):
                    self.best_runs[worst_index, 0] = fmin_w
                    self.best_runs[worst_index, 1] = rank_w
                    self.best_runs[worst_index, 2] = nrun_w
                    best_worker = True

                # Verify if worker is stalled
                stall_flag = self.stalled_worker(data)

                # Send stall flag to worker (Continue, Stop)
                if stall_flag is True and best_worker is True:
                    # Stalled but one of the best (Continue)
                    stall_flag = False
                elif stall_flag is True and best_worker is False:
                    # Stalled but one of the best (Stop and Reset)
                    # Initialise stall error history list in stalled rank
                    self.eps_his[rank_w] = []

                self.comm.send(stall_flag, dest=rank_w, tag=4000)

                # Update global convergence file '0\conv.dat'
                self.glob_conv(data, count[nrun_w - 1], best_fmin,
                               best_nrun, msg_count, best_name, best_init)

                # Update global convergence plot '0\conv.png'
                if self.live_plot:
                    if plot is True:
                        self.plot_glob_conv(count, best_nrun)
                    plot = True

            # Tell control a rank has finished its runs
            elif status.tag == 3000:
                self.comm.recv(source=status.source, tag=status.tag)
                self.ranks_alive -= 1
                print("Runs Alive: ", self.ranks_alive, '\n')

        # Exit controller
        print('Final Count', count)
        print('Global Best Run:', best_nrun)
        print("Exiting Controller: ", self.rank, '\n')
        sys.exit()

    def work(self):
        """
        Worker that communicates with each optimiser

        Returns:

        """

        rank_ready = True
        continue_rank = True

        while continue_rank:

            # Tell controller rank is ready
            self.comm.send(rank_ready, dest=0, tag=1000)

            # Receive optimiser and run counter
            # or stop flag is necessary
            self.optim_ind = self.comm.recv(source=0, tag=0)

            # Optimiser list is exceeded
            if self.optim_ind == -1:
                print("Exiting Rank: ", self.rank, '\n')
                break

            # Receive run id number
            self.run_counter = self.comm.recv(source=0, tag=1)

            # Update counter in files manager
            self.files_man.nrun = self.run_counter

            if self.work_out:
                # Create run folder
                self.files_man.paths_init()
                self.files_man.run_folder()

            # Run seed generated from rank seed
            run_seed = [self.rank_seed[self.run_counter]]

            # Initialise Behaviour
            self.args, beh_index = init_behaviour(self.args)

            # Initialise optimiser
            sub_optim = \
                self.optim_list[self.optim_ind](self.args,
                                                self.init,
                                                self.proj,
                                                self.files_man,
                                                run_seed)

            print("Run Initialised:"
                  " - Rank: ", self.rank,
                  " - Seed: ", run_seed,
                  " - Optimiser: ", sub_optim.name,
                  " - behaviour: ", beh_index,
                  " - Run: ", self.run_counter,
                  " - Population Size: ", sub_optim.pop_size, '\n\n')

            # Run optimiser
            sub_optim.run()

        # Tell controller that current rank finished its runs
        rank_ready = False
        self.comm.send(rank_ready, dest=0, tag=3000)

        # Delete optimiser object
        del sub_optim
        # Exit worker
        sys.exit()

    def save_repository(self, best_fmin, best_sol):
        """
        Stores the best seed repository data

        Args:
            best_fmin: Best objective evaluation
            best_sol: Best model

        Returns:

        """

        # concatenate fmin and sol
        rep_data = np.concatenate((np.array([best_fmin]), best_sol))
        rep_path = self.files_man.run_path + '/' + 'rep.dat'

        # Save best seed to supervisor model
        filename = self.files_man.super_path + '/' + 'model.dat'
        np.savetxt(filename, best_sol, fmt='%.10f', newline=" ")

        # Verify rep
        rep_rows = 0
        rep_exists = os.path.exists(rep_path)

        if rep_exists:
            # Determine number of lines
            rep_file = open(rep_path, 'r')
            rep_rows = len(rep_file.readlines())
            rep_file.close()

            if rep_rows >= self.n_rep:
                # read file and create np array from file
                rep = np.genfromtxt(rep_path, delimiter='\t')
    
                # find worst seed
                row_del = np.argmax(rep[:, 0])
    
                # Delete worst seed from rep
                rep = np.delete(rep, (row_del), axis=0)
    
                # rewrite file without worst seed
                np.savetxt(rep_path, rep, delimiter='\t', fmt='%.8f')

        if self.max_pop_from_rep > 0.0:
            # Append new seed
            rep_file = open(rep_path, 'a')
            i = 0
            while i < self.n_params:
                rep_file.write(str("{:.8f}".format(rep_data[i])))
                rep_file.write("\t")
                i += 1
            rep_file.write(str("{:.8f}".format(rep_data[i])))
            rep_file.write("\n")
            rep_file.close()

    def glob_conv(self, data, count, best_fmin, best_nrun, msg_count,
                  best_name, best_init):
        """
        Store the global convergence data

        Args:
            data: Worker state
            count: list of checkpoints counter per run
            best_fmin:objective evaluation by best model
            best_nrun: number of run of best model
            msg_count: number of messages sent
            best_name: Name of best optimiser at this state
            best_init: Flag of best model (0) not seeded (1) seeded


        Returns:

        """

        # Data into variables
        data_nrun = data[0]
        data_rank = data[1]
        data_fmin = data[2]
        data_best_sol = data[3]
        data_name = data[4]
        data_init_count = data[5]

        # Controller time
        state_time = time.time()

        measured_time = state_time - self.start_control_time

        hours, rem = divmod(measured_time, 3600)
        minutes, seconds = divmod(rem, 60)

        conv_path = self.files_man.run_path + '/' + 'conv.dat'
        conv_file = open(conv_path, "a")
        conv_file.write(str(data_nrun))  # 0 - nrun
        conv_file.write("\t")
        conv_file.write(str(data_rank))  # 1 - rank
        conv_file.write("\t")
        conv_file.write(str(count))  # 2 - count
        conv_file.write("\t")
        conv_file.write(str("{:.8f}".format(data_fmin)))  # 3 - fmin
        conv_file.write("\t")
        conv_file.write(str(msg_count))  # 4 - msg_count
        conv_file.write("\t")
        conv_file.write(
            str("{:.8f}".format(best_fmin)))  # 5 - global best fmin
        conv_file.write("\t")
        conv_file.write(str(best_nrun))  # 6 - global best nrun
        conv_file.write("\t")
        conv_file.write(str(data_init_count))  # 7 -  init from rep
        conv_file.write("\t")
        conv_file.write(str(best_init))  # 8 -  best init from rep flag
        conv_file.write("\t")
        conv_file.write(str("{:.8f}".format(self.eps_0)))  # 9 - Cost_Ref
        conv_file.write("\t")
        conv_file.write(
            str("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),
                                                int(minutes),
                                                seconds)))  # 10 - time
        conv_file.write("\t")
        conv_file.write(data_name)  # 11 - name
        conv_file.write("\t")
        conv_file.write(best_name)  # 12 - name
        conv_file.write("\n")
        conv_file.close()

    def plot_glob_conv(self, count, best_nrun):
        """
        Saves and displays the global convergence plot

        Args:
            count: checkpoint counter
            best_nrun: best run

        Returns:
            plot
        """

        plt.clf()
        plt.ion()
        plt.show()
        conv_path = self.files_man.run_path + '/' + 'conv.dat'
        glob_conv = np.genfromtxt(conv_path, delimiter='\t')
        for i in range(self.max_runs):
            nrun_ind = np.where(glob_conv[:, 0] == i + 1)
            fmin_array = glob_conv[nrun_ind, 3].flatten()
            gen_array = np.linspace(self.checkpoint,
                                    count[i] * self.checkpoint,
                                    num=int(count[i]))
            if (i + 1) == best_nrun:
                plt.loglog(gen_array, fmin_array, '-or', linewidth=2,
                           markersize=3,
                           label='Best Run: ' + str(best_nrun))
                plt.legend()
            else:
                plt.loglog(gen_array, fmin_array, '-ok', linewidth=2,
                           markersize=3)

        plt.xlabel('Generations')
        plt.ylabel('Cost')
        plt.title('Global Convergence')
        plt.grid(True)
        plt.pause(0.01)
        filename = self.files_man.run_path + '/' + 'conv_opt.png'
        plt.savefig(filename)

    def stalled_worker(self, data):
        """
        Function that decides to define a run as stalled

        Args:
            data: Worker state

        Returns:
            stall: Bool flag to continue or stop worker
        """

        # Worker data
        nrun = data[0]
        rank = data[1]
        opt_name = data[4]

        # 1. fmin received from worker
        eps = data[2]

        # 2. Store current fmin
        self.eps_his[rank].append(eps)

        his_first = self.eps_his[rank][0]  # First fmin received
        his_last = self.eps_his[rank][-1]  # Current fmin

        # 3. Remove history when worker is improving
        while his_last / his_first < 1 - self.stall_tol \
                and len(self.eps_his[rank]) > 0:
            # stall_tol smaller, more chance to remove history
            # more chance to continue (not killed)
            self.eps_his[rank].pop(0)

        # 4. Number of generations allowed n
        n = self.n_0
        if self.cost_ref_flag == 1:
            n = self.n_0 * max(1, (self.eps_0 / eps) ** self.p_n)

        # 5. Check if run is stalled
        if len(self.eps_his[rank]) >= n and self.kill_flag:

            # Limit history to n states
            while len(self.eps_his[rank]) > n:
                self.eps_his[rank].pop(0)

            # Stall flag set to kill run
            stall = True

            if opt_name not in self.optim_name_stall:
                self.optim_name_stall.append(opt_name)

            # Counter of stalls per optimiser
            index = self.optim_name_stall.index(opt_name)
            self.optim_eps_stall[index].append(eps)
            self.optim_stall_counter[index] += 1

            # Check if list of optimiser has captured all of them
            optim_list_bool = len(self.optim_name_stall) == self.n_optim_list

            # Check whether all optimisers reached n_stall
            optim_stall_bool = \
                all(i >= self.n_stall for i in self.optim_stall_counter)

            # Define cost reference level eps_0
            if self.cost_ref_flag == 0 and optim_list_bool and optim_stall_bool:

                # Reduce optim stall list to same n_stall
                red_optim_eps_stall = [sub[:self.n_stall] for sub in self.optim_eps_stall]

                # transform to np
                optim_eps_stall_array = np.array(red_optim_eps_stall)

                # Define eps_0 as mean of optimisers
                self.eps_0 = np.mean(optim_eps_stall_array[:, -1])
                self.cost_ref_flag = 1

            #print("Run:", nrun, "Rank:", rank, "Opt:", opt_name, "Stall_tol: ", self.stall_tol)

        else:
            # Stall flag set to continue
            stall = False

        return stall

    def log(self):
        """
        Function that saves the problem definition

        Returns:

        """

        log_path = self.files_man.run_path + '/' + 'log.dat'
        log_file = open(log_path, "a")
        log_file.write('Problem Definition - General' + '\n')
        log_file.write('Optimisers: ' + str(self.optim_list) + '\n')
        log_file.write('checkpoint: ' + str(self.checkpoint) + '\n')
        log_file.write('n_devices: ' + str(self.n_ranks) + '\n')
        log_file.write('max_runs: ' + str(self.max_runs) + '\n')
        log_file.write('n_rep: ' + str(self.n_rep) + '\n')
        log_file.write('n_param: ' + str(self.n_params) + '\n')
