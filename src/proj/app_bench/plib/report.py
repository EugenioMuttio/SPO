import numpy as np
from plib.plots import Plots


class Report(object):
    def __init__(self, args, files_man):

        # Argparse parameters
        self.args = args
        # Plot training
        self.nrun = args.nrun

        # Files path
        self.files_man = files_man
        self.super_path = files_man.super_path
        self.run_path = files_man.run_path
        self.run_id_path = files_man.run_id_path
        self.his_path = files_man.his_path
        self.plot_path = files_man.plot_path

        # Convergence path
        self.conv_path = self.run_path + '/' + 'conv.dat'
        self.conv = []

        self.global_conv_path = self.run_id_path + '/0/' + 'conv.dat'
        self.global_conv = []

        # Best Model path
        self.sol_path = self.super_path + '/' + 'model.dat'
        self.sol = []

        # Objective evaluation history path (MO)
        self.f_his_path = self.his_path + '/' + 'model_f_his.dat'
        self.f_his = []

        # Models repository path
        self.f_rep_path = self.run_path + '/' + 'f_rep.dat'
        self.f_rep = []

        # Read Model Best Solution
        self.best_sol_path = self.his_path + '/' + 'model_his.dat'
        #self.best_sol = np.genfromtxt(self.best_sol_path, delimiter="\t")

        # Best run
        self.best_run = 0

    def report_plot(self, proj):
        """
        Function activated when --report is 1.
        This function requires to define --nrun num, where num is the
        test experiment that is selected to plot results.

        Returns:
            A group of plots within a directory called 'Plots' inside
            the run selected.
        """

        # Read Convergence
        # self.conv = np.genfromtxt(self.conv_path, delimiter="\t")

        # Read Global Convergence
        self.global_conv = \
            np.genfromtxt(self.global_conv_path,
                          dtype=(
                              int, int, int, float, int, float, int, int, int,
                              float, 'S11', 'S10', 'S10'),
                          delimiter="\t")

        # Read best model
        self.sol = np.loadtxt(self.sol_path)

        # Plot folder
        self.files_man.plot_folder()

        # Plot object
        plot = Plots(self.args, self.files_man)

        # Global convergence simple
        # plot.global_convergence_simple(self.global_conv)

        # Global convergence fast (suggested)
        # The issue with this is that optimisers are plotted in groups,
        # Then, first optimiser will be back of the others
        plot.global_convergence_fast(self.global_conv)

        # Global convergence according to supervisor messages
        # PLots produced in the paper are produced with this function
        # It takes more time to produce the plots
        # plot.global_convergence(self.global_conv)

        # Plot global best trajectory:
        plot.path_2d(self.sol, proj)

        # Optim Analysis (Messages from the supervisor)
        # plot.optim_analysis(self.global_conv)

    def report_comparison(self, proj, prob_id, runs_file, best_run_id, opt_id,
                          avg_fmin, std_dev, n_comp):
        """
        Function activated when --report is 1.
        This function requires to define --nrun num, where num is the
        test experiment that is selected to plot results.

        Returns:
            A group of plots within a directory called 'Plots' inside
            the run selected.
        """

        # Read wrapper Convergence
        self.global_conv = \
            np.genfromtxt(self.global_conv_path,
                          dtype=(
                              int, int, int, float, int, float, int, int, int,
                              float, 'S11', 'S10', 'S10'),
                          delimiter="\t")

        # Read wrapper model
        self.sol = np.loadtxt(self.sol_path)

        # Plot object
        plot = Plots(self.args, self.files_man)

        # Comparison with embedded optimisers --------------------------------
        plot.comparison_plots(proj, prob_id, runs_file, best_run_id, opt_id,
                              self.global_conv, self.sol, avg_fmin, std_dev,
                              n_comp)

        plot.comparison_errorbar(prob_id, opt_id, avg_fmin, std_dev, n_comp)

        # SOTA comparison -----------------------------------------------------
        # plot.comparison_plots_sota(proj, prob_id, runs_file, best_run_id,
        # opt_id, self.global_conv, self.sol, avg_fmin, std_dev, n_comp)

    def report_avg(self, run_name, prob_id, n_runs, opt_flag, n_workers):
        """
        Statistics of the convergence of the runs.

        Args:
            run_name: str - Name of the run
            prob_id: str - Problem ID
            n_runs: int - Number of runs
            opt_flag: bool - Flag to indicate if the optimiser is embedded
            n_workers: int - Number of workers
        Returns:
            Statistics of the convergence of the runs.

        """

        riwi = 1
        for ri in range(1, n_runs + 1):
            # File Paths
            run_id = self.files_man.results_path + '/' + prob_id + \
                     run_name + '_' + str(ri) + '/0'

            # Convergence path
            conv_path = run_id + '/' + 'conv.dat'

            # Read Convergence
            conv = \
                np.genfromtxt(conv_path, delimiter="\t")

            if ri == 1:
                best_conv = conv[-1, :]
            else:
                best_conv = np.vstack((best_conv, conv[-1, :]))

            if opt_flag:
                for wi in range(1, n_workers + 1):
                    # File Paths
                    run_id = self.files_man.results_path + '/' + prob_id + \
                             run_name + '_' + str(ri) + '/' + str(wi)

                    # Convergence path
                    conv_path = run_id + '/' + 'conv.dat'

                    opt_conv = np.genfromtxt(conv_path, delimiter="\t")

                    if riwi == 1:
                        best_opt_conv = opt_conv[-1, :]
                    else:
                        best_opt_conv = np.vstack((best_opt_conv,
                                                   opt_conv[-1, :]))

                    riwi += 1

        min_best_fmin = np.min(best_conv[:, 5])
        avg_best_fmin = np.mean(best_conv[:, 5])
        std_fmin = np.std(best_conv[:, 5])
        median_fmin = np.median(best_conv[:, 5])
        max_best_fmin = np.max(best_conv[:, 5])

        ind_best_fmin = np.argmin(best_conv[:, 5])

        print('best fmin: ', min_best_fmin, ' index: ', ind_best_fmin + 1)
        print('Average best fmin: ', avg_best_fmin)
        print('std fmin: ', std_fmin)
        print('median fmin: ', median_fmin)
        print('max fmin: ', max_best_fmin)

        if opt_flag:
            avg_opt_fmin = np.mean(best_opt_conv[:, 2])
            std_opt_fmin = np.std(best_opt_conv[:, 2])
            median_opt_fmin = np.median(best_opt_conv[:, 2])
            max_opt_fmin = np.max(best_opt_conv[:, 2])

            print('Average opt fmin: ', avg_opt_fmin)
            print('std opt fmin: ', std_opt_fmin)
            print('median opt fmin: ', median_opt_fmin)
            print('max opt fmin: ', max_opt_fmin)

    def report_optim_analysis(self, run_name, prob_id, n_runs, n_comp):
        """
        Function to evaluate the message communication of the supervisor.

        Function activated when --report is 1.
        This function requires to define --nrun num, where num is the
        test experiment that is selected to plot results.

        Args:
            run_name (str): Name of the run.
            prob_id (str): Problem id.
            n_runs (int): Number of runs.
            n_comp (int): Number of comparisons.

        Returns:
            A group of plots within a directory called 'Plots' inside
            the run selected.
        """

        # Plot object
        plot = Plots(self.args, self.files_man)

        # Read wrapper Convergence
        global_conv = []

        for ri in range(1, n_runs+1):
            run_id = self.files_man.results_path + '/' + prob_id + \
                     run_name + '_' + str(ri) + '/0'
            conv_path = run_id + '/' + 'conv.dat'
            global_conv.append(
                np.genfromtxt(conv_path,
                              dtype=(
                                  int, int, int, float, int, float, int, int, int,
                                  float, 'S11', 'S10', 'S10'),
                              delimiter="\t"))

        plot.optim_analysis_comp(global_conv, prob_id, n_comp, n_runs)