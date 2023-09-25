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
        self.current_path = files_man.current_path
        self.run_id = files_man.run_id

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
        #self.conv = np.genfromtxt(self.conv_path, delimiter="\t")

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

        # Global convergence simple (suggested)
        #plot.global_convergence_simple(self.global_conv)

        # Global convergence fast (suggested)
        # The issue with this is that optimisers are plotted in groups,
        # Then, first optimiser will be back of the others
        #plot.global_convergence_fast(self.global_conv)

        # Global convergence according to supervisor messages
        # PLots produced in the paper are produced with this function
        # (Not suggested due to matplotlib computational time)
        # plot.global_convergence(self.global_conv)

        # Plot global best trajectory:
        #plot.path_2d(self.sol, proj)

        # Optim Analysis
        plot.optim_analysis(self.global_conv)

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

        plot.comparison_plots(proj, prob_id, runs_file, best_run_id, opt_id,
                              self.global_conv, self.sol, avg_fmin, std_dev, n_comp)

        plot.comparison_errorbar(prob_id, opt_id, avg_fmin, std_dev, n_comp)

    def report_avg(self, run_name, prob_id, n_runs, opt_flag, n_workers):


        riwi = 1
        for ri in range(1,n_runs+1):
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
                        best_opt_conv = np.vstack((best_opt_conv, opt_conv[-1, :]))

                    riwi += 1

        min_best_fmin = np.min(best_conv[:, 5])
        avg_best_fmin = np.mean(best_conv[:, 5])
        std_fmin = np.std(best_conv[:, 5])
        median_fmin = np.median(best_conv[:, 5])
        max_best_fmin = np.max(best_conv[:, 5])

        print('best fmin: ', min_best_fmin)
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


    def report_hyper(self, prob_id, n_comp, run_name, n_runs, fmin_lim):
        """
        Function activated when --report is 1.
        This function requires to define --nrun num, where num is the
        test experiment that is selected to plot results.

        Returns:
            A group of plots within a directory called 'Plots' inside
            the run selected.
        """

        # Plot object
        plot = Plots(self.args, self.files_man)

        # Read wrapper Convergence

        n_hyper = 6
        hyper_value_id = ['A', 'B', 'C', 'D', 'E']
        hyper_time = np.zeros((n_runs, 6, len(hyper_value_id)))
        hyper_value = np.zeros((n_runs, 6, len(hyper_value_id)))
        hyper_fmin = np.zeros((n_runs, 6, len(hyper_value_id)))

        for ri in range(1, n_runs + 1):
            for hi in range(1, n_hyper + 1):
                for id in range(len(hyper_value_id)):
                    run_id = self.files_man.results_path + '/' + prob_id[:-1] +  \
                             '_' + str(hi) + '/'  + run_name + hyper_value_id[id] + \
                             '_' + str(ri) + '/0'
                    conv_path = run_id + '/' + 'conv.dat'
                    conv_data = \
                        np.genfromtxt(conv_path,
                                      dtype=(
                                          int, int, int, float, int, float, int, int,
                                          int,
                                          float, 'S11', 'S10', 'S10'),
                                      delimiter="\t")

                    # Fmin
                    hyper_fmin[ri-1, hi-1, id] = conv_data[-1][5]

                    # Time to reach fmin_lim
                    n_data = conv_data.shape[0]
                    found = False
                    idata = 0
                    while(not found and idata < n_data):
                        if conv_data[idata][5] > fmin_lim:
                            idata += 1
                        else:
                            found = True
                            hyper_value[ri-1, hi-1, id] = conv_data[idata][5]
                            # Time
                            aux = conv_data[idata][10].decode('ASCII')
                            aux_b = aux.split(':')
                            aux_c = np.array(aux_b).astype(float)
                            run_time = aux_c.reshape((1, 3))
                            run_time_unit = run_time[0, 0] + \
                                            np.divide(run_time[0, 1], 60) + \
                                            np.divide(run_time[0, 2], 3600)
                            hyper_time[ri-1, hi-1, id] = run_time_unit

                    if not found:
                        hyper_value[ri-1, hi-1, id] = 15.0
                        hyper_time[ri-1, hi-1, id] = 15.0

        fmin = hyper_fmin[0, :, :]
        avg_fmin = np.zeros(fmin.shape)
        fmin_std_dev = np.zeros(fmin.shape)

        tmin = hyper_time[0, :, :]
        avg_tmin = np.zeros(tmin.shape)
        time_std_dev = np.zeros(tmin.shape)

        for ri in range(1, n_runs + 1):
            # Min value in matrix
            fmin = np.minimum(fmin, hyper_fmin[ri-1, :, :])
            tmin = np.minimum(tmin, hyper_time[ri-1, :, :])
            avg_fmin += hyper_fmin[ri-1, :, :]
            avg_tmin += hyper_time[ri-1, :, :]

        avg_fmin = np.divide(avg_fmin, n_runs)
        avg_tmin = np.divide(avg_tmin, n_runs)
        for i in range(tmin.shape[0]):
            for j in range(tmin.shape[1]):
                fmin_std_dev[i, j] = np.std(hyper_fmin[:, i, j])
                time_std_dev[i, j] = np.std(hyper_time[:, i, j])

        tmin_list = []
        avg_tmin_list = []
        time_std_dev_list = []
        name_list = []
        tmin_list.append(tmin)
        avg_tmin_list.append(avg_tmin)
        time_std_dev_list.append(time_std_dev)

        fmin_list = []
        avg_fmin_list = []
        std_dev_list = []
        name_list = []

        fmin_list.append(fmin)
        avg_fmin_list.append(avg_fmin)
        std_dev_list.append(fmin_std_dev)
        name_list.append('$N_{p}=200$')

        # Plot hyper
        plot.hyper_plots(prob_id, fmin_list, avg_fmin_list,
                         std_dev_list, name_list, n_comp, ylabel='Cost', ylim=[30, 45.0], col_n=0)
        # Plot hyper time
        plot.hyper_plots(prob_id, tmin_list, avg_tmin_list,
                         time_std_dev_list, name_list, n_comp, ylabel='Time (h)', ylim=[0, 20.0],col_n=2)




    def report_optim_analysis(self, run_name, prob_id, n_runs, n_comp):
        """
        Function activated when --report is 1.
        This function requires to define --nrun num, where num is the
        test experiment that is selected to plot results.

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


    def algorithmic_complexity(self, proj, n_worker_runs, n_workers, func_file):
        """
                Function activated when --report is 1.
                Computes the algorithmic complexity of the problem

                Returns:

                """

        eval_dict = {}
        for ri in range(1, n_worker_runs + 1):
            conv_path = self.run_id_path + '/' + str(ri) + '/conv.dat'
            # Read Convergence
            conv = np.genfromtxt(conv_path,
                                 dtype=('S11', int, int, float, 'S10'),
                                 delimiter="\t")
            key = conv[-1][0]

            if key not in eval_dict:
                eval_dict[key] = [conv[-1][2]]
            else:
                eval_dict[key].append(conv[-1][2])

        opt_eval_dict = {}
        for key in eval_dict:
            opt_eval_dict[key] = np.sum(eval_dict[key])

        # Sum of all evaluations
        total_eval = np.sum(list(opt_eval_dict.values()))

        # Verify function evaluation
        func_path = np.genfromtxt(func_file, delimiter="\n")

        func_eval = func_path[-1] * n_workers

        # Compute algorithmic complexity
        alg_complex_spo = (func_eval - total_eval) / total_eval
        print("function evaluations: ", func_eval)
        print("total evaluations: ", total_eval)
        print("SPO: Algorithmic complexity: ", alg_complex_spo)

        # Compute algorithmic complexity per algorithm
        for key in opt_eval_dict:
            alg_complex = (func_eval  - opt_eval_dict[key]) / func_eval * alg_complex_spo
            print("Algorithm: ", key, " Algorithmic complexity: ", alg_complex)

