import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import rcParams
from matplotlib import colors
from scipy.signal import convolve2d


rcParams.update({'figure.autolayout': True})
# Global Fonts
rcParams.update({
    "mathtext.fontset": "dejavuserif",
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})
#rcParams['text.usetex'] = True

class Plots(object):
    def __init__(self, args, files_man):

        # Args
        self.checkpoint = args.checkpoint

        # Files
        self.files_man = files_man
        self.run_path = files_man.run_path
        self.run_id_path = files_man.run_id_path
        self.plot_path_in = files_man.plot_path_in
        self.nrun = args.nrun
        self.results_path = files_man.results_path

        # Repository path located at zero (controller)
        self.rep_path = self.run_id_path + '/0/plots/'

        # Controller folder
        try:
            # Create target Directory
            os.mkdir(self.rep_path)

        except FileExistsError:
            print('Plot directory for controller already exists')

        # Plots format settings
        self.colors = {'red': np.array([255, 89, 94]) / 255,
                       'blue': np.array([25, 130, 196]) / 255,
                       'green': np.array([138, 201, 38]) / 255,
                       'yellow': np.array([255, 202, 58]) / 255,
                       'purple': np.array([106, 76, 147]) / 255,
                       'black': np.array([33, 37, 41]) / 255,
                       'grey': np.array([222, 226, 230]) / 255}

        self.opt_colors = {'MCS': np.array([208, 0, 0]) / 255,
                           'MCSV1': np.array([157, 2, 8]) / 255,
                           'MCSV2': np.array([106, 4, 15]) / 255,
                           'PymooPSO': np.array([0, 95, 115]) / 255,
                           'PymooPSOV1': np.array([10, 147, 150]) / 255,
                           'PymooPSOV2': np.array([148, 210, 189]) / 255,
                           'PymooGA': np.array([238, 155, 0]) / 255,
                           'PymooDE': np.array([202, 103, 2]) / 255,
                           'PymooCMAES': np.array([187, 62, 3]) / 255,
                           'Best': np.array([33, 37, 41]) / 255,
                           'SPO': np.array([0, 128, 0]) / 255}

        self.font_factor = 1.8

        self.fig_format = 'jpg'
        
        self.dpi = 150

        # Path Planning Optimisation Parameters
        self.xs = args.xs
        self.xt = args.xt
        self.ys = args.ys
        self.yt = args.yt
        self.n_param = args.n_param
        self.upper_limit = args.upper_limit
        self.lower_limit = args.lower_limit
        
        # For greater data the plot is reduced by checkpoints
        self.data_checkpoint = 1
        
        # Plotting parameters
        # Plot limits
        self.yl1, self.yl2 = 30.0, 300.0
        # Plot zoom limits
        self.ylz1, self.ylz2 = 31.25, 35.0
        
        # Figure size
        self.h_dim = 15.0
        self.v_dim = 6.0

    def global_convergence_simple(self, conv_data):
        """
        Plot the convergence plot (loss)
        Args:
            loss (ndarray): array with the convergence data
            (0) Gen (1) Eval (2) Fmin (3) Time in hr:min:sec
        """

        # Convergence data
        n_data = conv_data.shape[0]
        n_runs = []
        runs_gen = []
        runs_fmin = []
        max_gen = []
        fmin = []
        init_count = []
        best_init = []
        run_time = np.zeros((1, 3))
        opt_name = []
        best_name = []
        cost_ref = []

        num_data = int(n_data / self.data_checkpoint)
        data_index = np.linspace(0, n_data - 1, num=num_data, dtype=int)
        for di in data_index:
            # N runs
            n_runs.append(conv_data[di][0])

            # Runs generations
            msg_counter = conv_data[di][2] * self.checkpoint
            runs_gen.append(msg_counter)

            # Runs F min
            runs_fmin.append(conv_data[di][3])

            # Max Generations
            max_msg_counter = conv_data[di][4] * self.checkpoint
            max_gen.append(max_msg_counter)

            # Best F min
            fmin.append(conv_data[di][5])

            # Init from rep
            init_count.append(conv_data[di][7])

            # Best Init from rep
            best_init.append(conv_data[di][8])

            # Optimiser name
            cost_ref.append(conv_data[di][9])

            # Time
            aux = conv_data[di][10].decode('ASCII')
            aux_b = aux.split(':')
            aux_c = np.array(aux_b).astype(float)
            aux_d = aux_c.reshape((1, 3))
            run_time = np.concatenate((run_time, aux_d), axis=0)

            # Optimiser name
            opt_name.append(conv_data[di][11].decode('ASCII'))

            # Optimiser name
            best_name.append(conv_data[di][12].decode('ASCII'))

        # Best run
        best_run = int(conv_data[-1][6])

        # Minimum Loss
        min_loss_index = np.argmin(fmin)
        min_loss = fmin[min_loss_index]


        # Time
        max_hours = np.max(run_time[:, 0])
        max_min = np.max(run_time[:, 1])

        if max_hours >= 5:
            run_time_unit = run_time[:, 0] + \
                            np.divide(run_time[:, 1], 60) + \
                            np.divide(run_time[:, 2], 3600)
            time_unit = '[hrs]'

        elif max_hours < 5 and max_min > 5:
            run_time_unit = np.multiply(run_time[:, 0], 60) + \
                            run_time[:, 1] + \
                            np.divide(run_time[:, 2], 60)
            time_unit = '[min]'

        elif max_hours < 1 and max_min <= 5:
            run_time_unit = np.multiply(run_time[:, 0], 3600) + \
                            np.multiply(run_time[:, 1], 60) + \
                            run_time[:, 2]
            time_unit = '[s]'

        # N runs best
        max_nruns = max(n_runs)
        # nruns_best: [0] run [1] loss [2] init
        nruns_best = np.ones((max_nruns, 3)) * np.inf
        # nruns best optimiser
        nruns_opt = [None] * max_nruns

        # Reference level
        ref_level = list(set(cost_ref))
        max_runs_gen = max(runs_gen)
        max_run_time = max(run_time_unit)

        # PLOT y --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Generations', fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        # Lines
        ax.plot(runs_gen, runs_fmin, "o",
                color=self.colors['grey'], markersize=5)

        ax.plot(runs_gen, fmin, "^",
                color=self.colors['black'], markersize=5)

        # Reference level
        for ri in range(len(ref_level)):
            ax.plot([0, max_runs_gen],
                    [ref_level[ri], ref_level[ri]],
                    linestyle=(0, (5, 10)),
                    color=self.colors['red'], linewidth=2)

        ax.grid()
        plt.ylim(self.ylz1, self.ylz2)

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)

        fig.tight_layout()

        labels = [best_name[-1] +
                  '\nMin Cost: ' + str("{:2.3f}".format(min_loss)),
                  'SPO Ensemble',
                  r'Ref Cost $\bar{\epsilon}$']

        handles = [plt.plot([], '^', color=self.colors['black'],
                            markersize=7.5)[0],
                   plt.plot([], 'o', color=self.colors['grey'],
                            markersize=5)[0],
                   plt.plot([], color=self.colors['red'],
                            linestyle='--')[0]]

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False,
                         prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'ConvZoom_s.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)
        del fig, ax

        # SEMILOG y --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Generations', fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        # Lines
        ax.semilogy(runs_gen, runs_fmin, "o",
                color=self.colors['grey'], markersize=5)

        ax.semilogy(runs_gen, fmin, "^",
                color=self.colors['black'], markersize=5)

        # Reference level
        for ri in range(len(ref_level)):
            ax.plot([0, max_runs_gen],
                    [ref_level[ri], ref_level[ri]],
                    linestyle=(0, (5, 10)),
                    color=self.colors['red'], linewidth=2)

        ax.grid()
        plt.ylim(self.yl1, self.yl2)

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)

        fig.tight_layout()

        labels = [best_name[-1] +
                  '\nMin Cost: ' + str("{:2.3f}".format(min_loss)),
                  'SPO Ensemble',
                  r'Ref Cost $\bar{\epsilon}$']

        handles = [plt.plot([], '^', color=self.colors['black'],
                            markersize=7.5)[0],
                   plt.plot([], 'o', color=self.colors['grey'],
                            markersize=5)[0],
                   plt.plot([], color=self.colors['red'],
                            linestyle='--')[0]]

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False,
                         prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'Convlog_s.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)
        del fig, ax

        # LOGLOG Plot ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Generations', fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        # Lines
        ax.loglog(runs_gen, runs_fmin, "o",
                    color=self.colors['grey'], markersize=5)

        ax.loglog(runs_gen, fmin, "^",
                    color=self.colors['black'], markersize=5)

        # Reference level
        for ri in range(len(ref_level)):
            ax.plot([0, max_runs_gen],
                    [ref_level[ri], ref_level[ri]],
                    linestyle=(0, (5, 10)),
                    color=self.colors['red'], linewidth=2)

        ax.grid()
        plt.ylim(self.yl1, self.yl2)

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)

        fig.tight_layout()

        labels = [best_name[-1] +
                  '\nMin Cost: ' + str("{:2.3f}".format(min_loss)),
                  'SPO Ensemble',
                  r'Ref Cost $\bar{\epsilon}$']

        handles = [plt.plot([], '^', color=self.colors['black'],
                            markersize=7.5)[0],
                   plt.plot([], 'o', color=self.colors['grey'],
                            markersize=5)[0],
                   plt.plot([], color=self.colors['red'],
                            linestyle='--')[0]]

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False,
                         prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'Convloglog_s.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)
        del fig, ax

        # TIME SMILOGy ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Time ' + str(time_unit), fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        ax.plot(run_time_unit[:-1], runs_fmin, "o",
                    color=self.colors['grey'], markersize=5)

        ax.plot(run_time_unit[:-1], fmin, "o",
                    color=self.colors['black'], markersize=5)

        # Reference level
        for ri in range(len(ref_level)):
            ax.plot([0, max_run_time],
                        [ref_level[ri], ref_level[ri]],
                        linestyle=(0, (5, 10)),
                        color=self.colors['red'], linewidth=1)

        ax.grid()

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)

        plt.ylim(self.ylz1, self.ylz2)

        fig.tight_layout()

        handles = [plt.plot([], '^', color=self.colors['black'],
                            markersize=7.5)[0],
                   plt.plot([], 'o', color=self.colors['grey'],
                            markersize=5)[0],
                   plt.plot([], color=self.colors['red'],
                            linestyle='--')[0]]

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False, prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'ConvTimeZoom_s.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)

        del fig, ax

        # TIME SMILOGy ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Time ' + str(time_unit), fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        ax.semilogy(run_time_unit[:-1], runs_fmin, "o",
                    color=self.colors['grey'], markersize=5)

        ax.semilogy(run_time_unit[:-1], fmin, "o",
                    color=self.colors['black'], markersize=5)


        # Reference level
        for ri in range(len(ref_level)):
            ax.semilogy([0, max_run_time],
                        [ref_level[ri], ref_level[ri]],
                        linestyle=(0, (5, 10)),
                        color=self.colors['red'], linewidth=1)

        ax.grid()

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)

        plt.ylim(self.yl1, self.yl2)

        fig.tight_layout()

        handles = [plt.plot([], '^', color=self.colors['black'],
                            markersize=7.5)[0],
                   plt.plot([], 'o', color=self.colors['grey'],
                            markersize=5)[0],
                   plt.plot([], color=self.colors['red'],
                            linestyle='--')[0]]

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False, prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'ConvTimelog_s.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)

        del fig, ax

    def global_convergence_fast(self, conv_data):
        """
        Plot the convergence plot (loss)
        Args:
            loss (ndarray): array with the convergence data
            (0) Gen (1) Eval (2) Fmin (3) Time in hr:min:sec
        """

        # Convergence data
        n_data = conv_data.shape[0]
        n_runs = []
        runs_gen = []
        runs_fmin = []
        max_gen = []
        fmin = []
        init_count = []
        best_init = []
        run_time = np.zeros((1,3))
        opt_name = []
        best_name = []
        cost_ref = []

        # Max Time
        aux = conv_data[-1][10].decode('ASCII')
        aux_b = aux.split(':')
        aux_c = np.array(aux_b).astype(float)
        aux_d = aux_c.reshape((1, 3))
        aux_run_time = np.zeros((1, 3))
        aux_run_time = np.concatenate((aux_run_time, aux_d), axis=0)

        # Time
        max_hours = np.max(aux_run_time[:, 0])
        max_min = np.max(aux_run_time[:, 1])

        if max_hours >= 5:
            run_time_unit = aux_run_time[:, 0] + \
                            np.divide(aux_run_time[:, 1], 60) + \
                            np.divide(aux_run_time[:, 2], 3600)
            time_unit = '[hrs]'

        elif max_hours < 5 and max_min > 5:
            run_time_unit = np.multiply(aux_run_time[:, 0], 60) + \
                            aux_run_time[:, 1] + \
                            np.divide(aux_run_time[:, 2], 60)
            time_unit = '[min]'

        elif max_hours < 1 and max_min <= 5:
            run_time_unit = np.multiply(aux_run_time[:, 0], 3600) + \
                            np.multiply(aux_run_time[:, 1], 60) + \
                            aux_run_time[:, 2]
            time_unit = '[s]'

        # Dictionary with all data:
        # [0] n_runs [1] fmin [2] best run
        # [3] best fmin [4] seed runs [5] seed fmin

        data_dict = {}
        
        num_data = int(n_data / self.data_checkpoint)
        data_index = np.linspace(0, n_data - 1, num=num_data, dtype=int)

        fmin_aux = conv_data[0][5]
        gen_aux = conv_data[0][2] * self.checkpoint
        for di in data_index:
            
            # N runs
            n_runs.append(conv_data[di][0])

            # Runs generations
            msg_counter = conv_data[di][2] * self.checkpoint
            runs_gen.append(msg_counter)

            # Runs F min
            runs_fmin.append(conv_data[di][3])

            # Max Generations
            max_msg_counter = conv_data[di][4] * self.checkpoint
            max_gen.append(max_msg_counter)

            # Best F min
            fmin.append(conv_data[di][5])

            # Init from rep
            init_count.append(conv_data[di][7])
            
            # Best Init from rep
            best_init.append(conv_data[di][8])

            # Optimiser name
            cost_ref.append(conv_data[di][9])

            # Time
            aux = conv_data[di][10].decode('ASCII')
            aux_b = aux.split(':')
            aux_c = np.array(aux_b).astype(float)
            aux_d = aux_c.reshape((1, 3))
            run_time = np.concatenate((run_time, aux_d), axis=0)

            if max_hours >= 5:
                run_time_unit = run_time[:, 0] + \
                                np.divide(run_time[:, 1], 60) + \
                                np.divide(run_time[:, 2], 3600)

            elif max_hours < 5 and max_min > 5:
                run_time_unit = np.multiply(run_time[:, 0], 60) + \
                                run_time[:, 1] + \
                                np.divide(run_time[:, 2], 60)

            elif max_hours < 1 and max_min <= 5:
                run_time_unit = np.multiply(run_time[:, 0], 3600) + \
                                np.multiply(run_time[:, 1], 60) + \
                                run_time[:, 2]


            # Optimiser name
            opt_name.append(conv_data[di][11].decode('ASCII'))

            # Optimiser name
            best_name.append(conv_data[di][12].decode('ASCII'))

            # Store in dictionary
            if opt_name[di] not in data_dict:
                data_dict[opt_name[di]] = [[], [], [], [], [], [], [], [], []]

            data_dict[opt_name[di]][0].append(runs_gen[di])
            data_dict[opt_name[di]][1].append(runs_fmin[di])
            data_dict[opt_name[di]][2].append(run_time_unit[di])
            if fmin[di] <= fmin_aux:
                gen_aux_max = max(gen_aux, runs_gen[di])
                data_dict[opt_name[di]][3].append(gen_aux_max)
                data_dict[opt_name[di]][4].append(fmin[di])
                data_dict[opt_name[di]][5].append(run_time_unit[di])
                fmin_aux = fmin[di]
                gen_aux = runs_gen[di]
            if init_count[di] > 0:
                data_dict[opt_name[di]][6].append(runs_gen[di])
                data_dict[opt_name[di]][7].append(runs_fmin[di])
                data_dict[opt_name[di]][8].append(run_time_unit[di])

        # sorting dict by key

        my_keys = list(data_dict.keys())
        my_keys.sort()
        data_dict = {key: data_dict[key] for key in my_keys}

        # Minimum Loss
        min_loss_index = np.argmin(fmin)
        min_loss = fmin[min_loss_index]
            
        # N runs best
        max_nruns = max(n_runs)
        
        # Reference level
        ref_level = list(set(cost_ref))
        max_runs_gen = max(runs_gen)
        max_run_time = max(run_time_unit)

        # PLOT y --------------------------------------------------------
        def plot_runs(ax, data_dict, key):
            # Plotting function to optimise code
                ax.plot(data_dict[key][0], data_dict[key][1], 'o',
                        color=self.opt_colors[key], markersize=5)

                ax.plot(data_dict[key][3], data_dict[key][4], '^',
                        color=self.opt_colors[key], markersize=7.5)

                ax.plot(data_dict[key][6], data_dict[key][7], '.',
                        color=self.opt_colors['Best'], markersize=2)

        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Generations', fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        for key in data_dict:
            plot_runs(ax, data_dict, key)

        # Reference level
        for ri in range(len(ref_level)):
            ax.plot([0, max_runs_gen],
                    [ref_level[ri], ref_level[ri]],
                    linestyle=(0, (5, 10)),
                    color=self.opt_colors["Best"], linewidth=2)

        ax.grid()
        plt.ylim(self.ylz1, self.ylz2)

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)

        fig.tight_layout()

        labels = [best_name[-1] +
                  '\nMin Cost: ' + str("{:2.3f}".format(min_loss))]

        for key in data_dict:
            labels.append(key)

        labels.append(r'Ref Cost $\bar{\epsilon}$')

        handles = [plt.plot([], '^', color=self.opt_colors[best_name[-1]],
                   markersize=7.5)[0]]

        for key in data_dict:
            handles.append(plt.plot([], 'o', color=self.opt_colors[key],
                                    markersize=5)[0])

        handles.append(plt.plot([], color=self.opt_colors['Best'],
                            linestyle='--')[0])

        if best_init[-1] > 0:
            handles[0] = \
                plt.plot([], '^', color=self.opt_colors[best_name[-1]],
                         markersize=7.5)[0]

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False,
                         prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'ConvZoom_f.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)
        del fig, ax

        # SEMILOG y --------------------------------------------------------
        def plot_log_runs(ax, data_dict, key):
            # Plotting function to optimise code
                ax.semilogy(data_dict[key][0], data_dict[key][1], 'o',
                        color=self.opt_colors[key], markersize=5)

                ax.semilogy(data_dict[key][3], data_dict[key][4], '^',
                        color=self.opt_colors[key], markersize=7.5)

                ax.semilogy(data_dict[key][6], data_dict[key][7], '.',
                        color=self.opt_colors['Best'], markersize=2)

        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Generations', fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        for key in data_dict:
            plot_log_runs(ax, data_dict, key)

        # Reference level
        for ri in range(len(ref_level)):
            ax.semilogy([0, max_runs_gen],
                    [ref_level[ri], ref_level[ri]],
                    linestyle=(0, (5, 10)),
                    color=self.opt_colors["Best"], linewidth=2)

        ax.grid()
        plt.ylim(self.yl1, self.yl2)

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)

        fig.tight_layout()

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False,
                         prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'Convlog_f.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)
        del fig, ax

        # LOGLOG y --------------------------------------------------------
        def plot_loglog_runs(ax, data_dict, key):
            # Plotting function to optimise code
            ax.loglog(data_dict[key][0], data_dict[key][1], 'o',
                        color=self.opt_colors[key], markersize=5)

            ax.loglog(data_dict[key][3], data_dict[key][4], '^',
                        color=self.opt_colors[key], markersize=7.5)

            ax.loglog(data_dict[key][6], data_dict[key][7], '.',
                        color=self.opt_colors['Best'], markersize=2)

        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Generations', fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        for key in data_dict:
            plot_loglog_runs(ax, data_dict, key)

        # Reference level
        for ri in range(len(ref_level)):
            ax.loglog([0, max_runs_gen],
                    [ref_level[ri], ref_level[ri]],
                    linestyle=(0, (5, 10)),
                    color=self.opt_colors["Best"], linewidth=2)

        ax.grid()
        plt.ylim(self.yl1, self.yl2)

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)

        fig.tight_layout()

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False,
                         prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'Convloglog_f.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)
        del fig, ax

        # TIME Plot ----------------------------------------------------
        def plot_time_runs(ax, data_dict, key):
            # Plotting function to optimise code
                ax.plot(data_dict[key][2], data_dict[key][1], 'o',
                        color=self.opt_colors[key], markersize=5)

                ax.plot(data_dict[key][5], data_dict[key][4], '^',
                        color=self.opt_colors[key], markersize=7.5)

                ax.plot(data_dict[key][8], data_dict[key][7], '.',
                        color=self.opt_colors['Best'], markersize=2)

        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Time ' + str(time_unit), fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        for key in data_dict:
            plot_time_runs(ax, data_dict, key)

        # Reference level
        for ri in range(len(ref_level)):
            ax.plot([0, max_run_time],
                        [ref_level[ri], ref_level[ri]],
                        linestyle=(0, (5, 10)),
                        color=self.opt_colors["Best"], linewidth=1)

        ax.grid()

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)
        ax.xaxis.set_major_formatter('{:.1f}'.format)

        plt.ylim(self.ylz1, self.ylz2)

        fig.tight_layout()

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False, prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'ConvTimeZoom_f.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)

        del fig, ax

        # TIME SMILOGy ----------------------------------------------------

        def plot_time_runs(ax, data_dict, key):
            # Plotting function to optimise code
            ax.semilogy(data_dict[key][2], data_dict[key][1], 'o',
                    color=self.opt_colors[key], markersize=5)

            ax.semilogy(data_dict[key][5], data_dict[key][4], '^',
                    color=self.opt_colors[key], markersize=7.5)

            ax.semilogy(data_dict[key][8], data_dict[key][7], '.',
                    color=self.opt_colors['Best'], markersize=2)

        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Time ' + str(time_unit), fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        for key in data_dict:
            plot_time_runs(ax, data_dict, key)

        # Reference level
        for ri in range(len(ref_level)):
            ax.plot([0, max_run_time],
                    [ref_level[ri], ref_level[ri]],
                    linestyle=(0, (5, 10)),
                    color=self.opt_colors["Best"], linewidth=1)

        ax.grid()

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)
        ax.xaxis.set_major_formatter('{:.1f}'.format)

        plt.ylim(self.yl1, self.yl2)

        fig.tight_layout()

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False, prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'ConvTimelog_f.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)

        del fig, ax

    def global_convergence(self, conv_data):
        """
        Plot the convergence plot (loss)
        Args:
            loss (ndarray): array with the convergence data
            (0) Gen (1) Eval (2) Fmin (3) Time in hr:min:sec
        """

        # Convergence data
        n_data = conv_data.shape[0]
        n_runs = []
        runs_gen = []
        runs_fmin = []
        max_gen = []
        fmin = []
        init_count = []
        best_init = []
        run_time = np.zeros((1, 3))
        opt_name = []
        best_name = []
        cost_ref = []

        # Max Time
        aux = conv_data[-1][10].decode('ASCII')
        aux_b = aux.split(':')
        aux_c = np.array(aux_b).astype(float)
        aux_d = aux_c.reshape((1, 3))
        aux_run_time = np.zeros((1, 3))
        aux_run_time = np.concatenate((aux_run_time, aux_d), axis=0)

        # Time
        max_hours = np.max(aux_run_time[:, 0])
        max_min = np.max(aux_run_time[:, 1])

        if max_hours >= 5:
            run_time_unit = aux_run_time[:, 0] + \
                            np.divide(aux_run_time[:, 1], 60) + \
                            np.divide(aux_run_time[:, 2], 3600)
            time_unit = '[hrs]'

        elif max_hours < 5 and max_min > 5:
            run_time_unit = np.multiply(aux_run_time[:, 0], 60) + \
                            aux_run_time[:, 1] + \
                            np.divide(aux_run_time[:, 2], 60)
            time_unit = '[min]'

        elif max_hours < 1 and max_min <= 5:
            run_time_unit = np.multiply(aux_run_time[:, 0], 3600) + \
                            np.multiply(aux_run_time[:, 1], 60) + \
                            aux_run_time[:, 2]
            time_unit = '[s]'

        # Dictionary with all data:
        # [0] n_runs [1] fmin  [2] time 
        # [3] best run [4] best fmin [5] best time
        # [6] seed runs [7] seed fmin [8] seed time

        data_dict = {}

        num_data = int(n_data / self.data_checkpoint)
        data_index = np.linspace(0, n_data - 1, num=num_data, dtype=int)
        for di in data_index:
            # N runs
            n_runs.append(conv_data[di][0])

            # Runs generations
            msg_counter = conv_data[di][2] * self.checkpoint
            runs_gen.append(msg_counter)

            # Runs F min
            runs_fmin.append(conv_data[di][3])

            # Max Generations
            max_msg_counter = conv_data[di][4] * self.checkpoint
            max_gen.append(max_msg_counter)

            # Best F min
            fmin.append(conv_data[di][5])

            # Init from rep
            init_count.append(conv_data[di][7])

            # Best Init from rep
            best_init.append(conv_data[di][8])

            # Optimiser name
            cost_ref.append(conv_data[di][9])

            # Time
            aux = conv_data[di][10].decode('ASCII')
            aux_b = aux.split(':')
            aux_c = np.array(aux_b).astype(float)
            aux_d = aux_c.reshape((1, 3))
            run_time = np.concatenate((run_time, aux_d), axis=0)

            # Optimiser name
            opt_name.append(conv_data[di][11].decode('ASCII'))

            # Optimiser name
            best_name.append(conv_data[di][12].decode('ASCII'))

            # Store in dictionary
            if opt_name[di] not in data_dict:
                data_dict[opt_name[di]] = [[], [], [], [], [], [], [], [], []]

            data_dict[opt_name[di]][0].append(runs_gen[di])
            data_dict[opt_name[di]][1].append(runs_fmin[di])

            if runs_fmin[di] <= fmin[di]:
                data_dict[opt_name[di]][3].append(runs_gen[di])
                data_dict[opt_name[di]][4].append(runs_fmin[di])

            if init_count[di] > 0:
                data_dict[opt_name[di]][6].append(runs_gen[di])
                data_dict[opt_name[di]][7].append(runs_fmin[di])


        # Best run
        best_run = int(conv_data[-1][6])

        # Minimum Loss
        min_loss_index = np.argmin(fmin)
        min_loss = fmin[min_loss_index]

        # Time
        max_hours = np.max(run_time[:, 0])
        max_min = np.max(run_time[:, 1])

        if max_hours >= 5:
            run_time_unit = run_time[:, 0] + \
                            np.divide(run_time[:, 1], 60) + \
                            np.divide(run_time[:, 2], 3600)
            time_unit = '[hrs]'

        elif max_hours < 5 and max_min > 5:
            run_time_unit = np.multiply(run_time[:, 0], 60) + \
                            run_time[:, 1] + \
                            np.divide(run_time[:, 2], 60)
            time_unit = '[min]'

        elif max_hours < 1 and max_min <= 5:
            run_time_unit = np.multiply(run_time[:, 0], 3600) + \
                            np.multiply(run_time[:, 1], 60) + \
                            run_time[:, 2]
            time_unit = '[s]'

        # N runs best
        max_nruns = max(n_runs)
        # nruns_best: [0] run [1] loss [2] init
        nruns_best = np.ones((max_nruns, 3)) * np.inf
        # nruns best optimiser
        nruns_opt = [None] * max_nruns

        # Reference level
        ref_level = list(set(cost_ref))
        max_runs_gen = max(runs_gen)
        max_run_time = max(run_time_unit)

        for ri in range(max_nruns):
            nruns_best[ri, 0] = ri + 1

        # Best data for each run
        for di in range(num_data):
            if runs_fmin[di] <= nruns_best[n_runs[di] - 1, 1]:
                nruns_best[n_runs[di] - 1, 1] = runs_fmin[di]
                nruns_best[n_runs[di] - 1, 2] = init_count[di]
                nruns_opt[n_runs[di] - 1] = opt_name[di]

        # Clear None values
        mi = 0
        counter = len(nruns_opt)
        while mi < counter:
            if nruns_opt[mi] is None:
                nruns_best = np.delete(nruns_best, (mi), axis=0)
                nruns_opt.pop(mi)
                counter = len(nruns_opt)
                mi -= 1
            mi += 1

        # Obtaining runs corresponding data
        n_runs = np.array(n_runs)
        runs_fmin = np.array(runs_fmin)
        runs_gen = np.array(runs_gen)
        init_count = np.array(init_count)
        cost_ref = np.array(cost_ref)
        runs_data = []
        for ri in range(max_nruns):
            nrun_ind = np.where(n_runs == ri + 1)
            if nrun_ind[0].shape[0] != 0:
                fmin_per_run = runs_fmin[nrun_ind].flatten()
                gen_per_run = runs_gen[nrun_ind].flatten()
                time_per_run = run_time_unit[nrun_ind].flatten()
                init_count_per_run = init_count[nrun_ind].flatten()
                cost_ref_per_run = cost_ref[nrun_ind].flatten()
                runs_data.append(
                    [nrun_ind, gen_per_run, fmin_per_run, time_per_run,
                     opt_name[nrun_ind[0][0]], init_count_per_run,
                     cost_ref_per_run])

        # PLOT y --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Generations', fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        # Lines
        for ri in range(len(runs_data)):
            if runs_data[ri][5][0] > 0:
                # ax.semilogy(runs_data[ri][1], runs_data[ri][2], "o",
                #            markerfacecolor=opt_colors[runs_data[ri][4]],
                #            color=opt_colors["Best"], markersize=5)

                ax.plot(runs_data[ri][1], runs_data[ri][2], "o",
                        color=self.opt_colors[runs_data[ri][4]],
                        markersize=5)

                ax.plot(runs_data[ri][1], runs_data[ri][2], ".",
                        color=self.opt_colors["Best"], markersize=2)
            else:

                ax.plot(runs_data[ri][1], runs_data[ri][2], "o",
                        color=self.opt_colors[runs_data[ri][4]],
                        markersize=5)

        fmin_aux = fmin[0]
        gen_aux = runs_gen[0]
        for di in range(num_data):
            if fmin[di] <= fmin_aux:
                gen_aux_max = max(gen_aux, runs_gen[di])
                if best_init[di] > 0:
                    # ax.semilogy(gen_aux_max, fmin[di], '^',
                    #          color=opt_colors['Best'],
                    #          linewidth=1.5,
                    #          markerfacecolor=opt_colors[best_name[di]],
                    #          markeredgewidth=1.5, markersize=7.5)

                    ax.plot(gen_aux_max, fmin[di], '^',
                            color=self.opt_colors[best_name[di]],
                            markersize=7.5)

                    ax.plot(gen_aux_max, fmin[di], ".",
                            color=self.opt_colors["Best"], markersize=2)
                else:
                    ax.plot(gen_aux_max, fmin[di], '^',
                            color=self.opt_colors[best_name[di]],
                            markersize=7.5)

                fmin_aux = fmin[di]
                gen_aux = runs_gen[di]

        # Reference level
        for ri in range(len(ref_level)):
            ax.plot([0, max_runs_gen],
                    [ref_level[ri], ref_level[ri]],
                    linestyle=(0, (5, 10)),
                    color=self.opt_colors["Best"], linewidth=2)

        ax.grid()
        plt.ylim(self.ylz1, self.ylz2)

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)

        fig.tight_layout()

        labels = [best_name[-1] +
                  '\nMin Cost: ' + str("{:2.3f}".format(min_loss))]

        for key in data_dict:
            labels.append(key)

        labels.append(r'Ref Cost $\bar{\epsilon}$')

        handles = [plt.plot([], '^', color=self.opt_colors[best_name[-1]],
                            markersize=7.5)[0]]

        for key in data_dict:
            handles.append(plt.plot([], 'o', color=self.opt_colors[key],
                                    markersize=5)[0])

        handles.append(plt.plot([], color=self.opt_colors['Best'],
                                linestyle='--')[0])

        if best_init[-1] > 0:
            handles[0] = \
                plt.plot([], '^', color=self.opt_colors[best_name[-1]],
                         markersize=7.5)[0]

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False,
                         prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'ConvZoom.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)
        del fig, ax

        # SEMILOG y --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Generations', fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        # Lines
        for ri in range(len(runs_data)):
            if runs_data[ri][5][0] > 0:
                #ax.semilogy(runs_data[ri][1], runs_data[ri][2], "o",
                #            markerfacecolor=opt_colors[runs_data[ri][4]],
                #            color=opt_colors["Best"], markersize=5)

                ax.semilogy(runs_data[ri][1], runs_data[ri][2], "o",
                            color=self.opt_colors[runs_data[ri][4]], markersize=5)

                ax.semilogy(runs_data[ri][1], runs_data[ri][2], ".",
                            color=self.opt_colors["Best"], markersize=2)
            else:

                ax.semilogy(runs_data[ri][1], runs_data[ri][2], "o",
                            color=self.opt_colors[runs_data[ri][4]], markersize=5)

        fmin_aux = fmin[0]
        gen_aux = runs_gen[0]
        for di in range(num_data):
            if fmin[di] <= fmin_aux:
                gen_aux_max = max(gen_aux, runs_gen[di])
                if best_init[di] > 0:
                    #ax.semilogy(gen_aux_max, fmin[di], '^',
                    #          color=opt_colors['Best'],
                    #          linewidth=1.5,
                    #          markerfacecolor=opt_colors[best_name[di]],
                    #          markeredgewidth=1.5, markersize=7.5)

                    ax.semilogy(gen_aux_max, fmin[di], '^',
                              color=self.opt_colors[best_name[di]],
                              markersize=7.5)

                    ax.semilogy(gen_aux_max, fmin[di], ".",
                              color=self.opt_colors["Best"], markersize=2)
                else:
                    ax.semilogy(gen_aux_max, fmin[di], '^',
                              color=self.opt_colors[best_name[di]],
                              markersize=7.5)

                fmin_aux = fmin[di]
                gen_aux = runs_gen[di]

        # Reference level
        for ri in range(len(ref_level)):
            ax.semilogy([0, max_runs_gen],
                        [ref_level[ri], ref_level[ri]],
                        linestyle=(0, (5, 10)),
                        color=self.opt_colors["Best"], linewidth=2)

        ax.grid()
        plt.ylim(self.yl1, self.yl2)

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)

        fig.tight_layout()


        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False, prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'ConvTimelog.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)
        del fig, ax

        # LOGLOG Plot ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Generations', fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        for ri in range(len(runs_data)):
            if runs_data[ri][5][0] > 0:
                #ax.loglog(runs_data[ri][1], runs_data[ri][2], "o",
                #            markerfacecolor=opt_colors[runs_data[ri][4]],
                #            color=opt_colors["Best"], markersize=5)

                ax.loglog(runs_data[ri][1], runs_data[ri][2], "o",
                            color=self.opt_colors[runs_data[ri][4]], markersize=5)

                ax.loglog(runs_data[ri][1], runs_data[ri][2], ".",
                            color=self.opt_colors["Best"], markersize=2)
            else:

                ax.loglog(runs_data[ri][1], runs_data[ri][2], "o",
                            color=self.opt_colors[runs_data[ri][4]], markersize=5)

        fmin_aux = fmin[0]
        gen_aux = runs_gen[0]
        for di in range(num_data):
            if fmin[di] <= fmin_aux:
                gen_aux_max = max(gen_aux, runs_gen[di])
                if best_init[di] > 0:
                    #ax.loglog(gen_aux_max, fmin[di], '^',
                    #            color=opt_colors['Best'],
                    #            linewidth=1.5,
                    #            markerfacecolor=opt_colors[best_name[di]],
                    #            markeredgewidth=1.5, markersize=7.5)

                    ax.loglog(gen_aux_max, fmin[di], '^',
                              color=self.opt_colors[best_name[di]],
                              markersize=7.5)

                    ax.loglog(gen_aux_max, fmin[di], ".",
                                color=self.opt_colors["Best"], markersize=2)
                else:
                    ax.loglog(gen_aux_max, fmin[di], '^',
                                color=self.opt_colors[best_name[di]],
                                markersize=7.5)

                fmin_aux = fmin[di]
                gen_aux = runs_gen[di]

        # Reference level
        for ri in range(len(ref_level)):
            ax.loglog([0, max_runs_gen],
                      [ref_level[ri], ref_level[ri]],
                      linestyle=(0, (5, 10)),
                      color=self.opt_colors["Best"], linewidth=2)

        plt.ylim(self.yl1, self.yl2)
        ax.grid()
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.LogFormatterSciNotation())
        ax.get_yaxis().set_minor_formatter(
           matplotlib.ticker.LogFormatterSciNotation())
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.LogFormatterSciNotation())

        ax.tick_params(axis='both', labelsize=14 * self.font_factor)

        fig.tight_layout()

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False, prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'Convloglog.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)
        del fig, ax

        # Init --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Number of seeds', fontsize=16 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=16 * self.font_factor)

        for mi in range(len(nruns_opt)):
            ax.semilogy(nruns_best[mi, 2], nruns_best[mi, 1], "o",
                        color=self.opt_colors[nruns_opt[mi]], markersize=5)

        nruns_best_index = np.argmin(nruns_best[:, 1])

        ax.semilogy(nruns_best[nruns_best_index, 2],
                    nruns_best[nruns_best_index, 1], "^",
                    color=self.opt_colors['Best'],
                    markerfacecolor=self.opt_colors[nruns_opt[nruns_best_index]],
                    markeredgewidth=1.5, markersize=7.5)

        ax.grid()

        ax.tick_params(axis='both', labelsize=14 * self.font_factor)

        fig.tight_layout()

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False, prop={"size": 12}, ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'InitLoss.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)
        del fig, ax

        # TIME Plot ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Time ' + str(time_unit), fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        for ri in range(len(runs_data)):
            if runs_data[ri][5][0] > 0:
                # ax.semilogy(runs_data[ri][3], runs_data[ri][2], "o",
                #            markerfacecolor=opt_colors[runs_data[ri][4]],
                #            color=opt_colors["Best"], markersize=5)

                ax.plot(runs_data[ri][3], runs_data[ri][2], "o",
                        color=self.opt_colors[runs_data[ri][4]], markersize=5)

                ax.plot(runs_data[ri][3], runs_data[ri][2], ".",
                        color=self.opt_colors["Best"], markersize=2)
            else:

                ax.plot(runs_data[ri][3], runs_data[ri][2], "o",
                        color=self.opt_colors[runs_data[ri][4]], markersize=5)

            ax.plot(runs_data[ri][3], runs_data[ri][6], "_",
                    color=self.opt_colors["Best"], markersize=8)

        fmin_aux = fmin[0]
        run_time_unit_aux = run_time_unit[0]
        for di in range(num_data):

            if fmin[di] <= fmin_aux:
                if best_init[di] > 0:
                    # ax.semilogy(run_time_unit[di], fmin[di],
                    #            '^',
                    #            color=opt_colors['Best'],
                    #            linewidth=1.5,
                    #            markerfacecolor=opt_colors[best_name[di]],
                    #            markeredgewidth=1.5, markersize=7.5)

                    ax.plot(run_time_unit[di], fmin[di],
                            '^',
                            color=self.opt_colors[best_name[di]],
                            markersize=7.5)

                    ax.plot(run_time_unit[di], fmin[di], ".",
                            color=self.opt_colors["Best"], markersize=2)

                else:

                    ax.plot([run_time_unit_aux,
                             run_time_unit[di]],
                            [fmin_aux, fmin[di]],
                            '-',
                            color=self.opt_colors['Best'],
                            linewidth=2)

                    ax.plot(run_time_unit[di], fmin[di],
                            '^',
                            color=self.opt_colors[best_name[di]],
                            markersize=7.5)

                fmin_aux = fmin[di]
                run_time_unit_aux = run_time_unit[di]

        # Reference level
        for ri in range(len(ref_level)):
            ax.plot([0, max_run_time],
                        [ref_level[ri], ref_level[ri]],
                        linestyle=(0, (5, 10)),
                        color=self.opt_colors["Best"], linewidth=1)

        ax.grid()

        ax.tick_params(axis='both', labelsize=12 * self.font_factor)
        ax.xaxis.set_major_formatter('{:.1f}'.format)

        plt.ylim(self.ylz1, self.ylz2)

        fig.tight_layout()

        handles[0] = plt.plot([], '-^', color=self.opt_colors[best_name[-1]],
                              markersize=7.5)[0]

        if best_init[-1] > 0:
            handles[0] = \
                plt.plot([], '-^', color=self.opt_colors[best_name[-1]],
                         markersize=7.5)[0]

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False, prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'ConvTimeZoom.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)

        del fig, ax

        # TIME SMILOGy ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(self.h_dim, self.v_dim))

        ax.set_xlabel('Time ' + str(time_unit), fontsize=15 * self.font_factor)
        ax.set_ylabel('Cost', fontsize=15 * self.font_factor)

        for ri in range(len(runs_data)):
            if runs_data[ri][5][0] > 0:
                # ax.semilogy(runs_data[ri][3], runs_data[ri][2], "o",
                #            markerfacecolor=opt_colors[runs_data[ri][4]],
                #            color=opt_colors["Best"], markersize=5)

                ax.semilogy(runs_data[ri][3], runs_data[ri][2], "o",
                            color=self.opt_colors[runs_data[ri][4]], markersize=5)

                ax.semilogy(runs_data[ri][3], runs_data[ri][2], ".",
                            color=self.opt_colors["Best"], markersize=2)
            else:

                ax.semilogy(runs_data[ri][3], runs_data[ri][2], "o",
                            color=self.opt_colors[runs_data[ri][4]], markersize=5)

            ax.semilogy(runs_data[ri][3], runs_data[ri][6], "_",
                        color=self.opt_colors["Best"], markersize=8)

        fmin_aux = fmin[0]
        run_time_unit_aux = run_time_unit[0]
        for di in range(num_data):

            if fmin[di] <= fmin_aux:
                if best_init[di] > 0:
                    # ax.semilogy(run_time_unit[di], fmin[di],
                    #            '^',
                    #            color=opt_colors['Best'],
                    #            linewidth=1.5,
                    #            markerfacecolor=opt_colors[best_name[di]],
                    #            markeredgewidth=1.5, markersize=7.5)

                    ax.semilogy(run_time_unit[di], fmin[di],
                                '^',
                                color=self.opt_colors[best_name[di]],
                                markersize=7.5)

                    ax.semilogy(run_time_unit[di], fmin[di], ".",
                                color=self.opt_colors["Best"], markersize=2)

                else:

                    ax.semilogy([run_time_unit_aux,
                                 run_time_unit[di]],
                                [fmin_aux, fmin[di]],
                                '-',
                                color=self.opt_colors['Best'],
                                linewidth=2)

                    ax.semilogy(run_time_unit[di], fmin[di],
                                '^',
                                color=self.opt_colors[best_name[di]],
                                markersize=7.5)

                fmin_aux = fmin[di]
                run_time_unit_aux = run_time_unit[di]

        # Reference level
        for ri in range(len(ref_level)):
            ax.semilogy([0, max_run_time],
                        [ref_level[ri], ref_level[ri]],
                        linestyle=(0, (5, 10)),
                        color=self.opt_colors["Best"], linewidth=1)

        ax.grid()

        ax.tick_params(axis='both', labelsize=14 * self.font_factor)

        plt.ylim(self.yl1, self.yl2)

        fig.tight_layout()

        handles[0] = plt.plot([], '-^', color=self.opt_colors[best_name[-1]],
                              markersize=7.5)[0]

        if best_init[-1] > 0:
            handles[0] = \
                plt.plot([], '-^', color=self.opt_colors[best_name[-1]],
                         markersize=7.5)[0]

        lgd = plt.legend(handles=handles, labels=labels,
                         fancybox=False, prop={"size": 10 * self.font_factor},
                         ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.35, 0.99))

        figname = 'ConvTime.' + self.fig_format
        path = self.rep_path + figname
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)

        del fig, ax

    def path_2d(self, y, proj):

        nrun = self.nrun

        style = '-or'

        y[0] = self.ys
        y[-1] = self.yt

        # Circle Obstacles
        xobs = proj.xobs  # circle centre x-coords
        yobs = proj.yobs  # circle centre y-coords
        robs = proj.robs  # radii

        x = np.linspace(self.xs, self.xt, self.n_param)

        # Calculate real length
        length = np.sum(np.sqrt((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2))

        figname = 'Sol'
        path = self.plot_path_in + figname
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.plot(x, y, style, markersize=2,
                 label='Run ' + str(nrun) + ' - Length: '
                       + str("{:2.3f}".format(length)))

        for i in range(len(xobs)):
            circle = plt.Circle((xobs[i], yobs[i]), robs[i], color='k')
            matplotlib.pyplot.text(xobs[i], yobs[i], s=str(i+1),
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   color='w')
            ax.add_patch(circle)

        circle = plt.Circle((self.xs, self.ys), 0.3, color='b')
        ax.add_patch(circle)
        matplotlib.pyplot.text(0.85, 0.8, s='$A$',
                              horizontalalignment='center',
                              verticalalignment='center',
                              color='b',
                              fontsize=10 * self.font_factor)

        circle = plt.Circle((self.xt, self.yt), 0.3, color='r')
        ax.add_patch(circle)
        matplotlib.pyplot.text(29.3, 0.8, s='$B$',
                              horizontalalignment='center',
                              verticalalignment='center',
                              color='r',
                              fontsize=10 * self.font_factor)


        plt.xlim(self.xs, self.xt)
        plt.ylim(self.lower_limit, self.upper_limit)
        #plt.ylim(self.lower_limit, self.upper_limit)
        plt.gca().set_aspect('equal')
        #plt.title(title, fontdict=font1)
        plt.xlabel('x', fontsize=14 * self.font_factor)
        plt.ylabel('y', fontsize=14 * self.font_factor)
        ax.tick_params(axis='both', labelsize=12 * self.font_factor)

        lgd = plt.legend(fancybox=False, prop={"size": 14 * self.font_factor}, ncol=1,
                         frameon=False, loc='upper center',
                         bbox_to_anchor=(0.5, 1.1))
        
        # Save in controller as well
        path = self.rep_path + figname + '.' + self.fig_format
        fig.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig)

    def comparison_plots(self, proj, prob_id, runs_file, best_run_id, opt_id,
                         wrapper_conv, wrapper_sol, avg_fmin, std_dev, n_comp):


        n_runs = len(runs_file)

        # Plot path
        plot_path = self.results_path + '/' + prob_id + \
                    'Comp' + str(self.n_param) + '_' + str(n_comp)
        self.files_man.create_folder(plot_path)

        # Conv Plot
        #fig_conv, ax_conv = plt.subplots(figsize=(8, 6))

        # Time Plot
        fig_time, ax_time = plt.subplots(figsize=(12, 7))

        # Path Plot
        fig_path, ax_path = plt.subplots(figsize=(12, 8))

        # Circle Obstacles
        xobs = proj.xobs  # circle centre x-coords
        yobs = proj.yobs  # circle centre y-coords
        robs = proj.robs  # radii

        x = np.linspace(self.xs, self.xt, self.n_param)
        length = []
        t_aux = 15
        for ri in range(n_runs):

            # File Paths
            run_id = self.results_path + '/' + prob_id + \
                     runs_file[ri] + '/0'

            # Convergence path
            conv_path = run_id + '/' + 'conv.dat'

            # Read Convergence
            conv = \
                np.genfromtxt(conv_path,
                              dtype=(
                                  int, int, int, float, int, float, int, int,
                                  int,
                                  float, 'S11', 'S10', 'S10'),
                              delimiter="\t")

            num_data = int(len(conv) / self.data_checkpoint)

            n_gen = []
            fmin = []
            run_time = np.zeros((1, 3))

            for di in range(num_data):
                n_gen.append(conv[di][4] * self.checkpoint)
                fmin.append(conv[di][5])

                # Time
                aux = conv[di][10].decode('ASCII')
                aux_b = aux.split(':')
                aux_c = np.array(aux_b).astype(float)
                aux_d = aux_c.reshape((1, 3))
                run_time = np.concatenate((run_time, aux_d), axis=0)

            #ax_conv.loglog(n_gen, fmin, "-o",
            #               color=self.opt_colors[opt_id[ri]], markersize=5)

            max_hours = np.max(run_time[:, 0])
            max_min = np.max(run_time[:, 1])

            if max_hours >= 5:
                run_time_unit = run_time[:, 0] + \
                                np.divide(run_time[:, 1], 60) + \
                                np.divide(run_time[:, 2], 3600)
                time_unit = '[hrs]'

            elif max_hours < 5 and max_min > 5:
                run_time_unit = np.multiply(run_time[:, 0], 60) + \
                                run_time[:, 1] + \
                                np.divide(run_time[:, 2], 60)
                time_unit = '[min]'

            elif max_hours < 1 and max_min <= 5:
                run_time_unit = np.multiply(run_time[:, 0], 3600) + \
                                np.multiply(run_time[:, 1], 60) + \
                                run_time[:, 2]
                time_unit = '[s]'

            ax_time.loglog(run_time_unit[1:], fmin, "-o",
                           color=self.opt_colors[opt_id[ri]], markersize=5)

            ax_time.errorbar(t_aux, avg_fmin[ri],
                           yerr=std_dev[ri], marker="<",
                           ecolor=self.opt_colors[opt_id[ri]],
                           elinewidth=2,
                           markerfacecolor=self.opt_colors[opt_id[ri]],
                           color='k', markersize=8,
                           markeredgewidth=2)

            t_aux += 5

            # Model path
            model_path = self.results_path + '/' + prob_id + \
                     runs_file[ri] + '/' + best_run_id[ri] + '/' + 'model.dat'

            # Read best model
            y = np.loadtxt(model_path)

            y[0] = self.ys
            y[-1] = self.yt

            # Calculate real length
            length.append(np.sum(
                np.sqrt((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2)))

            ax_path.plot(x, y, '-o', markersize=4, linewidth=2,
                         color=self.opt_colors[opt_id[ri]])

        # Convergence Time ----------------------------------------------

        wrapper_gen = []
        wrapper_fmin = []
        wrapper_time = np.zeros((1, 3))

        num_data = int(len(wrapper_conv) / self.data_checkpoint)
        
        for di in range(num_data):
            wrapper_gen.append(wrapper_conv[di][4] * self.checkpoint)
            wrapper_fmin.append(wrapper_conv[di][5])

            # Time
            aux = wrapper_conv[di][10].decode('ASCII')
            aux_b = aux.split(':')
            aux_c = np.array(aux_b).astype(float)
            aux_d = aux_c.reshape((1, 3))
            wrapper_time = np.concatenate((wrapper_time, aux_d), axis=0)

        if max_hours >= 5:
            wrapper_time_unit = wrapper_time[:, 0] + \
                            np.divide(wrapper_time[:, 1], 60) + \
                            np.divide(wrapper_time[:, 2], 3600)
            time_unit = '[hrs]'

        elif max_hours < 5 and max_min > 5:
            wrapper_time_unit = np.multiply(wrapper_time[:, 0], 60) + \
                            wrapper_time[:, 1] + \
                            np.divide(wrapper_time[:, 2], 60)
            time_unit = '[min]'

        elif max_hours < 1 and max_min <= 5:
            wrapper_time_unit = np.multiply(wrapper_time[:, 0], 3600) + \
                            np.multiply(wrapper_time[:, 1], 60) + \
                            wrapper_time[:, 2]
            time_unit = '[s]'

        ax_time.loglog(wrapper_time_unit[1:], wrapper_fmin, "-o",
                       color=self.opt_colors['SPO'],
                       markersize=5)

        ax_time.errorbar(35, avg_fmin[-1],
                       yerr=std_dev[-1], marker="<",
                       ecolor=self.opt_colors['SPO'],
                       elinewidth=2,
                       markerfacecolor=self.opt_colors['SPO'],
                       color='k', markersize=8,
                       markeredgewidth=2)

        ax_time.set_xscale("log")
        ax_time.set_yscale("log")
        ax_time.set_xlabel('Time ' + str(time_unit), fontsize=16 * self.font_factor)
        ax_time.set_ylabel('Cost', fontsize=16 * self.font_factor)

        ax_time.set_ylim(30, 3000)
        ax_time.grid()

        ax_time.get_yaxis().set_major_formatter(
            matplotlib.ticker.LogFormatterSciNotation())
        ax_time.get_yaxis().set_minor_formatter(
            matplotlib.ticker.LogFormatterSciNotation())
        #ax_time.get_xaxis().set_major_formatter(
        #    matplotlib.ticker.LogFormatterSciNotation())
        ax_time.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax_time.tick_params(axis='both', labelsize=14 * self.font_factor)

        fig_time.tight_layout()

        labels = opt_id.copy()
        labels.append('SPO')

        labels.append('PymooPSO $\mu,\sigma$')
        labels.append('PymooGA $\mu,\sigma$')
        labels.append('PymooCMAES $\mu,\sigma$')
        labels.append('MCS $\mu,\sigma$')
        labels.append('SPO $\mu,\sigma$')

        handles = []

        for ri in opt_id:
            handles.append(plt.plot([], 'o',
                                    color=self.opt_colors[ri],
                                    markersize=5)[0])

        handles.append(plt.plot([], 'o',
                                linewidth=3,
                                color=self.opt_colors['SPO'],
                                markersize=5)[0])


        for ri in opt_id:
            handles.append(plt.errorbar([],[],[], marker="<",
                           color=self.opt_colors[ri],
                           ecolor=self.opt_colors[ri],
                           elinewidth=2,
                           markerfacecolor=self.opt_colors[ri],
                           markeredgecolor='k', markersize=8,
                           markeredgewidth=2)[0])

        handles.append(plt.errorbar([],[],[], marker="<",
                       color=self.opt_colors['SPO'],
                       ecolor=self.opt_colors['SPO'],
                       elinewidth=2,
                       markerfacecolor=self.opt_colors['SPO'],
                       markeredgecolor='k', markersize=8,
                       markeredgewidth=2)[0])

        lgd = fig_time.legend(handles=handles, labels=labels,
                         fancybox=False, prop={"size": 12 * self.font_factor}, ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.5, 0.98))

        figname = '/ConvCompTime.' + self.fig_format
        path = plot_path + figname
        fig_time.savefig(path, bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig_time)

        # Convergence ----------------------------------------------

        # ax_conv.loglog(wrapper_gen, wrapper_fmin, "-o",
        #                color=self.opt_colors['SPO'],
        #                markersize=5)
        #
        # ax_conv.set_xlabel('Generations', fontsize=16 * self.font_factor)
        # ax_conv.set_ylabel('Loss', fontsize=16 * self.font_factor)
        #
        # plt.ylim(min(wrapper_fmin), max(wrapper_fmin))
        # ax_conv.grid()
        #
        # ax_conv.get_yaxis().set_major_formatter(
        #     matplotlib.ticker.LogFormatterSciNotation())
        # ax_conv.get_yaxis().set_minor_formatter(
        #     matplotlib.ticker.LogFormatterSciNotation())
        # ax_conv.get_xaxis().set_major_formatter(
        #     matplotlib.ticker.LogFormatterSciNotation())
        #
        # ax_conv.tick_params(axis='both', labelsize=14 * self.font_factor)
        #
        # fig_conv.tight_layout()
        #
        # labels = opt_id
        #
        # handles = []
        #
        # for ri in opt_id:
        #     handles.append(plt.plot([], 'o',
        #                             color=self.opt_colors[ri],
        #                             markersize=5)[0])
        #
        # handles.append(plt.plot([], 'o',
        #                         linewidth=3,
        #                         markerfacecolor=self.opt_colors['SPO'],
        #                         color='k',
        #                         markersize=5,
        #                         markeredgewidth=1.5)[0])
        #
        # lgd = fig_conv.legend(handles=handles, labels=labels,
        #                       fancybox=False,
        #                       prop={"size": 12 * self.font_factor}, ncol=1,
        #                       frameon=False, loc='upper right',
        #                       bbox_to_anchor=(1.35, 0.95))
        #
        # figname = '/ConvComp.' + self.fig_format
        # path = plot_path + figname
        # fig_conv.savefig(path, bbox_extra_artists=(lgd,),
        #                  bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        # plt.close(fig_conv)

        # Path Plot  -------------------------------------------------------
        # circles
        y_w = wrapper_sol
        y_w[0] = self.ys
        y_w[-1] = self.yt
        # Wrapper model
        # Calculate real length
        length.append(np.sum(
            np.sqrt((x[:-1] - x[1:]) ** 2 + (y_w[:-1] - y_w[1:]) ** 2)))

        ax_path.plot(x, y_w, 'o', markersize=4, linewidth=3,
                     markeredgewidth=1.5,
                     color=self.opt_colors['SPO'])
                     #color='k', markerfacecolor=self.opt_colors['Wrapper'])

        for i in range(len(xobs)):
            circle = plt.Circle((xobs[i], yobs[i]), robs[i], color='k')
            ax_path.add_patch(circle)

        plt.xlim(self.xs, self.xt)
        plt.ylim(-15, 15)
        plt.gca().set_aspect('equal')

        ax_path.set_xlabel('x', fontsize=14 * self.font_factor)
        ax_path.set_ylabel('y', fontsize=14 * self.font_factor)
        ax_path.tick_params(axis='both', labelsize=14 * self.font_factor)

        #labels = []
        #for ri in range(len(opt_id)-1):
            #labels.append(opt_id[ri] + ' L: ' + str("{:2.3f}".format(length[ri])))

        labels = opt_id.copy()

        labels.append('SPO')

        handles = []

        for ri in opt_id:
            handles.append(plt.plot([], 'o',
                                    color=self.opt_colors[ri],
                                    markersize=5)[0])

        handles.append(plt.plot([], 'o',
                                linewidth=3,
                                color=self.opt_colors['SPO'],
                                markersize=5)[0])

        lgd = fig_path.legend(handles=handles, labels=labels,
                         fancybox=False, prop={"size": 12 * self.font_factor}, ncol=1,
                         frameon=False, loc='upper right',
                         bbox_to_anchor=(1.05, 0.97))

        figname = '/Sol.' + self.fig_format
        path = plot_path + figname
        fig_path.savefig(path, bbox_extra_artists=(lgd,),
                         bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig_path)

    def comparison_errorbar(self, prob_id,opt_id, avg_fmin, std_dev, n_comp):


        # Plot path
        plot_path = self.results_path + '/' + prob_id + \
                    'Comp' + str(self.n_param) + '_' + str(n_comp)
        self.files_man.create_folder(plot_path)

        # Errorbar ----------------------------------------------

        fig_error, ax_error = plt.subplots(figsize=(8, 7))

        x_pos = np.arange(len(opt_id))

        for ri in range(len(opt_id)):
            ax_error.errorbar(x_pos[ri], avg_fmin[ri], yerr=std_dev[ri],
                         marker="_",
                         ecolor=self.opt_colors[opt_id[ri]],
                         elinewidth=30,
                         markerfacecolor=self.opt_colors[opt_id[ri]],
                         color='w', markersize=30,
                         markeredgewidth=2)

        ax_error.errorbar(x_pos[-1] + 1, avg_fmin[-1],
                          yerr=std_dev[-1], marker="_",
                          ecolor=self.opt_colors['SPO'],
                          elinewidth=30,
                          markerfacecolor=self.opt_colors['SPO'],
                          color='w', markersize=30,
                          markeredgewidth=2)

        #ax_error.set_yscale("log")
        ax_error.yaxis.grid(True)

        labels = ['Pymoo\nPSO', 'Pymoo\nGA', 'Pymoo\nCMAES', 'MCS', 'SPO']
        x_pos = np.arange(len(labels))

        ax_error.set_ylabel('Cost', fontsize=16 * self.font_factor)
        ax_error.set_xticks(x_pos)
        ax_error.set_xticklabels(labels, rotation=90)
        ax_error.tick_params(axis='both', labelsize=14 * self.font_factor)

        figname = '/ConvError.' + self.fig_format
        path = plot_path + figname
        fig_error.savefig(path, bbox_inches='tight', format=self.fig_format, dpi=self.dpi)
        plt.close(fig_error)