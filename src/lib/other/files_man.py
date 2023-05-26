import os
import argparse
from pathlib import Path
from mpi4py import MPI


class FilesMan(object):
    def __init__(self, args):

        # Optimiser Parameters

        self.nrun = args.nrun

        self.run_id = args.run_id

        self.n_obj = args.n_obj

        # Path
        current_path_aux = Path(os.path.abspath(os.getcwd()))

        # Necessary to include results ouiutside src folder
        self.current_path = current_path_aux.parents[2]

        # Containing folders to save run information
        self.results_path = str(self.current_path) + '/results'

        # Call results folder
        self.results_folder()

        # Create sub folders with each initialisation Run
        self.init_run_id = ' '
        self.run_path = ' '
        self.plot_path = ' '
        self.plot_path_in = ' '
        self.his_path = ' '
        self.rep_his_path = ' '

        self.paths_init()

    def paths_init(self):

        # Create sub folders with each initialisation Run
        self.init_run_id = self.run_id + '/' + str(self.nrun)
        self.run_id_path = str(self.current_path) + '/results/' + self.run_id
        self.super_path = self.run_id_path + '/0/'
        self.run_path = str(self.current_path) + '/results/' + self.init_run_id
        self.plot_path = str(self.current_path) + '/results/' + self.run_id + '/' + str(self.nrun)
        self.plot_path_in = self.plot_path + '/plots/'
        self.his_path = self.run_path + '/his/'
        self.rep_his_path = self.his_path + '/rep/'

    def results_folder(self):
        """
        Creates directory where results will be allocated
        """

        # Create Results Folder

        # Get my rank
        rank = MPI.COMM_WORLD.Get_rank()
        try:
            # Create target Directory
            os.mkdir(self.results_path)
            if rank == 0:
                print('Directory "Results" created')

        except FileExistsError:
            if rank == 0:
                print('Directory "Results" already exists')

        run_path = str(self.current_path) + '/results/' + self.run_id

        try:
            # Create target Directory
            os.mkdir(run_path)
            if rank == 0:
                print('Directory {} created'.format(self.run_id))

        except FileExistsError:
            if rank == 0:
                print('Directory {} already exists'.format(self.run_id))

    def run_folder(self):
        """
        Creates directory for each run test within results folder
        where each result will be allocated
        """

        # Get my rank
        rank = MPI.COMM_WORLD.Get_rank()

        try:
            # Create target Directory
            os.mkdir(self.run_path)

        except FileExistsError:
            if rank == 0:
                print('Directory for Run {} already exists'.format(self.nrun))

        # History folder
        try:
            # Create target Directory
            os.mkdir(self.his_path)

        except FileExistsError:
            if rank == 0:
                print('History directory for Run {} '
                      'already exists'.format(self.nrun))

        # Repository History folder
        if self.n_obj > 1:
            try:
                # Create target Directory
                os.mkdir(self.rep_his_path)

            except FileExistsError:
                if rank == 0:
                    print('Repository History directory for Run'
                          ' {} already exists'.format(self.nrun))

    def plot_folder(self):
        """
        Creates directory within each run test folder named Plots
        where each plot will be allocated
        """
        try:
            # Create target Directory
            os.mkdir(self.plot_path + '/plots')
            print('Directory Plots created')
        except FileExistsError:
            print('Directory Plots already exists')


    def create_folder(self, path):
        """
        Creates directory within each run test folder named Plots
        where each plot will be allocated
        """
        try:
            # Create target Directory
            os.mkdir(path)
            print('Directory created')
        except FileExistsError:
            print('Directory already exists')