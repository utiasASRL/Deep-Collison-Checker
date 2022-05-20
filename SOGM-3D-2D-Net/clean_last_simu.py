#
#
#      0==============================0
#      |    Deep Collision Checker    |
#      0==============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to process data from rosbags
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import numpy as np
import os
from os import listdir, makedirs
from os.path import join, exists
import shutil

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


def main():

    ######
    # Init
    ######

    # Data path
    root_path = '../Data/Simulation_v2'

    # Path to the bag files
    runs_path = join(root_path, 'simulated_runs')
    
    # List sessions
    run_folders = np.sort([f for f in listdir(runs_path)])

    for run in run_folders[-1:]:

        last_run = join(runs_path, run)
        log_path = join(last_run, "logs-{:s}".format(run))

        # List files/folders to remove
        pathes_to_remove = [join(last_run, 'classified_frames')]
        pathes_to_remove += [join(last_run, 'raw_data.bag')]
        pathes_to_remove += [join(log_path, 'processed_data.pickle')]
        pathes_to_remove += [join(log_path, 'collider_data.pickle')]

        print('Removing:')
        for rm_path in pathes_to_remove:

            print(rm_path, exists(rm_path))

            if exists(rm_path):
                if os.path.isdir(rm_path):
                    shutil.rmtree(rm_path)
                else:
                    os.remove(rm_path)

    return


if __name__ == '__main__':

    main()
