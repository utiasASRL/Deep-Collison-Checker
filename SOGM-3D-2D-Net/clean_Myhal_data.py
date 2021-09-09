#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to clean the data of MyhalSim dataset
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
from genericpath import isdir
import numpy as np
from utils.ply import read_ply

#from mayavi import mlab
from os import listdir, makedirs, remove
from os.path import join, exists

import shutil


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utilities
#       \***************/
#


def load_gt_poses(simu_path, day):


    # Load gt from ply files
    gt_ply_file = join(simu_path, day, 'gt_pose.ply')
    data = read_ply(gt_ply_file)
    gt_T = np.vstack([data['pos_x'], data['pos_y'], data['pos_z']]).T
    gt_Q = np.vstack([data['rot_x'], data['rot_y'], data['rot_z'], data['rot_w']]).T

    # Times
    day_gt_t = data['time']

    # print(day_gt_t)
    # print(self.day_f_times[d])
    # plt.plot(day_gt_t, day_gt_t*0, '.')
    # plt.plot(self.day_f_times[d], self.day_f_times[d]*0, '.')
    # plt.show()
    # a = 1/0

    # Convert gt to homogenous rotation/translation matrix
    gt_R = scipyR.from_quat(gt_Q)
    gt_R = gt_R.as_matrix()
    day_gt_H = np.zeros((len(day_gt_t), 4, 4))
    day_gt_H[:, :3, :3] = gt_R
    day_gt_H[:, :3, 3] = gt_T
    day_gt_H[:, 3, 3] = 1

    return day_gt_t, day_gt_H


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    # In this script we take the data from the simulator which has a lot of redundancy
    # and we remove every unecessary file to make it a minimal dataset, such as one 
    # coming from another source that the simulator.

    # Composition of a minimal dataset:
    #
    #   Session_Name
    #   |
    #   |---logs-Session_Name
    #   |   |
    #   |   |---map_traj.ply         # (OPTIONAL) ply file containing the map poses (x, y, z, qx, qy, qz, qw)
    #   |   |---pointmap_00000.ply   # (OPTIONAL) ply file the map point cloud
    #   |
    #   |---sim_frames
    #   |   |
    #   |   |---XXXX.XXXX.ply  # ply file for each frame point cloud
    #   |   |---XXXX.XXXX.ply
    #   |   |--- ...
    #   |
    #   |---gt_pose.ply  # (OPTIONAL) ply file containing the groundtruth poses (x, y, z, qx, qy, qz, qw)
    #   |
    #   |---loc_pose.ply  # ply file containing the localization poses (x, y, z, qx, qy, qz, qw)
    #

    # Additional necessary file
    #
    #   - Lidar calibration file:
    #       Modify the file located at Data/Simulation/calibration/jackal_calib.csv according to your data
    #
    #
    #
    #


    # Path to the data
    data_path = '../Data/Simulation/simulated_runs'

    # List folders in the data
    simu_folders = [f for f in listdir(data_path)]

    # Container for all the pathes to be removed
    pathes_to_remove = []

    for simu_folder in simu_folders:

        simu_path = join(data_path, simu_folder)
        files_dirs = [f for f in listdir(simu_path)]

        for f in files_dirs:
            if f not in ['sim_frames', 'logs-' + simu_folder, 'gt_pose.ply', 'loc_pose.ply']:
                pathes_to_remove.append(join(simu_path, f))


        logs_path = join(simu_path, 'logs-' + simu_folder)

        if exists(logs_path):
            files_dirs = [f for f in listdir(logs_path)]
            for f in files_dirs:
                if f not in ['map_traj.ply', 'pointmap_00000.ply']:
                    pathes_to_remove.append(join(logs_path, f))

    print('Do you want to remove these pathes:')
    for rm_path in pathes_to_remove:
        print(rm_path)

    confirm = input('\nType \"y\" to confirm you want to delete all listed files\n')
    
    if (confirm == 'y'):
        for rm_path in pathes_to_remove:
            if isdir(rm_path):
                shutil.rmtree(rm_path)
            else:
                remove(rm_path)

