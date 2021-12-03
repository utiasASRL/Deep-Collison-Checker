#
#
#      0==============================0
#      |    Deep Collision Checker    |
#      0==============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to test any model on any dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import os
import torch
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir, remove, getcwd, makedirs
from sklearn.metrics import confusion_matrix
from slam.dev_slam import frame_H_to_points, interp_pose, rot_trans_diffs, normals_orientation, save_trajectory, RANSAC
from slam.cpp_slam import update_pointmap, polar_normals, point_to_map_icp, slam_on_sim_sequence, ray_casting_annot, get_lidar_visibility, slam_on_real_sequence
import time
import pickle
from matplotlib.widgets import Slider, Button

# My libs
from utils.ply import read_ply


import open3d as o3d


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def plot_rms():

    ######################################
    # Step 1: Choose what you want to plot
    ######################################

    path = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/all_rms.txt"

    with open(path, 'r') as f:
        lines = f.readlines()

    all_rms = []
    all_plane_rms = []
    for line in lines:
        data = line.split()
        f_ind = int(data[0])
        N = (len(data) - 1) // 2
        all_rms.append(np.array([float(d) for d in data[1:1+N]]))
        all_plane_rms.append(np.array([float(d) for d in data[1+N:]]))

    # for i, rms in enumerate(all_rms):
    #     print(i, len(rms), len(rms) == len(all_plane_rms[i]))
    
    print(len(all_rms))
    print(len(all_plane_rms))

    all_rms = all_rms[5000:]
    all_plane_rms = all_plane_rms[5000:]

    ################
    # Plot functions
    ################

    # Figure
    figA, axA = plt.subplots(1, 1, figsize=(10, 7))
    plt.subplots_adjust(bottom=0.25)

    # Plot last PR curve for each log
    plotsA = []
    num_showed = 10
    for i in range(num_showed):
        if i < num_showed - 1:
            plotsA += axA.plot(np.arange(len(all_plane_rms[i])), all_plane_rms[i], 'b-', linewidth=1)
        else:
            plotsA += axA.plot(np.arange(len(all_plane_rms[i])), all_plane_rms[i], 'r-', linewidth=3)
    
    # Customize the graph
    axA.grid(linestyle='-.', which='both')
    axA.set_xlim(0, 100)
    axA.set_ylim(0, 0.4)

    # Set names for axes
    plt.xlabel('iter')
    plt.ylabel('all_plane_rms')
    
    # Make a horizontal slider to control the frequency.
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    time_slider = Slider(ax=axtime,
                         label='ind',
                         valmin=0,
                         valmax=len(all_plane_rms) - num_showed - 1,
                         valinit=0,
                         valstep=1)

    # The function to be called anytime a slider's value changes
    def update_PR(val):
        time_ind = (int)(val)
        for plot_i, plot_obj in enumerate(plotsA):
            plot_obj.set_xdata(np.arange(len(all_plane_rms[time_ind + plot_i])))
            plot_obj.set_ydata(all_plane_rms[time_ind + plot_i])
        # axA.relim()
        # axA.autoscale_view()

    # register the update function with each slider
    time_slider.on_changed(update_PR)

    plt.show()

    return



# ----------------------------------------------------------------------------------------------------------------------
#
#           Main call
#       \***************/
#


if __name__ == '__main__':

    plot_rms()

    a = 1/0
