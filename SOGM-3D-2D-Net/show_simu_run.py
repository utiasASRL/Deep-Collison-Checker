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
from asyncio import selector_events
import numpy as np
import sys
import time
import os
from os import listdir, makedirs
from os.path import join, exists
import time as RealTime
import pickle
import json
import subprocess
import rosbag
import plyfile as ply
import shutil
from scipy.spatial.transform import Rotation as scipyR
from utils import bag_tools as bt

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from datasets.common import grid_subsampling

from ros_numpy import point_cloud2

from utils.ply import write_ply, read_ply


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utilities
#       \***************/
#

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    

def tukey(x, mu, dx):
    d = x - mu
    y = np.square(1 - np.square(d / dx))
    y[np.abs(d) > dx] = 0
    return y


def load_gt_poses(days_folder, days):

    gt_H = []
    gt_t = []
    for d, day in enumerate(days):

        t1 = time.time()

        # Load gt from ply files
        gt_ply_file = join(days_folder, day, 'gt_pose.ply')

        if not exists(gt_ply_file):
            print('No groundtruth poses found at ' + gt_ply_file)
            print('Using localization poses instead')
            gt_ply_file = join(days_folder, day, 'loc_pose.ply')

        if not exists(gt_ply_file):
            raise ValueError('No localization poses found at ' + gt_ply_file)

        data = read_ply(gt_ply_file)
        gt_T = np.vstack([data['pos_x'], data['pos_y'], data['pos_z']]).T
        gt_Q = np.vstack([data['rot_x'], data['rot_y'], data['rot_z'], data['rot_w']]).T

        # Times
        day_gt_t = data['time']

        # Convert gt to homogenous rotation/translation matrix
        gt_R = scipyR.from_quat(gt_Q)
        gt_R = gt_R.as_matrix()
        day_gt_H = np.zeros((len(day_gt_t), 4, 4))
        day_gt_H[:, :3, :3] = gt_R
        day_gt_H[:, :3, 3] = gt_T
        day_gt_H[:, 3, 3] = 1

        t2 = time.time()
        print('{:s} {:d}/{:d} Done in {:.1f}s'.format(
            day, d, len(days), t2 - t1))

        gt_t += [day_gt_t]
        gt_H += [day_gt_H]

        # # Remove frames that are not inside gt timings
        # mask = np.logical_and(day_f_times[d] > day_gt_t[0], day_f_times[d] < day_gt_t[-1])
        # day_f_names[d] = day_f_names[d][mask]
        # day_f_times[d] = day_f_times[d][mask]

    return gt_t, gt_H


def get_gt_plot_data(gt_t, gt_H, time_ind, duration=1.0):

    # Get speeds accross the run
    gt_xy = gt_H[:, :2, 3]
    gt_speeds = (gt_xy[1:, :] - gt_xy[:-1, :]) / np.expand_dims((gt_t[1:] - gt_t[:-1]), axis=1)
    gt_speeds = np.linalg.norm(gt_speeds, axis=1)
    gt_speeds = np.hstack((gt_speeds, gt_speeds[-1:]))
    gt_speeds = gt_speeds / np.max(gt_speeds)

    # Get alpha relative to plot duration
    t0 = gt_t[time_ind]
    gt_alpha = tukey(gt_t, t0, duration)
    mask = gt_alpha > 0.001

    # Get area of interest
    gt_alpha = gt_alpha[mask]
    gt_xy = gt_xy[mask, :]
    gt_speeds = gt_speeds[mask]

    # Define colormap
    resolution = 256
    slow_c = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    med_c = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    fast_c = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cmap_speed = np.vstack((np.linspace(slow_c, med_c, resolution), np.linspace(med_c, fast_c, resolution)))

    # Get color relative to speed
    gt_color = cmap_speed[np.around(gt_speeds * (cmap_speed.shape[0] - 1)).astype(np.int32)]

    return gt_xy, gt_color, gt_alpha


def get_actors_plot_data(actors_t, actors_xy, t0, duration=1.0):

    # # Get speeds accross the run
    # act_speeds = (act_xy[:, 1:, :] - act_xy[:, :-1, :]) / np.expand_dims((act_t[1:] - act_t[:-1]), axis=1)
    # act_speeds = np.linalg.norm(act_speeds, axis=1)
    # act_speeds = np.hstack((act_speeds, act_speeds[-1:]))
    # act_speeds = act_speeds / np.max(act_speeds)

    # Get alpha relative to plot duration
    act_alpha = tukey(actors_t, t0, duration)
    mask = act_alpha > 0.001

    # Get area of interest
    act_alpha = act_alpha[mask]
    act_xy = actors_xy[:, mask, :]
    # act_speeds = act_speeds[mask]

    # Define colormap
    resolution = 256
    med_c = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    fast_c = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cmap_speed = np.linspace(fast_c, med_c, resolution)

    # Get color relative to speed
    act_color = cmap_speed[np.linspace(0, cmap_speed.shape[0] - 1, act_alpha.shape[0], dtype=np.int32)]

    # Reshape to concatenate actors
    act_color = np.tile(np.expand_dims(act_color, 0), (act_xy.shape[0], 1, 1))
    act_alpha = np.tile(np.expand_dims(act_alpha, 0), (act_xy.shape[0], 1))
    act_xy = np.reshape(act_xy, (-1, 2))
    act_color = np.reshape(act_color, (-1, 3))
    act_alpha = np.reshape(act_alpha, (-1,))
    return act_xy, act_color, act_alpha


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
    
    # Manually select runs
    selected_runs = []

    # Automatic run selection [-2, -1] for the last two runs
    runs_ids = [-3, -2, -1]
    if len(selected_runs) < 1:
        run_folders = np.sort([f for f in listdir(runs_path)])
        selected_runs = run_folders[runs_ids]


    ###########
    # Load Data
    ###########

    # Get gt_poses
    print('Loading gt_poses')
    gt_t, gt_H = load_gt_poses(runs_path, selected_runs)
    print('OK')
    print()

    # Load the environment map
    print('Loading map')
    map_path = join(root_path, 'slam_offline/2020-10-02-13-39-05/map_update_0001.ply')
    data = read_ply(map_path)
    map_points = np.vstack((data['x'], data['y'], data['z'])).T
    map_normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
    map_scores = data['scores']
    print('OK')
    print()

    # Get a 2D footprint of the map
    print('Getting footprint')
    footprint = map_points[np.logical_and(map_points[:, 2] > 0.5, map_points[:, 2] < 1.0), :]
    footprint[:, 2] *= 0
    
    # Subsampling to a 2D PointCloud
    dl_2D = 0.06
    footprint = grid_subsampling(footprint, sampleDl=dl_2D)
    print('OK')
    print()
    
    ##################
    # Init actor poses
    ##################
    
    print('Load actor positions')

    actor_times = []
    actor_xy = []

    for i, run in enumerate(selected_runs):

        # Load actor poses
        t1 = time.time()
        poses_path = os.path.join(runs_path, run, "vehicles.txt")
        actor_poses = np.loadtxt(poses_path)

        # Extract xy positions and times
        actor_times.append(actor_poses[:, 0])
        actor_x = actor_poses[:, 1::7]
        actor_y = actor_poses[:, 2::7]
        xy = np.stack((actor_x, actor_y), axis=2)
        actor_xy.append(np.transpose(xy, (1, 0, 2)))
        
        t2 = time.time()
        print('Loaded {:s} in {:.1f}s'.format(run, t2 - t1))


    ###############################
    # Plot complete traj on the map
    ###############################

    plot1 = False
    if plot1:

        print('Plot complete traj on the map')

        # Plot traj
        for i, run in enumerate(selected_runs):
            plt.scatter(gt_H[i][:, 0, 3], gt_H[i][:, 1, 3], s=1, c=gt_t[i])
            plt.scatter(footprint[:, 0], footprint[:, 1], s=1, c=[0, 0, 0])
            # plt.xlim(-3, 3)
            # plt.ylim(-3, 3)
            plt.axis('equal')
            plt.show()

        print('OK')
        print()


    ######################
    # Slider for positions
    ######################

    plot2 = True
    if plot2:

        print('Plot complete traj on the map')

        # Plot traj
        for i, run in enumerate(selected_runs):

            # Set variables
            times = gt_t[i]
                
            # Figure
            figA, axA = plt.subplots(1, 1, figsize=(10, 7))
            plt.subplots_adjust(bottom=0.15)

            # Plot the map
            plt.scatter(footprint[:, 0], footprint[:, 1], s=1, c=[[0, 0, 0]])

            # Plot the positions
            gt_xy, gt_color, gt_alpha = get_gt_plot_data(gt_t[i], gt_H[i], 0)
            gt_color = np.hstack((gt_color, np.expand_dims(gt_alpha, 1)))
            plotsA = [axA.scatter(gt_xy[:, 0],
                                  gt_xy[:, 1],
                                  s=1.0,
                                  color=gt_color)]

            # Plot the actors
            act_xy, act_color, act_alpha = get_actors_plot_data(actor_times[i], actor_xy[i], 0)
            act_color = np.hstack((act_color, np.expand_dims(act_alpha, 1)))
            plotsB = [axA.scatter(act_xy[:, 0],
                                  act_xy[:, 1],
                                  s=1.0,
                                  color=act_color)]


            # Adjust plot
            plt.axis('equal')

            # Make a horizontal slider to control the time.
            axcolor = 'lightgoldenrodyellow'
            axtime = plt.axes([0.1, 0.06, 0.8, 0.03], facecolor=axcolor)
            time_slider = Slider(ax=axtime,
                                 label='ind',
                                 valmin=0,
                                 valmax=len(times) - 1,
                                 valinit=0,
                                 valstep=1)

            # The function to be called anytime a slider's value changes
            def update_time(val):
                print('OLe')
                time_ind = (int)(val)

                gt_xy, gt_color, gt_alpha = get_gt_plot_data(gt_t[i], gt_H[i], time_ind)
                gt_color = np.hstack((gt_color, np.expand_dims(gt_alpha, 1)))
                
                act_xy, act_color, act_alpha = get_actors_plot_data(actor_times[i], actor_xy[i], gt_t[i][time_ind])
                act_color = np.hstack((act_color, np.expand_dims(act_alpha, 1)))

                for plot_i, plot_obj in enumerate(plotsA):
                    plot_obj.set_offsets(gt_xy)
                    plot_obj.set_color(gt_color)

                for plot_i, plot_obj in enumerate(plotsB):
                    plot_obj.set_offsets(act_xy)
                    plot_obj.set_color(act_color)

            # register the update function with each slider
            time_slider.on_changed(update_time)

            plt.show()

        print('OK')
        print()


    return


if __name__ == '__main__':

    main()
