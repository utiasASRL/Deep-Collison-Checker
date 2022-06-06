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
from matplotlib.animation import FuncAnimation

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


def get_gt_plot_data(gt_t, gt_H, t0, duration=1.0):

    # Get speeds accross the run
    gt_xy = gt_H[:, :2, 3]
    gt_speeds = (gt_xy[1:, :] - gt_xy[:-1, :]) / np.expand_dims((gt_t[1:] - gt_t[:-1]), axis=1)
    gt_speeds = np.linalg.norm(gt_speeds, axis=1)
    gt_speeds = np.hstack((gt_speeds, gt_speeds[-1:]))
    gt_speeds = gt_speeds / np.max(gt_speeds)

    # Get alpha relative to plot duration
    gt_alpha = tukey(gt_t, t0, duration)
    mask = gt_alpha > 0.001

    # Get area of interest
    gt_alpha = gt_alpha[mask]
    gt_xy = gt_xy[mask, :]
    gt_speeds = gt_speeds[mask]

    # Define colormap
    resolution = 256
    slow_c = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    med_c = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    fast_c = np.array([0.0, 1.0, 0.0], dtype=np.float64)
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


def interp_actor_xy(actor_times, actor_xy, sorted_times):

    # Relevant actor times
    fut_i = np.searchsorted(actor_times, sorted_times)

    # Actor xy positions interpolated (T, n_actors, 2)
    prev_xy = actor_xy[:, fut_i - 1, :]
    next_xy = actor_xy[:, fut_i, :]
    prev_t = actor_times[fut_i - 1]
    next_t = actor_times[fut_i]
    alpha = (sorted_times - prev_t) / (next_t - prev_t)
    alpha = np.expand_dims(alpha, (0, 2))
    interp_xy = (1-alpha) * prev_xy + alpha * next_xy

    return interp_xy

# ----------------------------------------------------------------------------------------------------------------------
#
#           Plot Func
#       \***************/
#


def plot_complete_traj(selected_runs, gt_t, gt_H, footprint):
    
    ###############################
    # Plot complete traj on the map
    ###############################

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

    return


def plot_slider_traj(selected_runs, gt_t, gt_H, footprint, actor_times, actor_xy):
    
    ######################
    # Slider for positions
    ######################


    print('Plot slider traj in actors')

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
        plt.xlim(-21, 21)
        plt.ylim(-21, 21)

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
            time_ind = (int)(val)

            gt_xy, gt_color, gt_alpha = get_gt_plot_data(gt_t[i], gt_H[i], gt_t[i][time_ind])
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


def save_vid_traj(runs_path, selected_runs, gt_t, gt_H, footprint, actor_times, actor_xy, fps=60, speed=10, following=False):
    
    ####################################
    # Video of robot and actor positions
    ####################################

    print('Make video of traj and actors')

    # Plot traj
    for i, run in enumerate(selected_runs):
        
        vid_name = os.path.join(runs_path, run, "logs-{:s}/videos/traj_{:s}.mp4".format(run, run))

        if exists(vid_name):
            continue

        # Set variables
        # times = gt_t[i]
        # t0 = gt_t[i][0]
        # t1 = gt_t[i][-1]
        vid_times = np.arange(gt_t[i][0], gt_t[i][-1], speed / fps)

        # Figure
        figA, axA = plt.subplots(1, 1, figsize=(10, 7))

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
        plt.xlim(-21, 21)
        plt.ylim(-21, 21)

        # Animation function
        def animate(ti):

            frame_t = vid_times[(int)(ti)]

            gt_xy, gt_color, gt_alpha = get_gt_plot_data(gt_t[i], gt_H[i], frame_t)
            gt_color = np.hstack((gt_color, np.expand_dims(gt_alpha, 1)))

            act_xy, act_color, act_alpha = get_actors_plot_data(actor_times[i], actor_xy[i], frame_t)
            act_color = np.hstack((act_color, np.expand_dims(act_alpha, 1)))

            for plot_i, plot_obj in enumerate(plotsA):
                plot_obj.set_offsets(gt_xy)
                plot_obj.set_color(gt_color)

            for plot_i, plot_obj in enumerate(plotsB):
                plot_obj.set_offsets(act_xy)
                plot_obj.set_color(act_color)
            
            return plotsA + plotsB

        anim = FuncAnimation(figA, animate,
                             frames=np.arange(vid_times.shape[0]),
                             interval=1000 / fps,
                             blit=True)


        # Advanced display
        print('\nProcessing {:s}'.format(run))
        progress_n = 50
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        tt1 = time.time()
        
        # Progress display
        def progress_vid(current_frame: int, total_frames: int):
            print('', end='\r')
            print(fmt_str.format('#' * ((current_frame * progress_n) // total_frames), 100 * current_frame / total_frames), end='', flush=True)
            return
                             
        anim.save(vid_name, fps=fps, progress_callback=progress_vid)

        # Show a nice 100% progress bar
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\n')
        
        tt2 = time.time()
        print('Done in {:.1f}s'.format(tt2 - tt1))

        plt.close(figA)


    return


def plot_collision_dist(selected_runs, gt_t, gt_H, footprint, actor_times, actor_xy, all_times, all_success, all_nav_info):
    
    print('Plot distance from robot to closest actor')
    # Threshold
    high_d = 2.0
    risky_d = 1.0
    collision_d = 0.4


    ########################
    # Measure risk distances
    ########################

    # Get distances
    all_min_dists = []
    for i, run in enumerate(selected_runs):

        gt_xy = gt_H[i][:, :2, 3]  # [T2, 2]

        # Get interpolatted actor positions at gt_times
        interp_xy = interp_actor_xy(actor_times[i], actor_xy[i], gt_t[i])

        # Get distances
        diffs = np.expand_dims(gt_xy, 0) - interp_xy
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=0)

        # # Threshold
        # colli_mask = min_dists < collision_d
        # risky_mask = np.logical_and(min_dists > collision_d, min_dists < risky_d)
        # colli_index = np.sum(colli_mask.astype(np.int32)) / colli_mask.shape[0]
        # risky_index = np.sum(risky_mask.astype(np.int32)) / risky_mask.shape[0]

        # print('{:s} | {:7.1f}% {:7.2f}% {:5.0f}s {:5d} /{:d}'.format(run,
        #                                                              100 * risky_index,
        #                                                              100 * colli_index,
        #                                                              all_times[i][-1],
        #                                                              np.sum(np.array(all_success[i], dtype=np.int32)),
        #                                                              len(all_success[i])))

        # For visu do not show higher distances
        min_dists = np.minimum(min_dists, high_d)
        all_min_dists.append(min_dists)


    ###############
    # Result tables
    ###############

    SOGM = np.array(all_nav_info['SOGM']) == 'true'
    GTSOGM = np.array(all_nav_info['GTSOGM']) == 'true'
    LINEAR = np.array(all_nav_info['EXTRAPO']) == 'true'
    IGNORE = np.array(all_nav_info['IGNORE']) == 'true'

    REGULAR = np.logical_and(np.logical_not(SOGM), np.logical_not(GTSOGM))
    REGULAR = np.logical_and(REGULAR, np.logical_not(LINEAR))

    SOGM = np.logical_and(SOGM, np.logical_not(IGNORE))

    # Verify that there is no problem with masks
    test_v = np.sum(REGULAR.astype(np.int32))
    test_v += np.sum(SOGM.astype(np.int32))
    test_v += np.sum(IGNORE.astype(np.int32))
    test_v += np.sum(GTSOGM.astype(np.int32))
    test_v += np.sum(LINEAR.astype(np.int32))

    if (test_v != len(SOGM)):
        raise ValueError('Problem with result masks')

    ####################################
    # Specific results for SOGM ablation
    ####################################

    date0 = '2022-05-00-00-00-00'
    date1 = '2022-06-02-00-00-00'
    date2 = '2022-06-05-00-00-00'
    date3 = '2022-06-08-00-00-00'

    mask01 = np.logical_and(np.array(selected_runs) > date0, np.array(selected_runs) < date1)
    mask12 = np.logical_and(np.array(selected_runs) > date1, np.array(selected_runs) < date2)
    mask23 = np.logical_and(np.array(selected_runs) > date2, np.array(selected_runs) < date3)

    SOGM_p1 = np.logical_and(SOGM, mask23)
    SOGM_no_t = np.logical_and(SOGM, mask12)
    SOGM = np.logical_and(SOGM, mask01)


    ##############
    # Print tables
    ##############

    # This for a quick plot of the ablation study, make a spcific plot for it or just a table???
    # res_names = ['REGULAR', 'SOGM_p1', 'SOGM_no_t', 'SOGM', 'GTSOGM']
    # res_masks = [REGULAR, SOGM_p1, SOGM_no_t, SOGM, GTSOGM]

    res_names = ['REGULAR', 'IGNORE', 'LINEAR', 'SOGM', 'GTSOGM']
    res_masks = [REGULAR, IGNORE, LINEAR, SOGM, GTSOGM]

    packed_results = {r_name: [] for r_name in res_names}

    for r_i, res_name in enumerate(res_names):

        print('\n{:s}'.format(res_name))

        res_risky_index = []
        res_colli_index = []
        res_times = []

        for i, run in enumerate(selected_runs):

            if res_masks[r_i][i]:

                # Threshold
                min_dists = all_min_dists[i]
                colli_mask = min_dists < collision_d
                risky_mask = np.logical_and(min_dists > collision_d, min_dists < risky_d)
                colli_index = np.sum(colli_mask.astype(np.int32)) / colli_mask.shape[0]
                risky_index = np.sum(risky_mask.astype(np.int32)) / risky_mask.shape[0]

                print('{:s} | {:7.1f}% {:7.2f}% {:5.0f}s {:5d} /{:d}'.format(run,
                                                                             100 * risky_index,
                                                                             100 * colli_index,
                                                                             all_times[i][-1],
                                                                             np.sum(np.array(all_success[i], dtype=np.int32)),
                                                                             len(all_success[i])))
                                                                                    
                res_risky_index += [100 * risky_index]
                res_colli_index += [100 * colli_index]
                res_times += [all_times[i][-1]]

        print('-' * 56)
        print('  Avg on {:4d} runs  | {:7.1f}% {:7.2f}% {:5.0f}s'.format(len(res_risky_index),
                                                                         np.mean(res_risky_index),
                                                                         np.mean(res_colli_index),
                                                                         np.mean(res_times)))
        print('  Std on {:4d} runs  | {:7.1f}% {:7.2f}% {:5.0f}s'.format(len(res_risky_index),
                                                                         np.std(res_risky_index),
                                                                         np.std(res_colli_index),
                                                                         np.std(res_times)))

        packed_results[res_name].append(res_risky_index)
        packed_results[res_name].append(res_colli_index)
        packed_results[res_name].append(res_times)


    ##############
    # Result plots
    ##############

    # Colors for the box plots
    res_col = ['dimgrey', 'firebrick', 'darkorange', 'mediumblue', 'palegreen']

    fig, axs = plt.subplots(1, 3, figsize=(8, 3.2), sharex=False, sharey=False)
    fig.subplots_adjust(left=0.11, right=0.95, top=0.95, bottom=0.11)
    axs = list(axs.ravel())

    for ax_i, metric_name in enumerate(['% of Time in Risky Area', '% of Time in Collision Area', 'Time to Finish in seconds']):

        y_data = [[]]
        x_data = ['']
        colors = ['black']


        for res_i, res_name in enumerate(res_names):

            if res_i in [4, ]:
                x_data.append('')
                y_data.append([])
                colors.append('black')

            y_data.append(packed_results[res_name][ax_i])
            if res_i == 2:
                x_data.append(metric_name)
            else:
                x_data.append('')
            colors.append(res_col[res_i])

            # x_data.append('')
            # y_data.append([])
            # colors.append('black')

        x_data.append('')
        y_data.append([])
        colors.append('black')

        # Add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        axs[ax_i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

        # Axes range
        if ax_i in [1, ]:
            axs[ax_i].set_ylim(bottom=-6.9/20, top=6.9)

        # Title
        # axs[ax_i].set_title(loc_meth)
        # axs[ax_i].text(0.8, .93, metric_name,
        #                transform=axs[ax_i].get_xaxis_transform(),
        #                horizontalalignment='left', fontsize='medium',
        #                weight='roman',
        #                color='k')

        # Hide the grid behind plot objects
        axs[ax_i].set_axisbelow(True)

        bp = axs[ax_i].boxplot(y_data, labels=x_data,
                               patch_artist=True, showfliers=False, widths=0.8)
        plt.setp(bp['medians'], color='k')

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Legend
        if ax_i == 1:
            my_legends = axs[ax_i].legend(np.array(bp['boxes'])[[1, 2, 3, 4, 6]], tuple(res_names), fontsize='small')
            plt.setp(my_legends.legendHandles[-1], ls='--')
        
            
        # Dashed line for GT
        plt.setp(bp['boxes'][5:], ls='--')
        plt.setp(bp['whiskers'][10:], ls='--')

    # plt.savefig('Exp3.pdf')

    ###################
    # Alternative plots
    ###################
    
    # # Get pointcloud distances
    # measured_min_dists = []
    # for i, run in enumerate(selected_runs):

    #     # Load pickle file if already computed

    #     # Read all annotated 2D colli pointcloud. And measure dist (need to annotate everything)
    #     min_dists = np.zeros((1,))

    #     # For visu do not show higher distances
    #     min_dists = np.minimum(min_dists, high_d)
    #     all_min_dists.append(min_dists)

    print('\nCompute alternative metrics')
    t1 = time.time()
    
    alt_results = {r_name: [] for r_name in res_names}
    
    for r_i, res_name in enumerate(res_names):

        res_speeds = []
        res_stops = []
        res_lin_speeds = []
        res_backs = []
        for i, run in enumerate(selected_runs):
            if res_masks[r_i][i]:
                
                # Speeds
                gt_xy = gt_H[i][:, :2, 3]  # [T2, 2]
                velocity = (gt_xy[1:] - gt_xy[:-1]) / np.expand_dims((gt_t[i][1:] - gt_t[i][:-1]), 1)
                velocity = np.vstack((velocity[:1], velocity))
                speeds = np.linalg.norm(velocity, axis=1)
                stop_mask = speeds < 0.1
                stop_index = np.sum(stop_mask.astype(np.int32)) / stop_mask.shape[0]
                res_speeds += [np.mean(speeds)]
                res_stops += [100 * stop_index]

                # Linear speed (with respect to facing direction, can be negative)
                gt_facing = gt_H[i][:, :2, 0]
                gt_facing = gt_facing / np.expand_dims(np.linalg.norm(gt_facing, axis=1), 1)
                linear_speeds = np.sum(gt_facing * velocity, axis=1)
                back_mask = linear_speeds < -0.1
                back_index = np.sum(back_mask.astype(np.int32)) / back_mask.shape[0]
                res_lin_speeds += [np.mean(linear_speeds)]
                res_backs += [100 * back_index]


        alt_results[res_name].append(res_speeds)
        alt_results[res_name].append(res_stops)
        alt_results[res_name].append(res_lin_speeds)
        alt_results[res_name].append(res_backs)

    t2 = time.time()
    print('Done in {:.1f}s'.format(t2 - t1))

    fig2, axs2 = plt.subplots(1, 4, figsize=(8, 3.2), sharex=False, sharey=False)
    fig2.subplots_adjust(left=0.11, right=0.95, top=0.95, bottom=0.11)
    axs2 = list(axs2.ravel())

    for ax_i, metric_name in enumerate(['Avg Speed (m/s)', '% of time stopped', 'Avg Linear Speed (m/s)', '% of time backwards']):

        y_data = [[]]
        x_data = ['']
        colors = ['black']


        for res_i, res_name in enumerate(res_names):

            if res_i in [4, ]:
                x_data.append('')
                y_data.append([])
                colors.append('black')

            y_data.append(alt_results[res_name][ax_i])
            if res_i == 2:
                x_data.append(metric_name)
            else:
                x_data.append('')
            colors.append(res_col[res_i])

            # x_data.append('')
            # y_data.append([])
            # colors.append('black')

        x_data.append('')
        y_data.append([])
        colors.append('black')

        # Add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        axs2[ax_i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

        # # Axes range
        # if ax_i in [1, ]:
        #     axs2[ax_i].set_ylim(bottom=-6.9/20, top=6.9)

        # Title
        # axs2[ax_i].set_title(loc_meth)
        # axs2[ax_i].text(0.8, .93, metric_name,
        #                transform=axs2[ax_i].get_xaxis_transform(),
        #                horizontalalignment='left', fontsize='medium',
        #                weight='roman',
        #                color='k')

        # Hide the grid behind plot objects
        axs2[ax_i].set_axisbelow(True)

        bp = axs2[ax_i].boxplot(y_data, labels=x_data,
                                patch_artist=True, showfliers=False, widths=0.8)
        plt.setp(bp['medians'], color='k')

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Legend
        if ax_i == 3:
            my_legends = axs2[ax_i].legend(np.array(bp['boxes'])[[1, 2, 3, 4, 6]], tuple(res_names), fontsize='small')
            plt.setp(my_legends.legendHandles[-1], ls='--')
        
            
        # Dashed line for GT
        plt.setp(bp['boxes'][5:], ls='--')
        plt.setp(bp['whiskers'][10:], ls='--')

    # plt.savefig('Exp3.pdf')
    plt.show()

    print('OK')
    print()

    return


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
    # runs_and_comments = [['2022-05-19-02-16-58', 'med | 2D obst TEB'],
    #                      ['2022-05-19-03-46-31', 'med | Groundtruth SOGM'],
    #                      ['2022-05-19-12-38-14', 'med | Predicted SOGM'], ]
    # selected_runs = [r_c[0] for r_c in runs_and_comments]


    # Select runs betweemn two dates
    from_date = '2022-05-19-22-26-08'
    to_date = '2022-06-06-13-22-16'
    if len(selected_runs) < 1:
        selected_runs = np.sort([f for f in listdir(runs_path) if from_date <= f <= to_date])

    # Automatic run selection [-2, -1] for the last two runs
    if len(selected_runs) < 1:
        runs_ids = [i for i in range(-26, 0, 1)]
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
        poses_pkl_path = os.path.join(runs_path, run, "vehicles.pkl")
        xy = None
        if exists(poses_pkl_path):
            with open(poses_pkl_path, 'rb') as f:
                actor_t, xy = pickle.load(f)

        else:
            # Load sklow txt file actor poses
            poses_path = os.path.join(runs_path, run, "vehicles.txt")
            actor_poses = np.loadtxt(poses_path)

            # Extract xy positions and times
            actor_t = actor_poses[:, 0]
            actor_x = actor_poses[:, 1::7]
            actor_y = actor_poses[:, 2::7]
            xy = np.stack((actor_x, actor_y), axis=2)
            xy = np.transpose(xy, (1, 0, 2))
            
            with open(poses_pkl_path, 'wb') as f:
                pickle.dump((actor_t, xy), f)

        actor_times.append(actor_t)
        actor_xy.append(xy)
        
        t2 = time.time()
        print('Loaded {:s} in {:.1f}s'.format(run, t2 - t1))


    #################
    # Get time metric
    #################
    
    all_times = []
    all_success = []
    all_nav_info = {'TOUR': [],
                    'MAPPING': [],
                    'FILTER': [],
                    'GTCLASS': [],
                    'TEB': [],
                    'SOGM': [],
                    'GTSOGM': [],
                    'EXTRAPO': [],
                    'IGNORE': []}
    for i, run in enumerate(selected_runs):
        
        # Read log file
        log_path = os.path.join(runs_path, run, "logs-{:s}/log.txt".format(run))
        with open(log_path) as log_f:
            lines = log_f.readlines()
        lines = [line.rstrip() for line in lines]

        # Get interesting lines
        lines = [line for line in lines if line.startswith('Reached') or line.startswith('Failed')]
        success = [line.startswith('Reached') for line in lines]

        # First verify that tour is successful
        success_n = np.sum(np.array(success, dtype=np.int32))

        # Get goal timing
        times = [float(line.split('time: ')[-1][:-1]) for line in lines]
        all_times.append(times)
        all_success.append(success)

        # Get the nav info
        log_path = os.path.join(runs_path, run, "logs-{:s}/log_nav.txt".format(run))
        with open(log_path) as log_f:
            lines = log_f.readlines()
        lines = [line.rstrip() for line in lines]
        run_info = {line.split(': ')[0]: line.split(': ')[1] for line in lines if ': ' in line}

        for k, _ in all_nav_info.items():
            if k in run_info:
                all_nav_info[k].append(run_info[k])
            else:
                all_nav_info[k].append('false')

        s = run
        for k, v in all_nav_info.items():
            s += '  |  {:^10s}'.format(v[-1])

        print(s)

    s = run
    for k, v in all_nav_info.items():
        s += '  |  {:^10s}'.format(k)
    print(s)
    print('\n------------------------------------------\n')

    ################
    # Plot functions
    ################

    # plot_complete_traj(selected_runs, gt_t, gt_H, footprint)

    # save_vid_traj(runs_path, selected_runs, gt_t, gt_H, footprint, actor_times, actor_xy)

    plot_collision_dist(selected_runs, gt_t, gt_H, footprint, actor_times, actor_xy, all_times, all_success, all_nav_info)

    # plot_slider_traj(selected_runs, gt_t, gt_H, footprint, actor_times, actor_xy)


    return


if __name__ == '__main__':

    main()
