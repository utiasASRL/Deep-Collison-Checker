#
#
#      0==============================0
#      |    Deep Collision Checker    |
#      0==============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on MyhalSim dataset
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
from re import S
from turtle import Turtle
import numpy as np
from pandas import options
from utils.ply import read_ply, write_ply

#from mayavi import mlab
import imageio
import pickle
import time
from os import listdir, makedirs
from os.path import join, exists

import pyvista as pv

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
#from datasets.MyhalCollision import *
from scipy.spatial.transform import Rotation as scipyR
from slam.dev_slam import filter_frame_timestamps, cart2pol
from slam.PointMapSLAM import motion_rectified, rigid_transform

from scipy import ndimage

from ipywidgets import widgets, interactive

from train_MyhalCollision import MyhalCollisionConfig
from datasets.MyhalCollision import MyhalCollisionDataset


# ----------------------------------------------------------------------------------------------------------------------
#
#           Old
#       \*********/
#


def get_videos(dataset_path, my_days, map_day=None, use_annotated=True):

    #day_list = day_list[1:]

    # Path of the original simu
    runs_path = join(dataset_path, 'runs')
    points_folder = 'velodyne_frames'

    # Annotation path
    annot_path = join(dataset_path, 'annotation')

    # Labeled frame path
    labels_path = join(dataset_path, 'annotated_frames')
    scalar_field = 'classif'

    # Path where we save the videos
    res_path = join(dataset_path, 'annot_videos')
    if not exists(res_path):
        makedirs(res_path)

    # Should the cam be static or follow the robot
    following_cam = True

    # Are frame localized in world coordinates?
    localized_frames = False

    # Colormap
    if scalar_field in ['classif', 'pred']:
        colormap = np.array([[209, 209, 209],
                            [122, 122, 122],
                            [255, 255, 0],
                            [0, 98, 255],
                            [255, 0, 0]], dtype=np.float32) / 255
    else:
        colormap = np.array([[122, 122, 122],
                            [0, 251, 251],
                            [255, 0, 0],
                            [89, 248, 123],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 190, 0]], dtype=np.float32) / 255

    colormap_map = 1 - (1 - colormap) * 0.9

    # Load map
    map_folder = join(dataset_path, 'slam_offline', map_day)
    map_names = [f for f in listdir(map_folder) if f.startswith('map_update_')]
    map_names = np.sort(map_names)
    last_map = map_names[-1]
    data = read_ply(join(map_folder, last_map))

    # Get and filter map points
    map_points = np.vstack((data['x'], data['y'], data['z'])).T
    map_normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
    map_annots = data['classif']

    # Reorient normals for better vis??

    ##################
    # Mayavi animation
    ##################

    # Window for headless visu
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=2560, height=1440, left=0, top=0)

    for day in my_days:

        ########
        # Init #
        ########

        # Get annotated lidar frames
        points_day_path = join(runs_path, day, points_folder)
        labels_day_path = join(labels_path, day)

        # Frame names
        f_names = [f for f in listdir(points_day_path) if f[-4:] == '.ply']
        f_times = np.array([float(f[:-4]) for f in f_names], dtype=np.float64)
        f_names = np.array([join(points_day_path, f) for f in f_names])
        ordering = np.argsort(f_times)
        f_names = f_names[ordering]
        map_t = f_times[ordering]

        # Filter timestamps
        map_t, frame_names = filter_frame_timestamps(map_t, f_names)

        # Previously computed files
        # cpp_map_name = join(out_folder, 'map_{:s}.ply'.format(day))
        cpp_traj_name = join(annot_path, day, 'correct_traj_{:s}.pkl'.format(day))

        # Load traj
        if exists(cpp_traj_name):
            with open(cpp_traj_name, 'rb') as f:
                day_map_H = pickle.load(f)
        else:
            raise ValueError('Trajectory not computed')

        # ############################
        # # DEBUG

        # print('\n-----------------------------------------------\n')
        # print(day, len(f_names), len(day_map_t), '\n')

        # dts = day_map_t[1:] - day_map_t[:-1]

        # sorted_dts = np.sort(dts)
        # mean_dt = np.mean(dts)

        # print('Average dt = {:.3f}    Average FPS = {:.1f}'.format(mean_dt, 1/mean_dt))
        # print()
        # print('min dts', sorted_dts[:5], '    max dts', sorted_dts[-5:])

        # print()

        # hist, bin_edges = np.histogram(dts, bins=100)
        # plt.hist(dts, bins=100)
        # plt.show()

        # for i, f_name in enumerate(f_names[10:30]):
        #     print('{:s}   {:^20s}   {:f}'.format(f_name.split('/')[-2], f_name.split('/')[-1], day_map_t[10+i]))

        # continue
        # ############################

        ######
        # Go #
        ######

        # Load the first frame in the window
        vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        if localized_frames:
            H0 = np.zeros((4, 4))
            H1 = np.zeros((4, 4))
        else:
            H0 = np.copy(day_map_H[0])
            H1 = np.copy(day_map_H[0])

        # Only rotate frame to have a hoverer mode
        H0[:3, 3] -= H1[:3, 3]
        H1[:3, 3] *= 0

        # Load points
        points, ts, annots = load_points_and_annot(frame_names[0].split('/')[-1], points_day_path, labels_day_path, scalar_field)

        # Apply transform with motion distorsion
        rect_points = motion_rectified(points, ts, H0, H1)

        # Remove ground and unclassified
        rect_points = rect_points[annots > 1.5]
        annots = annots[annots > 1.5]

        # Update pcd
        pcd_update_from_data(rect_points, annots, pcd, colormap)
        vis.add_geometry(pcd)

        # Create a pcd for the map
        pcd_map = o3d.geometry.PointCloud()
        pcd_update_from_data(map_points, map_annots, pcd_map, colormap_map, normals=map_normals)
        vis.add_geometry(pcd_map)

        # Add Robot mesh
        pcd_robot = o3d.geometry.PointCloud()
        robot_pts, robot_annots, colormap_robot = robot_point_model()
        pcd_update_from_data(robot_pts, robot_annots, pcd_robot, colormap_robot)
        vis.add_geometry(pcd_robot)

        # Apply render options
        render_option = vis.get_render_option()
        render_option.light_on = True
        render_option.point_size = 3
        render_option.show_coordinate_frame = True

        # Prepare view point
        view_control = vis.get_view_control()
        if following_cam:
            target = day_map_H[0][:3, 3]
            front = target + np.array([0.0, -10.0, 15.0])
            view_control.set_front(front)
            view_control.set_lookat(target)
            view_control.set_up([0.0, 0.0, 1.0])
            view_control.set_zoom(0.2)
            pinhole0 = view_control.convert_to_pinhole_camera_parameters()
            follow_H0 = np.copy(pinhole0.extrinsic)
        else:
            traj_points = np.vstack([H[:3, 3] for H in day_map_H])
            target = np.mean(traj_points, axis=0)
            front = target + np.array([10.0, 10.0, 10.0])
            view_control.set_front(front)
            view_control.set_lookat(target)
            view_control.set_up([0.0, 0.0, 1.0])
            view_control.set_zoom(0.4)
            follow_H0 = None
            pinhole0 = None

        # Advanced display
        N = len(frame_names)
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nGenerating Open3D screenshots for ' + day)
        video_list = []
        for i, f_name in enumerate(frame_names):

            if i < 1:
                continue

            if i > len(day_map_H) - 1:
                break

            if localized_frames:
                H0 = np.zeros((4, 4))
                H1 = np.zeros((4, 4))
            else:
                H0 = np.copy(day_map_H[i - 1])
                H1 = np.copy(day_map_H[i])

            # Only rotate frame to have a hoverer mode
            T_map = np.copy(H1[:3, 3])
            H0[:3, 3] -= T_map
            H1[:3, 3] *= 0

            # Load points
            points, ts, annots = load_points_and_annot(f_name.split('/')[-1], points_day_path, labels_day_path, scalar_field)

            # Apply transform with motion distorsion
            rect_points = motion_rectified(points, ts, H0, H1)

            # Remove ground and unclassified
            rect_points = rect_points[annots > 1.5]
            annots = annots[annots > 1.5]

            # Update pcd
            pcd_update_from_data(rect_points, annots, pcd, colormap)
            vis.update_geometry(pcd)

            # Update pcd for the map
            new_map_pts = map_points - T_map
            pcd_update_from_data(new_map_pts, map_annots, pcd_map, colormap_map, normals=map_normals)
            vis.update_geometry(pcd_map)

            # Update Robot mesh
            robot_ts = ts[:robot_pts.shape[0]]
            new_robot_pts = motion_rectified(robot_pts, robot_ts, H1, H1)
            pcd_update_from_data(new_robot_pts, robot_annots, pcd_robot, colormap_robot)
            vis.update_geometry(pcd_robot)

            # New view point
            if following_cam:
                # third person mode
                follow_H = np.dot(follow_H0, np.linalg.inv(day_map_H[i]))
                # pinhole0.extrinsic = follow_H
                # view_control.convert_from_pinhole_camera_parameters(pinhole0)

            # Render
            vis.poll_events()
            vis.update_renderer()

            # Screenshot
            image = vis.capture_screen_float_buffer(True)

            npimage = (np.asarray(image) * 255).astype(np.uint8)
            if npimage.shape[0] % 2 == 1:
                npimage = npimage[:-1, :]
            if npimage.shape[1] % 2 == 1:
                npimage = npimage[:, :-1]
            video_list.append(npimage)
            # plt.imsave('test_{:d}.png'.format(i), image, dpi=1)

            print('', end='\r')
            print(fmt_str.format('#' * (((i + 1) * progress_n) // N), 100 * (i + 1) / N), end='', flush=True)

        # Show a nice 100% progress bar
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\n')

        # Path for saving
        video_path = join(res_path, 'video_{:s}_{:s}.mp4'.format(scalar_field, day))

        # Write video file
        print('\nWriting video file for ' + day)
        kargs = {'macro_block_size': None}
        with imageio.get_writer(video_path, mode='I', fps=30, quality=10, **kargs) as writer:
            N = len(video_list)
            for i, frame in enumerate(video_list):
                writer.append_data(frame)

                print('', end='\r')
                print(fmt_str.format('#' * (((i + 1) * progress_n) // N), 100 * (i + 1) / N), end='', flush=True)

        # Show a nice 100% progress bar
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\n')

    vis.destroy_window()


    return


def show_seq_slider(data_path, all_seqs, in_radius=8.0):

    colli_path = join(data_path, 'noisy_collisions')
    annot_path = join(data_path, 'annotation')
    
    # convertion from labels to colors
    colormap = np.array([[209, 209, 209],
                        [122, 122, 122],
                        [255, 255, 0],
                        [0, 98, 255],
                        [255, 0, 0]], dtype=np.float32) / 255

    # Parameters
    im_lim = in_radius / np.sqrt(2)
    
    # Variables
    global f_i
    f_i = 0

    all_pts = [[] for seq in all_seqs]
    all_colors = [[] for seq in all_seqs]
    all_labels = [[] for seq in all_seqs]
    for s_ind, seq in enumerate(all_seqs):

        #########
        # Loading
        #########

        # Get annotated lidar frames
        points_day_path = join(data_path, 'runs', seq, 'velodyne_frames')

        # Frame names
        f_names = [f for f in listdir(points_day_path) if f[-4:] == '.ply']
        f_times = np.array([float(f[:-4]) for f in f_names], dtype=np.float64)
        f_names = np.array([join(points_day_path, f) for f in f_names])
        ordering = np.argsort(f_times)
        f_names = f_names[ordering]
        map_t = f_times[ordering]

        # Filter timestamps
        map_t, frames_pathes = filter_frame_timestamps(map_t, f_names)

        s_frames = [f.split('/')[-1][:-4] for f in frames_pathes]

        # Advanced display
        N = len(s_frames)
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nGetting gt for ' + seq)

        # Previously computed files
        # cpp_map_name = join(out_folder, 'map_{:s}.ply'.format(day))
        cpp_traj_name = join(annot_path, seq, 'correct_traj_{:s}.pkl'.format(seq))

        # Load traj
        if exists(cpp_traj_name):
            with open(cpp_traj_name, 'rb') as f:
                day_map_H = pickle.load(f)
        else:
            raise ValueError('Trajectory not computed')

        for f_ind, frame in enumerate(s_frames):

            # Get groundtruth in 2D points format
            gt_file = join(colli_path, seq, frame + '_2D.ply')

            # Read points
            data = read_ply(gt_file)
            pts_2D = np.vstack((data['x'], data['y'])).T
            labels_2D = data['classif']

            # Recenter
            p0 = day_map_H[f_ind][:2, 3]
            centered_2D = (pts_2D - p0).astype(np.float32)

            # Remove outside boundaries of images
            img_mask = np.logical_and(centered_2D < im_lim, centered_2D > -im_lim)
            img_mask = np.logical_and(img_mask[:, 0], img_mask[:, 1])
            centered_2D = centered_2D[img_mask]
            labels_2D = labels_2D[img_mask]

            # Get the number of points per label (only present in image)
            label_v, label_n = np.unique(labels_2D, return_counts=True)
            label_count = np.zeros((colormap.shape[0],), dtype=np.int32)
            label_count[label_v] = label_n
            
            all_pts[s_ind].append(centered_2D)
            all_colors[s_ind].append(colormap[labels_2D])
            all_labels[s_ind].append(label_count)

            print('', end='\r')
            print(fmt_str.format('#' * (((f_ind + 1) * progress_n) // N), 100 * (f_ind + 1) / N), end='', flush=True)

        # Show a nice 100% progress bar
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\n')

        ##########
        # Plotting
        ##########

        # Figure
        figA, axA = plt.subplots(1, 1, figsize=(10, 7))
        plt.subplots_adjust(bottom=0.25)

        # Plot first frame of seq
        plotsA = [axA.scatter(all_pts[s_ind][0][:, 0],
                              all_pts[s_ind][0][:, 1],
                              s=2.0,
                              c=all_colors[s_ind][0])]

        # Show a circle of the loop closure area
        axA.add_patch(Circle((0, 0), radius=0.2,
                             edgecolor=[0.2, 0.2, 0.2],
                             facecolor=[1.0, 0.79, 0],
                             fill=True,
                             lw=1))

        plt.subplots_adjust(left=0.1, bottom=0.15)

        # # Customize the graph
        # axA.grid(linestyle='-.', which='both')
        axA.set_xlim(-im_lim, im_lim)
        axA.set_ylim(-im_lim, im_lim)
        axA.set_aspect('equal', adjustable='box')

        # Make a horizontal slider to control the frequency.
        axcolor = 'lightgoldenrodyellow'
        axtime = plt.axes([0.1, 0.04, 0.8, 0.02], facecolor=axcolor)
        time_slider = Slider(ax=axtime,
                             label='ind',
                             valmin=0,
                             valmax=len(all_pts[s_ind]) - 1,
                             valinit=0,
                             valstep=1)

        # The function to be called anytime a slider's value changes
        def update_PR(val):
            global f_i
            f_i = (int)(val)
            for plot_i, plot_obj in enumerate(plotsA):
                plot_obj.set_offsets(all_pts[s_ind][f_i])
                plot_obj.set_color(all_colors[s_ind][f_i])

        # register the update function with each slider
        time_slider.on_changed(update_PR)

        # Add an ax with the presence of dynamic points
        dyn_img = np.vstack(all_labels[s_ind]).T
        dyn_img = dyn_img[-1:]
        dyn_img[dyn_img > 10] = 10
        dyn_img[dyn_img > 0] += 10
        axdyn = plt.axes([0.1, 0.02, 0.8, 0.015])
        axdyn.imshow(dyn_img, cmap='OrRd', aspect='auto')
        axdyn.set_axis_off()

        # wanted_f = []

        # # Register event
        # def onkey(event):
        #     if event.key == 'enter':
        #         wanted_f.append(f_i)
        #         print('Added current frame to the wanted indices. Now containing:', wanted_f)

        #     elif event.key == 'backspace':
        #         if wanted_f:
        #             wanted_f.pop()
        #         print('removed last frame from the wanted indices. Now containing:', wanted_f)

        #     elif event.key == 'x':
        #         if wanted_f:
        #             remove_i = np.argmin([abs(i - f_i) for i in wanted_f])
        #             wanted_f.pop(remove_i)
        #         print('removed closest frame from the wanted indices. Now containing:', wanted_f)

        # cid = figA.canvas.mpl_connect('key_press_event', onkey)
        # print('\n---------------------------------------\n')
        # print('Instructions:\n')
        # print('> Press Enter to add current frame to the wanted indices.')
        # print('> Press Backspace to remove last frame added to the wanted indices.')
        # print('> Press x to to remove the closest frame to current one from the wanted indices.')
        # print('\n---------------------------------------\n')

        plt.show()


    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utilities
#       \***************/
#


def robot_point_model():

    min_x = -0.21
    max_x = 0.21
    min_y = -0.16
    max_y = 0.16
    nx, ny = (43, 33)
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    xv, yv = np.meshgrid(x, y)
    robot_pts = np.vstack((np.ravel(xv), np.ravel(yv), np.ravel(xv) * 0 - 0.6)).T
    robot_annots = (robot_pts[:, 0] * 0).astype(np.int32)
    colormap_robot = np.array([[0, 150, 0]], dtype=np.float32) / 255

    return robot_pts, robot_annots, colormap_robot


def filter_valid_frames(f_names, day_map_t, day_map_H):

    # Verify which frames we need:
    frame_names = []
    f_name_i = 0
    last_t = day_map_t[0] - 0.1
    remove_inds = []
    for i, t in enumerate(day_map_t):

        # Handle cases were we have two identical timestamps in map_t
        if np.abs(t - last_t) < 0.01:
            remove_inds.append(i)
            continue
        last_t = t

        f_name = '{:.6f}.ply'.format(t)
        while f_name_i < len(f_names) and not (
                f_names[f_name_i].endswith(f_name)):
            # print(f_names[f_name_i], ' skipped for ', f_name)
            f_name_i += 1

        if f_name_i >= len(f_names):
            break

        frame_names.append(f_names[f_name_i])
        f_name_i += 1

    # Remove the double inds form map_t and map_H
    day_map_t = np.delete(day_map_t, remove_inds, axis=0)
    day_map_H = np.delete(day_map_H, remove_inds, axis=0)

    return frame_names, day_map_t, day_map_H


def load_points_and_annot(f_name, points_path, annot_path, scalar_field):

    # Load points
    data = read_ply(join(points_path, f_name))
    pts = np.vstack((data['x'], data['y'], data['z'])).T
    ts = data['time']

    # Load annot
    data = read_ply(join(annot_path, f_name))
    annots = data[scalar_field]

    return pts, ts, annots


def pcd_update_from_ply(ply_name, pcd, H_frame, colormap, scalar_field='classif'):

    # Load first frame
    data = read_ply(ply_name)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    classif = data[scalar_field]

    if np.sum(np.abs(H_frame)) > 1e-3:
        points = np.hstack((points, np.ones_like(points[:, :1])))
        points = np.matmul(points, H_frame.T).astype(np.float32)[:, :3]

    # Get colors
    np_colors = colormap[classif, :]

    # Save as pcd
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)

    if 'nx' in data.dtype.names:
        normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
        if np.sum(np.abs(H_frame)) > 1e-3:
            world_normals = np.matmul(normals, H_frame[:3, :3].T).astype(np.float32)
        pcd.normals = o3d.utility.Vector3dVector(normals)


def pcd_update_from_data(points, annots, pcd, colormap, normals=None):


    # Get colors
    np_colors = colormap[annots, :]

    # Save as pcd
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)


def loading_session(dataset, s_ind, i_l, im_lim, colormap):

    seq = dataset.sequences[s_ind]

    print('\nInspecting session:', seq)
    print('********************' + '*' * len(seq))
    
    all_pts = []
    all_colors = []
    all_labels = []

    # Advanced display
    N = dataset.frames[s_ind].shape[0]
    progress_n = 50
    fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
    print('\nGetting 2D frames')
    for f_ind, frame in enumerate(dataset.frames[s_ind]):

        # Get groundtruth in 2D points format
        gt_file = join(dataset.colli_path[s_ind], frame + '_2D.ply')

        # Read points
        data = read_ply(gt_file)
        pts_2D = np.vstack((data['x'], data['y'])).T
        labels_2D = data['classif']
            
        # Special treatment to old simulation annotations
        if dataset.sim_sequences[s_ind]:
            times_2D = data['t']
            time_mask = np.logical_and(times_2D > -0.001, times_2D < 0.001)
            pts_2D = pts_2D[time_mask]
            labels_2D = labels_2D[time_mask]

        # Recenter
        p0 = dataset.poses[s_ind][f_ind][:2, 3]
        centered_2D = (pts_2D - p0).astype(np.float32)

        # Remove outside boundaries of images
        img_mask = np.logical_and(centered_2D < im_lim, centered_2D > -im_lim)
        img_mask = np.logical_and(img_mask[:, 0], img_mask[:, 1])
        centered_2D = centered_2D[img_mask]
        labels_2D = labels_2D[img_mask]

        # Get the number of points per label (only present in image)
        label_v, label_n = np.unique(labels_2D, return_counts=True)
        label_count = np.zeros((colormap.shape[0],), dtype=np.int32)
        label_count[label_v] = label_n

        # Do not count dynamic points if we are not in interesting area
        # Only for Myhal 5th floor
        # if not dataset.sim_sequences[s_ind]:
        #     if p0[1] < 6 and p0[0] < 4:
        #         label_count[-1] = 0

        all_pts.append(centered_2D)
        all_colors.append(colormap[labels_2D])
        all_labels.append(label_count)

        print('', end='\r')
        print(fmt_str.format('#' * (((f_ind + 1) * progress_n) // N), 100 * (f_ind + 1) / N), end='', flush=True)

    # Show a nice 100% progress bar
    print('', end='\r')
    print(fmt_str.format('#' * progress_n, 100), flush=True)
    print('\n')

    print('\nProcess wanted_inds')

    # Get the wanted indices
    class_mask = np.vstack(all_labels).T
    class_mask = class_mask[i_l:i_l + 1] > 10

    # Remove isolated inds with opening
    open_struct = np.ones((1, 31))
    class_mask_opened = ndimage.binary_opening(class_mask, structure=open_struct)

    # Remove the one where the person is disappearing or reappearing
    erode_struct = np.ones((1, 31))
    erode_struct[:, :13] = 0
    class_mask_eroded = ndimage.binary_erosion(class_mask_opened, structure=erode_struct)

    # Update selected inds for all sequences
    seq_mask = dataset.all_inds[:, 0] == s_ind
    
    print('    > Done')

    return all_pts, all_colors, all_labels, class_mask_opened, class_mask_eroded, seq_mask


def inspect_sogm_sessions(dataset_path, map_day, train_days, train_comments):

    print('\n')
    print('------------------------------------------------------------------------------')
    print('\n')
    print('Start session inspection')
    print('************************')
    print('\nInitial map run:', map_day)
    print('\nInspected runs:')
    for d, day in enumerate(train_days):
        print(' >', day)
    print('')

    # Reduce number of runs to inspect
    print('You can choose to inspect only the last X runs (enter nothing to inspect all runs)')
    n_runs = input("Enter the number X of runs to inspect:\n")

    if len(n_runs) > 0:
        n_runs = int(n_runs)
    else:
        n_runs = len(train_days)
    
    print('You choose to inspect only the last X runs')
    print(n_runs)

    if n_runs < len(train_days):
        train_days = train_days[-n_runs:]
        train_comments = train_comments[-n_runs:]

    config = MyhalCollisionConfig()

    # Initialize datasets (dummy validation)
    dataset = MyhalCollisionDataset(config,
                                    train_days,
                                    chosen_set='training',
                                    dataset_path=dataset_path,
                                    balance_classes=True)


    # convertion from labels to colors
    im_lim = config.radius_2D / np.sqrt(2)
    colormap = np.array([[209, 209, 209],
                        [122, 122, 122],
                        [255, 255, 0],
                        [0, 98, 255],
                        [255, 0, 0]], dtype=np.float32) / 255

    # We care about the dynamic class
    i_l = 4

    #########################
    # Create a display window
    #########################
    
    # Init data
    s_ind = 0
    data = loading_session(dataset, s_ind, i_l, im_lim, colormap)
    all_pts, all_colors, all_labels, class_mask_opened, class_mask_eroded, seq_mask = data

    figA, axA = plt.subplots(1, 1, figsize=(10, 7))
    plt.subplots_adjust(left=0.1, bottom=0.2)
    
    # Plot first frame of seq
    plotsA = [axA.scatter(all_pts[0][:, 0],
                          all_pts[0][:, 1],
                          s=2.0,
                          c=all_colors[0])]

    # Show a circle of the loop closure area
    axA.add_patch(Circle((0, 0), radius=0.2,
                         edgecolor=[0.2, 0.2, 0.2],
                         facecolor=[1.0, 0.79, 0],
                         fill=True,
                         lw=1))

    # # Customize the graph
    # axA.grid(linestyle='-.', which='both')
    axA.set_xlim(-im_lim, im_lim)
    axA.set_ylim(-im_lim, im_lim)
    axA.set_aspect('equal', adjustable='box')

    # Make a horizontal slider to control the frequency.
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.1, 0.1, 0.8, 0.015], facecolor=axcolor)
    time_slider = Slider(ax=axtime,
                         label='ind',
                         valmin=0,
                         valmax=len(all_pts) - 1,
                         valinit=0,
                         valstep=1)

    # The function to be called anytime a slider's value changes
    def update_points(val):
        global f_i
        f_i = (int)(val)
        for plot_i, plot_obj in enumerate(plotsA):
            plot_obj.set_offsets(all_pts[f_i])
            plot_obj.set_color(all_colors[f_i])

    # register the update function with each slider
    time_slider.on_changed(update_points)

    # Ax with the presence of dynamic points
    class_mask = np.zeros_like(dataset.all_inds[:, 0], dtype=bool)
    class_mask[dataset.class_frames[i_l]] = True
    seq_mask = dataset.all_inds[:, 0] == s_ind
    seq_class_frames = class_mask[seq_mask]
    seq_class_frames = np.expand_dims(seq_class_frames, 0)
    axdyn0 = plt.axes([0.1, 0.08, 0.8, 0.015])
    axdyn0.imshow(seq_class_frames, cmap='GnBu', aspect='auto')
    axdyn0.set_axis_off()

    # Ax with the presence of dynamic points at least 10
    dyn_img = np.vstack(all_labels).T
    dyn_img = dyn_img[-1:]
    dyn_img[dyn_img > 10] = 10
    dyn_img[dyn_img > 0] += 10
    axdyn1 = plt.axes([0.1, 0.06, 0.8, 0.015])
    axdyn1.imshow(dyn_img, cmap='OrRd', aspect='auto')
    axdyn1.set_axis_off()

    # Ax with opened
    axdyn2 = plt.axes([0.1, 0.04, 0.8, 0.015])
    axdyn2.imshow(class_mask_opened, cmap='OrRd', aspect='auto')
    axdyn2.set_axis_off()

    # Ax with eroded
    axdyn3 = plt.axes([0.1, 0.02, 0.8, 0.015])
    axdyn3.imshow(class_mask_eroded, cmap='OrRd', aspect='auto')
    axdyn3.set_axis_off()

    ###################
    # Saving function #
    ###################

    global selected_saves
    selected_saves = []

    # Register event
    def onkey(event):
        global f_i, selected_saves

        # Save current as ptcloud
        if event.key in ['p', 'P']:
            print('Saving in progress')

            seq_name = dataset.sequences[s_ind]
            frame_name = dataset.frames[s_ind][f_i]
            sogm_folder = join(dataset.original_path, 'inspect_images')
            print(sogm_folder)
            if not exists(sogm_folder):
                makedirs(sogm_folder)

            # Save pointcloud
            H0 = dataset.poses[s_ind][f_i - 1]
            H1 = dataset.poses[s_ind][f_i]
            data = read_ply(join(dataset.seq_path[s_ind], frame_name + '.ply'))
            f_points = np.vstack((data['x'], data['y'], data['z'])).T
            f_ts = data['time']
            world_points = motion_rectified(f_points, f_ts, H0, H1)

            data = read_ply(join(dataset.annot_path[s_ind], frame_name + '.ply'))
            sem_labels = data['classif']

            ply_name = join(sogm_folder, 'ply_{:s}_{:s}.ply'.format(seq_name, frame_name))
            write_ply(ply_name,
                      [world_points, sem_labels],
                      ['x', 'y', 'z', 'classif'])

            print('Done')

        # Save current as ptcloud video
        if event.key in ['g', 'G']:
            selected_saves.append([f_i, s_ind])
            print('New list of saves:', len(selected_saves))
            for f_ind_save, s_ind_save in selected_saves:
                print(f_ind_save, s_ind_save)


        # Save current as ptcloud video
        if event.key in ['h', 'H']:

            for f_ind_save, s_ind_save in selected_saves:

                video_i0 = -30
                video_i1 = 70
                if f_ind_save + video_i1 >= len(dataset.frames[s_ind_save]) or f_ind_save + video_i0 < 0:
                    print('Invalid f_i')
                    return

                sogm_folder = join(dataset.original_path, 'inspect_images')
                print(sogm_folder)
                if not exists(sogm_folder):
                    makedirs(sogm_folder)

                # Video path
                seq_name = dataset.sequences[s_ind_save]

                # Get the pointclouds
                vid_pts = []
                vid_labels = []
                vid_ts = []
                vid_H0 = []
                vid_H1 = []
                for vid_i in range(video_i0, video_i1):
                    frame_name = dataset.frames[s_ind_save][f_ind_save + vid_i]
                    H0 = dataset.poses[s_ind_save][f_ind_save + vid_i - 1]
                    H1 = dataset.poses[s_ind_save][f_ind_save + vid_i]
                    data = read_ply(join(dataset.seq_path[s_ind_save], frame_name + '.ply'))
                    f_points = np.vstack((data['x'], data['y'], data['z'])).T
                    f_ts = data['time']
                    data = read_ply(join(dataset.annot_path[s_ind_save], frame_name + '.ply'))
                    sem_labels = data['classif']

                    vid_pts.append(f_points)
                    vid_labels.append(sem_labels)
                    vid_ts.append(f_ts)
                    vid_H0.append(H0)
                    vid_H1.append(H1)

                map_folder = join(dataset.original_path, 'slam_offline', map_day)
                map_names = np.sort([f for f in listdir(map_folder) if f.startswith('map_update_')])
                last_map = join(map_folder, map_names[-1])

                # Zero height for poses in 2D
                vid_H0_2D = []
                for h0 in vid_H0:
                    h1 = np.copy(h0)
                    h1[2, 3] *= 0
                    vid_H0_2D.append(h1)

                # Get 2D points
                save_pts2D = True
                if save_pts2D:
                    data = loading_session(dataset, s_ind_save, i_l, im_lim, colormap)
                    all_pts_save, all_colors_save, _, _, _, _ = data
                    vid_pts2D = all_pts_save[f_ind_save + video_i0:f_ind_save + video_i1]
                    vid_colors2D = all_colors_save[f_ind_save + video_i0:f_ind_save + video_i1]
                    gif_name = join(sogm_folder, 'gif_{:s}_{:s}_0.gif'.format(seq_name, dataset.frames[s_ind_save][f_ind_save]))
                    plt_2D_gif(gif_name, vid_pts2D, vid_colors2D, im_lim)

                # Create video
                save_all_3D = False
                if save_all_3D:
                    video_path = join(sogm_folder, 'vid_{:s}_{:s}_0.mp4'.format(seq_name, dataset.frames[s_ind_save][f_ind_save]))
                    pyvista_vid_0(video_path,
                                  vid_pts,
                                  vid_labels,
                                  vid_H0,
                                  colored=False,
                                  visu_loc=False)
                    video_path = join(sogm_folder, 'vid_{:s}_{:s}_1.mp4'.format(seq_name, dataset.frames[s_ind_save][f_ind_save]))
                    pyvista_vid_0(video_path,
                                  vid_pts,
                                  vid_labels,
                                  vid_H0,
                                  colored=True,
                                  visu_loc=False)
                video_path = join(sogm_folder, 'vid_{:s}_{:s}_2.mp4'.format(seq_name, dataset.frames[s_ind_save][f_ind_save]))
                pyvista_vid_0(video_path,
                              vid_pts,
                              vid_labels,
                              vid_H0,
                              colored=True,
                              visu_loc=True)

                print('Done')

            selected_saves = []
            print('All saves finished')

        return

    cid = figA.canvas.mpl_connect('key_press_event', onkey)


    #############################
    # Create a interactive window
    #############################


    def update_display():

        # Redifine sliders
        time_slider.val = 0
        time_slider.valmin = 0
        time_slider.valmax = len(all_pts) - 1
        time_slider.ax.set_xlim(time_slider.valmin, time_slider.valmax)

        # Redraw masks
        class_mask = np.zeros_like(dataset.all_inds[:, 0], dtype=bool)
        class_mask[dataset.class_frames[i_l]] = True
        seq_mask = dataset.all_inds[:, 0] == s_ind
        seq_class_frames = class_mask[seq_mask]
        seq_class_frames = np.expand_dims(seq_class_frames, 0)
        axdyn0.imshow(seq_class_frames, cmap='GnBu', aspect='auto')
        axdyn1.imshow(dyn_img, cmap='OrRd', aspect='auto')
        axdyn2.imshow(class_mask_opened, cmap='OrRd', aspect='auto')
        axdyn3.imshow(class_mask_eroded, cmap='OrRd', aspect='auto')

        # Update points
        update_points(time_slider.val)
        
        plt.draw()

        return

    # One button for each session
    figB = plt.figure(figsize=(11, 5))
    rax = plt.axes([0.05, 0.05, 0.9, 0.9], facecolor='lightgrey')
    radio_texts = [s + ': ' + train_comments[i] for i, s in enumerate(train_days)]
    radio_texts_to_i = {s: i for i, s in enumerate(radio_texts)}
    radio = RadioButtons(rax, radio_texts)
    
    def radio_func(label):
        # Load current sequence data
        nonlocal all_pts, all_colors, all_labels, class_mask_opened, s_ind, class_mask_eroded, seq_mask
        s_ind = radio_texts_to_i[label]
        data = loading_session(dataset, s_ind, i_l, im_lim, colormap)
        all_pts, all_colors, all_labels, class_mask_opened, class_mask_eroded, seq_mask = data
        update_display()
        return

    radio.on_clicked(radio_func)

    plt.show()

    print('    > Done')

    print('\n')
    print('  +-----------------------------------+')
    print('  | Finished all the annotation tasks |')
    print('  +-----------------------------------+')
    print('\n')

    return


def pyvista_vid_0(video_path, vid_pts, vid_labels, vid_H0, colored=True, visu_loc=True):
    """
    PyVista video with only lidar frame colored or not
    """

    ################
    # Init Varaibles
    ################
    
    # Varaiable for poses
    pose_color1 = np.array([0, 255, 255, 0], dtype=np.float32) / 255
    pose_color2 = np.array([0, 255, 0, 255], dtype=np.float32) / 255
    cmap_pose = ListedColormap(np.linspace(pose_color1, pose_color2, 256))

    # Colormap
    colormap = np.array([[209, 209, 209],
                        [122, 122, 122],
                        [255, 255, 0],
                        [0, 98, 255],
                        [255, 0, 0]], dtype=np.float32) / 255
    
    # Make the colormap from the listed colors
    if colored:
        colormap1 = ListedColormap(colormap[1:])
    else:
        colormap1 = ListedColormap(colormap[:1])

    # Get poses as lines
    pose_centers = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0], ])
    pose_directions = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1], ])

    # Get all centers and directions in world space
    all_H0 = np.stack(vid_H0, axis=0)  # [N, 4, 4]
    all_p0 = all_H0[:, :3, 3]  # [N, 3]
    all_p0 = np.expand_dims(all_p0, 1)  # [N, 1, 3]
    pose_centers = np.expand_dims(pose_centers, 0)  # [1, 3, 3]
    pose_centers = all_p0 + pose_centers  # [N, 3, 3]
    pose_centers = np.reshape(pose_centers, (-1, 3))  # [N*3, 3]
    all_R0 = all_H0[:, :3, :3]  # [N, 3, 3]
    pose_directions = np.copy(all_R0.transpose((0, 2, 1)))
    pose_directions = np.reshape(pose_directions, (-1, 3))  # [N*3, 3]


    ##############
    # Init plotter
    ##############

    # Remove unclassified
    annots = vid_labels[0]
    rect_points = vid_pts[0][annots > 0.5]
    annots = annots[annots > 0.5]

    # Window for headless visu
    
    pl = pv.Plotter(notebook=False, off_screen=True, window_size=[1600, 912])

    if video_path.endswith(".gif"):
        pl.open_gif(video_path)
    else:
        pl.open_movie(video_path, framerate=30, quality=9)

    # Add initial points
    pl_points = pl.add_points(rect_points,
                              render_points_as_spheres=True,
                              scalars=annots,
                              cmap=colormap1,
                              point_size=5.0)

    # Move to first frame space
    if visu_loc:
        pose_centers0, pose_directions0 = rigid_transform(all_H0[0],
                                                          pose_centers,
                                                          normals=pose_directions,
                                                          inverse=True)
        scalars = np.linspace(0, len(vid_pts) - 1, pose_centers0.shape[0] * 31)
        scalars = np.maximum(1 - np.abs(scalars) / 10, 0)
        scalars[31:] *= 0
        scalars = scalars * 3.0 + 1.0
        arrows = pl.add_arrows(pose_centers0,
                               pose_directions0,
                               scalars=scalars,
                               mag=0.4,
                               opacity='linear',
                               cmap=cmap_pose)

    pl.set_background('white')

    pl.set_position([-14, 0, 12])
    pl.set_focus([0, 0, 0])
    pl.set_viewup([0, 0, 1])

    # pl.enable_eye_dome_lighting()
    # pl.show()


    #############
    # Start video
    #############

    pl.remove_scalar_bar()

    pl.show(auto_close=False)  # only necessary for an off-screen movie

    # Run through each frame
    # pl.write_frame()  # write initial data

    # Advanced display
    N = len(vid_pts)
    progress_n = 30
    fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
    print('\nGenerating PyVista Video')

    # Update scalars on each frame
    for i, pts in enumerate(vid_pts):
        if i < 1:
            continue

        # Remove unclassified
        annots = vid_labels[i]
        rect_points = pts[annots > 0.5]
        annots = annots[annots > 0.5]

        # pl.clear()
        pl.remove_actor(pl_points)
        if visu_loc:
            pl.remove_actor(arrows)

        pl_points = pl.add_points(rect_points,
                                  render_points_as_spheres=True,
                                  scalars=annots,
                                  cmap=colormap1,
                                  point_size=5.0)

        # Add localization
        if visu_loc:
            # Move to first frame space
            pose_centers0, pose_directions0 = rigid_transform(all_H0[i],
                                                              pose_centers,
                                                              normals=pose_directions,
                                                              inverse=True)

            scalars = np.arange(len(vid_pts))
            scalars = np.maximum(1 - np.abs(scalars - i) / 10, 0)
            scalars[i + 1:] *= 0
            scalars = scalars * 3.0 + 0.99
            scalars = np.expand_dims(scalars, 1)
            scalars = np.tile(scalars, (1, 3 * 15))
            scalars = np.ravel(scalars)
            arrows = pl.add_arrows(pose_centers0,
                                   pose_directions0,
                                   scalars=scalars,
                                   mag=0.4,
                                   opacity='linear',
                                   cmap=cmap_pose)

        pl.remove_scalar_bar()

        pl.write_frame()  # Write this frame
        
        print('', end='\r')
        print(fmt_str.format('#' * (((i + 1) * progress_n) // N), 100 * (i + 1) / N), end='', flush=True)

    # Show a nice 100% progress bar
    print('', end='\r')
    print(fmt_str.format('#' * progress_n, 100), flush=True)
    print('\n')

    # Be sure to close the plotter when finished
    pl.close()

    return


def pyvista_vid(video_path, vid_pts, vid_labels, vid_ts, vid_H0, vid_H1, map_path=None):
    
    
    # Colormap
    colormap = np.array([[209, 209, 209],
                        [122, 122, 122],
                        [255, 255, 0],
                        [0, 98, 255],
                        [255, 0, 0]], dtype=np.float32) / 255
    
    colormap_map = 1 - (1 - colormap) * 0.3


    # Make the colormap from the listed colors
    colormap1 = ListedColormap(colormap[1:])
    colormap2 = ListedColormap(colormap_map[1:])


    # Only rotate frame to have a hoverer mode
    H0 = np.copy(vid_H0[0])
    H1 = np.copy(vid_H1[0])
    # H0[:3, 3] -= H1[:3, 3]
    # H1[:3, 3] *= 0

    # Apply transform with motion distorsion
    rect_points0 = motion_rectified(vid_pts[0], vid_ts[0], H0, H1)
    annots = vid_labels[0]

    # Remove ground if we have map
    if (map_path):
        rect_points = rect_points0[annots > 1.5]
        annots = annots[annots > 1.5]
    else:
        rect_points = rect_points0[annots > 0.5]
        annots = annots[annots > 0.5]

    # Window for headless visu
    pl = pv.Plotter(window_size=[1600, 900])
    # filename = "pyvista_test.mp4"
    # pl.open_movie(filename)

    # Add map points
    if (map_path):
        data = read_ply(map_path)
        map_points = np.vstack((data['x'], data['y'], data['z'])).T
        # map_normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
        map_classif = data['classif']

        # # Only keep ground points for map
        # map_mask = np.logical_and(map_classif > 0.5, map_classif < 1.5)
        # map_points = map_points[map_mask]
        # # map_normals = map_normals[map_mask]
        # map_classif = map_classif[map_mask]

        # # Reduce to visible space
        # visible_mask = get_visble_pts(np.copy(map_points), rect_points0, (H0[:3, 3] + H1[:3, 3]) / 2)
        # map_points_vis = np.copy(map_points[visible_mask])
        # # map_normals_vis = map_normals[visible_mask]
        # map_classif_vis = np.copy(map_classif[visible_mask])
        
        pl.add_points(map_points,
                      render_points_as_spheres=False,
                      scalars=map_classif,
                      cmap=colormap2,
                      point_size=3.0)

                      

    # Add initial points
    pl.add_points(rect_points,
                  render_points_as_spheres=True,
                  scalars=annots,
                  cmap=colormap1,
                  point_size=5.0)

    # Add Robot mesh
    robot_pts, robot_annots, colormap_robot = robot_point_model()
    color_robot = colormap_robot[0]
    pl.add_points(robot_pts,
                  render_points_as_spheres=True,
                  color=color_robot,
                  point_size=5.0)

    pl.set_background('white')
    pl.enable_eye_dome_lighting()
    pl.show()

    a = 1/0


    # # only necessary for an off-screen movie    
    # pl.show(auto_close=False)  










    
    # Window for headless visu
    vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=True, width=2560, height=1440, left=0, top=0)
    # vis.create_window(visible=True, width=1920, height=1080, left=0, top=0)
    vis.create_window(visible=True, width=1600, height=900, left=0, top=0)
    # vis.create_window(visible=True, width=1280, height=720, left=0, top=0)
    
    # Load the first frame in the window
    vis.clear_geometries()
    pcd = o3d.geometry.PointCloud()
    

    # Only rotate frame to have a hoverer mode
    H0 = np.copy(vid_H0[0])
    H1 = np.copy(vid_H1[0])
    H0[:3, 3] -= H1[:3, 3]
    H1[:3, 3] *= 0

    # Apply transform with motion distorsion
    rect_points0 = motion_rectified(vid_pts[0], vid_ts[0], H0, H1)
    annots = vid_labels[0]

    # Remove ground if we have map
    if (map_path):
        rect_points = rect_points0[annots > 1.5]
        annots = annots[annots > 1.5]
    else:
        rect_points = rect_points0[annots > 0.5]
        annots = annots[annots > 0.5]

    # Update pcd
    pcd_update_from_data(rect_points, annots, pcd, colormap)
    vis.add_geometry(pcd)

    # Create a pcd for the map
    pcd_map = o3d.geometry.PointCloud()
    if (map_path):
        
        data = read_ply(map_path)
        map_points = np.vstack((data['x'], data['y'], data['z'])).T
        # map_normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
        map_classif = data['classif']

        # Only keep ground points for map
        map_mask = np.logical_and(map_classif > 0.5, map_classif < 1.5)
        map_points = map_points[map_mask]
        # map_normals = map_normals[map_mask]
        map_classif = map_classif[map_mask]

        # Reduce to visible space
        visible_mask = get_visble_pts(np.copy(map_points), rect_points0, (H0[:3, 3] + H1[:3, 3]) / 2)
        map_points_vis = np.copy(map_points[visible_mask])
        # map_normals_vis = map_normals[visible_mask]
        map_classif_vis = np.copy(map_classif[visible_mask])

        pcd_map = o3d.geometry.PointCloud()
        pcd_update_from_data(map_points_vis, map_classif_vis, pcd_map, colormap_map)
        vis.add_geometry(pcd_map)

    # Add Robot mesh
    pcd_robot = o3d.geometry.PointCloud()
    robot_pts, robot_annots, colormap_robot = robot_point_model()
    pcd_update_from_data(robot_pts, robot_annots, pcd_robot, colormap_robot)
    vis.add_geometry(pcd_robot)

    # Apply render options
    render_option = vis.get_render_option()
    render_option.light_on = True
    render_option.point_size = 3
    render_option.show_coordinate_frame = True

    # Prepare view point
    view_control = vis.get_view_control()
    target = H1[:3, 3]
    front = target + np.array([-3.0, -2.0, 4.0])  # scale of this does not matter it is just for direction
    view_control.set_front(front)
    view_control.set_lookat(target)
    view_control.set_up([0.0, 0.0, 1.0])
    view_control.set_zoom(0.15)
    # view_control.change_field_of_view(0.45)
    
    pinhole0 = view_control.convert_to_pinhole_camera_parameters()
    follow_H0 = np.copy(pinhole0.extrinsic)

    # Advanced display
    N = len(vid_pts)
    progress_n = 30
    fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
    print('\nGenerating Open3D screenshots')
    video_list = []
    for i, pts in enumerate(vid_pts):

        # if i < 1:
        #     continue

        if i > len(vid_H1) - 1:
            break

        H0 = np.copy(vid_H0[i])
        H1 = np.copy(vid_H1[i])

        p0 = np.copy(H1[:3, 3])

        # # Only rotate frame to have a hoverer mode
        # T_map = np.copy(H1[:3, 3])
        # H0[:3, 3] -= T_map
        # H1[:3, 3] *= 0

        # Apply transform with motion distorsion
        rect_points0 = motion_rectified(pts, vid_ts[i], H0, H1)
        annots = vid_labels[i]

        # Remove ground if we have map
        if (map_path):
            rect_points = rect_points0[annots > 1.5]
            annots = annots[annots > 1.5]
        else:
            rect_points = rect_points0[annots > 0.5]
            annots = annots[annots > 0.5]

        # Recenter for hoverer mode
        rect_points = rect_points - p0

        # Update pcd
        pcd_update_from_data(rect_points, annots, pcd, colormap)
        vis.update_geometry(pcd)

        # Update pcd for the map
        if (map_path):

            # Reduce to visible space
            visible_mask = get_visble_pts(np.copy(map_points), rect_points0, (H0[:3, 3] + H1[:3, 3]) / 2)
            map_points_vis = np.copy(map_points[visible_mask])
            # map_normals_vis = map_normals[visible_mask]
            map_classif_vis = np.copy(map_classif[visible_mask])

            pcd_update_from_data(map_points_vis - p0, map_classif_vis, pcd_map, colormap_map)
            vis.update_geometry(pcd_map)

        # Update Robot mesh
        H1_robot = np.copy(H1)
        robot_ts = vid_ts[i][:robot_pts.shape[0]]
        new_robot_pts = motion_rectified(np.copy(robot_pts), robot_ts, H1_robot, H1_robot)
        pcd_update_from_data(new_robot_pts - p0, robot_annots, pcd_robot, colormap_robot)
        vis.update_geometry(pcd_robot)

        # Render
        vis.poll_events()
        vis.update_renderer()

        # Screenshot
        image = vis.capture_screen_float_buffer(True)
        npimage = (np.asarray(image) * 255).astype(np.uint8)
        if npimage.shape[0] % 2 == 1:
            npimage = npimage[:-1, :]
        if npimage.shape[1] % 2 == 1:
            npimage = npimage[:, :-1]
        video_list.append(npimage)
        # plt.imsave('test_{:d}.png'.format(i), image, dpi=1)

        print('', end='\r')
        print(fmt_str.format('#' * (((i + 1) * progress_n) // N), 100 * (i + 1) / N), end='', flush=True)

    # Show a nice 100% progress bar
    print('', end='\r')
    print(fmt_str.format('#' * progress_n, 100), flush=True)
    print('\n')




    if video_path.endswith('.gif'):
        imageio.mimsave(video_path, video_list, fps=30)

    elif video_path.endswith('.mp4'):
        # Write video file
        print('\nWriting video file')
        kargs = {'macro_block_size': None}
        with imageio.get_writer(video_path, mode='I', fps=30, quality=10, **kargs) as writer:
            N = len(video_list)
            for i, frame in enumerate(video_list):
                writer.append_data(frame)

                print('', end='\r')
                print(fmt_str.format('#' * (((i + 1) * progress_n) // N), 100 * (i + 1) / N), end='', flush=True)

        # Show a nice 100% progress bar
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\n')

    else:
        raise ValueError('Unknown video extension: \"{:s}\"'.format(video_path[-4:]))
    
    vis.destroy_window()

    return


def open_3d_vid(video_path, vid_pts, vid_labels, vid_ts, vid_H0, vid_H1, map_path=None):
    
    
    # Colormap
    colormap = np.array([[209, 209, 209],
                        [122, 122, 122],
                        [255, 255, 0],
                        [0, 98, 255],
                        [255, 0, 0]], dtype=np.float32) / 255

    colormap_map = 1 - (1 - colormap) * 0.9
    
    # Window for headless visu
    vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=True, width=2560, height=1440, left=0, top=0)
    # vis.create_window(visible=True, width=1920, height=1080, left=0, top=0)
    vis.create_window(visible=True, width=1600, height=900, left=0, top=0)
    # vis.create_window(visible=True, width=1280, height=720, left=0, top=0)
    
    # Load the first frame in the window
    vis.clear_geometries()
    pcd = o3d.geometry.PointCloud()
    
    H0 = np.copy(vid_H0[0])
    H1 = np.copy(vid_H1[0])

    # Only rotate frame to have a hoverer mode
    H0[:3, 3] -= H1[:3, 3]
    H1[:3, 3] *= 0

    # Apply transform with motion distorsion
    rect_points0 = motion_rectified(vid_pts[0], vid_ts[0], H0, H1)
    annots = vid_labels[0]

    # Remove ground if we have map
    if (map_path):
        rect_points = rect_points0[annots > 1.5]
        annots = annots[annots > 1.5]
    else:
        rect_points = rect_points0[annots > 0.5]
        annots = annots[annots > 0.5]

    # Update pcd
    pcd_update_from_data(rect_points, annots, pcd, colormap)
    vis.add_geometry(pcd)

    # Create a pcd for the map
    pcd_map = o3d.geometry.PointCloud()
    if (map_path):
        
        data = read_ply(map_path)
        map_points = np.vstack((data['x'], data['y'], data['z'])).T
        # map_normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
        map_classif = data['classif']

        # Only keep ground points for map
        map_mask = np.logical_and(map_classif > 0.5, map_classif < 1.5)
        map_points = map_points[map_mask]
        # map_normals = map_normals[map_mask]
        map_classif = map_classif[map_mask]

        # Reduce to visible space
        visible_mask = get_visble_pts(np.copy(map_points), rect_points0, (H0[:3, 3] + H1[:3, 3]) / 2)
        map_points_vis = np.copy(map_points[visible_mask])
        # map_normals_vis = map_normals[visible_mask]
        map_classif_vis = np.copy(map_classif[visible_mask])

        pcd_map = o3d.geometry.PointCloud()
        pcd_update_from_data(map_points_vis, map_classif_vis, pcd_map, colormap_map)
        vis.add_geometry(pcd_map)

    # Add Robot mesh
    pcd_robot = o3d.geometry.PointCloud()
    robot_pts, robot_annots, colormap_robot = robot_point_model()
    pcd_update_from_data(robot_pts, robot_annots, pcd_robot, colormap_robot)
    vis.add_geometry(pcd_robot)

    # Apply render options
    render_option = vis.get_render_option()
    render_option.light_on = True
    render_option.point_size = 3
    render_option.show_coordinate_frame = True

    # Prepare view point
    view_control = vis.get_view_control()
    target = H1[:3, 3]
    front = target + np.array([-3.0, -2.0, 4.0])  # scale of this does not matter it is just for direction
    view_control.set_front(front)
    view_control.set_lookat(target)
    view_control.set_up([0.0, 0.0, 1.0])
    view_control.set_zoom(0.15)
    # view_control.change_field_of_view(0.45)
    
    pinhole0 = view_control.convert_to_pinhole_camera_parameters()
    follow_H0 = np.copy(pinhole0.extrinsic)

    # Advanced display
    N = len(vid_pts)
    progress_n = 30
    fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
    print('\nGenerating Open3D screenshots')
    video_list = []
    for i, pts in enumerate(vid_pts):

        # if i < 1:
        #     continue

        if i > len(vid_H1) - 1:
            break

        H0 = np.copy(vid_H0[i])
        H1 = np.copy(vid_H1[i])

        p0 = np.copy(H1[:3, 3])

        # # Only rotate frame to have a hoverer mode
        # T_map = np.copy(H1[:3, 3])
        # H0[:3, 3] -= T_map
        # H1[:3, 3] *= 0

        # Apply transform with motion distorsion
        rect_points0 = motion_rectified(pts, vid_ts[i], H0, H1)
        annots = vid_labels[i]

        # Remove ground if we have map
        if (map_path):
            rect_points = rect_points0[annots > 1.5]
            annots = annots[annots > 1.5]
        else:
            rect_points = rect_points0[annots > 0.5]
            annots = annots[annots > 0.5]

        # Recenter for hoverer mode
        rect_points = rect_points - p0

        # Update pcd
        pcd_update_from_data(rect_points, annots, pcd, colormap)
        vis.update_geometry(pcd)

        # Update pcd for the map
        if (map_path):

            # Reduce to visible space
            visible_mask = get_visble_pts(np.copy(map_points), rect_points0, (H0[:3, 3] + H1[:3, 3]) / 2)
            map_points_vis = np.copy(map_points[visible_mask])
            # map_normals_vis = map_normals[visible_mask]
            map_classif_vis = np.copy(map_classif[visible_mask])

            pcd_update_from_data(map_points_vis - p0, map_classif_vis, pcd_map, colormap_map)
            vis.update_geometry(pcd_map)

        # Update Robot mesh
        H1_robot = np.copy(H1)
        robot_ts = vid_ts[i][:robot_pts.shape[0]]
        new_robot_pts = motion_rectified(np.copy(robot_pts), robot_ts, H1_robot, H1_robot)
        pcd_update_from_data(new_robot_pts - p0, robot_annots, pcd_robot, colormap_robot)
        vis.update_geometry(pcd_robot)

        # Render
        vis.poll_events()
        vis.update_renderer()

        # Screenshot
        image = vis.capture_screen_float_buffer(True)
        npimage = (np.asarray(image) * 255).astype(np.uint8)
        if npimage.shape[0] % 2 == 1:
            npimage = npimage[:-1, :]
        if npimage.shape[1] % 2 == 1:
            npimage = npimage[:, :-1]
        video_list.append(npimage)
        # plt.imsave('test_{:d}.png'.format(i), image, dpi=1)

        print('', end='\r')
        print(fmt_str.format('#' * (((i + 1) * progress_n) // N), 100 * (i + 1) / N), end='', flush=True)

    # Show a nice 100% progress bar
    print('', end='\r')
    print(fmt_str.format('#' * progress_n, 100), flush=True)
    print('\n')




    if video_path.endswith('.gif'):
        imageio.mimsave(video_path, video_list, fps=30)

    elif video_path.endswith('.mp4'):
        # Write video file
        print('\nWriting video file')
        kargs = {'macro_block_size': None}
        with imageio.get_writer(video_path, mode='I', fps=30, quality=10, **kargs) as writer:
            N = len(video_list)
            for i, frame in enumerate(video_list):
                writer.append_data(frame)

                print('', end='\r')
                print(fmt_str.format('#' * (((i + 1) * progress_n) // N), 100 * (i + 1) / N), end='', flush=True)

        # Show a nice 100% progress bar
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\n')

    else:
        raise ValueError('Unknown video extension: \"{:s}\"'.format(video_path[-4:]))
    
    vis.destroy_window()

    return


def get_visble_pts(pts, ray_pts, ray_origin, min_radius=1.0, max_radius=40.0):

    # Center points on origin
    ray_pts = ray_pts - ray_origin
    pts = pts - ray_origin

    # angle and radius resolution
    da = 5.01 * np.pi / 180
    dr = 0.5

    # Create an angle/radius table
    angles = np.arange(0, 2 * np.pi, da)
    radiuses = np.arange(min_radius, max_radius, dr)
    Na = angles.shape[0]
    Nr = radiuses.shape[0]
    vis_table = np.zeros((Na, Nr), dtype=bool)

    # For each angle get the visible radius
    ray_rtp = cart2pol(ray_pts)
    ray_ri = np.floor((ray_rtp[:, 0] - min_radius) / dr).astype(np.int32)
    ray_ai = np.floor((ray_rtp[:, 2]) / da).astype(np.int32)
    ray_coords = np.vstack((ray_ai, ray_ri)).T

    # Remove rays outside area
    valid_mask = ray_ri > 0
    ray_coords = ray_coords[valid_mask, :]
    ray_ri = ray_ri[valid_mask]
    
    # Reduce too long ray
    long_mask = ray_ri >= Nr
    ray_coords[long_mask, 1] = Nr - 1

    vis_table[ray_coords[:, 0], ray_coords[:, 1]] = True

    # Get max radiuses
    radius_table = np.zeros_like(vis_table, dtype=np.float32)
    radius_table += np.expand_dims(radiuses+dr, 0)
    radius_table[np.logical_not(vis_table)] = 0
    max_radiuses = np.max(radius_table, axis=1)


    # Get the mask of visible points
    rtp = cart2pol(pts)
    ai = np.floor((rtp[:, 2]) / da).astype(np.int32)
    visi_mask = rtp[:, 0] < max_radiuses[ai]
    visi_mask[rtp[:, 0] < min_radius] = True
    return visi_mask


def plt_2D_gif(gif_name, all_pts, all_colors, im_lim):


    figA, axA = plt.subplots(1, 1, figsize=(10, 7))
    plt.subplots_adjust(left=0.1, bottom=0.2)
    
    # Plot first frame of seq
    plotss = [axA.scatter(all_pts[0][:, 0],
                          all_pts[0][:, 1],
                          s=2.0,
                          c=all_colors[0])]

    # Show a circle of the loop closure area
    axA.add_patch(Circle((0, 0), radius=0.2,
                         edgecolor=[0.2, 0.2, 0.2],
                         facecolor=[1.0, 0.79, 0],
                         fill=True,
                         lw=1))

    # # Customize the graph
    # axA.grid(linestyle='-.', which='both')
    axA.set_xlim(-im_lim, im_lim)
    axA.set_ylim(-im_lim, im_lim)
    axA.set_aspect('equal', adjustable='box')

    def animate(f_i):
        for plot_i, plot_obj in enumerate(plotss):
            plot_obj.set_offsets(all_pts[f_i])
            plot_obj.set_color(all_colors[f_i])
        return plotss

    anim = FuncAnimation(figA, animate,
                         frames=np.arange(len(all_pts)),
                         interval=50,
                         blit=True)
    anim.save(gif_name, fps=30)

    plt.close(figA)

    return





# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    dataset_path = '../Data/RealMyhal'
    train_days = ['2021-12-06_08-12-39',    # - \
                  '2021-12-06_08-38-16',    # -  \
                  '2021-12-06_08-44-07',    # -   > First runs with controller for mapping of the environment
                  '2021-12-06_08-51-29',    # -  /
                  '2021-12-06_08-54-58',    # - /
                  '2021-12-10_13-32-10',    # - \
                  '2021-12-10_13-26-07',    # -  \
                  '2021-12-10_13-17-29',    # -   > Session with normal TEB planner
                  '2021-12-10_13-06-09',    # -  /
                  '2021-12-10_12-53-37',    # - /
                  '2021-12-13_18-16-27',    # - \
                  '2021-12-13_18-22-11',    # -  \
                  '2021-12-15_19-09-57',    # -   > Session with normal TEB planner Tour A and B
                  '2021-12-15_19-13-03']    # -  /
    map_i = 3
    refine_i = np.array([0, 6, 7, 8])
    train_i = np.arange(len(train_days))[5:]
    val_inds = [0]
         
    map_day = train_days[map_i]
    refine_days = np.array(train_days)[refine_i]
    train_days = np.sort(np.array(train_days)[train_i])

    show_seq_slider(dataset_path, train_days)

    # # List of days we want to create video from
    # my_days = train_days_RandBounce

    # get_videos(my_days)

