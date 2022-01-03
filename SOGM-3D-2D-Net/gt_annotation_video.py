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
import numpy as np
from utils.ply import read_ply

#from mayavi import mlab
import imageio
import pickle
import time
from os import listdir, makedirs
from os.path import join, exists

import open3d as o3d
import matplotlib.pyplot as plt
#from datasets.MyhalCollision import *
from scipy.spatial.transform import Rotation as scipyR
from slam.dev_slam import filter_frame_timestamps
from slam.PointMapSLAM import motion_rectified


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
    robot_pts = np.vstack((np.ravel(xv), np.ravel(yv), np.ravel(xv) * 0 + 0.3)).T
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


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    train_days_RandBounce = ['2021-05-15-23-15-09',
                             '2021-05-15-23-33-25',
                             '2021-05-15-23-54-50',
                             '2021-05-16-00-44-53',
                             '2021-05-16-01-09-43',
                             '2021-05-16-20-37-47',
                             '2021-05-16-20-59-49',
                             '2021-05-16-21-22-30',
                             '2021-05-16-22-26-45',
                             '2021-05-16-22-51-06',
                             '2021-05-16-23-34-15',
                             '2021-05-17-01-21-44',
                             '2021-05-17-01-37-09',
                             '2021-05-17-01-58-57',
                             '2021-05-17-02-34-27',
                             '2021-05-17-02-56-02',
                             '2021-05-17-03-54-39',
                             '2021-05-17-05-26-10',
                             '2021-05-17-05-41-45']
                             
    train_days_RandWand = ['2021-05-17-14-04-52',
                           '2021-05-17-14-21-56',
                           '2021-05-17-14-44-46',
                           '2021-05-17-15-26-04',
                           '2021-05-17-15-50-45',
                           '2021-05-17-16-14-26',
                           '2021-05-17-17-02-17',
                           '2021-05-17-17-27-02',
                           '2021-05-17-17-53-42',
                           '2021-05-17-18-46-44',
                           '2021-05-17-19-02-37',
                           '2021-05-17-19-39-19',
                           '2021-05-17-20-14-57',
                           '2021-05-17-20-48-53',
                           '2021-05-17-21-36-22',
                           '2021-05-17-22-16-13',
                           '2021-05-17-22-40-46',
                           '2021-05-17-23-08-01',
                           '2021-05-17-23-48-22',
                           '2021-05-18-00-07-26',
                           '2021-05-18-00-23-15',
                           '2021-05-18-00-44-33',
                           '2021-05-18-01-24-07']

    train_days_RandFlow = ['2021-06-02-19-55-16',
                           '2021-06-02-20-33-09',
                           '2021-06-02-21-09-48',
                           '2021-06-02-22-05-23',
                           '2021-06-02-22-31-49',
                           '2021-06-03-03-51-03',
                           '2021-06-03-14-30-25',
                           '2021-06-03-14-59-20',
                           '2021-06-03-15-43-06',
                           '2021-06-03-16-48-18',
                           '2021-06-03-18-00-33',
                           '2021-06-03-19-07-19',
                           '2021-06-03-19-52-45',
                           '2021-06-03-20-28-22',
                           '2021-06-03-21-32-44',
                           '2021-06-03-21-57-08']



    # List of days we want to create video from
    my_days = train_days_RandBounce

    get_videos(my_days)

