#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
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


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utilities
#       \***************/
#

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


def pcd_update_from_ply(ply_name, pcd, H_frame, scalar_field='classif'):

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

    #day_list = day_list[1:]

    # Path of the original simu
    simu_path = '../Data/Simulation/simulated_runs/'

    # path of the classified frames
    path = '../Data/Simulation/annotated_frames'
    #path = '../Data/Simulation/predicted_frames'
    #path = '../Data/Simulation/simulated_runs'

    # Name of folder where frames are stored (if any)
    if path.endswith('annotated_frames') or path.endswith('predicted_frames'):
        folder_name = ''

    elif path.endswith('simulated_runs'):
        folder_name = 'sim_frames'
        #folder_name = 'classified_frames'

    else:
        raise ValueError('wrong path for frames video')

    # Path where we save the videos
    res_path = '../Data/Simulation/annot_videos'
    if not exists(res_path):
        makedirs(res_path)

    # Scalar field we want to show
    if path.endswith('annotated_frames'):
        scalar_field = 'classif'
        #scalar_field = 'labels'
        
    elif path.endswith('predicted_frames'):
        scalar_field = 'pred'

    elif path.endswith('simulated_runs'):
        scalar_field = 'cat'

    else:
        raise ValueError('wrong path for frames video')

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

    ##################
    # Mayavi animation
    ##################

    # Window for headless visu
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)

    for day in my_days:

        ########
        # Init #
        ########

        # Get annotated lidar frames
        if folder_name:
            frames_folder = join(path, day, folder_name)
        else:
            frames_folder = join(path, day)

        f_names = [f for f in listdir(frames_folder) if f[-4:] == '.ply']
        f_times = np.array([float(f[:-4]) for f in f_names], dtype=np.float64)
        f_names = np.array([join(frames_folder, f) for f in f_names])
        ordering = np.argsort(f_times)
        f_names = f_names[ordering]
        f_times = f_times[ordering]

        # Load mapping poses
        map_traj_file = join(simu_path, day, 'logs-'+day, 'map_traj.ply')
        data = read_ply(map_traj_file)
        map_T = np.vstack([data['pos_x'], data['pos_y'], data['pos_z']]).T
        map_Q = np.vstack([data['rot_x'], data['rot_y'], data['rot_z'], data['rot_w']]).T

        # Load info on 
        map_traj_file = join(simu_path, day, 'logs-'+day, 'map_traj.ply')

        # Times
        day_map_t = data['time']

        # Convert map to homogenous rotation/translation matrix
        map_R = scipyR.from_quat(map_Q)
        map_R = map_R.as_matrix()
        day_map_H = np.zeros((len(day_map_t), 4, 4))
        day_map_H[:, :3, :3] = map_R
        day_map_H[:, :3, 3] = map_T
        day_map_H[:, 3, 3] = 1

        # Filter valid frames
        f_names, day_map_t, day_map_H = filter_valid_frames(f_names, day_map_t, day_map_H)

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
            H_frame = np.zeros((4, 4))
        else:
            # H_frame = day_map_H[0]
            # Only rotate frame to have a hoverer mode
            H_frame = day_map_H[0]
            H_frame[:3, 3] *= 0

        pcd_update_from_ply(f_names[0], pcd, H_frame, scalar_field)
        vis.add_geometry(pcd)

        # Apply render options
        render_option = vis.get_render_option()
        render_option.light_on = False
        render_option.point_size = 5
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
        N = len(f_names)
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nGenerating Open3D screenshots for ' + day)
        video_list = []
        for i, f_name in enumerate(f_names):

            if i > len(day_map_H) - 1:
                break

            # New frame
            if localized_frames:
                H_frame = np.zeros((4, 4))
            else:
                # H_frame = day_map_H[i]
                # Only rotate frame to have a hoverer mode
                H_frame = day_map_H[i]
                H_frame[:3, 3] *= 0
            pcd_update_from_ply(f_name, pcd, H_frame, scalar_field=scalar_field)
            vis.update_geometry(pcd)

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
            image = vis.capture_screen_float_buffer(False)
            video_list.append((np.asarray(image) * 255).astype(np.uint8))
            #plt.imsave('test_{:d}.png'.format(i), image, dpi=1)

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
