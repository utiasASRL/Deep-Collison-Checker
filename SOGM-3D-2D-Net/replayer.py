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

from mpl_toolkits.mplot3d import Axes3D


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
        while f_name_i < len(f_names) and not (f_names[f_name_i].endswith(f_name)):
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


# ----------------------------------------------------------------------------------------------------------------------
#
#           Plot functions
#       \********************/
#


def open3d_video(my_days):
    # List of days we want to create video from

    #day_list = day_list[1:]

    # Path of the original simu
    simu_path = '../../../Myhal_Simulation/simulated_runs/'

    # path of the original frames
    lidar_folder = 'sim_frames'

    # path of the classified frames
    classif_folder = 'classified_frames'

    # Path where we save the videos
    res_path = '../../../Myhal_Simulation/annot_videos'
    if not exists(res_path):
        makedirs(res_path)

    # Scalar field we want to show
    # scalar_field = 'classif'
    # scalar_field = 'labels'
    # scalar_field = 'pred'
    scalar_field = 'cat'

    # Should the cam be static or follow the robot
    following_cam = True

    # Are frame localized in world coordinates?
    localized_frames = False

    # Colormap
    colormap = np.array([[209, 209, 209],
                        [122, 122, 122],
                        [255, 255, 0],
                        [0, 98, 255],
                        [255, 0, 0]], dtype=np.float32) / 255

    ##################
    # Open3D animation
    ##################

    # Window for headless visu
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)

    for day in my_days:

        ########
        # Init #
        ########

        # Load collider preds
        with open(join(simu_path, day, 'logs-' + day, 'collider_data.pickle'), 'rb') as f:
            collider_preds = pickle.load(f)

        # Load collider preds
        with open(join(simu_path, day, 'logs-' + day, 'teb_local_plans.pickle'), 'rb') as f:
            teb_local_plans = pickle.load(f)
            
        # for plan0 in teb_local_plans:
        #     print(plan0['header_stamp'], ['{:.3f}'.format(plan0['header_stamp'] + 10 * pose_tuple[2]) for pose_tuple in plan0['pose_list']])
        #     print('--------------')

        # Get annotated lidar frames
        lidar_path = join(simu_path, day, lidar_folder)
        classif_path = join(simu_path, day, classif_folder)

        f_names = [f for f in listdir(lidar_path) if f[-4:] == '.ply']
        f_times = np.array([float(f[:-4]) for f in f_names], dtype=np.float64)
        f_names = np.array([join(lidar_path, f) for f in f_names])
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
            H_frame = np.copy(day_map_H[0])
            H_frame[:3, 3] *= 0

        # Add frame point cloud
        pcd_update_from_ply(f_names[0], pcd, H_frame, colormap, scalar_field)
        vis.add_geometry(pcd)

        # Init traj point cloud
        traj_lines = o3d.geometry.LineSet()
        
        # Center z axis on current time stamp
        t_points = np.copy(day_map_H[:, :3, 3])
        t_points[:, 2] = np.copy(day_map_t - day_map_t[0])
        traj_lines.points = o3d.utility.Vector3dVector(t_points)
        t_lines = [[i, i+1] for i in range(t_points.shape[0] - 1)]
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(t_points),
                                        lines=o3d.utility.Vector2iVector(t_lines))
        traj_classif = (t_points[:-1, 2] > 0).astype(np.int32) + 1
        np_colors = colormap[traj_classif, :]
        traj_lines.colors = o3d.utility.Vector3dVector(np_colors)


        # Add traj to visu
        vis.add_geometry(traj_lines)
        
        # Apply render options
        render_option = vis.get_render_option()
        render_option.light_on = False
        render_option.point_size = 5
        render_option.show_coordinate_frame = True

        # Prepare view point
        view_control = vis.get_view_control()
        if following_cam:
            target = np.copy(day_map_H[0][:3, 3])
            front = target + np.array([5.0, -10.0, 15.0])
            view_control.set_front(front)
            view_control.set_lookat(target)
            view_control.set_up([0.0, 0.0, 1.0])
            view_control.set_zoom(0.2)
            pinhole0 = view_control.convert_to_pinhole_camera_parameters()
            follow_H0 = np.copy(pinhole0.extrinsic)
        else:
            traj_points = np.vstack([np.copy(H[:3, 3]) for H in day_map_H])
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
        print(N)
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
                H_frame = np.copy(day_map_H[i])
                H_frame[:3, 3] *= 0
            pcd_update_from_ply(f_name, pcd, H_frame, colormap, scalar_field=scalar_field)
            vis.update_geometry(pcd)

            # Update traj
            t_points = np.copy(day_map_H[:, :3, 3])
            if np.sum(np.abs(H_frame)) > 1e-3:
                t_points = np.hstack((t_points, np.ones_like(t_points[:, :1])))
                t_points = np.matmul(t_points, np.copy(day_map_H[i])).astype(np.float32)
                t_points = np.matmul(t_points, H_frame.T).astype(np.float32)[:, :3]
            t_points[:, 2] = np.copy(day_map_t - day_map_t[i])
            traj_lines.points = o3d.utility.Vector3dVector(t_points)
            traj_lines.lines = o3d.utility.Vector2iVector(t_lines)
            traj_classif = (t_points[:, 2] > 0).astype(np.int32) + 1
            np_colors = colormap[traj_classif, :]
            traj_lines.colors = o3d.utility.Vector3dVector(np_colors)
            vis.update_geometry(traj_lines)

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

    return


def traj_plan_compare(day):

    # Path of the original simu
    simu_path = '../../../Myhal_Simulation/simulated_runs/'

    # path of the original frames
    lidar_folder = 'sim_frames'

    # path of the classified frames
    classif_folder = 'classified_frames'

    # Path where we save the videos
    res_path = '../../../Myhal_Simulation/annot_videos'
    if not exists(res_path):
        makedirs(res_path)


    ########
    # Init #
    ########

    # Load collider preds
    with open(join(simu_path, day, 'logs-' + day, 'teb_local_plans.pickle'), 'rb') as f:
        teb_local_plans = pickle.load(f)
        
    # for plan0 in teb_local_plans:
    #     print(plan0['header_stamp'], ['{:.3f}'.format(plan0['header_stamp'] + 10 * pose_tuple[2]) for pose_tuple in plan0['pose_list']])
    #     print('--------------')

    # Get annotated lidar frames
    lidar_path = join(simu_path, day, lidar_folder)

    f_names = [f for f in listdir(lidar_path) if f[-4:] == '.ply']
    f_times = np.array([float(f[:-4]) for f in f_names], dtype=np.float64)
    f_names = np.array([join(lidar_path, f) for f in f_names])
    ordering = np.argsort(f_times)
    f_names = f_names[ordering]
    f_times = f_times[ordering]

    # Load mapping poses
    map_traj_file = join(simu_path, day, 'logs-'+day, 'map_traj.ply')
    data = read_ply(map_traj_file)
    map_T = np.vstack([data['pos_x'], data['pos_y'], data['pos_z']]).T
    map_Q = np.vstack([data['rot_x'], data['rot_y'], data['rot_z'], data['rot_w']]).T

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

    # Load gt_poses
    gt_t, gt_H = load_gt_poses(simu_path, day)
    
    # Init loc abd gt traj
    gt_traj = gt_H[:, :3, 3]
    gt_traj[:, 2] = gt_t
    loc_traj = day_map_H[:, :3, 3]
    loc_traj[:, 2] = day_map_t

    ###############
    # Mayavi visu #
    ###############

    # Create figure 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot new data feature
    global plan_i
    plan_i = 0
    plan = np.copy(np.array(teb_local_plans[plan_i]['pose_list'], dtype=np.float32))
    plan[:, 2] *= 10
    plan[:, 2] += teb_local_plans[plan_i]['header_stamp']

    plots = ax.plot(plan[:, 0],
                    plan[:, 1],
                    plan[:, 2], 'g-', lw=2.0)

    min_t = np.min(plan[:, 2]) - 1.0
    max_t = np.max(plan[:, 2]) + 1.0

    # Show 3D traj
    gt_mask = np.logical_and(gt_traj[:, 2] > min_t, gt_traj[:, 2] < max_t)
    sub_gt = gt_traj[gt_mask, :]
    plots += ax.plot(sub_gt[:, 0],
                     sub_gt[:, 1],
                     sub_gt[:, 2], 'b-', lw=1.0)
    
    loc_mask = np.logical_and(loc_traj[:, 2] > min_t, loc_traj[:, 2] < max_t)
    sub_loc = loc_traj[loc_mask, :]
    plots += ax.plot(sub_loc[:, 0],
                     sub_loc[:, 1],
                     sub_loc[:, 2], 'r--', lw=1.5)


    def onkey(event):
        global plan_i

        if event.key == 'left':
            plan_i = (plan_i - 1) % len(teb_local_plans)

        elif event.key == 'right':
            plan_i = (plan_i + 1) % len(teb_local_plans)

        elif event.key == 'g':
            plan_i = (plan_i - 10) % len(teb_local_plans)

        elif event.key == 'h':
            plan_i = (plan_i + 10) % len(teb_local_plans)

        plan = np.copy(np.array(teb_local_plans[plan_i]['pose_list'], dtype=np.float32))
        plan[:, 2] *= 10
        plan[:, 2] += teb_local_plans[plan_i]['header_stamp']

        plots[0].set_xdata(plan[:, 0])
        plots[0].set_ydata(plan[:, 1])
        plots[0].set_3d_properties(plan[:, 2])

        min_t = np.min(plan[:, 2]) - 1.0
        max_t = np.max(plan[:, 2]) + 1.0

        gt_mask = np.logical_and(gt_traj[:, 2] > min_t, gt_traj[:, 2] < max_t)
        sub_gt = gt_traj[gt_mask, :]

        plots[1].set_xdata(sub_gt[:, 0])
        plots[1].set_ydata(sub_gt[:, 1])
        plots[1].set_3d_properties(sub_gt[:, 2])
        
        loc_mask = np.logical_and(loc_traj[:, 2] > min_t, loc_traj[:, 2] < max_t)
        sub_loc = loc_traj[loc_mask, :]

        plots[2].set_xdata(sub_loc[:, 0])
        plots[2].set_ydata(sub_loc[:, 1])
        plots[2].set_3d_properties(sub_loc[:, 2])

        #ax.apply_aspect()
        #ax.autoscale_view()

        min_x = np.min(sub_gt[:, 0])
        max_x = np.max(sub_gt[:, 0])
        min_y = np.min(sub_gt[:, 1])
        max_y = np.max(sub_gt[:, 1])

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_t, max_t)
        plt.draw()

        return plots


    cid = fig.canvas.mpl_connect('key_press_event', onkey)


    plt.show()

    a = 1/0


    return


def plan_pred_compare(day):

    # Path of the original simu
    simu_path = '../../../Myhal_Simulation/simulated_runs/'

    # path of the original frames
    lidar_folder = 'sim_frames'

    # path of the classified frames
    classif_folder = 'classified_frames'

    # Path where we save the videos
    res_path = '../../../Myhal_Simulation/annot_videos'
    if not exists(res_path):
        makedirs(res_path)

    # Scalar field we want to show
    # scalar_field = 'classif'
    # scalar_field = 'labels'
    # scalar_field = 'pred'
    scalar_field = 'cat'

    # Should the cam be static or follow the robot
    following_cam = True

    # Are frame localized in world coordinates?
    localized_frames = False

    # Colormap
    colormap = np.array([[209, 209, 209],
                        [122, 122, 122],
                        [255, 255, 0],
                        [0, 98, 255],
                        [255, 0, 0]], dtype=np.float32) / 255

    ########
    # Init #
    ########

    # Load collider preds
    with open(join(simu_path, day, 'logs-' + day, 'collider_data.pickle'), 'rb') as f:
        collider_preds = pickle.load(f)

    # Load collider preds
    with open(join(simu_path, day, 'logs-' + day, 'teb_local_plans.pickle'), 'rb') as f:
        teb_local_plans = pickle.load(f)
        
    # for plan0 in teb_local_plans:
    #     print(plan0['header_stamp'], ['{:.3f}'.format(plan0['header_stamp'] + 10 * pose_tuple[2]) for pose_tuple in plan0['pose_list']])
    #     print('--------------')

    # Get annotated lidar frames
    lidar_path = join(simu_path, day, lidar_folder)
    classif_path = join(simu_path, day, classif_folder)

    f_names = [f for f in listdir(lidar_path) if f[-4:] == '.ply']
    f_times = np.array([float(f[:-4]) for f in f_names], dtype=np.float64)
    f_names = np.array([join(lidar_path, f) for f in f_names])
    ordering = np.argsort(f_times)
    f_names = f_names[ordering]
    f_times = f_times[ordering]

    # Load mapping poses
    map_traj_file = join(simu_path, day, 'logs-'+day, 'map_traj.ply')
    data = read_ply(map_traj_file)
    map_T = np.vstack([data['pos_x'], data['pos_y'], data['pos_z']]).T
    map_Q = np.vstack([data['rot_x'], data['rot_y'], data['rot_z'], data['rot_w']]).T

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

    # Load gt_poses
    gt_t, gt_H = load_gt_poses(simu_path, day)
    
    # Init loc abd gt traj
    gt_traj = gt_H[:, :3, 3]
    gt_traj[:, 2] = gt_t
    loc_traj = day_map_H[:, :3, 3]
    loc_traj[:, 2] = day_map_t

    ###############
    # Mayavi visu #
    ###############

    # Create figure for features
    from mayavi import mlab
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    activations = mlab.plot3d(gt_traj[:, 0],
                              gt_traj[:, 1],
                              gt_traj[:, 2],
                              gt_traj[:, 2],
                              figure=fig1)
    mlab.show()

    a = 1/0

    # Indices
    global file_i
    file_i = 0

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Plot new data feature
        t0 = teb_local_plans[file_i]['header_stamp']
        plan = np.array(teb_local_plans[file_i]['pose_list'], dtype=np.float32)
        print(plan.shape)
        plan[:, 2] *= 10
        plan[:, 2] += t0

        # Show point clouds colorized with activations
        activations = mlab.plot3d(plan[:, 0],
                                  plan[:, 1],
                                  plan[:, 2],
                                  plan[:, 2],
                                  scale_factor=3.0,
                                  scale_mode='none',
                                  figure=fig1)

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i

        if vtk_obj.GetKeyCode() in ['g', 'G']:

            file_i = (file_i - 1) % len(teb_local_plans)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:

            file_i = (file_i + 1) % len(teb_local_plans)
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()

    a = 1/0

    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    # Create video of the logs
    #open3d_video(['2021-05-31-16-20-51'])

    # Compare traj to teb plans
    #traj_plan_compare('2021-05-31-16-20-51')

    # Compare plan to preds
    plan_pred_compare('2021-05-31-16-20-51')
