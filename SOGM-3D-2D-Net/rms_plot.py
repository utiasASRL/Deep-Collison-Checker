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

    ################
    # Plot functions
    ################

    # Figure
    figA, axA = plt.subplots(1, 1, figsize=(10, 7))
    plt.subplots_adjust(bottom=0.25)

    # Plot last PR curve for each log
    plotsA = []
    num_showed = 20
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


def pairwise_registration(source, target, d_coarse, d_fine, perform_icp=False):

    if (perform_icp):
        print("Apply point-to-plane ICP")
        icp_coarse = o3d.pipelines.registration.registration_icp(source,
                                                                 target,
                                                                 d_coarse,
                                                                 np.identity(4),
                                                                 o3d.pipelines.registration.TransformationEstimationPointToPlane())
        icp_fine = o3d.pipelines.registration.registration_icp(source,
                                                               target,
                                                               d_fine,
                                                               icp_coarse.transformation,
                                                               o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transformation_icp = icp_fine.transformation

    else:
        transformation_icp = np.identity(4)

    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source,
                                                                                          target,
                                                                                          d_fine,
                                                                                          transformation_icp)
    return transformation_icp, information_icp


def full_registration(pcds, loop_closure_edges, d_coarse, d_fine):

    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)

    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    n_pcds = len(pcds)

    print("Build o3d.pipelines.registration.PoseGraph")

    # Add odom links
    print("Adding {:d} odom_edge".format(n_pcds))
    for source_id in range(n_pcds - 1):

        target_id = source_id + 1

        transformation_icp, information_icp = pairwise_registration(pcds[source_id],
                                                                    pcds[target_id],
                                                                    d_coarse,
                                                                    d_fine)

        odometry = np.dot(transformation_icp, odometry)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                         target_id,
                                                                         transformation_icp,
                                                                         information_icp,
                                                                         uncertain=False))

    # Add odom links
    print("Adding {:d} loop_edge".format(len(loop_closure_edges)))
    for source_id, target_id in loop_closure_edges:

        transformation_icp, information_icp = pairwise_registration(pcds[source_id],
                                                                    pcds[target_id],
                                                                    d_coarse,
                                                                    d_fine,
                                                                    perform_icp=True)

        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                         target_id,
                                                                         transformation_icp,
                                                                         information_icp,
                                                                         uncertain=True))
    return pose_graph


def test_loop_closure():

    # TODO: For graph slam look at this:
    #       open3d
    #           http://www.open3d.org/docs/0.12.0/tutorial/pipelines/multiway_registration.html
    #           https://github.com/isl-org/Open3D/blob/master/cpp/open3d/pipelines/registration/Registration.cpp (information matrix computation)
    #           http://www.open3d.org/docs/0.12.0/tutorial/reconstruction_system/system_overview.html
    #
    #       ceres
    #           http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment
    #
    #       g2o
    #           https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/icp/gicp_demo.cpp
    #           https://github.com/uoip/g2opy/blob/master/python/examples/gicp_demo.py
    #
    #

    # min_f_i = 50
    # max_f_i = 6501
    # f_i_stride = 50

    # Full loop closure
    min_f_i = 0
    max_f_i = 1000000
    f_i_stride = 1
    loop_edges = [[150, 6450],
                  [250, 6350],
                  [600, 6000],
                  [800, 5800]]
    loop_edges = np.array(loop_edges, dtype=np.int32)
    loop_edges = (loop_edges - min_f_i) // f_i_stride 
    
    # Ground closure
    min_f_i = 600
    max_f_i = 1400
    f_i_stride = 1
    loop_edges = np.zeros((0, 2))


    #################
    # LOAD ALL FRAMES
    #################

    # Load all frame corrected and aligned
    frame_path = "/home/hth/Deep-Collison-Checker/Data/Real/icp_frames/tmp/"
    frame_names = np.sort([join(frame_path, f) for f in listdir(frame_path) if f.endswith('.ply')])

    frame_names = frame_names[min_f_i:max_f_i:f_i_stride]
    
    # Advanced display
    N = len(frame_names)
    progress_n = 30
    fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
    print('\nLoading frames')

    all_points = []
    all_normals = []
    all_icp_scores = []
    all_norm_scores = []
    all_lens = []
    all_pcds = []
    for i, fname in enumerate(frame_names):

        data = read_ply(fname)
        all_points.append(np.vstack((data['x'], data['y'], data['z'])).T)
        all_normals.append(np.vstack((data['nx'], data['ny'], data['nz'])).T)
        all_icp_scores.append(data['f0'])
        all_norm_scores.append(data['f1'])
        all_lens.append(data.shape[0])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points[-1])
        pcd.normals = o3d.utility.Vector3dVector(all_normals[-1])
        all_pcds.append(pcd)

        print('', end='\r')
        print(fmt_str.format('#' * ((i * progress_n) // N), 100 * i / N), end='', flush=True)

    print('', end='\r')
    print(fmt_str.format('#' * progress_n, 100), flush=True)
    print('\n')

    print(np.array(all_lens, dtype=np.int32))

    ################
    # LOAD ALL POSES
    ################

    poses_path = "/home/hth/Deep-Collison-Checker/Data/Real/slam_offline/2021-11-16_19-42-45/map0_traj_2021-11-16_19-42-45.pkl"

    with open(poses_path, 'rb') as file:
        map_H = pickle.load(file)
    
    map_H = map_H[min_f_i:max_f_i:f_i_stride]

    ################
    # LOAD ALL TIMES
    ################

    original_path = '/home/hth/Deep-Collison-Checker/Data/Real/runs/2021-11-16_19-42-45/velodyne_frames'
    frame_times = np.sort(np.array([float(f[:-4]) for f in listdir(original_path) if f.endswith('.ply')], dtype=np.float64))
    frame_times = frame_times[min_f_i:max_f_i:f_i_stride]

    print(len(map_H), len(all_points), len(frame_times))

    ####################################################################################################################################################
    # Try to get better ground

    # Only get interesting frames

    voxel_size = 0.1
    d_coarse = voxel_size * 15
    d_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(all_pcds,
                                       loop_edges,
                                       d_coarse,
                                       d_fine)





    ####################################################################################################################################################


    #######################
    # Simple loop detection
    #######################

    closure_d = 1.0
    closure_d2 = closure_d * closure_d
    save_d = closure_d / 2
    save_d2 = save_d * save_d
    closure_t = 20.0
    closed_loops = 0

    sparse_positions = []
    sparse_f_inds = []
    for f_i, f_H in enumerate(map_H):
        current_position = f_H[:3, 3]
        if (len(sparse_positions) < 1):
            sparse_positions.append(current_position)
            sparse_f_inds.append(f_i)
        else:
            diff = current_position - sparse_positions[-1]
            if diff.dot(diff) > save_d2:
                sparse_positions.append(current_position)
                sparse_f_inds.append(f_i)

        closure_ind = -1
        if (closed_loops < 1):
            for i, p in enumerate(sparse_positions):

                if (frame_times[f_i] - frame_times[sparse_f_inds[i]] > closure_t):

                    diff = p - current_position
                    diff[2] = 0
                    d2 = diff.dot(diff)

                    if (d2 < closure_d2):
                        closure_ind = sparse_f_inds[i]
                        closed_loops += 1
                        print(closure_ind)
                        break

        # Here close
        if closure_ind > 0:
            print('Close loop here', f_i, frame_times[i])

            # print("Full registration ...")
            # voxel_size = 0.1
            # d_coarse = voxel_size * 15
            # d_fine = voxel_size * 1.5
            # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            #     pose_graph = full_registration(all_pcds,
            #                                    d_coarse,
            #                                    d_fine)


    # Or here close
    print('Close after??')
    print("Full registration ...")
    voxel_size = 0.1
    d_coarse = voxel_size * 15
    d_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(all_pcds,
                                       loop_edges,
                                       d_coarse,
                                       d_fine)
                                        
    print('Optimizing PoseGraph ...')
    option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=d_fine,
                                                                 edge_prune_threshold=0.25,
                                                                 reference_node=0)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(pose_graph,
                                                       o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                       o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                       option)

    print('Transform points and display')
    new_map_H = []
    for point_id in range(len(all_pcds)):
        all_pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        new_map_H.append(np.matmul(pose_graph.nodes[point_id].pose, map_H[point_id]))
    # o3d.visualization.draw_geometries(all_pcds)

    a = 1/0

    # Save new trajectory
    map_folder = '/home/hth/Deep-Collison-Checker/Data/Real/slam_offline/2021-11-16_19-42-45/'
    map_day = '2021-11-16_19-42-45'
    save_trajectory(join(map_folder, 'loopclosed_traj_{:s}.ply'.format(map_day)), new_map_H)
    new_traj_file = join(map_folder, 'loopclosed_traj_{:s}.pkl'.format(map_day))
    with open(new_traj_file, 'wb') as file:
        pickle.dump(new_map_H, file)

    ###########################
    # Try multiway registration
    ###########################



    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main call
#       \***************/
#


if __name__ == '__main__':

    plot_rms()

    # test_loop_closure()

    a = 1/0
