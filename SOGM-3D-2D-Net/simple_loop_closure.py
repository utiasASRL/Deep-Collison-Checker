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

    ############
    # Parameters
    ############

    # Data path
    root_path = "/home/hth/Deep-Collison-Checker/"
    frame_path = join(root_path, "Data/Real/icp_frames/tmp/")

    map_day = "2021-11-30_12-05-32"
    map_folder = join(root_path, "Data/Real/slam_offline", map_day)
    poses_path = join(map_folder, "map0_traj_{:s}.pkl".format(map_day))

    original_path = join(root_path, 'Data/Real/runs/{:s}/velodyne_frames'.format(map_day))


    # Reducing the number of frame optimized
    min_f_i = 0
    max_f_i = 10000000
    f_i_stride = 1
    # min_f_i = 50
    # max_f_i = 6501
    # f_i_stride = 50

    #################
    # LOAD LOOP EDGES
    #################

    # Number od points per poses in the saved ply
    n_per_poses = 15

    # Read them from a point picking list in cloud compare
    picking_path = join(map_folder, "picking_list.txt")
    picking_data = np.loadtxt(picking_path, delimiter=',')
    if (picking_data.shape[0] % 2 == 1):
        raise ValueError('Error: Odd number od points picked')
    picking_list = np.reshape(picking_data[:, 0].astype(np.int64), (-1, 2))

    # Rescale indices due to saving poses as multiple points
    picking_list = picking_list // n_per_poses

    # Addapt if we reduced the poses
    loop_edges = (picking_list - min_f_i) // f_i_stride
    
    #################
    # LOAD ALL FRAMES
    #################

    # Load all frame corrected and aligned
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

    # print(np.array(all_lens, dtype=np.int32))

    ################
    # LOAD ALL POSES
    ################

    with open(poses_path, 'rb') as file:
        map_H = pickle.load(file)
    
    map_H = map_H[min_f_i:max_f_i:f_i_stride]

    ################
    # LOAD ALL TIMES
    ################

    frame_times = np.sort(np.array([float(f[:-4]) for f in listdir(original_path) if f.endswith('.ply')], dtype=np.float64))
    frame_times = frame_times[min_f_i:max_f_i:f_i_stride]

    print(len(map_H), len(all_points), len(frame_times))

    # ####################################################################################################################################################
    # # Try to get better ground
    # # Only get interesting frames
    # voxel_size = 0.1
    # d_coarse = voxel_size * 15
    # d_fine = voxel_size * 1.5
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     pose_graph = full_registration(all_pcds,
    #                                    loop_edges,
    #                                    d_coarse,
    #                                    d_fine)
    # ####################################################################################################################################################


    # #######################
    # # Simple loop detection
    # #######################

    # closure_d = 1.0
    # closure_d2 = closure_d * closure_d
    # save_d = closure_d / 2
    # save_d2 = save_d * save_d
    # closure_t = 20.0
    # closed_loops = 0

    # sparse_positions = []
    # sparse_f_inds = []
    # for f_i, f_H in enumerate(map_H):
    #     current_position = f_H[:3, 3]
    #     if (len(sparse_positions) < 1):
    #         sparse_positions.append(current_position)
    #         sparse_f_inds.append(f_i)
    #     else:
    #         diff = current_position - sparse_positions[-1]
    #         if diff.dot(diff) > save_d2:
    #             sparse_positions.append(current_position)
    #             sparse_f_inds.append(f_i)

    #     closure_ind = -1
    #     if (closed_loops < 1):
    #         for i, p in enumerate(sparse_positions):

    #             if (frame_times[f_i] - frame_times[sparse_f_inds[i]] > closure_t):

    #                 diff = p - current_position
    #                 diff[2] = 0
    #                 d2 = diff.dot(diff)

    #                 if (d2 < closure_d2):
    #                     closure_ind = sparse_f_inds[i]
    #                     closed_loops += 1
    #                     print(closure_ind)
    #                     break

    #     # Here close
    #     if closure_ind > 0:
    #         print('Close loop here', f_i, frame_times[i])

    #         # print("Full registration ...")
    #         # voxel_size = 0.1
    #         # d_coarse = voxel_size * 15
    #         # d_fine = voxel_size * 1.5
    #         # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #         #     pose_graph = full_registration(all_pcds,
    #         #                                    d_coarse,
    #         #                                    d_fine)


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

    # Save new trajectory
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

    test_loop_closure()

    a = 1/0
