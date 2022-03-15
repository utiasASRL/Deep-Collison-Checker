#
#
#      0==============================0
#      |    Deep Collision Checker    |
#      0==============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling SemanticKitti dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
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
import sys
import time
import os
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np
import pickle
import yaml
import torch
from multiprocessing import Lock, Value
from datasets.common import PointCloudDataset, batch_neighbors
from torch.utils.data import Sampler
from utils.config import bcolors
from datasets.common import grid_subsampling

from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation as scipyR
from sklearn.neighbors import KDTree
from slam.PointMapSLAM import PointMap, extract_map_ground, extract_ground, motion_rectified
from slam.cpp_slam import update_pointmap, polar_normals, point_to_map_icp, slam_on_sim_sequence, ray_casting_annot, get_lidar_visibility, slam_on_real_sequence, merge_pointmaps
from slam.dev_slam import frame_H_to_points, interp_pose, rot_trans_diffs, normals_orientation, save_trajectory, RANSAC, filter_frame_timestamps, extract_flat_ground, in_plane
from utils.ply import read_ply, write_ply
from utils.mayavi_visu import save_future_anim, fast_save_future_anim, Box
from utils.sparse_morpho import sparse_point_opening, pepper_noise_removal
from simple_loop_closure import loop_closure

# OS functions
from os import listdir, makedirs
from os.path import exists, join, isdir, getsize


# ----------------------------------------------------------------------------------------------------------------------
#
#           Special dataset class for SLAM
#       \************************************/


class MyhalCollisionSlam:

    def __init__(self,
                 only_day_1=False,
                 first_day='',
                 last_day='',
                 day_list=None,
                 map_day=None,
                 dataset_path='../Data/Real',
                 verbose=1):

        # Name of the dataset
        self.name = 'MyhalCollisionSlam'

        # Data path
        self.original_path = dataset_path
        self.data_path = self.original_path
        self.days_folder = join(self.original_path, 'runs')
        self.frame_folder_name = 'velodyne_frames'
        self.map_day = map_day

        # List of days
        if day_list is not None:
            # Use explicit day list
            self.days = np.array(day_list)

        else:
            # Use first and last days
            self.days = np.sort([d for d in listdir(self.days_folder)])
            if len(first_day) == 0:
                first_day = self.days[0]
            if len(last_day) == 0:
                last_day = self.days[-1]
            self.days = np.sort(
                [d for d in self.days if first_day <= d <= last_day])

        # Parameters
        self.only_day_1 = only_day_1
        self.motion_distortion = False
        self.day_f_times = []
        self.day_f_names = []
        self.map_f_times = None
        self.map_f_names = None
        self.get_frames_names()

        ##################
        # Load calibration
        ##################

        # Calibration file from simulation
        calib_csv = join(self.original_path, 'calibration', 'jackal_calib.csv')
        calib = np.loadtxt(calib_csv, delimiter=',', dtype=str)
        T_base_A = [float(t) for t in calib[1, 2:5]]
        T_A_B = [float(t) for t in calib[4, 2:5]]
        T_B_velo = [float(t) for t in calib[7, 2:5]]
        Q_base_A = np.array([float(t) for t in calib[1, 5:9]])
        Q_A_B = np.array([float(t) for t in calib[4, 5:9]])
        Q_B_velo = np.array([float(t) for t in calib[7, 5:9]])

        # Transorm quaternions and translation into homogenus 4x4 matrices
        H_base_A = np.eye(4, dtype=np.float64)
        H_base_A[:3, :3] = scipyR.from_quat(Q_base_A).as_matrix()
        H_base_A[:3, 3] = T_base_A
        H_A_B = np.eye(4, dtype=np.float64)
        H_A_B[:3, :3] = scipyR.from_quat(Q_A_B).as_matrix()
        H_A_B[:3, 3] = T_A_B
        H_B_velo = np.eye(4, dtype=np.float64)
        H_B_velo[:3, :3] = scipyR.from_quat(Q_B_velo).as_matrix()
        H_B_velo[:3, 3] = T_B_velo
        self.H_velo_base = np.matmul(H_B_velo, np.matmul(H_A_B, H_base_A))
        self.H_base_velo = np.linalg.inv(self.H_velo_base)

        ###############
        # Load GT poses
        ###############

        # if verbose:
        #     print('\nLoading days groundtruth poses...')
        # t0 = time.time()
        # self.gt_t, self.gt_H = self.load_gt_poses()
        # t2 = time.time()
        # if verbose:
        #     print('Done in {:.1f}s\n'.format(t2 - t0))

        ################
        # Load loc poses
        ################

        if verbose:
            print('\nLoading days localization poses...')
        t0 = time.time()
        self.loc_t, self.loc_H = self.load_loc_poses()
        t2 = time.time()
        if verbose:
            print('Done in {:.1f}s\n'.format(t2 - t0))

        return

    def get_frames_names(self, verbose=1):

        # Loop on days
        self.day_f_times = []
        self.day_f_names = []
        for d, day in enumerate(self.days):

            # Get frame timestamps
            frames_folder = join(self.days_folder, day, self.frame_folder_name)
            if not exists(frames_folder):
                frames_folder = join(self.days_folder, day,
                                     'classified_frames')
            f_names = [f for f in listdir(frames_folder) if f[-4:] == '.ply']
            f_times = np.array([float(f[:-4]) for f in f_names],
                               dtype=np.float64)
            f_names = np.array([join(frames_folder, f) for f in f_names])
            ordering = np.argsort(f_times)
            f_names = f_names[ordering]
            f_times = f_times[ordering]
            self.day_f_times.append(f_times)
            self.day_f_names.append(f_names)

            if self.only_day_1 and d > -1:
                break

        # Handle map day
        frames_folder = join(self.days_folder, self.map_day,
                             self.frame_folder_name)
        f_names = [f for f in listdir(frames_folder) if f[-4:] == '.ply']
        f_times = np.array([float(f[:-4]) for f in f_names], dtype=np.float64)
        f_names = np.array([join(frames_folder, f) for f in f_names])
        ordering = np.argsort(f_times)
        self.map_f_names = f_names[ordering]
        self.map_f_times = f_times[ordering]

        return

    def load_gt_poses_old(self):

        #gt_files = np.sort([gt_f for gt_f in listdir(self.days) if gt_f[-4:] == '.csv'])
        gt_H = []
        gt_t = []
        for d, day in enumerate(self.days):

            # Out files
            gt_folder = join(self.data_path, 'slam_gt', day)
            if not exists(gt_folder):
                makedirs(gt_folder)

            t1 = time.time()

            gt_pkl_file = join(gt_folder, 'gt_poses.pkl')
            if exists(gt_pkl_file):
                # Read pkl
                with open(gt_pkl_file, 'rb') as f:
                    day_gt_t, day_gt_H = pickle.load(f)

            else:
                # File paths
                gt_csv = join(self.days_folder, day, 'gt_poses.csv')

                # Load gt
                day_gt_t = np.loadtxt(gt_csv,
                                      delimiter=',',
                                      usecols=0,
                                      skiprows=1,
                                      dtype=np.uint64)
                gt_T = np.loadtxt(gt_csv,
                                  delimiter=',',
                                  usecols=(5, 6, 7),
                                  skiprows=1,
                                  dtype=np.float32)
                gt_Q = np.loadtxt(gt_csv,
                                  delimiter=',',
                                  usecols=(8, 9, 10, 11),
                                  skiprows=1,
                                  dtype=np.float32)

                # Convert gt to homogenous rotation/translation matrix
                gt_R = scipyR.from_quat(gt_Q)
                gt_R = gt_R.as_matrix()
                day_gt_H = np.zeros((len(day_gt_t), 4, 4))
                day_gt_H[:, :3, :3] = gt_R
                day_gt_H[:, :3, 3] = gt_T
                day_gt_H[:, 3, 3] = 1

                # Save pickle
                with open(gt_pkl_file, 'wb') as f:
                    pickle.dump([day_gt_t, day_gt_H], f)

                t2 = time.time()
                print('{:s} {:d}/{:d} Done in {:.1f}s'.format(
                    day, d, len(self.days), t2 - t1))

            gt_t += [day_gt_t]
            gt_H += [day_gt_H]

            if self.only_day_1 and d > -1:
                break

        return gt_t, gt_H

    def load_gt_poses(self):

        gt_H = []
        gt_t = []
        for d, day in enumerate(self.days):

            # Out files
            gt_folder = join(self.data_path, 'slam_gt', day)
            if not exists(gt_folder):
                makedirs(gt_folder)

            t1 = time.time()

            # Load gt from ply files
            gt_ply_file = join(self.days_folder, day, 'gt_pose.ply')

            if not exists(gt_ply_file):
                print('No groundtruth poses found at ' + gt_ply_file)
                print('Using localization poses instead')
                gt_ply_file = join(self.days_folder, day, 'loc_pose.ply')

            if not exists(gt_ply_file):
                raise ValueError('No localization poses found at ' + gt_ply_file)

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

            t2 = time.time()
            print('{:s} {:d}/{:d} Done in {:.1f}s'.format(
                day, d, len(self.days), t2 - t1))

            gt_t += [day_gt_t]
            gt_H += [day_gt_H]

            # # Remove frames that are not inside gt timings
            # mask = np.logical_and(self.day_f_times[d] > day_gt_t[0], self.day_f_times[d] < day_gt_t[-1])
            # self.day_f_names[d] = self.day_f_names[d][mask]
            # self.day_f_times[d] = self.day_f_times[d][mask]

            if self.only_day_1 and d > -1:
                break

        return gt_t, gt_H

    def load_loc_poses(self):

        loc_H = []
        loc_t = []
        for d, day in enumerate(self.days):

            t1 = time.time()

            # Load loc from ply files
            loc_ply_file = join(self.days_folder, day, 'loc_pose.ply')

            if not exists(loc_ply_file):
                raise ValueError('No localization poses found at ' + loc_ply_file)

            data = read_ply(loc_ply_file)
            loc_T = np.vstack([data['pos_x'], data['pos_y'], data['pos_z']]).T
            loc_Q = np.vstack([data['rot_x'], data['rot_y'], data['rot_z'], data['rot_w']]).T

            # Times
            day_loc_t = data['time']

            # print(day_loc_t)
            # print(self.day_f_times[d])
            # plt.plot(day_loc_t, day_loc_t*0, '.')
            # plt.plot(self.day_f_times[d], self.day_f_times[d]*0, '.')
            # plt.show()
            # a = 1/0

            # Convert loc to homogenous rotation/translation matrix
            loc_R = scipyR.from_quat(loc_Q)
            loc_R = loc_R.as_matrix()
            day_loc_H = np.zeros((len(day_loc_t), 4, 4))
            day_loc_H[:, :3, :3] = loc_R
            day_loc_H[:, :3, 3] = loc_T
            day_loc_H[:, 3, 3] = 1

            t2 = time.time()
            print('{:s} {:d}/{:d} Done in {:.1f}s'.format(
                day, d, len(self.days), t2 - t1))

            loc_t += [day_loc_t]
            loc_H += [day_loc_H]

            if self.only_day_1 and d > -1:
                break

        return loc_t, loc_H

    def load_map_poses(self, day):

        # Load map from ply files
        map_ply_file = join(self.days_folder, day, 'logs-' + day,
                            'map_traj.ply')
        data = read_ply(map_ply_file)
        map_T = np.vstack([data['pos_x'], data['pos_y'], data['pos_z']]).T
        map_Q = np.vstack(
            [data['rot_x'], data['rot_y'], data['rot_z'], data['rot_w']]).T

        # Times
        day_map_t = data['time']

        # Convert map to homogenous rotation/translation matrix
        map_R = scipyR.from_quat(map_Q)
        map_R = map_R.as_matrix()
        day_map_H = np.zeros((len(day_map_t), 4, 4))
        day_map_H[:, :3, :3] = map_R
        day_map_H[:, :3, 3] = map_T
        day_map_H[:, 3, 3] = 1

        return day_map_t, day_map_H

    def load_map_gt_poses(self):

        ####################################
        # Load gt poses from mapping session
        ####################################

        # Load gt from ply files
        gt_ply_file = join(self.days_folder, self.map_day, 'gt_pose.ply')
        data = read_ply(gt_ply_file)
        gt_T = np.vstack([data['pos_x'], data['pos_y'], data['pos_z']]).T
        gt_Q = np.vstack(
            [data['rot_x'], data['rot_y'], data['rot_z'], data['rot_w']]).T

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

    def load_frame_points(self, frame_path):

        data = read_ply(frame_path)
        points = np.vstack((data['x'], data['y'], data['z'])).T

        # Safe check for points equal to zero
        hr = np.sqrt(np.sum(points[:, :2]**2, axis=1))
        if np.sum((hr < 1e-6).astype(np.int32)) > 0:
            points = points[hr > 1e-6]
            print('Warning: lidar frame with invalid points: frame_names[i].')
            a = 1 / 0
        return points

    def load_frame_points_labels(self, frame_path, label_path=None):

        data_p = read_ply(frame_path)
        points = np.vstack((data_p['x'], data_p['y'], data_p['z'])).T

        if label_path is None:
            data = data_p
        else:
            data = read_ply(label_path)

        if 'category' in data.dtype.names:
            labels = data['category']
        elif 'cat' in data.dtype.names:
            labels = data['cat']
        else:
            labels = -np.ones(points[:, 0].shape, dtype=np.int32)

        # Safe check for points equal to zero
        hr = np.sqrt(np.sum(points[:, :2]**2, axis=1))
        if np.sum((hr < 1e-6).astype(np.int32)) > 0:
            labels = labels[hr > 1e-6]
            points = points[hr > 1e-6]
            print('Warning: lidar frame with invalid points: frame_names[i].')
            a = 1 / 0

        return points, labels

    def debug_angular_velocity(self):

        for d, day in enumerate(self.days):

            # List of frames for this day
            frame_times = self.day_f_times[d]

            # List of groundtruth timestamps and poses
            gt_t = self.gt_t[d]
            gt_H = self.gt_H[d]

            all_frame_H = []
            for i, f_t in enumerate(frame_times):
                # Find closest gt poses
                gt_i1 = np.argmin(np.abs(gt_t - f_t))
                if f_t < gt_t[gt_i1]:
                    gt_i0 = gt_i1 - 1
                else:
                    gt_i0 = gt_i1
                    gt_i1 = gt_i0 + 1

                # Interpolate the ground truth pose at current time
                interp_t = (f_t - gt_t[gt_i0]) / (gt_t[gt_i1] - gt_t[gt_i0])
                frame_H = interp_pose(interp_t, gt_H[gt_i0], gt_H[gt_i1])

                # Transformation of the lidar (gt is in body frame)
                all_frame_H.append(np.matmul(frame_H, self.H_velo_base))

            all_frame_H = np.stack(all_frame_H, axis=0)
            dT, dR = rot_trans_diffs(all_frame_H)
            plt.figure('dT')
            plt.plot(dT)
            plt.figure('dR')
            plt.plot(dR * 180 / np.pi)
            plt.show()

        return

    def gt_mapping(self, map_voxel_size=0.08, save_group=50, verbose=1):

        #################
        # Init parameters
        #################

        # Out files
        out_folder = join(self.data_path, 'slam_gt')
        if not exists(out_folder):
            makedirs(out_folder)

        ##########################
        # Start first pass of SLAM
        ##########################

        for d, day in enumerate(self.days):

            out_day_folder = join(out_folder, day)
            if not exists(out_day_folder):
                makedirs(out_day_folder)

            for folder_name in ['trajectory', 'map', 'frames']:
                if not exists(join(out_day_folder, folder_name)):
                    makedirs(join(out_day_folder, folder_name))

            gt_slam_file = join(out_day_folder, 'gt_map_{:s}.pkl'.format(day))
            if exists(gt_slam_file):
                continue

            # List of frames for this day
            frame_names = self.day_f_names[d]
            frame_times = self.day_f_times[d]

            # List of groundtruth timestamps and poses
            gt_t = self.gt_t[d]
            gt_H = self.gt_H[d]

            # Initiate map
            transform_list = []
            pointmap = PointMap(dl=map_voxel_size)
            last_saved_frames = 0
            FPS = 0
            N = len(frame_names)

            # Test mapping
            for i, f_t in enumerate(frame_times):

                t = [time.time()]

                # Load ply format points
                points = self.load_frame_points(frame_names[i])

                t += [time.time()]

                # Get normals (dummy r_scale to avoid removing points as simulation scans are perfect)
                normals, planarity, linearity = polar_normals(points,
                                                              radius=1.5,
                                                              lidar_n_lines=31,
                                                              h_scale=0.5,
                                                              r_scale=1000.0)
                norm_scores = planarity + linearity

                # Remove outliers
                points = points[norm_scores > 0.1]
                normals = normals[norm_scores > 0.1]
                norm_scores = norm_scores[norm_scores > 0.1]

                # Filter out points according to main normal directions (Not necessary if normals are better computed)
                # norm_scores *= normal_filtering(normals)

                # Find closest gt poses
                gt_i1 = np.argmin(np.abs(gt_t - f_t))
                if f_t < gt_t[gt_i1]:
                    gt_i0 = gt_i1 - 1
                else:
                    gt_i0 = gt_i1
                    gt_i1 = gt_i0 + 1

                if gt_i1 >= len(gt_t):
                    break

                # Interpolate the ground truth pose at current time
                interp_t = (f_t - gt_t[gt_i0]) / (gt_t[gt_i1] - gt_t[gt_i0])
                frame_H = interp_pose(interp_t, gt_H[gt_i0], gt_H[gt_i1])

                # Transformation of the lidar (gt is in body frame)
                H_velo_world = np.matmul(frame_H, self.H_velo_base)
                transform_list.append(H_velo_world)

                # Apply transf
                world_points = np.hstack((points, np.ones_like(points[:, :1])))
                world_points = np.matmul(world_points, H_velo_world.T).astype(
                    np.float32)[:, :3]
                world_normals = np.matmul(
                    normals, H_velo_world[:3, :3].T).astype(np.float32)

                t += [time.time()]

                # Update map
                pointmap.update(world_points, world_normals, norm_scores)

                if i % save_group == 0:
                    filename = join(out_day_folder, 'map',
                                    'gt_map_{:03d}.ply'.format(i))
                    write_ply(filename, [
                        pointmap.points, pointmap.normals, pointmap.scores,
                        pointmap.counts
                    ], ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

                t += [time.time()]

                if verbose == 2:
                    ti = 0
                    print('Load ............ {:7.1f}ms'.format(
                        1000 * (t[ti + 1] - t[ti])))
                    ti += 1
                    print('Preprocessing ... {:7.1f}ms'.format(
                        1000 * (t[ti + 1] - t[ti])))
                    ti += 1
                    print('Align ........... {:7.1f}ms'.format(
                        1000 * (t[ti + 1] - t[ti])))
                    ti += 1
                    print('Mapping ......... {:7.1f}ms'.format(
                        1000 * (t[ti + 1] - t[ti])))

                if verbose > 0:
                    fmt_str = 'GT Mapping {:3d}  --- {:5.1f}% or {:02d}:{:02d}:{:02d} remaining at {:.1f}fps'
                    if i == 0:
                        FPS = 1 / (t[-1] - t[0])
                    else:
                        FPS += (1 / (t[-1] - t[0]) - FPS) / 10
                    remaining = int((N - (i + 1)) / FPS)
                    hours = remaining // 3600
                    remaining = remaining - 3600 * hours
                    minutes = remaining // 60
                    seconds = remaining - 60 * minutes
                    print(
                        fmt_str.format(i, 100 * (i + 1) / N, hours, minutes,
                                       seconds, FPS))

                # Save groups of 100 frames together
                if (i > last_saved_frames + save_group + 1):
                    all_points = []
                    all_labels = []
                    all_traj_pts = []
                    all_traj_clrs = []
                    all_traj_pts2 = []
                    all_traj_clrs2 = []
                    i0 = last_saved_frames
                    i1 = i0 + save_group
                    for save_i, save_f_t in enumerate(frame_times[i0:i1]):

                        # Load points
                        points, labels = self.load_frame_points_labels(
                            frame_names[i0 + save_i])

                        # Find closest gt poses
                        gt_i1 = np.argmin(np.abs(gt_t - save_f_t))
                        if save_f_t < gt_t[gt_i1]:
                            gt_i0 = gt_i1 - 1
                        else:
                            gt_i0 = gt_i1
                            gt_i1 = gt_i0 + 1

                        # Interpolate the ground truth pose at current time
                        interp_t = (save_f_t - gt_t[gt_i0]) / (gt_t[gt_i1] - gt_t[gt_i0])
                        world_H = interp_pose(interp_t, gt_H[gt_i0],
                                              gt_H[gt_i1])

                        # Transformation of the lidar (gt is in body frame)
                        H_velo_world = np.matmul(world_H, self.H_velo_base)

                        # Apply transf
                        world_pts = np.hstack(
                            (points, np.ones_like(points[:, :1])))
                        world_pts = np.matmul(
                            world_pts, H_velo_world.T).astype(np.float32)

                        # Save frame
                        world_pts[:, 3] = i0 + save_i
                        all_points.append(world_pts)
                        all_labels.append(labels)

                        # also save trajectory
                        traj_pts, traj_clrs = frame_H_to_points(H_velo_world,
                                                                size=0.1)
                        traj_pts = np.hstack(
                            (traj_pts,
                             np.ones_like(traj_pts[:, :1]) * (i0 + save_i)))
                        all_traj_pts.append(traj_pts.astype(np.float32))
                        all_traj_clrs.append(traj_clrs)

                        traj_pts, traj_clrs = frame_H_to_points(world_H,
                                                                size=1.1)
                        traj_pts = np.hstack(
                            (traj_pts,
                             np.ones_like(traj_pts[:, :1]) * (i0 + save_i)))
                        all_traj_pts2.append(traj_pts.astype(np.float32))
                        all_traj_clrs2.append(traj_clrs)

                    last_saved_frames += save_group
                    filename = join(out_day_folder, 'frames',
                                    'gt_aligned_{:05d}.ply'.format(i0))
                    write_ply(filename,
                              [np.vstack(all_points),
                               np.hstack(all_labels)],
                              ['x', 'y', 'z', 't', 'category'])
                    filename = join(out_day_folder, 'trajectory',
                                    'gt_traj_{:05d}.ply'.format(i0))
                    write_ply(
                        filename,
                        [np.vstack(all_traj_pts),
                         np.vstack(all_traj_clrs)],
                        ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

                    filename = join(out_day_folder, 'trajectory',
                                    'gt_traj_body_{:05d}.ply'.format(i0))
                    write_ply(
                        filename,
                        [np.vstack(all_traj_pts2),
                         np.vstack(all_traj_clrs2)],
                        ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

            #################
            # Post processing
            #################

            filename = join(out_day_folder, 'gt_map_{:s}.ply'.format(day))
            write_ply(filename, [
                pointmap.points, pointmap.normals, pointmap.scores,
                pointmap.counts
            ], ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

            all_points = []
            all_labels = []
            all_traj_pts = []
            all_traj_clrs = []
            i0 = last_saved_frames
            for save_i, save_f_t in enumerate(frame_times[i0:]):

                # Load points
                points, labels = self.load_frame_points_labels(
                    frame_names[i0 + save_i])

                # Find closest gt poses
                gt_i1 = np.argmin(np.abs(gt_t - save_f_t))
                if save_f_t < gt_t[gt_i1]:
                    gt_i0 = gt_i1 - 1
                else:
                    gt_i0 = gt_i1
                    gt_i1 = gt_i0 + 1

                if gt_i1 >= len(gt_t):
                    break

                # Interpolate the ground truth pose at current time
                interp_t = (save_f_t - gt_t[gt_i0]) / \
                    (gt_t[gt_i1] - gt_t[gt_i0])
                world_H = interp_pose(interp_t, gt_H[gt_i0], gt_H[gt_i1])

                # Transformation of the lidar (gt is in body frame)
                H_velo_world = np.matmul(world_H, self.H_velo_base)

                # Apply transf
                world_pts = np.hstack((points, np.ones_like(points[:, :1])))
                world_pts = np.matmul(world_pts,
                                      H_velo_world.T).astype(np.float32)

                # Save frame
                world_pts[:, 3] = i0 + save_i
                all_points.append(world_pts)
                all_labels.append(labels)

                # also save trajectory
                traj_pts, traj_clrs = frame_H_to_points(H_velo_world, size=0.1)
                traj_pts = np.hstack(
                    (traj_pts, np.ones_like(traj_pts[:, :1]) * (i0 + save_i)))
                all_traj_pts.append(traj_pts.astype(np.float32))
                all_traj_clrs.append(traj_clrs)

            last_saved_frames += save_group
            filename = join(out_day_folder, 'frames',
                            'gt_aligned_{:05d}.ply'.format(i0))
            write_ply(filename, [np.vstack(all_points),
                                 np.hstack(all_labels)],
                      ['x', 'y', 'z', 't', 'category'])
            filename = join(out_day_folder, 'trajectory',
                            'gt_traj_{:05d}.ply'.format(i0))
            write_ply(filename,
                      [np.vstack(all_traj_pts),
                       np.vstack(all_traj_clrs)],
                      ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

            # Save full trajectory
            all_traj_pts = []
            all_traj_clrs = []
            for save_i, save_H in enumerate(transform_list):

                # Save trajectory
                traj_pts, traj_clrs = frame_H_to_points(save_H, size=0.1)
                traj_pts = np.hstack(
                    (traj_pts, np.ones_like(traj_pts[:, :1]) * save_i))
                all_traj_pts.append(traj_pts.astype(np.float32))
                all_traj_clrs.append(traj_clrs)

            filename = join(out_day_folder, 'gt_traj_{:s}.ply'.format(day))
            write_ply(filename,
                      [np.vstack(all_traj_pts),
                       np.vstack(all_traj_clrs)],
                      ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

            # Save alignments
            with open(gt_slam_file, 'wb') as file:
                pickle.dump((frame_names[:len(transform_list)], transform_list,
                             1, pointmap), file)

    def loc_mapping(self, map_voxel_size=0.03, save_group=50, verbose=1):

        #################
        # Init parameters
        #################

        # Out files
        out_folder = join(self.data_path, 'slam_online')
        if not exists(out_folder):
            makedirs(out_folder)

        ##########################
        # Start first pass of SLAM
        ##########################

        for d, day in enumerate(self.days):

            out_day_folder = join(out_folder, day)
            if not exists(out_day_folder):
                makedirs(out_day_folder)

            for folder_name in ['trajectory', 'map', 'frames']:
                if not exists(join(out_day_folder, folder_name)):
                    makedirs(join(out_day_folder, folder_name))

            online_slam_file = join(out_day_folder,
                                    'loc_map_{:s}.pkl'.format(day))
            if exists(online_slam_file):
                continue

            # List of frames for this day
            frame_names = self.day_f_names[d]
            frame_times = self.day_f_times[d]

            # List of groundtruth timestamps and poses
            loc_t = self.loc_t[d]
            loc_H = self.loc_H[d]

            # Initiate map
            transform_list = []
            pointmap = PointMap(dl=map_voxel_size)
            last_saved_frames = 0
            FPS = 0
            N = len(frame_names)

            # Test mapping
            for i, f_t in enumerate(frame_times):

                t = [time.time()]

                # Load ply format points
                points = self.load_frame_points(frame_names[i])

                t += [time.time()]

                # Get normals (dummy r_scale to avoid removing points as simulation scans are perfect)
                normals, planarity, linearity = polar_normals(points,
                                                              radius=1.5,
                                                              lidar_n_lines=31,
                                                              h_scale=0.5,
                                                              r_scale=1000.0)
                norm_scores = planarity + linearity

                # Remove outliers
                points = points[norm_scores > 0.1]
                normals = normals[norm_scores > 0.1]
                norm_scores = norm_scores[norm_scores > 0.1]

                # Filter out points according to main normal directions (Not necessary if normals are better computed)
                # norm_scores *= normal_filtering(normals)

                # Find closest loc poses
                loc_i1 = np.argmin(np.abs(loc_t - f_t))
                if f_t < loc_t[loc_i1]:
                    loc_i0 = loc_i1 - 1
                else:
                    loc_i0 = loc_i1
                    loc_i1 = loc_i0 + 1

                if loc_i1 >= len(loc_t):
                    break

                # Interpolate the ground truth pose at current time
                interp_t = (f_t - loc_t[loc_i0]) / \
                    (loc_t[loc_i1] - loc_t[loc_i0])
                frame_H = interp_pose(interp_t, loc_H[loc_i0], loc_H[loc_i1])

                # Transformation of the lidar (loc is in body frame)
                H_velo_world = np.matmul(frame_H, self.H_velo_base)
                transform_list.append(H_velo_world)

                # Apply transf
                world_points = np.hstack((points, np.ones_like(points[:, :1])))
                world_points = np.matmul(world_points, H_velo_world.T).astype(
                    np.float32)[:, :3]
                world_normals = np.matmul(
                    normals, H_velo_world[:3, :3].T).astype(np.float32)

                t += [time.time()]

                # Update map
                pointmap.update(world_points, world_normals, norm_scores)

                if i % save_group == 0:
                    filename = join(out_day_folder, 'map',
                                    'loc_map_{:03d}.ply'.format(i))
                    write_ply(filename, [
                        pointmap.points, pointmap.normals, pointmap.scores,
                        pointmap.counts
                    ], ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

                t += [time.time()]

                if verbose == 2:
                    ti = 0
                    print('Load ............ {:7.1f}ms'.format(
                        1000 * (t[ti + 1] - t[ti])))
                    ti += 1
                    print('Preprocessing ... {:7.1f}ms'.format(
                        1000 * (t[ti + 1] - t[ti])))
                    ti += 1
                    print('Align ........... {:7.1f}ms'.format(
                        1000 * (t[ti + 1] - t[ti])))
                    ti += 1
                    print('Mapping ......... {:7.1f}ms'.format(
                        1000 * (t[ti + 1] - t[ti])))

                if verbose > 0:
                    fmt_str = 'GT Mapping {:3d}  --- {:5.1f}% or {:02d}:{:02d}:{:02d} remaining at {:.1f}fps'
                    if i == 0:
                        FPS = 1 / (t[-1] - t[0])
                    else:
                        FPS += (1 / (t[-1] - t[0]) - FPS) / 10
                    remaining = int((N - (i + 1)) / FPS)
                    hours = remaining // 3600
                    remaining = remaining - 3600 * hours
                    minutes = remaining // 60
                    seconds = remaining - 60 * minutes
                    print(
                        fmt_str.format(i, 100 * (i + 1) / N, hours, minutes,
                                       seconds, FPS))

                # Save groups of 100 frames together
                if (i > last_saved_frames + save_group + 1):
                    all_points = []
                    all_labels = []
                    all_traj_pts = []
                    all_traj_clrs = []
                    all_traj_pts2 = []
                    all_traj_clrs2 = []
                    i0 = last_saved_frames
                    i1 = i0 + save_group
                    for save_i, save_f_t in enumerate(frame_times[i0:i1]):

                        # Load points
                        points, labels = self.load_frame_points_labels(
                            frame_names[i0 + save_i])

                        # Find closest loc poses
                        loc_i1 = np.argmin(np.abs(loc_t - save_f_t))
                        if save_f_t < loc_t[loc_i1]:
                            loc_i0 = loc_i1 - 1
                        else:
                            loc_i0 = loc_i1
                            loc_i1 = loc_i0 + 1

                        # Interpolate the ground truth pose at current time
                        interp_t = (save_f_t - loc_t[loc_i0]) / (
                            loc_t[loc_i1] - loc_t[loc_i0])
                        world_H = interp_pose(interp_t, loc_H[loc_i0],
                                              loc_H[loc_i1])

                        # Transformation of the lidar (loc is in body frame)
                        H_velo_world = np.matmul(world_H, self.H_velo_base)

                        # Apply transf
                        world_pts = np.hstack(
                            (points, np.ones_like(points[:, :1])))
                        world_pts = np.matmul(
                            world_pts, H_velo_world.T).astype(np.float32)

                        # Save frame
                        world_pts[:, 3] = i0 + save_i
                        all_points.append(world_pts)
                        all_labels.append(labels)

                        # also save trajectory
                        traj_pts, traj_clrs = frame_H_to_points(H_velo_world,
                                                                size=0.1)
                        traj_pts = np.hstack(
                            (traj_pts,
                             np.ones_like(traj_pts[:, :1]) * (i0 + save_i)))
                        all_traj_pts.append(traj_pts.astype(np.float32))
                        all_traj_clrs.append(traj_clrs)

                        traj_pts, traj_clrs = frame_H_to_points(world_H,
                                                                size=1.1)
                        traj_pts = np.hstack(
                            (traj_pts,
                             np.ones_like(traj_pts[:, :1]) * (i0 + save_i)))
                        all_traj_pts2.append(traj_pts.astype(np.float32))
                        all_traj_clrs2.append(traj_clrs)

                    last_saved_frames += save_group
                    filename = join(out_day_folder, 'frames',
                                    'loc_aligned_{:05d}.ply'.format(i0))
                    write_ply(filename,
                              [np.vstack(all_points),
                               np.hstack(all_labels)],
                              ['x', 'y', 'z', 't', 'category'])
                    filename = join(out_day_folder, 'trajectory',
                                    'loc_traj_{:05d}.ply'.format(i0))
                    write_ply(
                        filename,
                        [np.vstack(all_traj_pts),
                         np.vstack(all_traj_clrs)],
                        ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

                    filename = join(out_day_folder, 'trajectory',
                                    'loc_traj_body_{:05d}.ply'.format(i0))
                    write_ply(
                        filename,
                        [np.vstack(all_traj_pts2),
                         np.vstack(all_traj_clrs2)],
                        ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

            #################
            # Post processing
            #################

            filename = join(out_day_folder, 'loc_map_{:s}.ply'.format(day))
            write_ply(filename, [
                pointmap.points, pointmap.normals, pointmap.scores,
                pointmap.counts
            ], ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

            all_points = []
            all_labels = []
            all_traj_pts = []
            all_traj_clrs = []
            i0 = last_saved_frames
            for save_i, save_f_t in enumerate(frame_times[i0:]):

                # Load points
                points, labels = self.load_frame_points_labels(
                    frame_names[i0 + save_i])

                # Find closest loc poses
                loc_i1 = np.argmin(np.abs(loc_t - save_f_t))
                if save_f_t < loc_t[loc_i1]:
                    loc_i0 = loc_i1 - 1
                else:
                    loc_i0 = loc_i1
                    loc_i1 = loc_i0 + 1

                if loc_i1 >= len(loc_t):
                    break

                # Interpolate the ground truth pose at current time
                interp_t = (save_f_t - loc_t[loc_i0]) / \
                    (loc_t[loc_i1] - loc_t[loc_i0])
                world_H = interp_pose(interp_t, loc_H[loc_i0], loc_H[loc_i1])

                # Transformation of the lidar (loc is in body frame)
                H_velo_world = np.matmul(world_H, self.H_velo_base)

                # Apply transf
                world_pts = np.hstack((points, np.ones_like(points[:, :1])))
                world_pts = np.matmul(world_pts,
                                      H_velo_world.T).astype(np.float32)

                # Save frame
                world_pts[:, 3] = i0 + save_i
                all_points.append(world_pts)
                all_labels.append(labels)

                # also save trajectory
                traj_pts, traj_clrs = frame_H_to_points(H_velo_world, size=0.1)
                traj_pts = np.hstack(
                    (traj_pts, np.ones_like(traj_pts[:, :1]) * (i0 + save_i)))
                all_traj_pts.append(traj_pts.astype(np.float32))
                all_traj_clrs.append(traj_clrs)

            last_saved_frames += save_group
            filename = join(out_day_folder, 'frames',
                            'loc_aligned_{:05d}.ply'.format(i0))
            write_ply(filename, [np.vstack(all_points),
                                 np.hstack(all_labels)],
                      ['x', 'y', 'z', 't', 'category'])
            filename = join(out_day_folder, 'trajectory',
                            'loc_traj_{:05d}.ply'.format(i0))
            write_ply(filename,
                      [np.vstack(all_traj_pts),
                       np.vstack(all_traj_clrs)],
                      ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

            # Save full trajectory
            all_traj_pts = []
            all_traj_clrs = []
            for save_i, save_H in enumerate(transform_list):

                # Save trajectory
                traj_pts, traj_clrs = frame_H_to_points(save_H, size=0.1)
                traj_pts = np.hstack(
                    (traj_pts, np.ones_like(traj_pts[:, :1]) * save_i))
                all_traj_pts.append(traj_pts.astype(np.float32))
                all_traj_clrs.append(traj_clrs)

            filename = join(out_day_folder, 'loc_traj_{:s}.ply'.format(day))
            write_ply(filename,
                      [np.vstack(all_traj_pts),
                       np.vstack(all_traj_clrs)],
                      ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

            # Save alignments
            with open(online_slam_file, 'wb') as file:
                pickle.dump((frame_names[:len(transform_list)], transform_list,
                             1, pointmap), file)

    def old_pointmap_slam(self,
                          map_voxel_size=0.08,
                          frame_stride=1,
                          frame_voxel_size=0.2,
                          max_pairing_dist=0.5,
                          save_group=50,
                          verbose=1):

        #################
        # Init parameters
        #################

        # Out files
        out_folder = join(self.data_path, 'slam_offline')
        if not exists(out_folder):
            makedirs(out_folder)

        ##########################
        # Start first pass of SLAM
        ##########################

        for d, day in enumerate(self.days):

            out_day_folder = join(out_folder, day)
            if not exists(out_day_folder):
                makedirs(out_day_folder)

            for folder_name in ['trajectory', 'map', 'frames']:
                if not exists(join(out_day_folder, folder_name)):
                    makedirs(join(out_day_folder, folder_name))

            mapping_file = join(out_day_folder,
                                'PointSLAM_{:s}.pkl'.format(day))
            if exists(mapping_file):
                continue

            # List of groundtruth timestamps and poses
            gt_t = self.gt_t[d]
            gt_H = self.gt_H[d]

            # List of frames for this day
            frame_names = self.day_f_names[d]
            frame_times = self.day_f_times[d]

            # List of transformation we are trying to optimize
            transform_list = [np.eye(4) for _ in frame_names]
            previous_nR = np.eye(3)

            # Initiate map
            pointmap = PointMap(dl=map_voxel_size)
            last_saved_frames = 0
            FPS = 0
            N = len(frame_names)

            # Test mapping
            for i, f_t in enumerate(frame_times):

                t = [time.time()]

                # Load ply format points
                points = self.load_frame_points(frame_names[i])

                t += [time.time()]

                # Get normals (dummy r_scale to avoid removing points as simulation scans are perfect)
                normals, planarity, linearity = polar_normals(points,
                                                              radius=1.5,
                                                              lidar_n_lines=31,
                                                              h_scale=0.5,
                                                              r_scale=1000.0)
                norm_scores = planarity + linearity

                # Remove outliers
                points = points[norm_scores > 0.1]
                normals = normals[norm_scores > 0.1]
                norm_scores = norm_scores[norm_scores > 0.1]

                # Filter out points according to main normal directions (Not necessary if normals are better computed)
                # norm_scores *= normal_filtering(normals)

                # Subsample to reduce number of points (Use map class for spherical grid subs)
                if frame_voxel_size > 0:
                    tmpmap = PointMap(dl=frame_voxel_size)
                    tmpmap.update(points, normals, norm_scores)
                    sub_points = tmpmap.points
                    sub_normals = tmpmap.normals
                    sub_norm_scores = tmpmap.scores
                else:
                    sub_points = points
                    sub_normals = normals
                    sub_norm_scores = norm_scores

                # # Init pose based on normal repartition
                # n_cloud, smooth_counts, n_R = normals_orientation(sub_normals)
                #
                # # Correct the rotation symmetry errors
                # dots = np.matmul(previous_nR.T, n_R)
                # angle_diffs = np.arccos(np.abs(np.clip(dots, -1.0, 1.0)))
                # assoc = np.argmin(angle_diffs, axis=1)
                # sym_correction = (dots > 0).astype(np.float32) * 2 - 1
                # sym_correction = np.array([sym_correction[i, assoc[i]] for i in range(3)])
                # n_R = n_R[:, assoc] * sym_correction

                # n_H = np.eye(4)
                # n_H[:3, :3] = n_R
                # traj_pts, traj_clrs = frame_H_to_points(n_H, size=1.5)
                # write_ply('nnn_NORMAL_ALIGN_{:d}.ply'.format(i),
                #           [traj_pts, traj_clrs],
                #           ['x', 'y', 'z', 'red', 'green', 'blue'])
                # write_ply('nnn_NORMAL_HIST_{:d}.ply'.format(i),
                #           [n_cloud, smooth_counts],
                #           ['x', 'y', 'z', 'counts'])
                # if i > 17:
                #     a = 1 / 0
                #
                # # Get rotation
                # init_dR = np.matmul(n_R, previous_nR.T)
                #
                # # Save for next iteration
                # previous_nR = n_R

                # Find closest gt poses
                gt_i1 = np.argmin(np.abs(gt_t - f_t))
                if f_t < gt_t[gt_i1]:
                    gt_i0 = gt_i1 - 1
                else:
                    gt_i0 = gt_i1
                    gt_i1 = gt_i0 + 1

                # Interpolate the ground truth pose at current time
                interp_t = (f_t - gt_t[gt_i0]) / (gt_t[gt_i1] - gt_t[gt_i0])
                frame_H = interp_pose(interp_t, gt_H[gt_i0], gt_H[gt_i1])
                gt_H_velo_world = np.matmul(frame_H, self.H_velo_base)

                if i < 1:

                    H_velo_world = gt_H_velo_world

                else:

                    # init_dH = np.eye(4)
                    # init_dH[:3, :3] = init_dR.T
                    # init_H = np.matmul(transform_list[i-1], init_dH)

                    # world_points = np.hstack((points, np.ones_like(points[:, :1])))
                    # world_points = np.matmul(world_points, transform_list[i - 1].T).astype(np.float32)[:, :3]
                    # write_ply('debug_init.ply',
                    #           [world_points],
                    #           ['x', 'y', 'z'])
                    #
                    # world_points = np.hstack((points, np.ones_like(points[:, :1])))
                    # world_points = np.matmul(world_points, init_H.T).astype(np.float32)[:, :3]
                    # write_ply('debug_init2.ply',
                    #           [world_points],
                    #           ['x', 'y', 'z'])

                    all_H, rms, planar_rms = point_to_map_icp(
                        sub_points,
                        sub_norm_scores,
                        pointmap.points,
                        pointmap.normals,
                        pointmap.scores,
                        init_H=transform_list[i - 1],
                        init_phi=0,
                        motion_distortion=False,
                        n_samples=1000,
                        max_pairing_dist=20.0,
                        max_iter=50,
                        rotDiffThresh=0.1 * np.pi / 180,
                        transDiffThresh=0.01,
                        avg_steps=2)

                    # print(i, f_t, int(1000 * (f_t - frame_times[i - 1])), 'ms')
                    # print(planar_rms[0], '=>', planar_rms[-1], 'in', len(planar_rms), 'iter')

                    # test_H = np.stack([transform_list[i-1], all_H[-1], all_H[-1]], 0)
                    # dT, dR = rot_trans_diffs(test_H)
                    #
                    # print('-------------    Init dR: ',
                    #       180 / np.pi * np.arccos((np.trace(init_dR) - 1) / 2))
                    # print('-------------  Converged: ',
                    #       180 / np.pi * dR[0])

                    # print('++++++++++++++++++++++++++++++++++++++++++>     ',
                    #       np.mean(dT[-10:]), 180 * np.mean(dR[-10:]) / np.pi)
                    # if i > 10:
                    #     plt.figure('dT')
                    #     plt.plot(dT)
                    #     plt.figure('dR')
                    #     plt.plot(dR)
                    #     plt.show()
                    #     a = 1/0

                    # Second pass with lower max paring dist
                    all_H, rms, planar_rms = point_to_map_icp(
                        sub_points,
                        sub_norm_scores,
                        pointmap.points,
                        pointmap.normals,
                        pointmap.scores,
                        init_H=all_H[-1],
                        init_phi=0,
                        motion_distortion=False,
                        n_samples=1000,
                        max_pairing_dist=max_pairing_dist,
                        max_iter=50,
                        rotDiffThresh=0.1 * np.pi / 180,
                        transDiffThresh=0.01,
                        avg_steps=5)
                    # print(planar_rms[0], '=>', planar_rms[-1], 'in', len(planar_rms), 'iter')

                    lR = all_H[-1, :3, :3]
                    ortho = np.matmul(lR, lR.T)

                    if (np.max(ortho - np.eye(3)) > 0.01):

                        np.set_printoptions(suppress=True, precision=2)
                        print('****************')
                        for tH in all_H[-2:]:
                            print(tH)
                        print('****************')
                        gt_H_velo_world *= 0

                    H_velo_world = all_H[-1]

                # Apply transf
                transform_list[i] = H_velo_world
                world_points = np.hstack((points, np.ones_like(points[:, :1])))
                world_points = np.matmul(world_points, H_velo_world.T).astype(
                    np.float32)[:, :3]
                world_normals = np.matmul(
                    normals, H_velo_world[:3, :3].T).astype(np.float32)

                # Check for failure
                T_error, R_error = rot_trans_diffs(
                    np.stack([gt_H_velo_world, H_velo_world], 0))
                if i > 1 and (T_error > 1.0 or R_error > 10 * np.pi / 180):

                    print(
                        '\n*************************************************\n',
                        'Error in the mapping:\n', 'T_error = ', T_error,
                        'R_error = ', R_error,
                        'Stopping mapping and saving debug frames\n',
                        '\n*************************************************\n'
                    )

                    write_ply('debug_converge.ply', [world_points],
                              ['x', 'y', 'z'])

                    filename = 'debug_map.ply'
                    write_ply(filename, [
                        pointmap.points, pointmap.normals, pointmap.scores,
                        pointmap.counts
                    ], ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

                    world_points = np.hstack(
                        (points, np.ones_like(points[:, :1])))
                    world_points = np.matmul(world_points,
                                             transform_list[i - 1].T).astype(
                                                 np.float32)[:, :3]
                    write_ply('debug_init.ply', [world_points],
                              ['x', 'y', 'z'])
                    a = 1 / 0

                t += [time.time()]

                # Update map
                pointmap.update(world_points, world_normals, norm_scores)

                if i % save_group == 0:
                    filename = join(out_day_folder, 'map',
                                    'debug_map_{:03d}.ply'.format(i))
                    write_ply(filename, [
                        pointmap.points, pointmap.normals, pointmap.scores,
                        pointmap.counts
                    ], ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

                t += [time.time()]

                if verbose == 2:
                    ti = 0
                    print('Load ............ {:7.1f}ms'.format(
                        1000 * (t[ti + 1] - t[ti])))
                    ti += 1
                    print('Preprocessing ... {:7.1f}ms'.format(
                        1000 * (t[ti + 1] - t[ti])))
                    ti += 1
                    print('Align ........... {:7.1f}ms'.format(
                        1000 * (t[ti + 1] - t[ti])))
                    ti += 1
                    print('Mapping ......... {:7.1f}ms'.format(
                        1000 * (t[ti + 1] - t[ti])))

                if verbose > 0:
                    fmt_str = 'Mapping {:3d}  --- {:5.1f}% or {:02d}:{:02d}:{:02d} remaining at {:.1f}fps'
                    if i == 0:
                        FPS = 1 / (t[-1] - t[0])
                    else:
                        FPS += (1 / (t[-1] - t[0]) - FPS) / 10
                    remaining = int((N - (i + 1)) / FPS)
                    hours = remaining // 3600
                    remaining = remaining - 3600 * hours
                    minutes = remaining // 60
                    seconds = remaining - 60 * minutes
                    print(
                        fmt_str.format(i, 100 * (i + 1) / N, hours, minutes,
                                       seconds, FPS))

                # Save groups of 100 frames together
                if (i > last_saved_frames + save_group + 1):
                    all_points = []
                    all_traj_pts = []
                    all_traj_clrs = []
                    all_traj_pts2 = []
                    all_traj_clrs2 = []
                    i0 = last_saved_frames
                    i1 = i0 + save_group
                    for save_i, save_f_t in enumerate(frame_times[i0:i1]):

                        # Load points
                        points = self.load_frame_points(frame_names[i0 + save_i])

                        # Transformation of the lidar (gt is in body frame)
                        H_velo_world = transform_list[i0 + save_i]

                        # Apply transf
                        world_pts = np.hstack(
                            (points, np.ones_like(points[:, :1])))
                        world_pts = np.matmul(
                            world_pts, H_velo_world.T).astype(np.float32)

                        # Save frame
                        world_pts[:, 3] = i0 + save_i
                        all_points.append(world_pts)

                        # also save trajectory
                        traj_pts, traj_clrs = frame_H_to_points(H_velo_world,
                                                                size=0.1)
                        traj_pts = np.hstack(
                            (traj_pts,
                             np.ones_like(traj_pts[:, :1]) * (i0 + save_i)))
                        all_traj_pts.append(traj_pts.astype(np.float32))
                        all_traj_clrs.append(traj_clrs)

                    last_saved_frames += save_group
                    filename = join(out_day_folder, 'frames',
                                    'debug_aligned_{:05d}.ply'.format(i0))
                    write_ply(filename, [np.vstack(all_points)],
                              ['x', 'y', 'z', 't'])
                    filename = join(out_day_folder, 'trajectory',
                                    'debug_traj_{:05d}.ply'.format(i0))
                    write_ply(
                        filename,
                        [np.vstack(all_traj_pts),
                         np.vstack(all_traj_clrs)],
                        ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

            #################
            # Post processing
            #################

            filename = join(out_day_folder, 'map_{:s}.ply'.format(day))
            write_ply(filename, [
                pointmap.points, pointmap.normals, pointmap.scores,
                pointmap.counts
            ], ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

            all_points = []
            all_traj_pts = []
            all_traj_clrs = []
            i0 = last_saved_frames
            for save_i, save_f_t in enumerate(frame_times[i0:]):

                # Load points
                points = self.load_frame_points(frame_names[i0 + save_i])

                # Transformation of the lidar (gt is in body frame)
                H_velo_world = transform_list[i0 + save_i]

                # Apply transf
                world_pts = np.hstack((points, np.ones_like(points[:, :1])))
                world_pts = np.matmul(world_pts,
                                      H_velo_world.T).astype(np.float32)

                # Save frame
                world_pts[:, 3] = i0 + save_i
                all_points.append(world_pts)

                # also save trajectory
                traj_pts, traj_clrs = frame_H_to_points(H_velo_world, size=0.1)
                traj_pts = np.hstack(
                    (traj_pts, np.ones_like(traj_pts[:, :1]) * (i0 + save_i)))
                all_traj_pts.append(traj_pts.astype(np.float32))
                all_traj_clrs.append(traj_clrs)

            last_saved_frames += save_group
            filename = join(out_day_folder, 'frames',
                            'debug_aligned_{:05d}.ply'.format(i0))
            write_ply(filename, [np.vstack(all_points)], ['x', 'y', 'z', 't'])
            filename = join(out_day_folder, 'trajectory',
                            'debug_traj_{:05d}.ply'.format(i0))
            write_ply(filename,
                      [np.vstack(all_traj_pts),
                       np.vstack(all_traj_clrs)],
                      ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

            # Save full trajectory
            all_traj_pts = []
            all_traj_clrs = []
            for save_i, save_H in enumerate(transform_list):

                # Save trajectory
                traj_pts, traj_clrs = frame_H_to_points(save_H, size=0.1)
                traj_pts = np.hstack(
                    (traj_pts, np.ones_like(traj_pts[:, :1]) * save_i))
                all_traj_pts.append(traj_pts.astype(np.float32))
                all_traj_clrs.append(traj_clrs)

            filename = join(out_day_folder, 'map_traj_{:s}.ply'.format(day))
            write_ply(filename,
                      [np.vstack(all_traj_pts),
                       np.vstack(all_traj_clrs)],
                      ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

            # Save alignments
            filename = join(out_day_folder, 'PointSLAM_{:s}.pkl'.format(day))
            with open(filename, 'wb') as file:
                pickle.dump(
                    (frame_names, transform_list, frame_stride, pointmap),
                    file)

    def pointmap_slam(self,
                      map_voxel_size=0.03,
                      frame_voxel_size=0.1,
                      max_pairing_dist=2.0,
                      icp_samples=400,
                      verbose=1):

        # Out files
        out_folder = join(self.data_path, 'slam_offline')
        if not exists(out_folder):
            makedirs(out_folder)

        for d, day in enumerate(self.days):

            out_day_folder = join(out_folder, day)
            if not exists(out_day_folder):
                makedirs(out_day_folder)

            # List of groundtruth timestamps and poses
            gt_t = self.gt_t[d]
            gt_H = self.gt_H[d]

            # List of frames for this day
            frame_names = self.day_f_names[d]
            frame_times = self.day_f_times[d]

            if d == 0:

                all_H = slam_on_sim_sequence(frame_names,
                                             frame_times,
                                             gt_H,
                                             gt_t,
                                             out_day_folder,
                                             map_voxel_size=map_voxel_size,
                                             frame_voxel_size=frame_voxel_size,
                                             motion_distortion=False,
                                             icp_samples=icp_samples,
                                             icp_pairing_dist=max_pairing_dist,
                                             H_velo_base=self.H_velo_base)

            else:

                # Load map data
                map_file = join(out_folder,
                                'map_{:s}.ply'.format(self.days[0]))
                data = read_ply(map_file)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
                scores = data['f0']

                all_H = slam_on_sim_sequence(frame_names,
                                             frame_times,
                                             gt_H,
                                             gt_t,
                                             out_day_folder,
                                             init_points=points,
                                             init_normals=normals,
                                             init_scores=scores,
                                             map_voxel_size=map_voxel_size,
                                             frame_voxel_size=frame_voxel_size,
                                             motion_distortion=False,
                                             icp_samples=icp_samples,
                                             icp_pairing_dist=max_pairing_dist,
                                             H_velo_base=self.H_velo_base)

            save_trajectory(join(out_folder, 'traj_{:s}.ply'.format(day)),
                            all_H)

            # Save a map of the single day (not including init map)
            pointmap = self.map_from_frames(frame_names,
                                            all_H,
                                            map_voxel_size=map_voxel_size)

            filename = join(out_day_folder, 'single_map_{:s}.ply'.format(day))
            write_ply(filename, [
                pointmap.points, pointmap.normals, pointmap.scores,
                pointmap.counts
            ], ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

            save_trajectory(join(out_folder, 'traj_{:s}.ply'.format(day)),
                            all_H)

            # Save alignments
            filename = join(out_folder, 'slam_{:s}.pkl'.format(day))
            with open(filename, 'wb') as file:
                pickle.dump(
                    (frame_names, all_H[:len(frame_names)], 1, pointmap), file)

    def map_from_frames(self,
                        frame_names,
                        frames_H,
                        map_voxel_size=0.03,
                        verbose=1):
        """Create a map with frames already align by a previous method"""

        # Init pointmap
        pointmap = PointMap(dl=map_voxel_size)
        FPS = 0
        N = len(frame_names)

        # Test mapping
        for i, f_H in enumerate(frames_H):

            t = [time.time()]

            # Load ply format points
            points = self.load_frame_points(frame_names[i])

            t += [time.time()]

            # Get normals (dummy r_scale to avoid removing points as simulation scans are perfect)
            normals, planarity, linearity = polar_normals(points,
                                                          radius=1.5,
                                                          lidar_n_lines=31,
                                                          h_scale=0.5,
                                                          r_scale=1000.0)
            norm_scores = planarity + linearity

            # Remove outliers
            points = points[norm_scores > 0.1]
            normals = normals[norm_scores > 0.1]
            norm_scores = norm_scores[norm_scores > 0.1]

            # Apply transf
            world_points = np.hstack((points, np.ones_like(points[:, :1])))
            world_points = np.matmul(world_points,
                                     f_H.T).astype(np.float32)[:, :3]
            world_normals = np.matmul(normals,
                                      f_H[:3, :3].T).astype(np.float32)

            t += [time.time()]

            # Update map
            pointmap.update(world_points, world_normals, norm_scores)

            t += [time.time()]

            if verbose > 0:
                fmt_str = 'Mapping {:3d}  --- {:5.1f}% or {:02d}:{:02d}:{:02d} remaining at {:.1f}fps'
                if i == 0:
                    FPS = 1 / (t[-1] - t[0])
                else:
                    FPS += (1 / (t[-1] - t[0]) - FPS) / 10
                remaining = int((N - (i + 1)) / FPS)
                hours = remaining // 3600
                remaining = remaining - 3600 * hours
                minutes = remaining // 60
                seconds = remaining - 60 * minutes
                print(
                    fmt_str.format(i, 100 * (i + 1) / N, hours, minutes,
                                   seconds, FPS))

        return pointmap

    def loop_closure_edges_prompt(self, lim_box, init_H):
            
        global picked_loop_edges
        picked_loop_edges = []

        # define Matplotlib figure and axis
        fig, ax = plt.subplots()

        margin = (lim_box.x2 - lim_box.x1) * 0.1
        ax.set_xlim(lim_box.x1 - margin, lim_box.x2 + margin)
        ax.set_ylim(lim_box.y1 - margin, lim_box.y2 + margin)
        ax.set_aspect('equal', adjustable='box')

        # add rectangle to plot
        ax.add_patch(lim_box.plt_rect(edgecolor='black', facecolor='cyan', fill=False, lw=2))

        # Plot the trajectory with color
        traj_color = np.arange(init_H.shape[0])
        coll = ax.scatter(init_H[:, 0, 3], init_H[:, 1, 3], c=traj_color, marker='.', picker=10)
        plt.grid(True)

        def on_pick(event):
            global picked_loop_edges

            # Cluster points close together
            max_gap = 30
            group = []
            centers = []
            for ind in event.ind:
                
                if len(group) < 1:
                    # Start a new group
                    group.append(ind)
                
                else:
                    # Fill group
                    if (ind - group[-1]) < max_gap:
                        group.append(ind)

                    # Finish group
                    else:
                        centers.append(group[len(group) // 2])
                        group = [ind]
            
            # Add last group
            centers.append(group[len(group) // 2])

            # Create edges
            picked = False
            if len(centers) <= 1:
                print("not enough picked trajectory points in the vicinity to create an edge: ", centers)
            elif len(centers) == 2:
                print("Adding the following loop closure edge: ", centers)
                picked_loop_edges.append(centers)
                picked = True
            else:
                print("To many picked trajectory points in the vicinity (TODO HANDLE THIS CASE): ", centers)

            if (picked):
                # Show a circle of the loop closure area
                ax.add_patch(Circle((event.mouseevent.xdata, event.mouseevent.ydata), radius=1.0,
                                    edgecolor='red',
                                    fill=False,
                                    lw=0.5))
                plt.draw()


            return

        # fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('pick_event', on_pick)

        # display plot
        plt.show()

        return np.array(picked_loop_edges, dtype=np.int32)

    def init_map(self, map_dl=0.03, occup_threshold=0.8):

        # Map is initiated by removing movable points with its own trajectory
        #   > Step 0: REdo the point map from scratch
        #   > Step 1: annotate short-term movables for each frame
        #   > Step 2: Do a pointmap slam without the short-term movables (enforce horizontal planar ground)
        #   > Step 3: Apply loop closure on the poses of this second slam
        #   > Step 4: With the corrected poses and the full point clouds, create a barycentre pointmap
        #   > Step 5: Remove the short-term movables from this good map.


        print('\n')
        print('Build Initial Map')
        print('*****************')
        print('\n')
        print('Using run', self.map_day)

        # Folder where the incrementally updated map is stored
        map_folder = join(self.data_path, 'slam_offline', self.map_day)
        if not exists(map_folder):
            makedirs(map_folder)

        initial_map_file = join(map_folder, 'map_update_{:04}.ply'.format(0))
        if exists(initial_map_file):
            print('\n----- Recovering map from previous file')
            print('\n    > Done')
            return

        # Load hardcoded map limits
        map_lim_file = join(self.data_path, 'calibration/map_limits.txt')
        if exists(map_lim_file):
            map_limits = np.loadtxt(map_lim_file)
        else:
            map_limits = None

        lim_box = Box(map_limits[0, 0], map_limits[1, 0], map_limits[0, 1], map_limits[1, 1])
        min_z = map_limits[2, 0]
        max_z = map_limits[2, 1]

        ##########################
        # Initial odometry mapping
        ##########################

        print('\n----- Initial odometry on the mapping session')

        # Get frame names and times
        map_t = np.array([np.float64(f.split('/')[-1][:-4]) for f in self.map_f_names], dtype=np.float64)
        map_t, frame_names = filter_frame_timestamps(map_t, self.map_f_names)

        # Check if initial mapping 
        init_map_pkl = join(map_folder, 'map0_traj_{:s}.pkl'.format(self.map_day))
        if exists(init_map_pkl):
            with open(init_map_pkl, 'rb') as file:
                map_H = pickle.load(file)

            print('\n    > Done (recovered from previous file)')

        else:

            map_H = slam_on_real_sequence(frame_names,
                                          map_t,
                                          map_folder,
                                          map_voxel_size=map_dl,
                                          frame_voxel_size=3 * map_dl,
                                          motion_distortion=True,
                                          filtering=False,
                                          force_flat_ground=True,
                                          verbose_time=5.0,
                                          icp_samples=600,
                                          icp_pairing_dist=2.0,
                                          icp_planar_dist=0.12,
                                          icp_max_iter=100,
                                          icp_avg_steps=5,
                                          saving_for_loop_closure=True)

            # Save the trajectory
            save_trajectory(join(map_folder, 'map0_traj_{:s}.ply'.format(self.map_day)), map_H)
            with open(init_map_pkl, 'wb') as file:
                pickle.dump(map_H, file)

            # Rename the saved map file
            old_name = join(map_folder, 'map_{:s}.ply'.format(self.map_day))
            new_name = join(map_folder, 'map0_{:s}.ply'.format(self.map_day))
            os.rename(old_name, new_name)

            print('\n    > Done')

        ######################
        # Loop closure utility
        ######################

        print('\n----- Loop closure')
        
        new_traj_file = join(map_folder, 'loopclosed_traj_{:s}.pkl'.format(self.map_day))
        if exists(new_traj_file):
            with open(new_traj_file, 'rb') as file:
                loop_H = pickle.load(file)
                
            print('\n    > Done (recovered from previous file)')

        else:

            # Get loop edges with a simple UI
            print('\nPicking loop edges')
            loop_edges = self.loop_closure_edges_prompt(lim_box, map_H)
            
            print('\nResult:')
            print(loop_edges)
            
            print('\nPerform loop closure')

            # Close the loop
            loop_H = loop_closure(self.map_day, self.data_path, loop_edges)
            
            # Save new trajectory
            save_trajectory(join(map_folder, 'loopclosed_traj_{:s}.ply'.format(self.map_day)), loop_H)
            with open(new_traj_file, 'wb') as file:
                pickle.dump(loop_H, file)
        
            print('\n    > Done')
        

        print('\n----- Build loop closed map')

        loop_closed_map_name = join(map_folder, 'loopclosed_map0_{:s}.ply'.format(self.map_day))
        if not exists(loop_closed_map_name):

            odom_H = [np.linalg.inv(odoH) for odoH in loop_H]
            odom_H = np.stack(odom_H, 0)
            _ = slam_on_real_sequence(frame_names,
                                      map_t,
                                      map_folder,
                                      map_voxel_size=map_dl,
                                      frame_voxel_size=3 * map_dl,
                                      motion_distortion=True,
                                      filtering=False,
                                      verbose_time=5.0,
                                      icp_samples=600,
                                      icp_pairing_dist=2.0,
                                      icp_planar_dist=0.12,
                                      icp_max_iter=0,
                                      icp_avg_steps=5,
                                      odom_H=odom_H)

            # Rename the saved map file
            old_name = join(map_folder, 'map_{:s}.ply'.format(self.map_day))
            os.rename(old_name, loop_closed_map_name)
            print('\n    > Done')

        else:
            print('\n    > Done (recovered from previous file)')


        #####################################
        # Annotate short-term on original map
        #####################################

        print('\n----- Remove dynamic points from map')

        first_annot_name = join(map_folder, 'movable_final.ply')
        if exists(first_annot_name):

            print('\nLoad annot')
            data = read_ply(first_annot_name)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
            movable_prob = data['movable']
            movable_count = data['counts']
            print('OK')
            
            print('\n    > Done (recovered from previous file)')

        else:

            print('\nPrepare pointmap')

            # Get the map
            map_original_name = join(map_folder, 'loopclosed_map0_' + self.map_day + '.ply')
            if not exists(map_original_name):
                map_original_name = join(map_folder, 'map0_' + self.map_day + '.ply')
            data = read_ply(map_original_name)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

            print('\nStart ray casting')

            # Get short term movables
            movable_prob, movable_count = ray_casting_annot(frame_names,
                                                            points,
                                                            normals,
                                                            loop_H,
                                                            theta_dl=0.33 * np.pi / 180,
                                                            phi_dl=0.4 * np.pi / 180,
                                                            map_dl=map_dl,
                                                            verbose_time=5,
                                                            motion_distortion_slices=16)

            movable_prob = movable_prob / (movable_count + 1e-6)
            movable_prob[movable_count < 1e-6] = -1

            # Extract ground ransac
            ground_mask = extract_map_ground(points,
                                             normals,
                                             map_folder,
                                             vertical_thresh=10.0,
                                             dist_thresh=0.2,
                                             remove_dist=0.21)

            # Do not remove ground points
            movable_prob[ground_mask] = 0

            print('\nSave movable probs')

            # Save it
            # write_ply(first_annot_name,
            write_ply(join(map_folder, 'movable_final.ply'),
                      [points, normals, movable_prob, movable_count],
                      ['x', 'y', 'z', 'nx', 'ny', 'nz', 'movable', 'counts'])

            print('\n    > Done')



        #####################
        # Enhance map quality
        #####################

        # TODO: C++ function for loop closure and flatten ground with ceres and pt2pl loss
        
        print('\n----- Get a finer barycenter map')
        
        correct_H = loop_H

        # Realign frames on the loopclosed map with forcing ground + IMLS ICP formulation
        fine_map_name = join(map_folder, 'fine_map0_{:s}.ply'.format(self.map_day))
        if not exists(fine_map_name):
            
            data = read_ply(loop_closed_map_name)
            scores = data['f1']
            ground_mask = np.zeros(movable_prob.shape, dtype=bool)

            n_finer = 2
            for _ in range(n_finer):

                # remove movables
                still_mask = np.logical_and(movable_prob > -0.1, movable_prob < occup_threshold)
                still_mask = np.logical_or(still_mask, ground_mask)
                still_mask = np.logical_and(still_mask, points[:, 0] > lim_box.x1)
                still_mask = np.logical_and(still_mask, points[:, 0] < lim_box.x2)
                still_mask = np.logical_and(still_mask, points[:, 1] > lim_box.y1)
                still_mask = np.logical_and(still_mask, points[:, 1] < lim_box.y2)
                still_mask = np.logical_and(still_mask, points[:, 2] > min_z)
                still_mask = np.logical_and(still_mask, points[:, 2] < max_z)
                points = points[still_mask]
                normals = normals[still_mask]
                scores = scores[still_mask]

                # Align frames with barycenter
                odom_H = [np.linalg.inv(odoH) for odoH in correct_H]
                odom_H = np.stack(odom_H, 0)
                correct_H = slam_on_real_sequence(frame_names,
                                                  map_t,
                                                  map_folder,
                                                  init_points=points,
                                                  init_normals=normals,
                                                  init_scores=scores,
                                                  map_voxel_size=map_dl,
                                                  frame_voxel_size=3 * map_dl,
                                                  motion_distortion=True,
                                                  filtering=False,
                                                  force_flat_ground=True,
                                                  barycenter_map=True,
                                                  update_init_map=False,
                                                  verbose_time=5,
                                                  icp_samples=600,
                                                  icp_pairing_dist=2.0,
                                                  icp_planar_dist=0.12,
                                                  icp_max_iter=100,
                                                  icp_avg_steps=5,
                                                  odom_H=odom_H)
                                                  
                # Update the map points with the refined ones
                data = read_ply(join(map_folder, 'barymap_{:s}.ply'.format(self.map_day)))
                points = np.vstack((data['x'], data['y'], data['z'])).T
                normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
                scores = data['f1']

                # Get short term movables
                movable_prob, movable_count = ray_casting_annot(frame_names,
                                                                points,
                                                                normals,
                                                                correct_H,
                                                                theta_dl=0.33 * np.pi / 180,
                                                                phi_dl=0.4 * np.pi / 180,
                                                                map_dl=map_dl,
                                                                verbose_time=5,
                                                                motion_distortion_slices=16)

                movable_prob = movable_prob / (movable_count + 1e-6)
                movable_prob[movable_count < 1e-6] = -1

                # Ground is flat
                ground_mask = extract_flat_ground(points,
                                                  dist_thresh=0.25,
                                                  remove_dist=0.24)

            # Rename the saved map file
            old_name = join(map_folder, 'barymap_{:s}.ply'.format(self.map_day))
            os.rename(old_name, fine_map_name)
            
            # Save the final movables
            write_ply(join(map_folder, 'fine_movable_final.ply'),
                      [points, normals, movable_prob, movable_count],
                      ['x', 'y', 'z', 'nx', 'ny', 'nz', 'movable', 'counts'])
            
            # Save the new corrected trajectory
            save_trajectory(join(map_folder, 'correct_traj_{:s}.ply'.format(self.map_day)), correct_H)
            with open(join(map_folder, 'correct_traj_{:s}.pkl'.format(self.map_day)), 'wb') as file:
                pickle.dump(correct_H, file)

        print('\n    > Done')

        ##################
        # Finalize the map
        ##################

        print('\n----- Finalize the map')       
        
        initial_map_file = join(map_folder, 'map_update_{:04}.ply'.format(0))
        if not exists(initial_map_file):
            
            # Get the finer map and movables
            fine_map_name = join(map_folder, 'fine_map0_{:s}.ply'.format(self.map_day))
            data = read_ply(fine_map_name)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
            scores = data['f1']

            data = read_ply(join(map_folder, 'fine_movable_final.ply'))
            movable_prob = data['movable']
            movable_count = data['counts']

            # Get flat ground
            ground_mask = extract_flat_ground(points,
                                              dist_thresh=0.25,
                                              remove_dist=0.24)

            # Remove movable except on ground
            still_mask = np.logical_and(movable_prob > -0.1, movable_prob < occup_threshold)
            still_mask = np.logical_or(still_mask, ground_mask)
            still_mask = np.logical_and(still_mask, points[:, 0] > lim_box.x1)
            still_mask = np.logical_and(still_mask, points[:, 0] < lim_box.x2)
            still_mask = np.logical_and(still_mask, points[:, 1] > lim_box.y1)
            still_mask = np.logical_and(still_mask, points[:, 1] < lim_box.y2)
            still_mask = np.logical_and(still_mask, points[:, 2] > min_z)
            still_mask = np.logical_and(still_mask, points[:, 2] < max_z)
            points = points[still_mask]
            normals = normals[still_mask]
            scores = scores[still_mask]
            movable_prob = movable_prob[still_mask]

            # Now get ground again without movable
            ground_mask = extract_flat_ground(points,
                                              dist_thresh=0.25,
                                              remove_dist=0.24)

            # This time remove points with the refined ground
            still_mask = np.logical_and(movable_prob > -0.1, movable_prob < occup_threshold)
            still_mask = np.logical_or(still_mask, ground_mask)
            points = points[still_mask]
            normals = normals[still_mask]
            scores = scores[still_mask]
            movable_prob = movable_prob[still_mask]
            ground_mask = ground_mask[still_mask]

            # make the ground points flat
            points[ground_mask, 2] = 0

            # Annotate everything as still except ground (rest of the points will be removed)
            annots = np.zeros(movable_prob.shape, dtype=np.int32) + 2
            annots[ground_mask] = 1

            write_ply(initial_map_file,
                      [points, normals, scores, annots],
                      ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'classif'])

        print('\n    > Done')

        return

    def refine_map(self, refine_days, map_dl=0.03, min_rays=10, occup_threshold=0.8, merging_new_points=False):
        """
        Remove moving objects via ray-tracing. (Step 1 in the annotation process)
        """

        print('\n')
        print('------------------------------------------------------------------------------')
        print('\n')
        print('Refine Initial Map')
        print('******************')
        print('\nInitial map run:', self.map_day)
        print('\nRefine runs:')
        for d, day in enumerate(refine_days):
            print(' >', day)
        print('\n')

        # Folder where the incrementally updated map is stored
        map_folder = join(self.data_path, 'slam_offline', self.map_day)
        if not exists(map_folder):
            makedirs(map_folder)

        # List of the updated maps
        map_names = [f for f in listdir(map_folder) if f.startswith('map_update_')]

        # First check if we never initiated the map
        if len(map_names) == 0:
            raise ValueError('Map not initialized')

        # Now check if these days were already used for updating
        day_movable_names = [f for f in listdir(map_folder) if f.startswith('last_movables_')]
        day_movable_names = ['_'.join(f[:-4].split('_')[-2:]) for f in day_movable_names]

        seen_inds = []
        for d, day in enumerate(refine_days):
            if day in day_movable_names:
                seen_inds.append(d)
        if len(seen_inds) == len(refine_days):
            print('Refinement with the given days has already been Done')
            return

        # Get the latest update of the map
        map_names = np.sort(map_names)
        last_map = map_names[-1]
        last_update_i = int(last_map[:-4].split('_')[-1])

        # Load map
        print('Load last update of the map: ', last_map)
        data = read_ply(join(map_folder, last_map))
        map_points = np.vstack((data['x'], data['y'], data['z'])).T
        map_normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
        map_scores = data['scores']
        print('OK')


        #################
        # Frame alignment
        #################
        
        # Load hardcoded map limits
        map_lim_file = join(self.data_path, 'calibration/map_limits.txt')
        if exists(map_lim_file):
            map_limits = np.loadtxt(map_lim_file)
        else:
            map_limits = None

        lim_box = Box(map_limits[0, 0], map_limits[1, 0], map_limits[0, 1], map_limits[1, 1])
        min_z = map_limits[2, 0]
        max_z = map_limits[2, 1]

        print('\n----- Align refinement runs')
        
        # Localize against with precise ICP for each session
        all_correct_H = []
        all_frame_names = []
        for d, day in enumerate(refine_days):

            # Out folder
            out_folder = join(self.data_path, 'annotation', day)
            if not exists(out_folder):
                makedirs(out_folder)

            # Frame names
            frames_folder = join(self.days_folder, day, self.frame_folder_name)
            f_names = [f for f in listdir(frames_folder) if f[-4:] == '.ply']
            f_times = np.array([float(f[:-4]) for f in f_names], dtype=np.float64)
            f_names = np.array([join(frames_folder, f) for f in f_names])
            ordering = np.argsort(f_times)
            f_names = f_names[ordering]
            map_t = f_times[ordering]

            # Filter timestamps
            map_t, frame_names = filter_frame_timestamps(map_t, f_names)

            # Previously computed files
            # cpp_map_name = join(out_folder, 'map_{:s}.ply'.format(day))
            cpp_traj_name = join(out_folder, 'correct_traj_{:s}.pkl'.format(day))

            # Align frames on the map
            if not exists(cpp_traj_name):

                correct_H = slam_on_real_sequence(frame_names,
                                                  map_t,
                                                  out_folder,
                                                  init_points=map_points,
                                                  init_normals=map_normals,
                                                  init_scores=map_scores,
                                                  map_voxel_size=map_dl,
                                                  frame_voxel_size=3 * map_dl,
                                                  motion_distortion=True,
                                                  force_flat_ground=True,
                                                  barycenter_map=True,
                                                  update_init_map=True,
                                                  verbose_time=5.0,
                                                  icp_samples=800,
                                                  icp_pairing_dist=2.5,
                                                  icp_planar_dist=0.20,
                                                  icp_max_iter=100,
                                                  icp_avg_steps=5)

                # Verify that there was no error
                test = np.sum(np.abs(correct_H), axis=(1, 2)) > 1e-6
                if not np.all(test):
                    num_computed = np.sum(test.astype(np.int32))
                    raise ValueError('PointSlam returned without only {:d} poses computed out of {:d}'.format(num_computed, test.shape[0]))
                             
                # Save traj
                save_trajectory(join(out_folder, 'correct_traj_{:s}.ply'.format(day)), correct_H)
                with open(cpp_traj_name, 'wb') as file:
                    pickle.dump(correct_H, file)
                    

            else:

                # Load traj
                with open(cpp_traj_name, 'rb') as f:
                    correct_H = pickle.load(f)
            
            all_correct_H.append(correct_H)
            all_frame_names.append(frame_names)

        print('\n    > Done')


        ##########################
        # Adding points to the map
        ##########################



        # If the initial map is good enough you can pass this part 
        if not merging_new_points:
            
            print('\n----- No new points added to the initial map')
            print('\n    > Done')

        else:

            print('\n----- Add points to the map')

            merged_map_file = join(map_folder, 'merged_map.ply')
            merged_list_file = join(map_folder, 'merged_list.txt')

            # check if already merged
            already_merged = False
            if exists(merged_list_file):
                merged_days = np.loadtxt(merged_list_file, dtype=str)
                if (merged_days.shape[0] == refine_days.shape[0]):
                    already_merged = np.all(np.sort(refine_days) == np.sort(merged_days))
            already_merged = already_merged and exists(merged_map_file)

            if already_merged:

                # Reload already merged map
                data = read_ply(merged_map_file)
                map_points = np.vstack((data['x'], data['y'], data['z'])).T
                map_normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
                map_scores = data['scores']

                print('\n    > Done (recovered from previous file)')

            else:

                # New merge map
                for d, day in enumerate(refine_days):

                    # Load the barymap
                    out_folder = join(self.data_path, 'annotation', day)
                    fine_map_name = join(out_folder, 'barymap_{:s}.ply'.format(day))
                    data = read_ply(fine_map_name)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
                    scores = data['f1']

                    # remove points to far away
                    still_mask = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)
                    still_mask = np.logical_and(still_mask, points[:, 0] > lim_box.x1)
                    still_mask = np.logical_and(still_mask, points[:, 0] < lim_box.x2)
                    still_mask = np.logical_and(still_mask, points[:, 1] > lim_box.y1)
                    still_mask = np.logical_and(still_mask, points[:, 1] < lim_box.y2)
                    points = points[still_mask]
                    normals = normals[still_mask]
                    scores = scores[still_mask]

                    # Simple function that merges two point maps and reduce the number of point by keeping only the best score per voxel
                    #  1 > Compute distances with original
                    #  2 > Compute double distance back to get the point we need to add
                    #  3 > (slightly inferior to make sure we do not add point where there are already in the map)
                    #  4 > Merging maps by adding points
                    # TIP: if update normals=True, use very high merge dist to only update teh normals with the best scores
                    map_points, map_normals, map_scores, merge_i = merge_pointmaps(map_points,
                                                                                   map_normals,
                                                                                   map_scores,
                                                                                   add_points=points,
                                                                                   add_normals=normals,
                                                                                   add_scores=scores,
                                                                                   map_voxel_size=map_dl,
                                                                                   merge_dist=0.5,
                                                                                   barycenter_map=False)

                    # write_ply(join(map_folder, 'merged_map_{:03d}.ply'.format(d)),
                    #           [map_points, map_normals, map_scores, merge_i],
                    #           ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'merge_i'])
                    # a = 1/0

                write_ply(merged_map_file,
                          [map_points, map_normals, map_scores],
                          ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores'])

                np.savetxt(merged_list_file, refine_days, fmt='%s')

                print('\n    > Done')


        ###################
        # Movable detection
        ###################

        # Get remove point form each day independently
        # Otherwise if a table is there in only one day, it will not be removed.

        print('\n----- Get movable points')

        all_movables_probs = []
        for d, day in enumerate(refine_days):

            # No update if this day have already been seen
            movable_path = join(map_folder, 'last_movables_{:s}.ply'.format(day))

            if exists(movable_path):

                data = read_ply(movable_path)
                movable_prob = data['movable']
                movable_count = data['counts']

            else:

                correct_H = all_correct_H[d]
                frame_names = all_frame_names[d]

                # Get short term movables
                movable_prob, movable_count = ray_casting_annot(frame_names,
                                                                map_points,
                                                                map_normals,
                                                                correct_H,
                                                                theta_dl=0.33 * np.pi / 180,
                                                                phi_dl=0.4 * np.pi / 180,
                                                                map_dl=map_dl,
                                                                verbose_time=5.0,
                                                                motion_distortion_slices=16)

                movable_prob = movable_prob / (movable_count + 1e-6)
                movable_prob[movable_count < min_rays] -= 2

                # Save it
                print('Saving')
                write_ply(movable_path,
                          [map_points, map_normals, movable_prob, movable_count],
                          ['x', 'y', 'z', 'nx', 'ny', 'nz', 'movable', 'counts'])

            all_movables_probs.append(movable_prob)


        # Combine movable probs from days
        all_movables_probs = np.stack(all_movables_probs, 0)
        all_movables_probs = np.max(all_movables_probs, axis=0)
        # write_ply(join(map_folder, 'last_movables_combined.ply'),
        #           [map_points, map_normals, all_movables_probs],
        #           ['x', 'y', 'z', 'nx', 'ny', 'nz', 'movable'])
        print('\n    > Done')


        ##################
        # Save refined map
        ##################


        print('\n----- Save refined map')

        # Get flat ground
        ground_mask = extract_flat_ground(map_points,
                                          dist_thresh=0.25,
                                          remove_dist=0.24)

        # Remove movable except on ground
        still_mask = np.logical_and(all_movables_probs > -0.1, all_movables_probs < occup_threshold)
        still_mask = np.logical_or(still_mask, ground_mask)
        still_mask = np.logical_and(still_mask, map_points[:, 0] > lim_box.x1)
        still_mask = np.logical_and(still_mask, map_points[:, 0] < lim_box.x2)
        still_mask = np.logical_and(still_mask, map_points[:, 1] > lim_box.y1)
        still_mask = np.logical_and(still_mask, map_points[:, 1] < lim_box.y2)
        still_mask = np.logical_and(still_mask, map_points[:, 2] > min_z)
        still_mask = np.logical_and(still_mask, map_points[:, 2] < max_z)
        map_points = map_points[still_mask]
        map_normals = map_normals[still_mask]
        map_scores = map_scores[still_mask]
        all_movables_probs = all_movables_probs[still_mask]

        # Now get ground again without movable
        ground_mask = extract_flat_ground(map_points,
                                          dist_thresh=0.25,
                                          remove_dist=0.24)

        # This time remove map_points with the refined ground
        still_mask = np.logical_and(all_movables_probs > -0.1, all_movables_probs < occup_threshold)
        still_mask = np.logical_or(still_mask, ground_mask)
        map_points = map_points[still_mask]
        map_normals = map_normals[still_mask]
        map_scores = map_scores[still_mask]
        all_movables_probs = all_movables_probs[still_mask]
        ground_mask = ground_mask[still_mask]

        # make the ground map_points flat
        map_points[ground_mask, 2] = 0

        # Annotate everything as still except ground (rest of the map_points will be removed)
        annots = np.zeros(all_movables_probs.shape, dtype=np.int32) + 2
        annots[ground_mask] = 1

        filename = join(map_folder, 'map_update_{:04}.ply'.format(last_update_i + 1))
        write_ply(filename,
                  [map_points, map_normals, map_scores, annots],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'classif'])



        print('\n    > Done')

        return

    def collision_annotation(self, dl_2D=0.03, start_T=-1.0, future_T=5.0, noise_margin=0.3, pepper_margin=0.06, debug_noise=True):

        ###############
        # STEP 0 - Init
        ###############
            
        print('\n')
        print('------------------------------------------------------------------------------')
        print('\n')
        print('Start SOGM generation')
        print('*********************')
        print('\nInitial map run:', self.map_day)
        print('\nAnnotated runs:')
        for d, day in enumerate(self.days):
            print(' >', day)
        print('')

        # # Classes
        # label_names = {0: 'uncertain',
        #                1: 'ground',
        #                2: 'still',
        #                3: 'longT',
        #                4: 'shortT'}

                        
        # # Folder where the incrementally updated map is stored
        # map_folder = join(self.data_path, 'slam_offline', self.map_day)

        # # List of the updated maps
        # map_names = [f for f in listdir(map_folder) if f.startswith('map_update_')]

        # # Get the latest update of the map
        # map_names = np.sort(map_names)
        # last_map = map_names[-1]
        # last_update_i = int(last_map[:-4].split('_')[-1])

        # # Load map
        # data = read_ply(join(map_folder, last_map))
        # map_points = np.vstack((data['x'], data['y'], data['z'])).T
        # map_normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

        # # Get ground
        # vertical_angle = np.arccos(np.abs(np.clip(map_normals[:, 2], -1.0, 1.0)))
        # plane_mask = vertical_angle < 10.0 * np.pi / 180
        # plane_P, plane_N, _ = RANSAC(map_points[plane_mask], threshold_in=0.1)
        # ground_plane = np.append(plane_N, np.dot(plane_N, plane_P))

        ##############
        # LOOP ON DAYS
        ##############

        fps = 0
        fps_regu = 0.9
        last_t = time.time()

        # Get remove point form each day independently
        # Otherwise if a table is there in only one day, it will not be removed.
        for d, day in enumerate(self.days):

            print('\n----- Collisions detection day {:s}'.format(day))

            ####################
            # Step 1: Load stuff
            ####################

            # Annot folder
            annot_folder = join(self.data_path, 'annotation', day)
            frame_folder = join(self.data_path, 'annotated_frames', day)

            # Out folder
            out_folder = join(self.data_path, 'collisions', day)
            if not exists(out_folder):
                makedirs(out_folder)
            noisy_folder = join(self.data_path, 'noisy_collisions', day)
            if debug_noise and not exists(noisy_folder):
                makedirs(noisy_folder)


            # Load poses
            f_names = self.day_f_names[d]
            map_t = np.array([np.float64(f.split('/')[-1][:-4]) for f in f_names], dtype=np.float64)

            # Verify if collisions are already computed
            computed_colli = [f for f in listdir(out_folder) if f.endswith('_2D.ply')]
            if len(computed_colli) == len(f_names):
                print('    > Done (recovered from previous file)')
                continue

            cpp_traj_name = join(annot_folder, 'correct_traj_{:s}.pkl'.format(day))
            with open(cpp_traj_name, 'rb') as f:
                map_H = pickle.load(f)

            # Verify which frames we need:
            frame_names = []
            f_name_i = 0
            last_t = map_t[0] - 0.1
            remove_inds = []
            for i, t in enumerate(map_t):

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
            map_t = np.delete(map_t, remove_inds, axis=0)
            map_H = np.delete(map_H, remove_inds, axis=0)

            # Load the annotated map
            annot_name = join(annot_folder, 'annotated_{:s}.ply'.format(day))
            data = read_ply(annot_name)
            day_points = np.vstack((data['x'], data['y'], data['z'])).T
            categories = data['classif']

            # Get the ground plane with RANSAC
            plane_P, plane_N, _ = RANSAC(day_points[categories == 1, :], threshold_in=0.1)

            # Load traj
            cpp_traj_name = join(annot_folder,
                                 'correct_traj_{:s}.pkl'.format(day))
            with open(cpp_traj_name, 'rb') as f:
                correct_H = pickle.load(f)


            ##############################
            # Step 3: Collisions per frame
            ##############################
            

            # Debug folder
            debug_n = 50
            debug_folder = join(self.data_path, 'debug_colli', day)
            if not exists(debug_folder):
                makedirs(debug_folder)

            # First convert the full annotation map to 2D for noise removal
            flat_day_pts = day_points[categories > 1.5, :]
            flat_day_pts[:, 2] *= 0
            flat_day_annots = categories[categories > 1.5]

            # Subsampling to a 2D PointCloud
            day_pts_2D, day_annots_2D = grid_subsampling(flat_day_pts,
                                                         labels=flat_day_annots,
                                                         sampleDl=dl_2D)

            # Here save as 2D slices
            annot_2D_name = join(annot_folder, 'flat_annot_{:s}.ply'.format(day))
            write_ply(annot_2D_name,
                      [day_pts_2D[:, :2], day_annots_2D],
                      ['x', 'y', 'classif'])
            
            # Prepare data for the noise removal
            day_static_pts = day_pts_2D[day_annots_2D.squeeze() < 3.5]
            day_static_tree = KDTree(day_static_pts)

            pts_FIFO = []
            annot_FIFO = []
            name_FIFO = []

            # Create KDTree on the map
            N = len(frame_names)
            for i, f_name in enumerate(frame_names):

                t = [time.time()]

                # Load points
                data = read_ply(f_name)
                f_points = np.vstack((data['x'], data['y'], data['z'])).T
                f_ts = data['time']

                # Load annot
                annot_name = join(frame_folder, f_name.split('/')[-1])
                data = read_ply(annot_name)
                f_annot = data['classif']

                # Apply transform with motion distorsion
                if (i < 1):
                    H0 = correct_H[i]
                    H1 = correct_H[i]
                else:
                    H0 = correct_H[i - 1]
                    H1 = correct_H[i]
                world_pts = motion_rectified(f_points, f_ts, H0, H1)

                # Mask of the future points
                f_mask = f_annot > 1.5
                
                # Do not use ground and uncertain points
                flat_pts = world_pts[f_mask, :]
                flat_annot = f_annot[f_mask]
                flat_ts = f_ts[f_mask]
                flat_pts[:, 2] *= 0
                
                # third coordinate is now time
                flat_pts[:, 2] = flat_ts

                # Safe check
                if (np.max(flat_ts) - np.min(flat_ts) < 2 * dl_2D):
                    raise ValueError('Problem, when subsampling 2D collision points: dT < 2 * dl. Use scaling to correct this')

                # Subsampling to a 2D PointCloud
                pts_2D, annots_2D = grid_subsampling(flat_pts,
                                                     labels=flat_annot,
                                                     sampleDl=dl_2D)
                                                     
                # mask of movables and dynamic
                static_mask = np.squeeze(np.logical_or(annots_2D == 3, annots_2D == 2))
                dynamic_mask = np.squeeze(annots_2D == 4)

                # Apply first noise removal with whole day static points
                
                # Like image binary opening but on sparse point positions in 3D
                # 1. negatives "eat" positives
                # 2. Remaining positives "eat back"
                opened_dyn_mask = sparse_point_opening(pts_2D,
                                                       dynamic_mask,
                                                       negative_tree=day_static_tree,
                                                       erode_d=noise_margin,
                                                       dilate_d=noise_margin - dl_2D)

                

                # Remove noise (dynamic points among static points)
                invalid_mask = np.logical_and(dynamic_mask, np.logical_not(opened_dyn_mask))
                valid_mask = np.logical_not(invalid_mask)
                pts_2D = pts_2D[valid_mask]
                annots_2D = annots_2D[valid_mask]
                dynamic_mask = dynamic_mask[valid_mask]

                # Second noise removal via erosion.
                # 1. Convert to image
                # 2. Perform image opening
                if np.any(dynamic_mask):
                    clean_mask = pepper_noise_removal(pts_2D,
                                                      sampleDl=dl_2D,
                                                      pepper_margin=pepper_margin,
                                                      positive_mask=dynamic_mask)

                    # Remove noise (dynamic points among static points)
                    invalid_mask = np.logical_and(dynamic_mask, np.logical_not(clean_mask))
                    valid_mask = np.logical_not(invalid_mask)
                    clean_pts_2D = pts_2D[valid_mask]
                    clean_annots_2D = annots_2D[valid_mask]

                # TODO: DO the closing / opening on 3 layers:
                #       1) This will eliminate large areas that diaspear right after
                #       2) This should help to better keep people because the blob is continuous in the 3 layers
                #
                #   OR, maybe using tracking of blob for this????? OR maybe the tracking should be after this noise removal???
                #
                # TODO: When choosing training examples, create a mask of valid training inds, (similar to balance_class)
                #       The valid frames are the one with more than 10 dynamic points and not isolated (the prev / next X (X=4?) frames 
                #       also have more than 10 dynamic points)
                #


                # Here save as 2D slices
                if debug_noise:
                    ply_debug_name = join(noisy_folder, f_name[:-4].split('/')[-1] + '_2D.ply')
                    annot_noise = np.copy(annots_2D)
                    annot_noise[invalid_mask] = 1
                    write_ply(ply_debug_name,
                              [pts_2D, annot_noise],
                              ['x', 'y', 't', 'classif'])

                # Here save as 2D slices
                ply_2D_name = join(out_folder, f_name[:-4].split('/')[-1] + '_2D.ply')
                write_ply(ply_2D_name,
                          [clean_pts_2D, clean_annots_2D],
                          ['x', 'y', 't', 'classif'])

                # # Get visibility
                # lidar_ranges = get_lidar_visibility(world_pts,
                #                                     correct_H[i][:3, 3].astype(np.float32),
                #                                     ground_plane,
                #                                     n_angles=720,
                #                                     z_min=0.4,
                #                                     z_max=1.5,
                #                                     dl_2D=0.12)


                # print(lidar_ranges.shape)
                # print(lidar_ranges.dtype)
                # print(np.min(lidar_ranges), np.max(lidar_ranges))
                # import matplotlib.pyplot as plt
                # plt.imshow(lidar_ranges)
                # plt.show()
                # a = 1/0

                # Add 2D points to FIFO (for visu purposes)
                pts_FIFO.append(pts_2D)
                annot_FIFO.append(annots_2D)
                name_FIFO.append(f_name[:-4].split('/')[-1])
                if float(name_FIFO[-1]) - float(name_FIFO[0]) > future_T - start_T:

                    # Get origin time
                    t_orig = float(name_FIFO[0]) - start_T
                    ind_orig = np.argmin([np.abs(float(future_name) - t_orig) for future_name in name_FIFO])

                    # Stack all point with timestamps (only if object is seen in first frame)
                    stacked_pts = np.zeros((0, pts_FIFO[0].shape[1]), pts_FIFO[0].dtype)
                    stacked_annot = np.zeros((0, annot_FIFO[0].shape[1]), annot_FIFO[0].dtype)
                    for future_pts, future_name, future_annot in zip(pts_FIFO, name_FIFO, annot_FIFO):
                        new_points = np.copy(future_pts)
                        new_points[:, 2] += float(future_name) - float(name_FIFO[ind_orig])
                        stacked_pts = np.vstack((stacked_pts, new_points))
                        stacked_annot = np.vstack((stacked_annot, future_annot))

                    # Save as a 2D point cloud
                    if i % debug_n == debug_n - 1:
                        ply_2D_name = join(debug_folder, name_FIFO[ind_orig] + '_stacked.ply')
                        write_ply(ply_2D_name,
                                  [stacked_pts, stacked_annot],
                                  ['x', 'y', 't', 'classif'])

                    # Get rid of oldest points
                    pts_FIFO.pop(0)
                    annot_FIFO.pop(0)
                    name_FIFO.pop(0)

                if i % debug_n == debug_n - 1:

                    ply_name = join(debug_folder, f_name.split('/')[-1])
                    write_ply(ply_name,
                              [world_pts, f_annot],
                              ['x', 'y', 'z', 'classif'])

                # Timing
                t += [time.time()]
                fps = fps_regu * fps + (1.0 - fps_regu) / (t[-1] - t[0])

                if (t[-1] - last_t > 5.0):
                    print('Collisions {:s} {:5d} --- {:5.1f}%% at {:.1f} fps'.format(day,
                                                                                     i + 1,
                                                                                     100 * (i + 1) / N,
                                                                                     fps))
                    last_t = t[-1]


            print('    > Done (recovered from previous file)')
            
        print('\n')
        print('  +-----------------------------------+')
        print('  | Finished all the annotation tasks |')
        print('  +-----------------------------------+')
        print('\n')
        return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class MyhalCollisionDataset(PointCloudDataset):
    """Class to handle MyhalCollision dataset."""

    def __init__(self,
                 config,
                 day_list,
                 chosen_set='training',
                 dataset_path='../Data/Real',
                 balance_classes=True,
                 add_sim_path='',
                 add_sim_days=None,
                 load_data=True):
        PointCloudDataset.__init__(self, 'MyhalCollision')

        ##########################
        # Parameters for the files
        ##########################

        # Dataset folder
        self.path = '../Data/KPConv_data'

        # Original data path
        self.original_path = dataset_path

        # Type of task conducted on this dataset
        self.dataset_task = 'collision_prediction'

        # Training or test set
        self.set = chosen_set

        # Get a list of sequences
        if self.set == 'training':
            self.sequences = day_list
        elif self.set == 'validation':
            self.sequences = day_list
        elif self.set == 'test':
            self.sequences = day_list
        else:
            raise ValueError('Unknown set for MyhalCollision data: ', self.set)

        # List all files in each sequence
        self.seq_path = []
        self.colli_path = []
        self.annot_path = []
        self.frames = []
        for seq in self.sequences:

            self.seq_path.append(join(self.original_path, 'runs', seq, 'velodyne_frames'))
            self.colli_path.append(join(self.original_path, 'collisions', seq))
            self.annot_path.append(join(self.original_path, 'annotated_frames', seq))

            frames = np.array([vf[:-4] for vf in listdir(self.seq_path[-1]) if vf.endswith('.ply')])
            order = np.argsort([float(ff) for ff in frames])
            frames = frames[order]
            self.frames.append(frames)
        


        ############################
        # Additional simulation data
        ############################
        
        if add_sim_path and add_sim_days is not None:

            # Simulation folder
            self.sim_path = add_sim_path

            # Specify which seq is simulation
            self.sim_sequences = [False for _ in self.sequences]

            # Add simulation sequences
            self.sequences = np.hstack((self.sequences, add_sim_days)) 
            self.sim_sequences += [True for _ in add_sim_days]

            # Add sim files
            for seq in add_sim_days:

                self.seq_path.append(join(self.sim_path, 'simulated_runs', seq, 'sim_frames'))
                self.colli_path.append(join(self.sim_path, 'collisions', seq))
                self.annot_path.append(join(self.sim_path, 'annotated_frames', seq))

                frames = np.array([vf[:-7] for vf in listdir(self.colli_path[-1]) if vf.endswith('2D.ply')])
                order = np.argsort([float(ff) for ff in frames])
                frames = frames[order]
                self.frames.append(frames)
               
        else:
            
            self.sim_path = ''
            self.sim_sequences = [False for _ in self.sequences]

        self.sim_sequences = np.array(self.sim_sequences)

        ###########################
        # Object classes parameters
        ###########################

        # Read labels
        config_file = join(self.path, 'myhal_sym.yaml')

        with open(config_file, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            learning_map_inv = doc['learning_map_inv']
            learning_map = doc['learning_map']
            self.learning_map = np.zeros(
                (np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map.items():
                self.learning_map[k] = v

            self.learning_map_inv = np.zeros(
                (np.max([k for k in learning_map_inv.keys()]) + 1),
                dtype=np.int32)
            for k, v in learning_map_inv.items():
                self.learning_map_inv[k] = v

        # Dict from labels to names
        self.label_to_names = {k: all_labels[v] for k, v in learning_map_inv.items()}

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.sort([0])

        ##################
        # Other parameters
        ##################

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        if not load_data:
            return

        ##################
        # Load calibration
        ##################

        # Init variables
        self.poses = []
        self.all_inds = np.zeros((0, ))
        self.class_proportions = None
        self.class_frames = []
        self.val_confs = []

        # Load everything
        self.load_calib_poses()

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize frame potentials
        self.potentials = torch.from_numpy(np.random.rand(self.all_inds.shape[0]) * 0.1 + 0.1)
        self.potentials.share_memory_()

        # If true, the same amount of frames is picked per class
        self.balance_classes = balance_classes

        # Choose batch_num in_R and max_in_p depending on validation or training
        if self.set == 'training':
            self.batch_num = config.batch_num
            self.max_in_p = config.max_in_points
            self.in_R = config.in_radius
        else:
            self.batch_num = config.val_batch_num
            self.max_in_p = config.max_val_points
            self.in_R = config.val_radius

        # shared epoch indices and classes (in case we want class balanced sampler)
        if self.set == 'training':
            N = int(np.ceil(config.epoch_steps * self.batch_num * 1.1))
        else:
            N = int(np.ceil(config.validation_size * self.batch_num * 1.1))
        self.epoch_i = torch.from_numpy(np.zeros((1, ), dtype=np.int64))
        self.epoch_inds = torch.from_numpy(np.zeros((N, ), dtype=np.int64))
        self.epoch_labels = torch.from_numpy(np.zeros((N, ), dtype=np.int32))
        self.epoch_i.share_memory_()
        self.epoch_inds.share_memory_()
        self.epoch_labels.share_memory_()

        self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
        self.worker_waiting.share_memory_()
        self.worker_lock = Lock()

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.frames)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        pl2D_list = []
        img_list = []
        f_list = []
        l_list = []
        fi_list = []
        p0_list = []
        s_list = []
        R_list = []
        r_inds_list = []
        r_mask_list = []
        val_labels_list = []
        batch_n = 0

        while True:

            t += [time.time()]

            with self.worker_lock:

                # Get potential minimum
                ind = int(self.epoch_inds[self.epoch_i])
                wanted_label = int(self.epoch_labels[self.epoch_i])

                # Update epoch indice
                self.epoch_i += 1
                if self.epoch_i >= int(self.epoch_inds.shape[0]):
                    self.epoch_i -= int(self.epoch_inds.shape[0])

            s_ind, f_ind = self.all_inds[ind]

            t += [time.time()]

            # Verify that we have enough input frames and future frames
            if f_ind - self.config.n_frames < 0:
                continue
            if f_ind + self.config.n_2D_layers + 1 >= len(self.frames[s_ind]):
                continue

            #########################
            # Merge n_frames together
            #########################

            # Initiate merged points
            merged_points = np.zeros((0, 3), dtype=np.float32)
            merged_labels = np.zeros((0, ), dtype=np.int32)
            merged_coords = np.zeros((0, self.config.n_frames + 1), dtype=np.float32)

            # Get center of the first frame in world coordinates
            p_origin = np.zeros((1, 4))
            p_origin[0, 3] = 1
            pose0 = self.poses[s_ind][f_ind]
            # pose0_inv = np.linalg.inv(pose0)
            p0 = p_origin.dot(pose0.T)[:, :3]
            p0 = np.squeeze(p0)
            o_pts = None
            o_labels = None

            t += [time.time()]

            num_merged = 0
            while num_merged < self.config.n_frames and f_ind - num_merged >= 0:

                # Current frame pose
                merge_ind = f_ind - num_merged
                if self.sim_sequences[s_ind] or (merge_ind < 1):
                    H0 = self.poses[s_ind][merge_ind]
                    H1 = self.poses[s_ind][merge_ind]
                else:
                    H0 = self.poses[s_ind][merge_ind - 1]
                    H1 = self.poses[s_ind][merge_ind]
                # H1_inv = np.linalg.inv(H1)

                # Load points
                data = read_ply(join(self.seq_path[s_ind], self.frames[s_ind][merge_ind] + '.ply'))
                f_points = np.vstack((data['x'], data['y'], data['z'])).T
                if self.sim_sequences[s_ind]:
                    # Apply simple transform
                    f_ts = np.arange(data['x'].shape[0], dtype=np.float32)
                    world_points = motion_rectified(f_points, f_ts, H0, H1)
                else:
                    # Apply transform with motion distorsion
                    f_ts = data['time']
                    world_points = motion_rectified(f_points, f_ts, H0, H1)

                # Load annot
                if self.set == 'test':
                    # Fake labels
                    sem_labels = np.zeros((f_points.shape[0],), dtype=np.int32)

                else:
                    data = read_ply(join(self.annot_path[s_ind], self.frames[s_ind][merge_ind] + '.ply'))
                    sem_labels = data['classif']
                        
                # In case of validation, keep the points in memory
                if self.set in ['validation', 'test'] and num_merged == 0:
                    o_pts = world_points.astype(np.float32)
                    o_labels = sem_labels.astype(np.int32)

                # In case radius smaller than 5m, chose new center on a point of the wanted class or not
                if self.in_R < 5.0 and num_merged == 0:
                    if self.balance_classes:
                        wanted_ind = np.random.choice(np.where(sem_labels == wanted_label)[0])
                    else:
                        wanted_ind = np.random.choice(f_points.shape[0])
                    p0 = world_points[wanted_ind, :3]

                # Eliminate points further than config.in_radius
                mask = np.sum(np.square(world_points - p0), axis=1) < self.in_R**2
                mask_inds = np.where(mask)[0].astype(np.int32)

                # Shuffle points
                rand_order = np.random.permutation(mask_inds)
                world_points = world_points[rand_order, :3]
                sem_labels = sem_labels[rand_order]

                # Stack features
                features = np.zeros((world_points.shape[0], self.config.n_frames + 1), dtype=np.float32)
                features[:, num_merged] = 1
                features[:, -1] = num_merged

                # Increment merge count
                merged_points = np.vstack((merged_points, world_points))
                merged_labels = np.hstack((merged_labels, sem_labels))
                merged_coords = np.vstack((merged_coords, features))
                num_merged += 1

            t += [time.time()]

            ###################
            # Data Augmentation
            ###################

            # Then center on p0
            merged_points_c = (merged_points - p0).astype(np.float32)

            # Too see yielding speed with debug timings method, collapse points (reduce mapping time to nearly 0)
            #merged_points = merged_points[:100, :]
            #merged_labels = merged_labels[:100]
            #merged_points *= 0.1

            # Subsample merged frames
            in_pts, in_fts, in_lbls = grid_subsampling(merged_points_c,
                                                       features=merged_coords,
                                                       labels=merged_labels,
                                                       sampleDl=self.config.first_subsampling_dl)

            t += [time.time()]

            # Number collected
            n = in_pts.shape[0]

            # Safe check
            if n < 2:
                continue

            # Randomly drop some points (augmentation process and safety for GPU memory consumption)
            if n > self.max_in_p:
                input_inds = np.random.choice(n, size=self.max_in_p, replace=False)
                in_pts = in_pts[input_inds, :]
                in_fts = in_fts[input_inds, :]
                in_lbls = in_lbls[input_inds]
                n = input_inds.shape[0]

            t += [time.time()]

            # Before augmenting, compute reprojection inds (only for validation and test)
            if self.set in ['validation', 'test']:

                # get val_points that are in range
                o_pts_c = o_pts - p0
                reproj_mask = np.sum(np.square(o_pts_c), axis=1) < (0.99 * self.in_R)**2

                # Project predictions on the frame points
                search_tree = KDTree(in_pts, leaf_size=50)
                proj_inds = search_tree.query(o_pts_c[reproj_mask, :],
                                              return_distance=False)
                proj_inds = np.squeeze(proj_inds).astype(np.int32)

            else:
                proj_inds = np.zeros((0, ))
                reproj_mask = np.zeros((0, ))

            t += [time.time()]

            # Data augmentation
            in_pts, scale, R = self.augmentation_transform(in_pts)

            t += [time.time()]

            ##########################
            # Compute 3D-2D projection
            ##########################

            # Compute projection indices and future images
            pools_2D, future_imgs = self.get_input_2D(in_pts, in_fts, in_lbls, s_ind, f_ind, p0, R, scale)
            #pools_2D = self.get_input_2D_cpp()

            # Check if Failed (probably because the cloud had 0 points)
            if pools_2D is None:
                continue

            # Get current visible pixels in the future images
            if self.config.use_visibility:
                future_visible_mask = self.get_future_visibility(in_pts, in_fts, in_lbls, s_ind, f_ind, p0, R, scale)
            else:
                future_visible_mask = None

            # Stack batch
            p_list += [in_pts]
            pl2D_list += [pools_2D.astype(np.int64)]
            img_list += [future_imgs]
            f_list += [in_fts]
            l_list += [np.squeeze(in_lbls)]
            fi_list += [[s_ind, f_ind]]
            p0_list += [p0]
            s_list += [scale]
            R_list += [R]
            r_inds_list += [proj_inds]
            r_mask_list += [reproj_mask]
            val_labels_list += [o_labels]

            t += [time.time()]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

        ###################
        # Concatenate batch
        ###################

        # First adjust pools_2D for batch pooling
        batch_N = np.sum([p.shape[0] for p in p_list])
        batch_n = 0
        for b_i, pl2D in enumerate(pl2D_list):
            mask = pl2D == p_list[b_i].shape[0]
            pl2D[mask] = batch_N
            pl2D[np.logical_not(mask)] += batch_n
            batch_n += p_list[b_i].shape[0]
        stacked_pools_2D = np.stack(pl2D_list, axis=0)

        # Concatenate the rest of the batch
        stacked_points = np.concatenate(p_list, axis=0)
        stacked_imgs = np.stack(img_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        frame_inds = np.array(fi_list, dtype=np.int32)
        frame_centers = np.stack(p0_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list],
                                 dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features (Use reflectance, input height or all coordinates)
        stacked_features = np.ones_like(stacked_points[:, :1],
                                        dtype=np.float32)

        if self.config.in_features_dim == 1:
            pass

        elif self.config.in_features_dim == self.config.n_frames:
            # Use the frame indicators
            stacked_features = features[:, :self.config.n_frames]

        elif self.config.in_features_dim == 3:
            # Use only the three frame indicators
            stacked_features = features[:, :3]

        elif self.config.in_features_dim == 4:
            # Use the ones + the three frame indicators
            stacked_features = np.hstack((stacked_features, features[:, :3]))

        else:
            raise ValueError('Only accepted input dimensions are 1, 2 and 4 (without and with XYZ)')

        t += [time.time()]

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points, stack_lengths)

        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [stacked_imgs, stacked_pools_2D]
        input_list += [stacked_features, labels.astype(np.int64)]
        input_list += [scales, rots, frame_inds, frame_centers, r_inds_list, r_mask_list, val_labels_list]

        t += [time.time()]

        # Display timings
        debugT = False
        if debugT:
            print('\n************************\n')
            print('Timings:')
            ti = 0
            N = 9
            mess = 'Init ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Lock ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Init ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Load ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Subs ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Drop ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Reproj .... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Augment ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Stack ..... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += N * (len(stack_lengths) - 1) + 1
            print('concat .... {:5.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print('input ..... {:5.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print('stack ..... {:5.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print('\n************************\n')

            # Timings: (in test configuration)
            # Lock ...... 0.1ms
            # Init ...... 0.0ms
            # Load ...... 40.0ms
            # subs ...... 143.6ms
            # drop ...... 4.6ms
            # reproj .... 297.4ms
            # augment ... 7.5ms
            # stack ..... 0.0ms
            # concat .... 1.4ms
            # input ..... 816.0ms
            # stack ..... 0.0ms

            # TODO: Where can we gain time for the robot real time test?
            #  > Load: no disk read necessary + pose useless if we only use one frame for testing
            #  > Drop: We can drop even more points. Random choice could be faster without replace=False
            #  > reproj: No reprojection needed
            #  > Augment: See which data agment we want at test time
            #  > input: MAIN BOTTLENECK. We need to see if we can do faster, maybe with some parallelisation. neighbors
            #           and subsampling accelerated with lidar frame order

        return [self.config.num_layers] + input_list

    def get_input_2D(self, in_pts, in_fts, in_lbls, s_ind, f_ind, p0, R, scale):

        # C++ wrappers to get the projections indexes (with shadow pools)
        # Temporarily use the 3D neighbors wrappers

        # Max number of points pooled to a grid cell
        max_2D_pools = 20

        # Project points on 2D plane
        support_pts = np.copy(in_pts)
        support_pts[:, 2] *= 0

        # Create grid
        grid_ticks = np.arange(-self.config.in_radius / np.sqrt(2),
                               self.config.in_radius / np.sqrt(2),
                               self.config.dl_2D)
        xx, yy = np.meshgrid(grid_ticks, grid_ticks)
        L_2D = xx.shape[0]
        pool_points = np.vstack((np.ravel(xx), np.ravel(yy), np.ravel(yy) * 0)).astype(np.float32).T

        # Get pooling indices
        pool_inds = batch_neighbors(pool_points,
                                    support_pts,
                                    [pool_points.shape[0]],
                                    [support_pts.shape[0]],
                                    self.config.dl_2D / np.sqrt(2))

        # Remove excedent => get to shape [L_2D*L_2D, max_2D_pools]
        if pool_inds.shape[1] < max_2D_pools:
            diff_d = max_2D_pools - pool_inds.shape[1]
            pool_inds = np.pad(pool_inds,
                               ((0, 0), (0, diff_d)),
                               'constant',
                               constant_values=support_pts.shape[0])

        else:
            pool_inds = pool_inds[:, :max_2D_pools]

        # Reshape into 2D grid
        pools_2D = np.reshape(pool_inds, (L_2D, L_2D, max_2D_pools))

        #########################
        # Load groundtruth future
        #########################

        # Path of points and labels
        if self.set == 'test':
            future_imgs = np.zeros((0, 0, 0), dtype=np.float32)
        else:
            
            # Verify time are synchronized between frames and predictions
            future_dt = self.config.T_2D / self.config.n_2D_layers
            frame_dt = 0.1


            # Assertion not true for simulation => Do that assertion once on a mean in the beginining
            # assert(abs(future_dt - (float(self.frames[s_ind][f_ind]) - float(self.frames[s_ind][f_ind - 1]))) < 0.03)

            # In case of simulation, the whole data is already stacked
            if self.sim_sequences[s_ind]:

                # Get groundtruth in 2D points format
                gt_file = join(self.colli_path[s_ind], self.frames[s_ind][f_ind] + '_2D.ply')

                # Read points
                data = read_ply(gt_file)
                pts_2D = np.vstack((data['x'], data['y'])).T
                times_2D = data['t']
                labels_2D = data['classif']

            else:

                # We load one more before and after to be sure to have all data
                pts_2D = []
                times_2D = []
                labels_2D = []
                orig_t = float(self.frames[s_ind][f_ind])
                i_2D = - self.config.n_frames
                f_t0 = -1
                while (f_t0 < self.config.T_2D):

                    # We need too far into the future, invalid input
                    if (f_ind + i_2D >= self.frames[s_ind].shape[0]):
                        return None, None

                    # Get groundtruth in 2D points format
                    future_name = self.frames[s_ind][f_ind + i_2D]
                    gt_file = join(self.colli_path[s_ind], future_name + '_2D.ply')

                    # Read points
                    data = read_ply(gt_file)
                    pts_2D.append(np.vstack((data['x'], data['y'])).T)
                    labels_2D.append(data['classif'])

                    # Handle time
                    f_t0 = float(future_name) - orig_t
                    times_2D.append(data['t'] + f_t0)

                    i_2D += 1

                pts_2D = np.vstack(pts_2D)
                times_2D = np.hstack(times_2D)
                labels_2D = np.hstack(labels_2D)

            # Center on p0 and apply same augmentation
            pts_2D = (pts_2D - p0[:2]).astype(np.float32)
            pts_2D = np.hstack((pts_2D, np.zeros_like(pts_2D[:, :1])))
            pts_2D = np.sum(np.expand_dims(pts_2D, 2) * R, axis=1) * scale

            # List the timestamps we want for each SOGM layers
            init_stamps = (np.arange(self.config.n_frames).astype(np.float32) - (self.config.n_frames - 1)) * frame_dt
            future_stamps = np.arange(future_dt, self.config.T_2D + 0.1 * future_dt, future_dt)
            timestamps = np.hstack((init_stamps, future_stamps))

            # For each time get the closest annotation
            future_imgs = []
            try:
                for future_t in timestamps:

                    # Valid points for this timestamps are in the time range dt/2
                    # TODO: Here different valid times for different classes
                    valid_mask = np.abs(times_2D - future_t) < frame_dt / 2
                    extension = 1
                    while np.sum(valid_mask) < 1 and extension < 5:
                        extension += 1
                        valid_mask = np.abs(times_2D - future_t) < frame_dt * extension / 2

                    valid_pts = pts_2D[valid_mask, :]
                    valid_labels = labels_2D[valid_mask]
                    # valid_times = times_2D[valid_mask]

                    # Get pooling indices to image
                    pool2D_inds = batch_neighbors(pool_points,
                                                  valid_pts,
                                                  [pool_points.shape[0]],
                                                  [valid_pts.shape[0]],
                                                  self.config.dl_2D / np.sqrt(2))

                    # Pool labels (shape = [L_2D*L_2D, max_neighb])
                    valid_labels = np.hstack((valid_labels, np.ones_like(valid_labels[:1] * -1)))
                    future_labels = valid_labels[pool2D_inds]
                    future_2 = np.sum((future_labels == self.name_to_label['still']).astype(np.float32), axis=1)
                    future_3 = np.sum((future_labels == self.name_to_label['longT']).astype(np.float32), axis=1)
                    future_4 = np.sum((future_labels == self.name_to_label['shortT']).astype(np.float32), axis=1)

                    # Reshape into 2D grid
                    future_2 = np.reshape(future_2, (L_2D, L_2D))
                    future_3 = np.reshape(future_3, (L_2D, L_2D))
                    future_4 = np.reshape(future_4, (L_2D, L_2D))

                    # Append
                    future_imgs.append(np.stack((future_2, future_3, future_4), axis=2))

            except RuntimeError as e:
                # Temporary bug fix when no neighbors at all we just skip this one
                print('------------------')
                print('ERROR in the future groundtruth generation. either a frame is missing or timings are not synchronized correctly. Error message:')
                print(e)
                print('------------------')
                return None, None

            # Stack future images
            future_imgs = np.stack(future_imgs, 0)

            # For permanent and long term objects, merge all futures
            future_imgs[:, :, :, 0] = np.sum(future_imgs[:, :, :, 0], axis=0, keepdims=True) / (self.config.n_2D_layers + self.config.n_frames)
            future_imgs[:, :, :, 1] = np.sum(future_imgs[:, :, :, 1], axis=0, keepdims=True) / (self.config.n_2D_layers + self.config.n_frames)

            # Hardcoded value of 10 shortT points
            future_imgs[:, :, :, 2] = np.clip(future_imgs[:, :, :, 2], 0, 10) / 10

            # Normalize all class future
            for i in range(future_imgs.shape[-1]):
                if np.max(future_imgs[:, :, :, i]) > 1.0:
                    future_imgs[:, :, :, i] = future_imgs[:, :, :, i] / (np.max(future_imgs[:, :, :, i]) + 1e-9)

            # Get in range [0.5, 1] instead of [0, 1]
            f_mask = future_imgs > 0.01
            future_imgs[f_mask] = 0.5 * (future_imgs[f_mask] + 1.0)

            input_classes = np.sum(future_imgs[0, :, :, :], axis=(0, 1)) > 0

            ###########################################################################################
            #DEBUG
            debug = self.config.input_threads == 0  # and False
            if debug:
                print('Precesnce of each input class: ', input_classes)
                debug = debug and np.all(input_classes)  # and 4 < s_ind < 8
            if debug:
                f1 = np.zeros_like(support_pts[:, 0])
                f1 = np.hstack((f1, np.zeros_like(f1[:1])))
                NN = 300
                rand_neighbs = np.random.choice(pool_inds.shape[0], size=NN)
                for rd_i in rand_neighbs:
                    f1[pool_inds[rd_i]] = rd_i

                print(support_pts.shape, f1.shape, in_fts.shape, in_lbls.shape)
                write_ply('results/t_supps.ply',
                          [support_pts, f1[:-1], in_fts[:, -1], in_lbls],
                          ['x', 'y', 'z', 'f1', 'f2', 'classif'])

                print(in_pts.shape, f1.shape, in_fts.shape, in_lbls.shape)
                write_ply('results/t_supps3D.ply',
                          [in_pts, f1[:-1], in_fts[:, -1], in_lbls],
                          ['x', 'y', 'z', 'f1', 'f2', 'classif'])

                n_neighbs = np.sum((pool_inds < support_pts.shape[0]).astype(np.float32), axis=1)
                print(pool_points.shape, n_neighbs.shape)
                write_ply('results/t_pools.ply',
                          [pool_points, n_neighbs],
                          ['x', 'y', 'z', 'n_n'])

                import matplotlib.pyplot as plt
                from matplotlib.animation import FuncAnimation
                pp = []
                for i in range(self.config.n_2D_layers + self.config.n_frames):
                    pool_points[:, 2] = timestamps[i]
                    pp.append(np.copy(pool_points))
                pp = np.concatenate(pp, axis=0)
                print(pp.shape, np.ravel(future_imgs[:, :, :, 2]).shape)
                write_ply('results/gt_pools.ply',
                          [pp, np.ravel(future_imgs[:, :, :, 2])],
                          ['x', 'y', 'z', 'f1'])

                print(pts_2D.shape, times_2D.shape, labels_2D.shape)
                write_ply('results/gt_pts.ply',
                          [pts_2D[:, :2], times_2D, labels_2D],
                          ['x', 'y', 't', 'f1'])

                # Function that saves future images as gif
                # fig1, anim = save_future_anim('results/gt_anim.gif', future_imgs)
                fast_save_future_anim('results/gt_anim.gif', future_imgs, zoom=5, correction=True)

                fig2, ax = plt.subplots()
                n_neighbs = np.sum((pools_2D < support_pts.shape[0]).astype(np.float32), axis=-1)
                n_neighbs = n_neighbs / max_2D_pools
                imgplot = plt.imshow(n_neighbs)
                plt.savefig('results/t_input_proj.png')
                plt.show()

        return pools_2D, future_imgs

    def get_future_visibility(self, in_pts, in_fts, in_lbls, s_ind, f_ind, p0, R, scale):

        # Get future poses
        future_poses = [self.poses[s_ind][f_ind + i] for i in range(self.config.n_2D_layers + 1)]


        


    


        return

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.poses = []

        if self.set in ['training', 'validation']:

            for s_ind, seq in enumerate(self.sequences):

                if not exists(join(self.path, seq)):
                    makedirs(join(self.path, seq))

                if self.sim_sequences[s_ind]:
                    data_path = self.sim_path
                else:
                    data_path = self.original_path

                in_folder = join(data_path, 'annotation', seq)
                in_file = join(in_folder, 'correct_traj_{:s}.pkl'.format(seq))
                with open(in_file, 'rb') as f:
                    transform_list = pickle.load(f)

                # Remove poses of ignored frames
                annot_path = join(data_path, 'annotated_frames', seq)
                annot_frames = np.array([vf[:-4] for vf in listdir(annot_path) if vf.endswith('.ply')])
                order = np.argsort([float(a_f) for a_f in annot_frames])
                annot_frames = annot_frames[order]
                pose_dict = {k: v for k, v in zip(annot_frames, transform_list)}
                self.poses.append([pose_dict[f] for f in self.frames[s_ind]])

        else:

            for s_ind, (seq, seq_frames) in enumerate(zip(self.sequences, self.frames)):
                dummy_poses = np.expand_dims(np.eye(4), 0)
                dummy_poses = np.tile(dummy_poses, (len(seq_frames), 1, 1))
                self.poses.append(dummy_poses)

        ###################################
        # Prepare the indices of all frames
        ###################################

        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.frames)])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.frames])
        self.all_inds = np.vstack((seq_inds, frame_inds)).T

        ################################################
        # For each class list the frames containing them
        ################################################

        if self.set in ['training', 'validation']:

            class_frames_bool = np.zeros((0, self.num_classes), dtype=np.bool)
            self.class_proportions = np.zeros((self.num_classes, ), dtype=np.int32)

            for s_ind, (seq, seq_frames) in enumerate(zip(self.sequences, self.frames)):

                frame_mode = 'movable'
                seq_stat_file = join(self.path, seq, 'stats_{:s}.pkl'.format(frame_mode))

                # Check if inputs have already been computed
                # if False and exists(seq_stat_file):
                if exists(seq_stat_file):
                    # Read pkl
                    with open(seq_stat_file, 'rb') as f:
                        seq_class_frames, seq_proportions = pickle.load(f)

                else:

                    # Initiate dict
                    print('Preparing seq {:s} class frames. (Long but one time only)'.format(seq))

                    # Class frames as a boolean mask
                    seq_class_frames = np.zeros((len(seq_frames), self.num_classes), dtype=np.bool)

                    # Proportion of each class
                    seq_proportions = np.zeros((self.num_classes, ), dtype=np.int64)

                    # Read all frames
                    for f_ind, frame_name in enumerate(seq_frames):

                        # Path of points and labels
                        velo_file = join(self.colli_path[s_ind], frame_name + '_2D.ply')

                        # Read labels
                        data = read_ply(velo_file)
                        sem_labels = data['classif']

                        # Special treatment to old simulation annotations
                        if self.sim_sequences[s_ind]:
                            sem_times = data['t']
                            sem_mask = np.logical_and(sem_times > -0.001, sem_times < 0.001)
                            sem_labels = sem_labels[sem_mask]
                            
                        # Get present labels and their frequency
                        unique, counts = np.unique(sem_labels, return_counts=True)

                        # Add this frame to the frame lists of all class present
                        frame_labels = np.array([self.label_to_idx[ll] for ll in unique], dtype=np.int32)
                        seq_class_frames[f_ind, frame_labels] = True

                        # Add proportions
                        seq_proportions[frame_labels] += counts

                    # Save pickle
                    with open(seq_stat_file, 'wb') as f:
                        pickle.dump([seq_class_frames, seq_proportions], f)

                class_frames_bool = np.vstack((class_frames_bool, seq_class_frames))
                self.class_proportions += seq_proportions

            # Transform boolean indexing to int indices.
            self.class_frames = []
            for i, c in enumerate(self.label_values):
                if c in self.ignored_labels:
                    self.class_frames.append(torch.zeros((0, ), dtype=torch.int64))
                else:
                    integer_inds = np.where(class_frames_bool[:, i])[0]
                    self.class_frames.append(torch.from_numpy(integer_inds.astype(np.int64)))

        # Add variables for validation
        if self.set in ['validation', 'test']:
            self.val_points = []
            self.val_labels = []
            self.val_confs = []
            self.val_2D_reconstruct = []
            self.val_2D_future = []

            for s_ind, seq_frames in enumerate(self.frames):
                self.val_confs.append(np.zeros((len(seq_frames), self.num_classes, self.num_classes)))
                self.val_2D_reconstruct.append(np.ones((len(seq_frames), 3)))
                self.val_2D_future.append(np.ones((len(seq_frames), self.config.n_2D_layers)))

        return

    def load_points(self, s_ind, f_ind):

        velo_file = join(self.seq_path[s_ind], self.frames[s_ind][f_ind] + '.ply')

        # Read points
        data = read_ply(velo_file)
        return np.vstack((data['x'], data['y'], data['z'])).T


class MyhalCollisionSampler(Sampler):
    """Sampler for MyhalCollision"""

    def __init__(self, dataset: MyhalCollisionDataset, manual_training_frames=False, debug_custom_inds=False):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

        # Choose training frames with specific manual rules
        if (manual_training_frames):

            # convertion from labels to colors
            im_lim = self.dataset.config.in_radius / np.sqrt(2)
            colormap = np.array([[209, 209, 209],
                                [122, 122, 122],
                                [255, 255, 0],
                                [0, 98, 255],
                                [255, 0, 0]], dtype=np.float32) / 255

            for i_l, ll in enumerate(self.dataset.label_values):
                if ll == 4:

                    # Variable containg the selected inds for this class (will replace self.dataset.class_frames[i_l])
                    selected_mask = np.zeros_like(self.dataset.all_inds[:, 0], dtype=bool)

                    # Advanced display
                    N = self.dataset.all_inds.shape[0]
                    progress_n = 50
                    fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
                    print('\nGetting custom training inds')

                    all_pts = [[] for frames in self.dataset.frames]
                    all_colors = [[] for frames in self.dataset.frames]
                    all_labels = [[] for frames in self.dataset.frames]
                    tot_i = 0
                    for s_ind, seq in enumerate(self.dataset.sequences):
                        for f_ind, frame in enumerate(self.dataset.frames[s_ind]):

                            # Get groundtruth in 2D points format
                            gt_file = join(self.dataset.colli_path[s_ind], frame + '_2D.ply')

                            # Read points
                            data = read_ply(gt_file)
                            pts_2D = np.vstack((data['x'], data['y'])).T
                            labels_2D = data['classif']
                                
                            # Special treatment to old simulation annotations
                            if self.dataset.sim_sequences[s_ind]:
                                times_2D = data['t']
                                time_mask = np.logical_and(times_2D > -0.001, times_2D < 0.001)
                                pts_2D = pts_2D[time_mask]
                                labels_2D = labels_2D[time_mask]

                            # Recenter
                            p0 = self.dataset.poses[s_ind][f_ind][:2, 3]
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
                            # if not self.dataset.sim_sequences[s_ind]:
                            #     if p0[1] < 6 and p0[0] < 4:
                            #         label_count[-1] = 0

                            all_pts[s_ind].append(centered_2D)
                            all_colors[s_ind].append(colormap[labels_2D])
                            all_labels[s_ind].append(label_count)

                            print('', end='\r')
                            print(fmt_str.format('#' * (((tot_i + 1) * progress_n) // N), 100 * (tot_i + 1) / N), end='', flush=True)
                            tot_i += 1

                        # Get the wanted indices
                        class_mask = np.vstack(all_labels[s_ind]).T
                        class_mask = class_mask[i_l:i_l + 1] > 10

                        # Remove isolated inds with opening
                        open_struct = np.ones((1, 31))
                        class_mask_opened = ndimage.binary_opening(class_mask, structure=open_struct)
                        
                        # Remove the one where the person is disappearing or reappearing
                        erode_struct = np.ones((1, 31))
                        erode_struct[:, :13] = 0
                        class_mask_eroded = ndimage.binary_erosion(class_mask_opened, structure=erode_struct)

                        # Update selected inds for all sequences
                        seq_mask = self.dataset.all_inds[:, 0] == s_ind
                        selected_mask[seq_mask] = np.squeeze(class_mask_eroded)

                        if debug_custom_inds:
                            
                            # Figure
                            figA, axA = plt.subplots(1, 1, figsize=(10, 7))
                            plt.subplots_adjust(left=0.1, bottom=0.2)

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

                            # Ax with the presence of dynamic points
                            class_mask = np.zeros_like(self.dataset.all_inds[:, 0], dtype=bool)
                            class_mask[self.dataset.class_frames[i_l]] = True
                            seq_mask = self.dataset.all_inds[:, 0] == s_ind
                            seq_class_frames = class_mask[seq_mask]
                            seq_class_frames = np.expand_dims(seq_class_frames, 0)
                            axdyn = plt.axes([0.1, 0.08, 0.8, 0.015])
                            axdyn.imshow(seq_class_frames, cmap='GnBu', aspect='auto')
                            axdyn.set_axis_off()

                            # Ax with the presence of dynamic points at least 10
                            dyn_img = np.vstack(all_labels[s_ind]).T
                            dyn_img = dyn_img[-1:]
                            dyn_img[dyn_img > 10] = 10
                            dyn_img[dyn_img > 0] += 10
                            axdyn = plt.axes([0.1, 0.06, 0.8, 0.015])
                            axdyn.imshow(dyn_img, cmap='OrRd', aspect='auto')
                            axdyn.set_axis_off()
                            
                            # Ax with opened
                            axdyn = plt.axes([0.1, 0.04, 0.8, 0.015])
                            axdyn.imshow(class_mask_opened, cmap='OrRd', aspect='auto')
                            axdyn.set_axis_off()
                            
                            # Ax with eroded
                            axdyn = plt.axes([0.1, 0.02, 0.8, 0.015])
                            axdyn.imshow(class_mask_eroded, cmap='OrRd', aspect='auto')
                            axdyn.set_axis_off()

                            plt.show()

                    selected_inds = np.where(selected_mask)[0]
                    self.dataset.class_frames[i_l] = torch.from_numpy(selected_inds.astype(np.int64))

                    # Show a nice 100% progress bar
                    print('', end='\r')
                    print(fmt_str.format('#' * progress_n, 100), flush=True)
                    print('\n')

                    if debug_custom_inds:
                        a = 1/0

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        if self.dataset.balance_classes:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0
            self.dataset.epoch_labels *= 0

            # Number of sphere centers taken per class in each cloud
            num_centers = self.dataset.epoch_inds.shape[0]

            # Generate a list of indices balancing classes and respecting potentials
            gen_indices = []
            gen_classes = []
            for i_l, ll in enumerate(self.dataset.label_values):
                if ll not in self.dataset.ignored_labels:

                    # Get the proportion of points picked in each class
                    proportions = self.dataset.config.balance_proportions
                    sim_class_n = 0
                    sim_ratio = None
                    if len(proportions) == len(self.dataset.label_values):
                        class_n = int(np.floor(num_centers * proportions[i_l] / np.sum(proportions))) + 1

                    elif len(proportions) == len(self.dataset.label_values) + 1:
                        sim_ratio = proportions[-1]
                        if (sim_ratio > 1.0 or sim_ratio < 0.0):
                            raise ValueError('Simulation example ratio must be between 0 and 1')
                        if not self.dataset.sim_path:
                            print('WARNING: No simualtion data found but a simulation ratio is defined in the balance proportions')
                            sim_ratio = 0
                        tot_n = int(np.floor(num_centers * proportions[i_l] / np.sum(proportions[:-1]))) + 1
                        sim_class_n = int(np.floor(tot_n * sim_ratio))
                        class_n = tot_n - sim_class_n
                    else:
                        used_classes = self.dataset.num_classes - len(self.dataset.ignored_labels)
                        class_n = num_centers // used_classes + 1

                    if (class_n + sim_class_n < 2):
                        continue

                    if sim_ratio is not None:
                        # Get the indices of simulation and normal frames separately
                        class_inds = self.dataset.class_frames[i_l]
                        seq_inds = self.dataset.all_inds[class_inds, 0]
                        sim_mask = self.dataset.sim_sequences[seq_inds]
                        sim_inds = class_inds[sim_mask]
                        real_inds = class_inds[np.logical_not(sim_mask)]
                    else:
                        sim_inds = None
                        real_inds = self.dataset.class_frames[i_l]

                    for c_inds, c_n in [[real_inds, class_n], [sim_inds, sim_class_n]]:

                        if c_inds is not None:

                            # Get the potentials of the frames containing this class
                            class_potentials = self.dataset.potentials[c_inds]

                            # Get the indices to generate thanks to potentials
                            if c_n < class_potentials.shape[0]:
                                _, class_indices = torch.topk(class_potentials,
                                                              c_n,
                                                              largest=False)
                            else:
                                class_indices = torch.randperm(class_potentials.shape[0])
                            class_indices = c_inds[class_indices]

                            # Add the indices to the generated ones
                            gen_indices.append(class_indices)
                            gen_classes.append(class_indices * 0 + ll)


            # Stack the chosen indices of all classes
            gen_indices = torch.cat(gen_indices, dim=0)
            gen_classes = torch.cat(gen_classes, dim=0)

            # Shuffle generated indices
            rand_order = torch.randperm(gen_indices.shape[0])[:num_centers]
            gen_indices = gen_indices[rand_order]
            gen_classes = gen_classes[rand_order]

            # Update potentials (Change the order for the next epoch)
            self.dataset.potentials[gen_indices] = torch.ceil(self.dataset.potentials[gen_indices])
            self.dataset.potentials[gen_indices] += torch.from_numpy(np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1)

            # Update epoch inds
            self.dataset.epoch_inds += gen_indices
            self.dataset.epoch_labels += gen_classes.type(torch.int32)

        else:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0
            self.dataset.epoch_labels *= 0

            # Number of sphere centers taken per class in each cloud
            num_centers = self.dataset.epoch_inds.shape[0]

            # Get the list of indices to generate thanks to potentials
            if num_centers < self.dataset.potentials.shape[0]:
                _, gen_indices = torch.topk(self.dataset.potentials,
                                            num_centers,
                                            largest=False,
                                            sorted=True)
            else:
                gen_indices = torch.randperm(self.dataset.potentials.shape[0])

            # Update potentials (Change the order for the next epoch)
            self.dataset.potentials[gen_indices] = torch.ceil(self.dataset.potentials[gen_indices])
            self.dataset.potentials[gen_indices] += torch.from_numpy(np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1)

            # Update epoch inds
            self.dataset.epoch_inds += gen_indices

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N

    def calib_max_in(self,
                     config,
                     dataloader,
                     untouched_ratio=0.8,
                     verbose=True,
                     force_redo=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration of max_in_points value (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load max_in_limit dictionary
        max_in_lim_file = join(self.dataset.path, 'max_in_limits.pkl')
        if exists(max_in_lim_file):
            with open(max_in_lim_file, 'rb') as file:
                max_in_lim_dict = pickle.load(file)
        else:
            max_in_lim_dict = {}

        # Check if the max_in limit associated with current parameters exists
        if self.dataset.balance_classes:
            sampler_method = 'balanced'
        else:
            sampler_method = 'random'
        key = '{:s}_{:d}_{:.3f}_{:.3f}'.format(
            sampler_method, self.dataset.config.n_frames, self.dataset.in_R,
            self.dataset.config.first_subsampling_dl)
        if not redo and key in max_in_lim_dict:
            self.dataset.max_in_p = max_in_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check max_in limit dictionary')
            if key in max_in_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(max_in_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ########################
            # Batch calib parameters
            ########################

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            all_lengths = []
            N = 1000

            #####################
            # Perform calibration
            #####################

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Control max_in_points value
                    all_lengths += batch.lengths[0]

                    # Convergence
                    if len(all_lengths) > N:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if t - last_display > 1.0:
                        last_display = t
                        message = 'Collecting {:d} in_points: {:5.1f}%'
                        print(message.format(N, 100 * len(all_lengths) / N))

                if breaking:
                    break

            self.dataset.max_in_p = int(
                np.percentile(all_lengths, 100 * untouched_ratio))

            if verbose:

                # Create histogram
                a = 1

            # Save max_in_limit dictionary
            print('New max_in_p = ', self.dataset.max_in_p)
            max_in_lim_dict[key] = self.dataset.max_in_p
            with open(max_in_lim_file, 'wb') as file:
                pickle.dump(max_in_lim_dict, file)

        # Update value in config
        if self.dataset.set == 'training':
            config.max_in_points = self.dataset.max_in_p
        else:
            config.max_val_points = self.dataset.max_in_p

        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return

    def calibration(self,
                    dataloader,
                    untouched_ratio=0.9,
                    verbose=False,
                    force_redo=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if self.dataset.balance_classes:
            sampler_method = 'balanced'
        else:
            sampler_method = 'random'
        key = '{:s}_{:d}_{:.3f}_{:.3f}_{:d}_{:d}'.format(
            sampler_method, self.dataset.config.n_frames, self.dataset.in_R,
            self.dataset.config.first_subsampling_dl, self.dataset.batch_num,
            self.dataset.max_in_p)
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:s}_{:d}_{:d}_{:.3f}_{:.3f}'.format(
                sampler_method, self.dataset.config.n_frames,
                self.dataset.max_in_p, dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:s}_{:d}_{:d}_{:.3f}_{:.3f}'.format(
                    sampler_method, self.dataset.config.n_frames,
                    self.dataset.max_in_p, dl, r)
                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v,
                                                    bcolors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(
                np.ceil(4 / 3 * np.pi *
                        (self.dataset.config.deform_radius + 1)**3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n),
                                    dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Save input pointcloud sizes to control max_in_points
            cropped_n = 0
            all_n = 0

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            #####################
            # Perform calibration
            #####################

            #self.dataset.batch_limit[0] = self.dataset.max_in_p * (self.dataset.batch_num - 1)

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Control max_in_points value
                    are_cropped = batch.lengths[0] > self.dataset.max_in_p - 1
                    cropped_n += torch.sum(are_cropped.type(
                        torch.int32)).item()
                    all_n += int(batch.lengths[0].shape[0])

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.frame_inds)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit[0] += Kp * error

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(
                            np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i, estim_b, int(self.dataset.batch_limit[0])))

                if breaking:
                    break

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum <
                                 (untouched_ratio * cumsum[hist_n - 1, :]),
                                 axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(
                            color, neighb_hists[layer, neighb_size],
                            bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Control max_in_points value
            print('\n**************************************************\n')
            if cropped_n > 0.3 * all_n:
                color = bcolors.FAIL
            else:
                color = bcolors.OKGREEN
            print('Current value of max_in_points {:d}'.format(self.dataset.max_in_p))
            print('  > {:}{:.1f}% inputs are cropped{:}'.format(color, 100 * cropped_n / all_n, bcolors.ENDC))
            if cropped_n > 0.3 * all_n:
                print('\nTry a higher max_in_points value\n')
                #raise ValueError('Value of max_in_points too low')
            print('\n**************************************************\n')

            # Save batch_limit dictionary
            key = '{:s}_{:d}_{:.3f}_{:.3f}_{:d}_{:d}'.format(
                sampler_method, self.dataset.config.n_frames,
                self.dataset.in_R, self.dataset.config.first_subsampling_dl,
                self.dataset.batch_num, self.dataset.max_in_p)
            batch_lim_dict[key] = float(self.dataset.batch_limit[0])
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * \
                    (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:s}_{:d}_{:d}_{:.3f}_{:.3f}'.format(
                    sampler_method, self.dataset.config.n_frames,
                    self.dataset.max_in_p, dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[
                    layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)

        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class MyhalCollisionSamplerTest(MyhalCollisionSampler):
    """Specific Sampler for MyhalCollision Tests which limits the predicted frame indices to certain ones"""

    def __init__(self, dataset: MyhalCollisionDataset, wanted_frame_inds):
        MyhalCollisionSampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.frame_inds = torch.from_numpy(np.array(wanted_frame_inds, dtype=np.int64))

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        # Initiate current epoch ind
        self.dataset.epoch_i *= 0
        self.dataset.epoch_inds *= 0
        self.dataset.epoch_labels *= 0

        # Number of sphere centers taken per class in each cloud
        num_centers = self.dataset.epoch_inds.shape[0]

        # Verification
        if (num_centers <= self.frame_inds.shape[0]):
            raise ValueError('Number of elements asked too high compared to the validation size. \
                Increase config.validation_size to correct this')

        # Repeat the wanted inds enough times
        num_repeats = num_centers // self.frame_inds.shape[0] + 1
        repeated_inds = self.frame_inds.repeat(num_repeats)

        # Update epoch inds
        self.dataset.epoch_inds += repeated_inds[:num_centers]

        # Generator loop
        for i in range(self.frame_inds.shape[0]):
            yield i


class MyhalCollisionCustomBatch:
    """Custom batch definition with memory pinning for MyhalCollision"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = int(input_list[0])

        # Extract input tensors from the list of numpy array
        ind = 1
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.future_2D = torch.from_numpy(input_list[ind])
        ind += 1
        self.pools_2D = torch.from_numpy(input_list[ind])
        ind += 1
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.frame_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.frame_centers = torch.from_numpy(input_list[ind])
        ind += 1
        self.reproj_inds = input_list[ind]
        ind += 1
        self.reproj_masks = input_list[ind]
        ind += 1
        self.val_labels = input_list[ind]

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [
            in_tensor.pin_memory() for in_tensor in self.neighbors
        ]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [
            in_tensor.pin_memory() for in_tensor in self.upsamples
        ]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.future_2D = self.future_2D.pin_memory()
        self.pools_2D = self.pools_2D.pin_memory()
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.frame_inds = self.frame_inds.pin_memory()
        self.frame_centers = self.frame_centers.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.future_2D = self.future_2D.to(device)
        self.pools_2D = self.pools_2D.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.frame_inds = self.frame_inds.to(device)
        self.frame_centers = self.frame_centers.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i + 1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(
                            self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def MyhalCollisionCollate(batch_data):
    return MyhalCollisionCustomBatch(batch_data)
