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
from multiprocessing import Lock
from datasets.common import PointCloudDataset, batch_neighbors
from torch.utils.data import Sampler
from utils.config import bcolors
from datasets.common import grid_subsampling

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as scipyR
from sklearn.neighbors import KDTree
from slam.PointMapSLAM import PointMap, extract_map_ground, extract_ground
from slam.cpp_slam import update_pointmap, polar_normals, point_to_map_icp, slam_on_sim_sequence, ray_casting_annot, get_lidar_visibility, slam_on_real_sequence
from slam.dev_slam import frame_H_to_points, interp_pose, rot_trans_diffs, normals_orientation, save_trajectory, RANSAC
from utils.ply import read_ply, write_ply
from utils.mayavi_visu import save_future_anim, fast_save_future_anim

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
                 verbose=1):

        # Name of the dataset
        self.name = 'MyhalCollisionSlam'

        # Data path
        self.original_path = '../Data/Real'
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

    def refine_map(self, map_dl=0.03, min_rays=10, occup_threshold=0.9):
        """
        Remove moving objects via ray-tracing. (Step 1 in the annotation process)
        """

        # Folder where the incrementally updated map is stored
        map_folder = join(self.data_path, 'slam_offline', self.map_day)
        if not exists(map_folder):
            makedirs(map_folder)

        # List of the updated maps
        map_names = [f for f in listdir(map_folder) if f.startswith('map_update_')]

        # First check if we never initiated the map
        if len(map_names) == 0:

            self.init_map(map_folder)

            # Stop computation here as we have nothing to train on
            a = 1 / 0

        else:

            print('Error: Refine map not modified for motion distortion yet')

            a = 1/0

        # Now check if these days were already used for updating
        day_movable_names = [
            f for f in listdir(map_folder) if f.startswith('last_movables_')
        ]
        day_movable_names = [f[:-4].split('_')[-1] for f in day_movable_names]
        seen_inds = []
        for d, day in enumerate(self.days):
            if day in day_movable_names:
                seen_inds.append(d)
        if len(seen_inds) == len(self.days):
            print('Not updating if no new days are given')
            return

        print('Start Map Update')

        # Get the latest update of the map
        map_names = np.sort(map_names)
        last_map = map_names[-1]
        last_update_i = int(last_map[:-4].split('_')[-1])

        # Load map
        print('Load last update')
        data = read_ply(join(map_folder, last_map))
        map_points = np.vstack((data['x'], data['y'], data['z'])).T
        map_normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
        map_scores = data['scores']
        map_counts = data['counts']
        print('OK')

        # Get remove point form each day independently
        # Otherwise if a table is there in only one day, it will not be removed.
        print('Movable detection')
        all_movables_probs = []
        for d, day in enumerate(self.days):

            # No update if this day have already been seen
            if d in seen_inds:
                continue

            # Load poses
            map_t, map_H = self.load_map_poses(day)
            f_names = self.day_f_names[d]

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
                    #print(f_names[f_name_i], ' skipped for ', f_name)
                    f_name_i += 1

                if f_name_i >= len(f_names):
                    break

                frame_names.append(f_names[f_name_i])
                f_name_i += 1

            # Remove the double inds form map_t and map_H
            map_t = np.delete(map_t, remove_inds, axis=0)
            map_H = np.delete(map_H, remove_inds, axis=0)

            # Get short term movables
            movable_prob, movable_count = ray_casting_annot(
                frame_names,
                map_points,
                map_normals,
                map_H,
                theta_dl=1.29 * np.pi / 180,
                phi_dl=0.1 * np.pi / 180,
                map_dl=map_dl,
                motion_distortion=False)
            movable_prob = movable_prob / (movable_count + 1e-6)
            movable_prob[movable_count < min_rays] -= 2
            all_movables_probs.append(movable_prob)

            # Save it
            print('Saving')
            write_ply(join(map_folder, 'last_movables_{:s}.ply'.format(day)),
                      [map_points, map_normals, movable_prob, movable_count],
                      ['x', 'y', 'z', 'nx', 'ny', 'nz', 'movable', 'counts'])

        print('OK')

        # Combine movable probs from days
        all_movables_probs = np.stack(all_movables_probs, 0)
        all_movables_probs = np.max(all_movables_probs, axis=0)
        write_ply(join(map_folder, 'last_movables_combined.ply'),
                  [map_points, map_normals, all_movables_probs],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz', 'movable'])

        # Extract ground ransac
        print('Get ground')
        ground_mask = extract_ground(map_points,
                                     map_normals,
                                     map_folder,
                                     vertical_thresh=10.0,
                                     dist_thresh=0.1,
                                     remove_dist=0.1)

        # Do not remove ground points
        all_movables_probs[ground_mask] = 0

        # Remove points with high movable probability from the map
        mask = all_movables_probs < occup_threshold
        filename = join(map_folder,
                        'map_update_{:04}.ply'.format(last_update_i + 1))
        write_ply(filename, [
            map_points[mask], map_normals[mask], map_scores[mask],
            map_counts[mask]
        ], ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])
        print('OK')

        return

    def init_map(self, map_folder, map_dl=0.03):

        # Map is initiated by removing movable points with its own trajectory
        #   > Step 0: REdo the point map from scratch
        #   > Step 1: annotate short-term movables for each frame
        #   > Step 2: Do a pointmap slam without the short-term movables (enforce horizontal planar ground)
        #   > Step 3: Apply loop closure on the poses of this second slam
        #   > Step 4: With the corrected poses and the full point clouds, create a barycentre pointmap
        #   > Step 5: Remove the short-term movables from this good map.


        #################
        # Initial mapping
        #################

        try:
            # Get map_poses
            from_nav = True
            map_t, map_H = self.load_map_poses(self.map_day)

        except FileNotFoundError as e:

            # If not available perfrom an initial mapping
            from_nav = False
            map_t = np.array([np.float64(f.split('/')[-1][:-4]) for f in self.map_f_names], dtype=np.float64)
            init_map_pkl = join(map_folder, 'map0_traj_{:s}.pkl'.format(self.map_day))
            loop_pkl = join(map_folder, 'loopclosed_traj_{:s}.pkl'.format(self.map_day))


            #DEBUG
            if exists(loop_pkl):
                with open(loop_pkl, 'rb') as file:
                    loop_H = pickle.load(file)

                map_H = loop_H

                loop_closed_map_name = join(map_folder, 'loopclosed_map0_{:s}.ply'.format(self.map_day))
                if not exists(loop_closed_map_name):

                    odom_H = [np.linalg.inv(odoH) for odoH in loop_H]
                    odom_H = np.stack(odom_H, 0)
                    _ = slam_on_real_sequence(self.map_f_names,
                                              map_t,
                                              map_folder,
                                              map_voxel_size=map_dl,
                                              frame_voxel_size=3 * map_dl,
                                              motion_distortion=True,
                                              filtering=False,
                                              verbose_time=5.0,
                                              icp_samples=600,
                                              icp_pairing_dist=2.0,
                                              icp_planar_dist=0.3,
                                              icp_max_iter=0,
                                              icp_avg_steps=5,
                                              odom_H=odom_H)

                    # Rename the saved map file
                    old_name = join(map_folder, 'map_{:s}.ply'.format(self.map_day))
                    os.rename(old_name, loop_closed_map_name)
                    

            else:

                if exists(init_map_pkl):
                    with open(init_map_pkl, 'rb') as file:
                        map_H = pickle.load(file)

                else:

                    map_H = slam_on_real_sequence(self.map_f_names,
                                                  map_t,
                                                  map_folder,
                                                  map_voxel_size=map_dl,
                                                  frame_voxel_size=3 * map_dl,
                                                  motion_distortion=True,
                                                  filtering=False,
                                                  verbose_time=5.0,
                                                  icp_samples=600,
                                                  icp_pairing_dist=2.0,
                                                  icp_planar_dist=0.3,
                                                  icp_max_iter=100,
                                                  icp_avg_steps=5)

                    # Save the trajectory
                    save_trajectory(join(map_folder, 'map0_traj_{:s}.ply'.format(self.map_day)), map_H)
                    with open(init_map_pkl, 'wb') as file:
                        pickle.dump(map_H, file)

                    # Rename the saved map file
                    old_name = join(map_folder, 'map_{:s}.ply'.format(self.map_day))
                    new_name = join(map_folder, 'map0_{:s}.ply'.format(self.map_day))
                    os.rename(old_name, new_name)


        #####################################
        # Annotate short-term on original map
        #####################################

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

            # Skip ply frames that are not present in the timestamps of map_t
            f_name = '{:.6f}.ply'.format(t)
            while f_name_i < len(self.map_f_names) and not (self.map_f_names[f_name_i].endswith(f_name)):
                f_name_i += 1

            if f_name_i >= len(self.map_f_names):
                break

            frame_names.append(self.map_f_names[f_name_i])
            f_name_i += 1

        # Remove the double inds form map_t and map_H
        map_t = np.delete(map_t, remove_inds, axis=0)
        map_H = np.delete(map_H, remove_inds, axis=0)

        print('OK')

        first_annot_name = join(map_folder, 'movable_final.ply')
        if exists(first_annot_name):

            print('Load annot')
            # pointmap = PointMap(map_dl)
            data = read_ply(first_annot_name)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
            movable_prob = data['movable']
            movable_count = data['counts']
            print('OK')

        else:

            print('Load pointmap')

            # Get the map
            if from_nav:
                map_original_name = join(self.data_path, 'simulated_runs',
                                         self.map_day, 'logs-' + self.map_day,
                                         'pointmap_00000.ply')
            else:
                map_original_name = join(map_folder, 'loopclosed_map0_' + self.map_day + '.ply')
                if not exists(map_original_name):
                    map_original_name = join(map_folder, 'map0_' + self.map_day + '.ply')

            data = read_ply(map_original_name)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
            scores = data['f0']

            print('OK')
            print('Update pointmap')

            # pointmap = PointMap(map_dl)
            # pointmap.update(points, normals, scores)

            print('OK')
            print('Start ray casting')

            # Get short term movables TODO: HERE implement motion distortion and correct double linked list bug
            movable_prob, movable_count = ray_casting_annot(frame_names,
                                                            points,
                                                            normals,
                                                            map_H,
                                                            theta_dl=1.29 * np.pi / 180,
                                                            phi_dl=0.1 * np.pi / 180,
                                                            map_dl=map_dl,
                                                            verbose_time=5.0,
                                                            motion_distortion_slices=12)

            movable_prob = movable_prob / (movable_count + 1e-6)
            movable_prob[movable_count < 1e-6] = -1

            print('Ray_casting done')

            # Save it
            write_ply(join(map_folder, 'movable_final.ply'),
                      [points, normals, movable_prob, movable_count],
                      ['x', 'y', 'z', 'nx', 'ny', 'nz', 'movable', 'counts'])

        #####################
        # Reproject on frames
        # ###################

        # Extract ground ransac
        print('Get ground')
        ground_mask = extract_map_ground(points,
                                         normals,
                                         map_folder,
                                         vertical_thresh=10.0,
                                         dist_thresh=0.3,
                                         remove_dist=0.29)

        # Do not remove ground points
        movable_prob[ground_mask] = 0

        # Folder where we save the first annotated_frames
        annot_folder = join(map_folder, 'tmp_frames')
        if not exists(annot_folder):
            makedirs(annot_folder)

        # Create KDTree on the map
        print('Reprojection of map day {:s}'.format(self.map_day))
        map_tree = None
        N = len(frame_names)
        new_frame_names = []
        last_t = time.time()
        fps = 0
        fps_regu = 0.9
        for i, f_name in enumerate(frame_names):

            # Check if we already did reprojection
            new_f_name = join(annot_folder, f_name.split('/')[-1])
            new_frame_names.append(new_f_name)
            if exists(new_f_name):
                continue
            elif i < 1:
                map_tree = KDTree(points)

            t = [time.time()]

            # Load points
            points = self.load_frame_points(f_name)

            # Apply transf
            world_pts = np.hstack((points, np.ones_like(points[:, :1])))
            world_pts = np.matmul(world_pts, map_H[i].T).astype(np.float32)[:, :3]

            # Get closest map points
            neighb_inds = np.squeeze(map_tree.query(world_pts, return_distance=False))
            frame_movable_prob = movable_prob[neighb_inds]
            frame_ground_mask = ground_mask[neighb_inds]

            # Save frame with annotation
            categories = np.zeros(frame_movable_prob.shape, np.int32)
            categories[frame_movable_prob > 0.7] = 4
            categories[frame_ground_mask] = 1
            write_ply(new_f_name, [categories], ['cat'])

            t += [time.time()]
            fps = fps_regu * fps + (1.0 - fps_regu) / (t[-1] - t[0])

            if (t[-1] - last_t > 5.0):
                print('Reproj {:s} {:5d} --- {:5.1f}%% at {:.1f} fps'.format(self.map_day,
                                                                             i + 1,
                                                                             100 * (i + 1) / N,
                                                                             fps))
                last_t = t[-1]
        print('OK')

        # ###########################################
        # # Aligned the transformation on groundtruth
        # ###########################################
        # # This is not cheating as if we did not add ground truth, the map would be the ground truth

        # gt_t, gt_H = self.load_map_gt_poses()

        # # Do not align with the first frame in case it is not a good one
        # f_i0 = 3
        # f_t0 = map_t[f_i0]

        # # Find closest gt poses
        # gt_i1 = np.argmin(np.abs(gt_t - f_t0))
        # if f_t0 < gt_t[gt_i1]:
        #     gt_i0 = gt_i1 - 1
        # else:
        #     gt_i0 = gt_i1
        #     gt_i1 = gt_i0 + 1

        # # Interpolate the ground truth pose at current time
        # interp_t = (f_t0 - gt_t[gt_i0]) / (gt_t[gt_i1] - gt_t[gt_i0])
        # frame_H = interp_pose(interp_t, gt_H[gt_i0], gt_H[gt_i1])
        # gt_H_velo_world = np.matmul(frame_H, self.H_velo_base)

        # # Align everything
        # correction_H = np.matmul(gt_H_velo_world, np.linalg.inv(map_H[f_i0]))
        # map_H = [np.matmul(correction_H, mH) for mH in map_H]
        # map_H = np.stack(map_H, 0)

        # save_trajectory(
        #     join(map_folder, 'init_traj_{:s}.ply'.format(self.map_day)), map_H)

        # # Save the grounbdtruth traj but of the velodyne pose
        # gt_H_velo_world = []
        # for f_i, f_t in enumerate(map_t):

        #     # Find closest gt poses
        #     gt_i1 = np.argmin(np.abs(gt_t - f_t))

        #     if gt_i1 == 0 or gt_i1 == len(gt_t) - 1:
        #         continue

        #     if f_t < gt_t[gt_i1]:
        #         gt_i0 = gt_i1 - 1
        #     else:
        #         gt_i0 = gt_i1
        #         gt_i1 = gt_i0 + 1

        #     # Interpolate the ground truth pose at current time
        #     interp_t = (f_t - gt_t[gt_i0]) / (gt_t[gt_i1] - gt_t[gt_i0])
        #     frame_H = interp_pose(interp_t, gt_H[gt_i0], gt_H[gt_i1])
        #     gt_H_velo_world.append(np.matmul(frame_H, self.H_velo_base))

        # gt_H_velo_world = np.stack(gt_H_velo_world, 0)
        # save_trajectory(
        #     join(map_folder, 'gt_traj_{:s}.ply'.format(self.map_day)),
        #     gt_H_velo_world)

        ##########################
        # Make a horizontal ground
        ##########################

        # TODO HERE: Well it os not so horizontal...
        # Maybe use something like imls for noise reduction on this flat surface
        # instead of forcing globally flat ground, force local flat ground!

        # Find the global transform that makes the ground flat and with height z = 0
        global_correct_H = np.eye(4)
        
        # Align all ground point on a horizontal plane (height does not matter so much as we correct it later)
        ground_points = points[ground_mask]
        ground_pointsh = np.hstack((ground_points, np.ones_like(ground_points[:, :1])))
        ground_points = np.matmul(ground_pointsh, global_correct_H.T).astype(np.float32)[:, :3]
        
        # Remove noise on the ground
        ground_z = 0
        ground_points[:, 2] = ground_z

        # Only vertical normals
        ground_normals = np.zeros_like(ground_points)
        ground_normals[:, 2] = 1.0

        # High scores to keep normals
        ground_scores = np.ones(ground_points.shape[0],
                                dtype=np.float32) * 0.99

        # Add the first frame in init point to avoid optimization problems
        # Hopefully during mapping the fist frame should not have motion distortion
        f_i0 = 1
        points, labels = self.load_frame_points_labels(frame_names[f_i0], new_frame_names[f_i0])
        normals, planarity, linearity = polar_normals(points,
                                                      radius=1.5,
                                                      lidar_n_lines=32,
                                                      h_scale=0.5,
                                                      r_scale=1000.0)
        norm_scores = planarity + linearity
        points = points[norm_scores > 0.1]
        labels = labels[norm_scores > 0.1]
        normals = normals[norm_scores > 0.1]

        # Eliminate moving and ground points
        points = points[labels == 0]
        normals = normals[labels == 0]

        # Reorient the first frame to be on the ground
        frame_H = np.matmul(global_correct_H, map_H[f_i0])
        world_points = np.hstack((points, np.ones_like(points[:, :1])))
        world_points = np.matmul(world_points, frame_H.T).astype(np.float32)[:, :3]
        world_normals = np.matmul(normals, frame_H[:3, :3].T).astype(np.float32)
        world_scores = np.ones(world_points.shape[0], dtype=np.float32) * 0.1

        # Add ground to init points
        world_points = np.vstack((world_points, ground_points))
        world_normals = np.vstack((world_normals, ground_normals))
        world_scores = np.hstack((world_scores, ground_scores))

        # Save for debug
        write_ply(join(map_folder, 'horizontal_ground.ply'),
                  [world_points, world_normals],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz'])

        print('OK')

        #############################################################
        # Pointmap slam without short-term and with horizontal ground
        # ###########################################################

        initial_map_file = join(map_folder, 'map_update_{:04}.ply'.format(0))
        if not exists(initial_map_file):

            # Odometry is given as Scanner to Odom so we have to invert matrices
            odom_H = [np.linalg.inv(odoH) for odoH in map_H]
            odom_H = np.stack(odom_H, 0)

            correct_H = slam_on_real_sequence(frame_names,
                                              map_t,
                                              map_folder,
                                              init_points=world_points,
                                              init_normals=world_normals,
                                              init_scores=world_scores,
                                              map_voxel_size=map_dl,
                                              frame_voxel_size=3 * map_dl,
                                              motion_distortion=True,
                                              filtering=True,
                                              verbose_time=5,
                                              icp_samples=600,
                                              icp_pairing_dist=2.0,
                                              icp_planar_dist=0.3,
                                              icp_max_iter=100,
                                              icp_avg_steps=5,
                                              odom_H=odom_H)

            # Apply offset so that traj is aligned with groundtruth
            # correct_H = correct_H

            # Save the new corrected trajectory
            save_trajectory(join(map_folder, 'correct_traj_{:s}.ply'.format(self.map_day)), correct_H)
            with open(join(map_folder, 'correct_traj_{:s}.pkl'.format(self.map_day)), 'wb') as file:
                pickle.dump(correct_H, file)

            # TODO: C++ function for creating a map with spherical barycenters of frames?
            #       We have to do it anyway for the annotation process after
            # # Create a point map with all the points and sphere barycentres.

            # Instead just load c++ map and save it as the final result
            data = read_ply(join(map_folder, 'map_{:s}.ply'.format(self.map_day)))
            pointmap.points = np.vstack((data['x'], data['y'], data['z'])).T
            pointmap.normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
            pointmap.scores = data['f1']
            pointmap.counts = data['f0']
            write_ply(initial_map_file,
                      [pointmap.points, pointmap.normals, pointmap.scores, pointmap.counts],
                      ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

        else:

            # Load traj
            with open(join(map_folder, 'correct_traj_{:s}.pkl'.format(self.map_day)), 'rb') as f:
                correct_H = pickle.load(f)

        # Optional stuff done for the ICRA video
        if False:
            odom_H = [np.linalg.inv(odoH) for odoH in correct_H]
            odom_H = np.stack(odom_H, 0)
            slam_on_sim_sequence(frame_names,
                                 map_t,
                                 map_H,
                                 map_t,
                                 map_folder,
                                 map_voxel_size=map_dl,
                                 frame_voxel_size=3 * map_dl,
                                 motion_distortion=False,
                                 filtering=True,
                                 icp_samples=600,
                                 icp_pairing_dist=2.0,
                                 icp_planar_dist=0.3,
                                 icp_max_iter=0,
                                 icp_avg_steps=5,
                                 odom_H=odom_H)

        return

    def collision_annotation(self, dl_2D=0.03, start_T=-1.0, future_T=5.0):

        ###############
        # STEP 0 - Init
        ###############

        # Classes
        print('Start Collision Annotation')
        label_names = {0: 'uncertain',
                       1: 'ground',
                       2: 'still',
                       3: 'longT',
                       4: 'shortT'}

                        
        # Folder where the incrementally updated map is stored
        map_folder = join(self.data_path, 'slam_offline', self.map_day)

        # List of the updated maps
        map_names = [f for f in listdir(map_folder) if f.startswith('map_update_')]

        # Get the latest update of the map
        map_names = np.sort(map_names)
        last_map = map_names[-1]
        last_update_i = int(last_map[:-4].split('_')[-1])

        # Load map
        print('Load last update')
        data = read_ply(join(map_folder, last_map))
        map_points = np.vstack((data['x'], data['y'], data['z'])).T
        map_normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
        map_scores = data['scores']
        map_counts = data['counts']
        print('OK')

        # Get ground
        vertical_angle = np.arccos(np.abs(np.clip(map_normals[:, 2], -1.0, 1.0)))
        plane_mask = vertical_angle < 10.0 * np.pi / 180
        plane_P, plane_N, _ = RANSAC(map_points[plane_mask], threshold_in=0.1)
        ground_plane = np.append(plane_N, np.dot(plane_N, plane_P))

        ##############
        # LOOP ON DAYS
        ##############

        print(self.days)
        fps = 0
        fps_regu = 0.9
        last_t = time.time()

        # Get remove point form each day independently
        # Otherwise if a table is there in only one day, it will not be removed.
        for d, day in enumerate(self.days):

            print('--- Collisions detection day {:s}'.format(day))

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

            # Load poses
            f_names = self.day_f_names[d]
            map_t = np.array([np.float64(f.split('/')[-1][:-4]) for f in f_names], dtype=np.float64)

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

            input_FIFO = []
            pts_FIFO = []
            annot_FIFO = []
            name_FIFO = []

            # Create KDTree on the map
            print('Start loop')
            N = len(frame_names)
            for i, f_name in enumerate(frame_names):

                t = [time.time()]

                # Load points with annotation
                ply_name = join(frame_folder, f_name.split('/')[-1])
                data = read_ply(ply_name)
                f_points = np.vstack((data['x'], data['y'], data['z'])).T
                f_annot = data['classif']

                # Apply transf
                world_pts = np.hstack((f_points, np.ones_like(f_points[:, :1])))
                world_pts = np.matmul(world_pts, correct_H[i].T).astype(np.float32)[:, :3]

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

                # Mask of the future points
                not_ground = np.abs(np.dot((world_pts - plane_P), plane_N)) > 0.2
                f_mask = np.logical_and(not_ground, f_annot > 1.5)
                
                # Do not use ground and uncertain points
                flat_pts = world_pts[f_mask, :]
                flat_annot = f_annot[f_mask]
                flat_pts[:, 2] *= 0
                
                # Subsampling to a 2D PointCloud
                pts_2D, annots_2D = grid_subsampling(flat_pts,
                                                     labels=flat_annot,
                                                     sampleDl=dl_2D)

                # Add 2D points to FIFO
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
                        new_points[:, 2] = float(future_name) - float(name_FIFO[ind_orig])
                        stacked_pts = np.vstack((stacked_pts, new_points))
                        stacked_annot = np.vstack((stacked_annot, future_annot))

                    # Save as a 2D point cloud
                    ply_2D_name = join(out_folder, name_FIFO[ind_orig] + '_2D.ply')
                    write_ply(ply_2D_name,
                              [stacked_pts, stacked_annot],
                              ['x', 'y', 't', 'classif'])

                    # Get rid of oldest points
                    pts_FIFO.pop(0)
                    annot_FIFO.pop(0)
                    name_FIFO.pop(0)

                debug_n = 50
                if i % debug_n == debug_n - 1:

                    ply_name = join(out_folder, f_name.split('/')[-1])
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

        print('OK')
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
                 balance_classes=True,
                 load_data=True):
        PointCloudDataset.__init__(self, 'MyhalCollision')

        ##########################
        # Parameters for the files
        ##########################

        # Dataset folder
        self.path = '../Data/KPConv_data'

        # Original data path
        self.original_path = '../Data/Real'

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
        self.frames = []
        for seq in self.sequences:
            if self.set == 'test':
                velo_path = join(self.original_path, 'runs', seq, 'velodyne_frames')
            else:
                velo_path = join(self.original_path, 'collisions', seq)
            frames = np.array([vf[:-7] for vf in listdir(velo_path) if vf.endswith('_2D.ply')])
            order = np.argsort([float(ff) for ff in frames])
            frames = frames[order]
            self.frames.append(frames)

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
        self.label_to_names = {
            k: all_labels[v]
            for k, v in learning_map_inv.items()
        }

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

        # test

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

            #########################
            # Merge n_frames together
            #########################

            # Initiate merged points
            merged_points = np.zeros((0, 3), dtype=np.float32)
            merged_labels = np.zeros((0, ), dtype=np.int32)
            merged_coords = np.zeros((0, self.config.n_frames + 1), dtype=np.float32)

            # In case of validation also keep original point and reproj indices

            # Get center of the first frame in world coordinates
            p_origin = np.zeros((1, 4))
            p_origin[0, 3] = 1
            pose0 = self.poses[s_ind][f_ind]
            pose0_inv = np.linalg.inv(pose0)
            p0 = p_origin.dot(pose0.T)[:, :3]
            p0 = np.squeeze(p0)
            o_pts = None
            o_labels = None

            t += [time.time()]

            num_merged = 0
            while num_merged < self.config.n_frames and f_ind - num_merged >= 0:

                # Current frame pose
                pose = self.poses[s_ind][f_ind - num_merged]
                pose_inv = np.linalg.inv(pose)

                # Path of points and labels
                if self.set == 'test':
                    seq_path = join(self.original_path, 'simulated_runs', self.sequences[s_ind], 'sim_frames')
                else:
                    seq_path = join(self.original_path, 'annotated_frames', self.sequences[s_ind])
                velo_file = join(seq_path, self.frames[s_ind][f_ind - num_merged] + '.ply')

                # Read points (in original lidar coordinates)
                data = read_ply(velo_file)
                points = np.vstack((data['x'], data['y'], data['z'])).T

                # Place in world coordinates (so that vertical projection works even in case of tilt)
                hpoints = np.hstack((points, np.ones_like(points[:, :1])))
                world_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)[:, :3]
                #hpoints = np.hstack((world_points, np.ones_like(world_points[:, :1])))
                #points0 = np.sum(np.expand_dims(hpoints, 2) * pose0_inv.T, axis=1)[:, :3]

                if self.set == 'test':
                    # Fake labels
                    #sem_labels = np.zeros((points.shape[0],), dtype=np.int32)
                    sem_labels = data['cat']
                else:
                    # Read labels
                    sem_labels = data['classif']

                # In case of validation, keep the points in memory
                if self.set in ['validation', 'test'] and num_merged == 0:
                    o_pts = world_points[:, :3].astype(np.float32)
                    o_labels = sem_labels.astype(np.int32)

                # In case radius smaller than 5m, chose new center on a point of the wanted class or not
                if self.in_R < 5.0 and num_merged == 0:
                    if self.balance_classes:
                        wanted_ind = np.random.choice(np.where(sem_labels == wanted_label)[0])
                    else:
                        wanted_ind = np.random.choice(points.shape[0])
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
                input_inds = np.random.choice(n,
                                              size=self.max_in_p,
                                              replace=False)
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

            # Check ifFailed (probably because the cloud had 0 points)
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
            raise ValueError(
                'Only accepted input dimensions are 1, 2 and 4 (without and with XYZ)'
            )

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

            # Get groundtruth in 2D points format
            seq_path = join(self.original_path, 'collisions', self.sequences[s_ind])
            gt_file = join(seq_path, self.frames[s_ind][f_ind] + '_2D.ply')

            # Read points
            data = read_ply(gt_file)
            pts_2D = np.vstack((data['x'], data['y'])).T
            times_2D = data['t']
            labels_2D = data['classif']

            # Center on p0 and apply same augmentation
            pts_2D = (pts_2D - p0[:2]).astype(np.float32)
            pts_2D = np.hstack((pts_2D, np.zeros_like(pts_2D[:, :1])))
            pts_2D = np.sum(np.expand_dims(pts_2D, 2) * R, axis=1) * scale

            # For each time get the closest annotation
            future_dt = self.config.T_2D / self.config.n_2D_layers
            timestamps = np.arange(-(self.config.n_frames - 1) * future_dt, self.config.T_2D + 0.5 * future_dt, future_dt)
            future_dt = (timestamps[1] - timestamps[0]) / 2
            future_imgs = []
            try:
                for future_t in timestamps:

                    # Valid points for this timestamps are in the time range dt/2
                    # TODO: Here different valid times for different classes
                    valid_mask = np.abs(times_2D - future_t) < future_dt / 2
                    extension = 1
                    while np.sum(valid_mask) < 1 and extension < 5:
                        extension += 1
                        valid_mask = np.abs(times_2D - future_t) < future_dt * extension / 2

                    valid_pts = pts_2D[valid_mask, :]
                    valid_labels = labels_2D[valid_mask]
                    valid_times = times_2D[valid_mask]

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
                    future_imgs.append(np.stack((future_2, future_3, future_4), axis=2))

            except RuntimeError:
                # Temporary bug fix when no neighbors at all we just skip this one
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
            debug = self.config.input_threads == 0 and False
            if debug:
                print('Precesnce of each input class: ', input_classes)
                debug = debug and np.all(input_classes) and 4 < s_ind < 8
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
                for i in range(self.config.n_2D_layers + 1):
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
                fig1, anim = save_future_anim('results/gt_anim.gif', future_imgs)
                fast_save_future_anim('results/gt_anim.gif', future_imgs, zoom=10)

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

                in_folder = join(self.original_path, 'annotation', seq)
                in_file = join(in_folder, 'correct_traj_{:s}.pkl'.format(seq))
                with open(in_file, 'rb') as f:
                    transform_list = pickle.load(f)

                # Remove poses of ignored frames
                annot_path = join(self.original_path, 'annotated_frames', seq)
                annot_frames = np.array([vf[:-4] for vf in listdir(annot_path) if vf.endswith('.ply')])
                order = np.argsort([float(a_f) for a_f in annot_frames])
                annot_frames = annot_frames[order]
                pose_dict = {k: v for k, v in zip(annot_frames, transform_list)}
                self.poses.append([pose_dict[f] for f in self.frames[s_ind]])

        else:

            for s_ind, (seq, seq_frames) in enumerate(
                    zip(self.sequences, self.frames)):
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
            self.class_proportions = np.zeros((self.num_classes, ),
                                              dtype=np.int32)

            for s_ind, (seq, seq_frames) in enumerate(
                    zip(self.sequences, self.frames)):

                frame_mode = 'movable'
                seq_stat_file = join(self.path, seq,
                                     'stats_{:s}.pkl'.format(frame_mode))

                # Check if inputs have already been computed
                if False and exists(seq_stat_file):
                    # Read pkl
                    with open(seq_stat_file, 'rb') as f:
                        seq_class_frames, seq_proportions = pickle.load(f)

                else:

                    # Initiate dict
                    print(
                        'Preparing seq {:s} class frames. (Long but one time only)'
                        .format(seq))

                    # Class frames as a boolean mask
                    seq_class_frames = np.zeros(
                        (len(seq_frames), self.num_classes), dtype=np.bool)

                    # Proportion of each class
                    seq_proportions = np.zeros((self.num_classes, ),
                                               dtype=np.int32)

                    # Sequence path
                    seq_path = join(self.original_path, 'annotated_frames',
                                    seq)

                    # Read all frames
                    for f_ind, frame_name in enumerate(seq_frames):

                        # Path of points and labels
                        velo_file = join(seq_path, frame_name + '.ply')

                        # Read labels
                        data = read_ply(velo_file)
                        sem_labels = data['classif']

                        # Get present labels and there frequency
                        unique, counts = np.unique(sem_labels,
                                                   return_counts=True)

                        # Add this frame to the frame lists of all class present
                        frame_labels = np.array(
                            [self.label_to_idx[l] for l in unique],
                            dtype=np.int32)
                        seq_class_frames[f_ind, frame_labels] = True

                        # Add proportions
                        seq_proportions[frame_labels] += counts

                    # Save pickle
                    with open(seq_stat_file, 'wb') as f:
                        pickle.dump([seq_class_frames, seq_proportions], f)

                class_frames_bool = np.vstack(
                    (class_frames_bool, seq_class_frames))
                self.class_proportions += seq_proportions

            # Transform boolean indexing to int indices.
            self.class_frames = []
            for i, c in enumerate(self.label_values):
                if c in self.ignored_labels:
                    self.class_frames.append(
                        torch.zeros((0, ), dtype=torch.int64))
                else:
                    integer_inds = np.where(class_frames_bool[:, i])[0]
                    self.class_frames.append(
                        torch.from_numpy(integer_inds.astype(np.int64)))

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

        # Path of points and labels
        if self.set == 'test':
            seq_path = join(self.original_path, 'simulated_runs',
                            self.sequences[s_ind], 'sim_frames')
        else:
            seq_path = join(self.original_path, 'annotated_frames',
                            self.sequences[s_ind])

        velo_file = join(seq_path, self.frames[s_ind][f_ind] + '.ply')

        # Read points
        data = read_ply(velo_file)
        return np.vstack((data['x'], data['y'], data['z'])).T


class MyhalCollisionSampler(Sampler):
    """Sampler for MyhalCollision"""

    def __init__(self, dataset: MyhalCollisionDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

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
            for i, c in enumerate(self.dataset.label_values):
                if c not in self.dataset.ignored_labels:

                    # Get the potentials of the frames containing this class
                    class_potentials = self.dataset.potentials[
                        self.dataset.class_frames[i]]

                    # Get the indices to generate thanks to potentials
                    used_classes = self.dataset.num_classes - \
                        len(self.dataset.ignored_labels)
                    class_n = num_centers // used_classes + 1
                    if class_n < class_potentials.shape[0]:
                        _, class_indices = torch.topk(class_potentials,
                                                      class_n,
                                                      largest=False)
                    else:
                        class_indices = torch.randperm(
                            class_potentials.shape[0])
                    class_indices = self.dataset.class_frames[i][class_indices]

                    # Add the indices to the generated ones
                    gen_indices.append(class_indices)
                    gen_classes.append(class_indices * 0 + c)

                    # Update potentials
                    self.dataset.potentials[class_indices] = np.ceil(
                        self.dataset.potentials[class_indices])
                    self.dataset.potentials[class_indices] += np.random.rand(
                        class_indices.shape[0]) * 0.1 + 0.1

            # Stack the chosen indices of all classes
            gen_indices = torch.cat(gen_indices, dim=0)
            gen_classes = torch.cat(gen_classes, dim=0)

            # Shuffle generated indices
            rand_order = torch.randperm(gen_indices.shape[0])[:num_centers]
            gen_indices = gen_indices[rand_order]
            gen_classes = gen_classes[rand_order]

            # Update potentials (Change the order for the next epoch)
            self.dataset.potentials[gen_indices] = torch.ceil(
                self.dataset.potentials[gen_indices])
            self.dataset.potentials[gen_indices] += torch.from_numpy(
                np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1)

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

        print(
            '\nStarting Calibration of max_in_points value (use verbose=True for more details)'
        )
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
                    counts = [
                        np.sum(neighb_mat.numpy() < neighb_mat.shape[0],
                               axis=1) for neighb_mat in batch.neighbors
                    ]
                    hists = [
                        np.bincount(c, minlength=hist_n)[:hist_n]
                        for c in counts
                    ]
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
                        print(
                            message.format(i, estim_b,
                                           int(self.dataset.batch_limit[0])))

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
            print('Current value of max_in_points {:d}'.format(
                self.dataset.max_in_p))
            print('  > {:}{:.1f}% inputs are cropped{:}'.format(
                color, 100 * cropped_n / all_n, bcolors.ENDC))
            if cropped_n > 0.3 * all_n:
                print('\nTry a higher max_in_points value\n'.format(
                    100 * cropped_n / all_n))
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
