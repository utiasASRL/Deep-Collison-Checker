#
#
#      0====================0
#      |    PointMapSLAM    |
#      0====================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Definition of classes and function for PointMapSLAM.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 14/05/2020
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp, Rotation
import pickle

# OS functions
from os import listdir, makedirs, rename
from os.path import exists, join, isdir

# Other function of the project
from utils.ply import read_ply, write_ply
from slam.cpp_slam import update_pointmap, polar_normals, point_to_map_icp, \
    slam_on_sim_sequence, ray_casting_annot, slam_on_real_sequence
from slam.dev_slam import normal_filtering, bundle_icp, frame_H_to_points, estimate_normals_planarity_debug, \
    cart2pol, get_odometry, ssc_to_homo, extract_ground, save_trajectory

#from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.neighbors import KDTree

from utils.metrics import fast_confusion


class PointMap:

    def __init__(self, dl=1.0):

        self.dl = dl
        self.points = None
        self.normals = None
        self.scores = None
        self.counts = None
        self.movable_prob = None
        self.still_prob = None
        self.movable_count = None
        self.planes = None
        self.plane_inds = None

    def update(self, points, normals, scores):

        self.points, self.normals, self.scores, self.counts = update_pointmap(points, normals, scores,
                                                                              map_points=self.points,
                                                                              map_normals=self.normals,
                                                                              map_scores=self.scores,
                                                                              map_counts=self.counts,
                                                                              map_dl=self.dl)

    def update_movable_old(self, points, R, T):

        # Safe check
        if points.shape[0] < 2:
            return

        # Init movable scores
        if self.movable_prob is None:
            self.movable_prob = np.zeros_like(self.scores)
            self.still_prob = np.zeros_like(self.scores)
            self.movable_count = np.zeros_like(self.scores)

        # Project map_points in the frame coordiantes
        map_pts0 = self.points - T.astype(np.float32)
        map_pts0 = np.matmul(map_pts0, R.astype(np.float32))

        # Project all points into polar coordinates
        polar_points = cart2pol(points)
        polar_map0 = cart2pol(map_pts0)

        # Crop
        theta1, phi1 = np.min(polar_points[:, 1:], axis=0)
        theta2, phi2 = np.max(polar_points[:, 1:], axis=0)
        polar_mask = np.logical_and(polar_map0[:, 2] > phi1, polar_map0[:, 2] < phi2)
        polar_mask = np.logical_and(polar_mask, polar_map0[:, 1] > theta1)
        polar_mask = np.logical_and(polar_mask, polar_map0[:, 1] < theta2)
        polar_map = polar_map0[polar_mask]

        # Project in a grid
        theta_res = 1.29 * np.pi / 180
        phi_res_ratio = 0.2
        dr_threshold1 = 0.05
        dr_threshold2 = 0.15

        polar_points[:, 2] *= 1 / phi_res_ratio
        polar_map[:, 2] *= 1 / phi_res_ratio

        # Compute voxel indice for each frame point
        grid_indices = (np.floor(polar_points[:, 1:] / theta_res)).astype(int)

        # Limits of the grid
        min_grid_indices = np.amin(grid_indices, axis=0)
        max_grid_indices = np.amax(grid_indices, axis=0)

        # Number of cells in each direction
        deltaX, deltaY = max_grid_indices - min_grid_indices + 1
        frame_img = np.zeros((deltaX * deltaY, 3), np.float32)

        # Scalar equivalent to grid indices
        grid_indices0 = grid_indices - min_grid_indices
        scalar_indices = grid_indices0[:, 0] + grid_indices0[:, 1] * deltaX
        frame_img[scalar_indices, :] = polar_points
        # TODO: not good switch to C++

        # Compute voxel indice for each map point
        grid_indices = (np.floor(polar_map[:, 1:] / theta_res)).astype(int)
        grid_indices0 = grid_indices - min_grid_indices
        scalar_indices = grid_indices0[:, 0] + grid_indices0[:, 1] * deltaX

        # eliminate points that are not in a pixel of the frame
        map_frame_bool = frame_img[scalar_indices, 0] > 1.0

        # Compute radius difference
        map_frame_diff = frame_img[scalar_indices, 0] - polar_map[:, 0]

        # Do not consider points of the map that are behind the frame (they are not seen by a ray)
        map_frame_bool = np.logical_and(map_frame_bool, map_frame_diff > - dr_threshold2)

        # Get the boolean mask on the whole map
        map_frame_diff = map_frame_diff[map_frame_bool]
        polar_mask[polar_mask] = map_frame_bool

        # Update movable probability
        movable_prob = (map_frame_diff - dr_threshold1) / (dr_threshold2 - dr_threshold1)
        movable_prob = np.maximum(np.minimum(movable_prob, 1), 0)
        still_prob = (np.abs(map_frame_diff) - dr_threshold2) / (dr_threshold1 - dr_threshold2)
        still_prob = np.maximum(np.minimum(still_prob, 1), 0)
        self.movable_prob[polar_mask] += movable_prob
        self.still_prob[polar_mask] += still_prob
        self.movable_count[polar_mask] += 1

        #
        # write_ply('ttt_xyz.ply',
        #           [points, polar_points[:, 2]],
        #           ['x', 'y', 'z', 'phi'])
        #
        # map_pts = map_pts0[polar_mask]
        # write_ply('ttt_xyz_map.ply',
        #           [map_pts, map_frame_diff],
        #           ['x', 'y', 'z', 'f'])
        #
        # write_ply('ttt_rtp.ply',
        #           [polar_points, polar_points[:, 2]],
        #           ['x', 'y', 'z', 'phi'])
        #
        # polar_map = polar_map0[polar_mask]
        # polar_map[:, 2] *= 1/phi_res_ratio
        # write_ply('ttt_rtp_map.ply',
        #           [polar_map, map_frame_diff],
        #           ['x', 'y', 'z', 'f'])
        #
        # write_ply('ttt_rtp_img.ply',
        #           [frame_img, frame_img[:, 2]],
        #           ['x', 'y', 'z', 'phi'])

        return

    def update_movable(self, points, R, T,
                       theta_res=0.2 * 1.29 * np.pi / 180,
                       phi_res=0.1 * 1.29 * np.pi / 180,
                       verbose=0):

        t = [time.time()]

        # Safe check
        if points.shape[0] < 2:
            return

        # Init movable scores
        if self.movable_prob is None:
            self.movable_prob = np.zeros_like(self.scores)
            self.still_prob = np.zeros_like(self.scores)
            self.movable_count = np.zeros_like(self.scores)

        t += [time.time()]

        # C++ ray_casting annot
        new_movable_prob, new_movable_mask = map_frame_comp(points,
                                                            self.points,
                                                            R,
                                                            T,
                                                            theta_res,
                                                            phi_res,
                                                            self.dl)

        t += [time.time()]

        # Update movable probability
        self.movable_prob[new_movable_mask] += new_movable_prob[new_movable_mask]
        self.still_prob[new_movable_mask] += 1 - new_movable_prob[new_movable_mask]
        self.movable_count[new_movable_mask] += 1

        t += [time.time()]

        if verbose > 0:

            ti = 0
            print('Init .... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print('cpp ..... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print('update .. {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            print('*************************')

        return

    def frame_localization(self, frame_pts, frame_normals):

        ############
        # Parameters
        ############

        # Threshold on normal variation
        norm_thresh = 20 * np.pi / 180

        ##########################
        # Detect planes in the map
        ##########################

        if True or self.planes is None:

            print('Search planes in map')
            self.planes, self.plane_inds = map_plane_growing(self.points,
                                                             self.normals,
                                                             norm_thresh=20 * np.pi / 180,
                                                             dist_thresh=0.1,
                                                             min_points=100,
                                                             max_planes=1000,
                                                             map_dl=self.dl)
            print('Found: ', self.planes.shape[0], 'planes')

            # Compute area of convex hull of each plane
            print('Get areas')
            self.plane_areas = []
            features = np.zeros(self.plane_inds.shape, np.float32)
            for plane_i, plane in enumerate(self.planes):
                plane_points = self.points[self.plane_inds == plane_i]
                self.plane_areas += [plane_area(plane_points, plane[:3])]
                features[self.plane_inds == plane_i] = self.plane_areas[-1]
            print('Done')

            write_ply('cc_region_growing_map.ply',
                      [self.points, self.normals, self.plane_inds, features],
                      ['x', 'y', 'z', 'nx', 'ny', 'nz', 'planes', 'area'])

        ############################
        # Detect planes in the frame
        ############################

        print('Search planes in frame')
        planes, plane_inds = lidar_plane_growing(frame_pts,
                                                 frame_normals,
                                                 norm_thresh=20 * np.pi / 180,
                                                 dist_thresh=0.1,
                                                 min_points=100,
                                                 max_planes=1000)
        print('Found: ', planes.shape[0], 'planes')

        # Compute area of convex hull of each plane
        print('Get areas')
        frame_areas = []
        features = np.zeros(plane_inds.shape, np.float32)
        for plane_i, plane in enumerate(planes):
            plane_points = frame_pts[plane_inds == plane_i]
            frame_areas += [plane_area(plane_points, plane[:3])]
            features[plane_inds == plane_i] = frame_areas[-1]

        print('Done')

        write_ply('cc_region_growing_frame.ply',
                  [frame_pts, frame_normals, plane_inds, features],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz', 'planes', 'area'])

        a = 1 / 0
        # TODO:
        #  Get planes from the map too
        #  Then do the association
        #     - start with biggest frame plane (geometrically)
        #     - This limits the number of possible plane from the map it can be associated with
        #     - Get same possible associations for second and third (non parallel) biggest planes
        #     - Loop on all possible asso and get the best candidate
        #   => Rank candidates by their relative size (the closest to the lidar plane size as long as it is bigger)

        print(planes.shape, planes.dtype)

        return planes, plane_inds

    def apply_mask(self, mask):
        if self.points is not None:
            self.points = self.points[mask]
        if self.normals is not None:
            self.normals = self.normals[mask]
        if self.scores is not None:
            self.scores = self.scores[mask]
        if self.counts is not None:
            self.counts = self.counts[mask]
        if self.movable_prob is not None:
            self.movable_prob = self.movable_prob[mask]
        if self.still_prob is not None:
            self.still_prob = self.still_prob[mask]
        if self.movable_count is not None:
            self.movable_count = self.movable_count[mask]

    def reset(self):
        self.points = None
        self.normals = None
        self.scores = None
        self.counts = None
        self.movable_prob = None
        self.still_prob = None
        self.movable_count = None
        self.planes = None
        self.plane_inds = None


def apply_motion_distortion(points, phi0, last_H, new_H, normals=None):

    # Get the phi of the current transform
    phi = (3 * np.pi / 2 - np.arctan2(points[:, 1], points[:, 0]).astype(np.float32)) % (2 * np.pi)
    phi1 = np.max(phi)

    # Create a slerp interpolation function for the rotation part of the transform
    R1 = last_H[:3, :3]
    R2 = new_H[:3, :3]
    key_rots = Rotation.from_matrix(np.stack((R1, R2), axis=0))
    slerp = Slerp([phi0, phi1], key_rots)
    interp_R = slerp(phi).as_matrix()

    # Create linear interpolation for translation
    T0 = last_H[:3, 3:]
    T1 = new_H[:3, 3:]
    phit = (phi - phi0) / (phi1 - phi0)
    interp_T = (1 - phit) * T0 + phit * T1

    world_pts = np.expand_dims(points, 1)
    world_pts = np.matmul(world_pts, interp_R.transpose((0, 2, 1))).astype(np.float32)

    world_pts = np.squeeze(world_pts)
    world_pts += interp_T.T

    if normals is None:
        return world_pts, phi1
    else:
        world_normals = np.expand_dims(normals, 1)
        world_normals = np.matmul(world_normals, interp_R.transpose((0, 2, 1))).astype(np.float32)
        world_normals = np.squeeze(world_normals)
        return world_pts, world_normals, phi1


def get_frame_slices(points, phi0, last_H, new_H, n_slices):

    # Get the phi of the current transform
    phi = (3 * np.pi / 2 - np.arctan2(points[:, 1], points[:, 0]).astype(np.float32)) % (2 * np.pi)
    phi1 = np.min(phi)
    phi2 = np.max(phi)

    # Get slice limits
    slices_limits = np.linspace(phi1, phi2, num=n_slices + 1)

    slices_phi = []
    slices_points = []
    for i, l1 in enumerate(slices_limits[:-1]):
        l2 = slices_limits[i + 1]
        slices_phi.append((l2 + l1) / 2)
        slice_mask = np.logical_and(phi > l1, phi < l2)
        slices_points.append(points[slice_mask, :])
    slices_phi = np.array(slices_phi, dtype=np.float32)

    # Create a slerp interpolation function for the rotation part of the transform
    R0 = last_H[:3, :3]
    R2 = new_H[:3, :3]
    key_rots = Rotation.from_matrix(np.stack((R0, R2), axis=0))
    slerp = Slerp([phi0, phi2], key_rots)
    slices_R = slerp(slices_phi).as_matrix()

    # Create linear interpolation for translation
    T0 = last_H[:3, 3:]
    T2 = new_H[:3, 3:]
    phit = (slices_phi - phi0) / (phi2 - phi0)
    slices_T = (1 - phit) * T0 + phit * T2

    return slices_points, slices_R, slices_T.T, phi2


def pointmap_slam_v0debug(verbose=2):

    ############
    # Parameters
    ############

    data_path, days, day_f_times = get_test_frames(area_center=np.array([-240, -527, 12]),
                                                   area_radius=30.0,
                                                   only_day_1=True,
                                                   verbose=1)

    # Out files
    out_folder = join(data_path, 'day_ply')
    if not exists(out_folder):
        makedirs(out_folder)

    # Stride (nb of frames skipped for transformations)
    frame_stride = 2

    # Normal estimation parameters
    score_thresh = 0.99

    # Pointcloud filtering parameters
    map_voxel_size = 0.1
    frame_voxel_size = 0.1

    # Group of frames saved together
    init_steps = 10
    save_group = 100

    ##########################
    # Start first pass of SLAM
    ##########################

    for d, day in enumerate(days):

        # List of frames we are trying to map
        frames_folder = join(data_path, 'raw_ply', day)
        f_times = [f_t for f_t in day_f_times[d][::frame_stride]]
        frame_names = [join(frames_folder, '{:.0f}.ply'.format(f_t)) for f_t in f_times]

        # List of transformation we are trying to optimize
        transform_list = [np.eye(4) for _ in f_times]

        # # Bundle ICP to initialize frame positions
        # init_transforms = bundle_icp(frame_names[:init_steps],
        #                              bundle_size=6,
        #                              score_thresh=0.99,
        #                              frame_voxel_size=-1,
        #                              verbose=1)

        # Initiate map
        pointmap = PointMap(dl=map_voxel_size)
        phi0 = 0
        last_saved_frames = 0
        FPS = 0
        N = len(f_times)

        # Test mapping
        for i, init_H in enumerate(transform_list):

            t = [time.time()]

            # Load ply format points
            data = read_ply(frame_names[i])
            points = np.vstack((data['x'], data['y'], data['z'])).T

            t += [time.time()]

            # Get normals
            normals, planarity, linearity = polar_normals(points, radius=1.5, h_scale=0.5)
            norm_scores = planarity + linearity

            write_ply('__clean_normals_frame.ply'.format(i),
                      [points, normals, norm_scores],
                      ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores'])

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
                points = tmpmap.points
                normals = tmpmap.normals
                norm_scores = tmpmap.scores

                write_ply('__subsample_normals_frame.ply'.format(i),
                          [points, normals, norm_scores],
                          ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores'])

            t += [time.time()]

            if i < 1:
                world_H = transform_list[i]
            else:
                all_H, rms, planar_rms = point_to_map_icp(points, norm_scores,
                                                          pointmap.points,
                                                          pointmap.normals,
                                                          pointmap.scores,
                                                          init_H=transform_list[i - 1],
                                                          init_phi=phi0,
                                                          n_samples=1000,
                                                          max_pairing_dist=0.2,
                                                          max_iter=50,
                                                          rotDiffThresh=0.004,
                                                          transDiffThresh=0.02,
                                                          avg_steps=5)
                world_H = all_H[-1]

            world_pts = np.hstack((points, np.ones_like(points[:, :1])))
            world_pts = np.matmul(world_pts, world_H.T).astype(np.float32)[:, :3]
            write_ply('__align_distort_frame.ply'.format(i),
                      [world_pts],
                      ['x', 'y', 'z'])

            world_points, world_normals, phi1 = apply_motion_distortion(points,
                                                                        phi0,
                                                                        transform_list[i - 1],
                                                                        world_H,
                                                                        normals=normals)
            if i > 0:
                write_ply('__map.ply'.format(i),
                          [pointmap.points, pointmap.normals, pointmap.scores, pointmap.counts],
                          ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])
            write_ply('__aligned_undistort_frame.ply'.format(i),
                      [world_points, world_normals],
                      ['x', 'y', 'z', 'nx', 'ny', 'nz'])

            print(world_H * np.linalg.inv(transform_list[i - 1]))
            if i > 145:
                a = 1 / 0

            # TODO: At one point traj fail why? We see that traj frame are deformed so maybe transform is not rotation
            #  anymore... Why ? Related to double free or courrptio nerror?

            # TODO: frame stride=1 does not work why?

            # write_ply('testundistort.ply',
            #           [world_pts, phi, phit],
            #           ['x', 'y', 'z', 'phi', 't'])

            # Save phi for next frame
            phi0 = phi1 - frame_stride * 2 * np.pi

            # Apply transf
            transform_list[i] = world_H
            # world_pts = np.hstack((points, np.ones_like(points[:, :1])))
            # world_pts = np.matmul(world_pts, world_H.T).astype(np.float32)[:, :3]
            # world_normals = np.matmul(normals, world_H[:3, :3].T).astype(np.float32)

            t += [time.time()]

            # Update map
            # TODO: Update map with all the points and not only filtered frame?
            # TODO: update only with closer points?
            pointmap.update(world_points, world_normals, norm_scores)

            # TODO: reduce map size as we advance

            t += [time.time()]

            if verbose == 2:
                ti = 0
                print('Load ............ {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
                ti += 1
                print('Preprocessing ... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
                ti += 1
                print('Align ........... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
                ti += 1
                print('Mapping ......... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))

                if i % 10 == 0:
                    write_ply('debug_map_{:03d}.ply'.format(i),
                              [pointmap.points, pointmap.normals, pointmap.scores, pointmap.counts],
                              ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

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
                print(fmt_str.format(i,
                                     100 * (i + 1) / N,
                                     hours, minutes, seconds,
                                     FPS))

            # Save groups of 100 frames together
            if (i > last_saved_frames + save_group + 1):
                all_points = []
                all_traj_pts = []
                all_traj_clrs = []
                i0 = last_saved_frames
                i1 = i0 + save_group
                phi0 = 0
                for i, world_H in enumerate(transform_list[i0: i1]):
                    # Load ply format points
                    data = read_ply(frame_names[i0 + i])
                    points = np.vstack((data['x'], data['y'], data['z'])).T

                    # Apply transf
                    world_pts, phi1 = apply_motion_distortion(points,
                                                              phi0,
                                                              transform_list[i - 1],
                                                              world_H)
                    # Save phi for next frame
                    phi0 = phi1 - frame_stride * 2 * np.pi

                    # Save frame
                    world_pts = np.hstack((world_pts, np.ones_like(world_pts[:, :1])))
                    world_pts[:, 3] = i0 + i
                    all_points.append(world_pts)

                    # also save trajectory
                    traj_pts, traj_clrs = frame_H_to_points(world_H, size=0.1)
                    traj_pts = np.hstack((traj_pts, np.ones_like(traj_pts[:, :1]) * (i0 + i)))
                    all_traj_pts.append(traj_pts.astype(np.float32))
                    all_traj_clrs.append(traj_clrs)

                last_saved_frames += save_group
                filename = 'debug_icp_{:05d}.ply'.format(i0)
                write_ply(filename,
                          [np.vstack(all_points)],
                          ['x', 'y', 'z', 't'])
                filename = 'debug_icp_{:05d}_traj.ply'.format(i0)
                write_ply(filename,
                          [np.vstack(all_traj_pts), np.vstack(all_traj_clrs)],
                          ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

        #################
        # Post processing
        #################

        all_points = []
        all_traj_pts = []
        all_traj_clrs = []
        i0 = last_saved_frames
        phi0 = 0
        for i, world_H in enumerate(transform_list[i0:]):
            # Load ply format points
            data = read_ply(frame_names[i0 + i])
            points = np.vstack((data['x'], data['y'], data['z'])).T

            # Apply transf
            world_pts, phi1 = apply_motion_distortion(points,
                                                      phi0,
                                                      transform_list[i - 1],
                                                      world_H)
            # Save phi for next frame
            phi0 = phi1 - frame_stride * 2 * np.pi

            # Save frame
            world_pts = np.hstack((world_pts, np.ones_like(world_pts[:, :1])))
            world_pts[:, 3] = i0 + i
            all_points.append(world_pts)

            # also save trajectory
            traj_pts, traj_clrs = frame_H_to_points(world_H, size=0.1)
            traj_pts = np.hstack((traj_pts, np.ones_like(traj_pts[:, :1]) * (i0 + i)))
            all_traj_pts.append(traj_pts.astype(np.float32))
            all_traj_clrs.append(traj_clrs)

        last_saved_frames += save_group
        filename = 'debug_icp_{:05d}.ply'.format(i0)
        write_ply(filename,
                  [np.vstack(all_points)],
                  ['x', 'y', 'z', 't'])
        filename = 'debug_icp_{:05d}_traj.ply'.format(i0)
        write_ply(filename,
                  [np.vstack(all_traj_pts), np.vstack(all_traj_clrs)],
                  ['x', 'y', 'z', 't', 'red', 'green', 'blue'])


def pointmap_slam_v1(verbose=1, dataset_name='NCLT'):

    ############
    # Parameters
    ############

    if dataset_name == 'NCLT':
        data_path, days, day_f_times = get_test_frames(area_center=np.array([-220, -527, 12]),
                                                       area_radius=25.0,
                                                       only_day_1=True,
                                                       verbose=1)
    else:
        data_path, days, day_f_times = get_MyhalSim_frames()

    # Out files
    out_folder = join(data_path, 'day_ply')
    if not exists(out_folder):
        makedirs(out_folder)

    # Stride (nb of frames skipped for transformations)
    frame_stride = 2
    # TODO: frame stride=1 does not work why?
    #  Reason is motion distortion. First phi of current frame is very close to last phi of last frame, therefore,
    #  with motion distortion, ICP cannot move these points from their position. If because of wrong motion distortion
    #  (not linear) these points are not aligned, then they will create a bias that is going to make the ICP diverge

    # Normal estimation parameters
    score_thresh = 0.99

    # Pointcloud filtering parameters
    map_voxel_size = 0.08
    frame_voxel_size = 0.2

    # Group of frames saved together
    save_group = 50

    ##########################
    # Start first pass of SLAM
    ##########################

    for d, day in enumerate(days):

        out_day_folder = join(out_folder, day)
        if not exists(out_day_folder):
            makedirs(out_day_folder)

        for folder_name in ['trajectory', 'map', 'frames']:
            if not exists(join(out_day_folder, folder_name)):
                makedirs(join(out_day_folder, folder_name))

        # List of frames we are trying to map
        frames_folder = join(data_path, 'raw_ply', day)
        f_times = [f_t for f_t in day_f_times[d][::frame_stride]]
        frame_names = [join(frames_folder, '{:.0f}.ply'.format(f_t)) for f_t in f_times]

        # List of transformation we are trying to optimize
        transform_list = [np.eye(4) for _ in f_times]

        # Initiate map
        pointmap = PointMap(dl=map_voxel_size)
        phi0 = 0
        last_saved_frames = 0
        last_saved_phi = 0
        FPS = 0
        N = len(f_times)

        # Test mapping
        for i, init_H in enumerate(transform_list):

            t = [time.time()]

            # Load ply format points
            data = read_ply(frame_names[i])
            points = np.vstack((data['x'], data['y'], data['z'])).T

            t += [time.time()]

            # Safe check for points equql to zero
            hr = np.sqrt(np.sum(points[:, :2]**2, axis=1))
            points = points[hr > 1e-6]

            if np.sum((hr < 1e-6).astype(np.int32)) > 0:
                print('Warning: lidar frame with invalid points: frame_names[i].\nCorrected')
                write_ply(frame_names[i],
                          [points],
                          ['x', 'y', 'z'])

            # Get normals
            normals, planarity, linearity = polar_normals(points, radius=1.5, h_scale=0.5)
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

            t += [time.time()]

            if i < 1:
                world_H = transform_list[i]

                # Apply transf (no motion distortion for first frame)
                world_points = (np.matmul(points, world_H[:3, :3].T) + world_H[:3, 3]).astype(np.float32)
                world_normals = (np.matmul(normals, world_H[:3, :3].T)).astype(np.float32)

                # Get the first phi1 for motion distortion
                phi = (3 * np.pi / 2 - np.arctan2(points[:, 1], points[:, 0]).astype(np.float32)) % (2 * np.pi)
                phi1 = np.max(phi)

            else:
                if i < 2:
                    motion_distortion = False
                else:
                    motion_distortion = True

                all_H, rms, planar_rms = point_to_map_icp(sub_points, sub_norm_scores,
                                                          pointmap.points,
                                                          pointmap.normals,
                                                          pointmap.scores,
                                                          init_H=transform_list[i - 1],
                                                          init_phi=phi0,
                                                          motion_distortion=motion_distortion,
                                                          n_samples=1000,
                                                          max_pairing_dist=0.2,
                                                          max_iter=50,
                                                          rotDiffThresh=0.004,
                                                          transDiffThresh=0.02,
                                                          avg_steps=5)
                world_H = all_H[-1]

                # Apply transf
                transform_list[i] = world_H
                world_points, world_normals, phi1 = apply_motion_distortion(points,
                                                                            phi0,
                                                                            transform_list[i - 1],
                                                                            world_H,
                                                                            normals=normals)

            if i > 3000:
                a = 1 / 0

            # Save phi for next frame
            phi0 = phi1 - frame_stride * 2 * np.pi

            t += [time.time()]

            # TODO: problem with motion distortion, at mapping 120 - 130. Motion distortion is wrong... Why?
            #   => the motion distortion is not linear (if robot encounters a step => bump in traj) or here because
            #      segway is oscillating

            # Reset map for the first motion distorted clouds
            if i < 2:
                pointmap.reset()

            # Update map
            # TODO: Update map with all the points and not only filtered frame?
            # TODO: update only with closer points?
            pointmap.update(world_points, world_normals, norm_scores)

            if i % 10 == 0:
                filename = join(out_day_folder, 'map', 'debug_map_{:03d}.ply'.format(i))
                write_ply(filename,
                          [pointmap.points, pointmap.normals, pointmap.scores, pointmap.counts],
                          ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

            t += [time.time()]

            if verbose == 2:
                ti = 0
                print('Load ............ {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
                ti += 1
                print('Preprocessing ... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
                ti += 1
                print('Align ........... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
                ti += 1
                print('Mapping ......... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))

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
                print(fmt_str.format(i,
                                     100 * (i + 1) / N,
                                     hours, minutes, seconds,
                                     FPS))

            # Save groups of 100 frames together
            if (i > last_saved_frames + save_group + 1):
                all_points = []
                all_traj_pts = []
                all_traj_clrs = []
                i0 = last_saved_frames
                i1 = i0 + save_group
                tmp_phi0 = last_saved_phi
                for i, world_H in enumerate(transform_list[i0:i1]):
                    # Load ply format points
                    data = read_ply(frame_names[i0 + i])
                    points = np.vstack((data['x'], data['y'], data['z'])).T

                    # Apply transf
                    world_pts, phi1 = apply_motion_distortion(points,
                                                              tmp_phi0,
                                                              transform_list[i0 + i - 1],
                                                              world_H)
                    # Save phi for next frame
                    tmp_phi0 = phi1 - frame_stride * 2 * np.pi

                    # Save frame
                    world_pts = np.hstack((world_pts, np.ones_like(world_pts[:, :1])))
                    world_pts[:, 3] = i0 + i
                    all_points.append(world_pts)

                    # also save trajectory
                    traj_pts, traj_clrs = frame_H_to_points(world_H, size=0.1)
                    traj_pts = np.hstack((traj_pts, np.ones_like(traj_pts[:, :1]) * (i0 + i)))
                    all_traj_pts.append(traj_pts.astype(np.float32))
                    all_traj_clrs.append(traj_clrs)

                last_saved_frames += save_group
                last_saved_phi = tmp_phi0
                filename = join(out_day_folder, 'frames', 'debug_icp_{:05d}.ply'.format(i0))
                write_ply(filename,
                          [np.vstack(all_points)],
                          ['x', 'y', 'z', 't'])
                filename = join(out_day_folder, 'trajectory', 'debug_icp_{:05d}_traj.ply'.format(i0))
                write_ply(filename,
                          [np.vstack(all_traj_pts), np.vstack(all_traj_clrs)],
                          ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

        #################
        # Post processing
        #################

        all_points = []
        all_traj_pts = []
        all_traj_clrs = []
        i0 = last_saved_frames
        tmp_phi0 = last_saved_phi
        for i, world_H in enumerate(transform_list[i0:]):
            # Load ply format points
            data = read_ply(frame_names[i0 + i])
            points = np.vstack((data['x'], data['y'], data['z'])).T

            # Apply transf
            world_pts, phi1 = apply_motion_distortion(points,
                                                      tmp_phi0,
                                                      transform_list[i0 + i - 1],
                                                      world_H)
            # Save phi for next frame
            tmp_phi0 = phi1 - frame_stride * 2 * np.pi

            # Save frame
            world_pts = np.hstack((world_pts, np.ones_like(world_pts[:, :1])))
            world_pts[:, 3] = i0 + i
            all_points.append(world_pts)

            # also save trajectory
            traj_pts, traj_clrs = frame_H_to_points(world_H, size=0.1)
            traj_pts = np.hstack((traj_pts, np.ones_like(traj_pts[:, :1]) * (i0 + i)))
            all_traj_pts.append(traj_pts.astype(np.float32))
            all_traj_clrs.append(traj_clrs)

        last_saved_frames += save_group
        filename = join(out_day_folder, 'frames', 'debug_icp_{:05d}.ply'.format(i0))
        write_ply(filename,
                  [np.vstack(all_points)],
                  ['x', 'y', 'z', 't'])
        filename = join(out_day_folder, 'trajectory', 'debug_icp_{:05d}_traj.ply'.format(i0))
        write_ply(filename,
                  [np.vstack(all_traj_pts), np.vstack(all_traj_clrs)],
                  ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

        # Save alignments
        filename = join(out_day_folder, 'PointSLAM_{:s}.pkl'.format(day))
        with open(filename, 'wb') as file:
            pickle.dump((frame_names,
                         transform_list,
                         frame_stride,
                         pointmap), file)


def pointmap_slam(dataset,
                  frame_stride=2,
                  map_voxel_size=0.08,
                  frame_voxel_size=0.2,
                  save_group=50,
                  score_thresh=0.99,
                  verbose=1):

    #################
    # Init parameters
    #################

    # Out files
    out_folder = join(dataset.data_path, 'slam_offline')
    if not exists(out_folder):
        makedirs(out_folder)

    ##########################
    # Start first pass of SLAM
    ##########################

    for d, day in enumerate(dataset.days):

        if dataset.only_day_1 and d > 0:
            break

        out_day_folder = join(out_folder, day)
        if not exists(out_day_folder):
            makedirs(out_day_folder)

        for folder_name in ['trajectory', 'map', 'frames']:
            if not exists(join(out_day_folder, folder_name)):
                makedirs(join(out_day_folder, folder_name))

        # List of frames we are trying to map
        frame_names = dataset.day_f_names[d][::frame_stride]

        # List of transformation we are trying to optimize
        transform_list = [np.eye(4) for _ in frame_names]

        # Initiate map
        pointmap = PointMap(dl=map_voxel_size)
        phi0 = 0
        last_saved_frames = 0
        last_saved_phi = 0
        FPS = 0
        N = len(frame_names)

        # Test mapping
        for i, init_H in enumerate(transform_list):

            t = [time.time()]

            # Load ply format points
            points = dataset.load_frame_points(frame_names[i])

            t += [time.time()]

            # Get normals
            normals, planarity, linearity = polar_normals(points, radius=1.5, h_scale=0.5)
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

            t += [time.time()]

            if i < 1:
                world_H = transform_list[i]

                # Apply transf (no motion distortion for first frame)
                world_points = (np.matmul(points, world_H[:3, :3].T) + world_H[:3, 3]).astype(np.float32)
                world_normals = (np.matmul(normals, world_H[:3, :3].T)).astype(np.float32)

                # Get the first phi1 for motion distortion
                phi = (3 * np.pi / 2 - np.arctan2(points[:, 1], points[:, 0]).astype(np.float32)) % (2 * np.pi)
                phi1 = np.max(phi)

            else:
                if i < 2:
                    motion_distortion = False
                else:
                    motion_distortion = True

                all_H, rms, planar_rms = point_to_map_icp(sub_points, sub_norm_scores,
                                                          pointmap.points,
                                                          pointmap.normals,
                                                          pointmap.scores,
                                                          init_H=transform_list[i - 1],
                                                          init_phi=phi0,
                                                          motion_distortion=motion_distortion,
                                                          n_samples=1000,
                                                          max_pairing_dist=0.2,
                                                          max_iter=50,
                                                          rotDiffThresh=0.004,
                                                          transDiffThresh=0.02,
                                                          avg_steps=5)
                world_H = all_H[-1]

                # Apply transf
                transform_list[i] = world_H
                world_points, world_normals, phi1 = apply_motion_distortion(points,
                                                                            phi0,
                                                                            transform_list[i - 1],
                                                                            world_H,
                                                                            normals=normals)

            # Save phi for next frame
            phi0 = phi1 - frame_stride * 2 * np.pi

            t += [time.time()]

            # TODO: problem with motion distortion, at mapping 120 - 130. Motion distortion is wrong... Why?
            #   => the motion distortion is not linear (if robot encounters a step => bump in traj) or here because
            #      segway is oscillating

            # Reset map for the first motion distorted clouds
            if i < 2:
                pointmap.reset()

            # Update map
            # TODO: Update map with all the points and not only filtered frame?
            # TODO: update only with closer points?
            pointmap.update(world_points, world_normals, norm_scores)

            if i % 10 == 0:
                filename = join(out_day_folder, 'map', 'debug_map_{:03d}.ply'.format(i))
                write_ply(filename,
                          [pointmap.points, pointmap.normals, pointmap.scores, pointmap.counts],
                          ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

            t += [time.time()]

            if verbose == 2:
                ti = 0
                print('Load ............ {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
                ti += 1
                print('Preprocessing ... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
                ti += 1
                print('Align ........... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
                ti += 1
                print('Mapping ......... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))

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
                print(fmt_str.format(i,
                                     100 * (i + 1) / N,
                                     hours, minutes, seconds,
                                     FPS))

            # Save groups of 100 frames together
            if (i > last_saved_frames + save_group + 1):
                all_points = []
                all_traj_pts = []
                all_traj_clrs = []
                i0 = last_saved_frames
                i1 = i0 + save_group
                tmp_phi0 = last_saved_phi
                for i, world_H in enumerate(transform_list[i0:i1]):

                    # Load points
                    points = dataset.load_frame_points(frame_names[i0 + i])

                    # Apply transf
                    world_pts, phi1 = apply_motion_distortion(points,
                                                              tmp_phi0,
                                                              transform_list[i0 + i - 1],
                                                              world_H)
                    # Save phi for next frame
                    tmp_phi0 = phi1 - frame_stride * 2 * np.pi

                    # Save frame
                    world_pts = np.hstack((world_pts, np.ones_like(world_pts[:, :1])))
                    world_pts[:, 3] = i0 + i
                    all_points.append(world_pts)

                    # also save trajectory
                    traj_pts, traj_clrs = frame_H_to_points(world_H, size=0.1)
                    traj_pts = np.hstack((traj_pts, np.ones_like(traj_pts[:, :1]) * (i0 + i)))
                    all_traj_pts.append(traj_pts.astype(np.float32))
                    all_traj_clrs.append(traj_clrs)

                last_saved_frames += save_group
                last_saved_phi = tmp_phi0
                filename = join(out_day_folder, 'frames', 'debug_icp_{:05d}.ply'.format(i0))
                write_ply(filename,
                          [np.vstack(all_points)],
                          ['x', 'y', 'z', 't'])
                filename = join(out_day_folder, 'trajectory', 'debug_icp_{:05d}_traj.ply'.format(i0))
                write_ply(filename,
                          [np.vstack(all_traj_pts), np.vstack(all_traj_clrs)],
                          ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

        #################
        # Post processing
        #################

        filename = join(out_day_folder, 'map', 'debug_map_final.ply')
        write_ply(filename,
                  [pointmap.points, pointmap.normals, pointmap.scores, pointmap.counts],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz', 'scores', 'counts'])

        all_points = []
        all_traj_pts = []
        all_traj_clrs = []
        i0 = last_saved_frames
        tmp_phi0 = last_saved_phi
        for i, world_H in enumerate(transform_list[i0:]):

            # Load points
            points = dataset.load_frame_points(frame_names[i0 + i])

            # Apply transf
            world_pts, phi1 = apply_motion_distortion(points,
                                                      tmp_phi0,
                                                      transform_list[i0 + i - 1],
                                                      world_H)
            # Save phi for next frame
            tmp_phi0 = phi1 - frame_stride * 2 * np.pi

            # Save frame
            world_pts = np.hstack((world_pts, np.ones_like(world_pts[:, :1])))
            world_pts[:, 3] = i0 + i
            all_points.append(world_pts)

            # also save trajectory
            traj_pts, traj_clrs = frame_H_to_points(world_H, size=0.1)
            traj_pts = np.hstack((traj_pts, np.ones_like(traj_pts[:, :1]) * (i0 + i)))
            all_traj_pts.append(traj_pts.astype(np.float32))
            all_traj_clrs.append(traj_clrs)

        last_saved_frames += save_group
        filename = join(out_day_folder, 'frames', 'debug_icp_{:05d}.ply'.format(i0))
        write_ply(filename,
                  [np.vstack(all_points)],
                  ['x', 'y', 'z', 't'])
        filename = join(out_day_folder, 'trajectory', 'debug_icp_{:05d}_traj.ply'.format(i0))
        write_ply(filename,
                  [np.vstack(all_traj_pts), np.vstack(all_traj_clrs)],
                  ['x', 'y', 'z', 't', 'red', 'green', 'blue'])

        # Save alignments
        filename = join(out_day_folder, 'PointSLAM_{:s}.pkl'.format(day))
        with open(filename, 'wb') as file:
            pickle.dump((frame_names,
                         transform_list,
                         frame_stride,
                         pointmap), file)


def annotation_on_gt_OLD(dataset, movable_threshold=0.9, still_threshold=0.5):

    #################################
    # Annotate each day independantly
    #################################

    # List of the annotated maps of the days
    day_maps = []
    day_frame_names = []
    day_transform_lists = []
    day_frame_stride = []
    long_term_probs = []
    short_term_probs = []
    classifs = []
    for d, day in enumerate(dataset.days):

        if dataset.only_day_1 and d > 0:
            break

        # Out files
        out_folder = join(dataset.data_path, 'annotation', day)
        if not exists(out_folder):
            makedirs(out_folder)

        # Load gt mapping
        in_folder = join(dataset.data_path, 'slam_gt', day)
        with open(join(in_folder, 'gt_map_{:s}.pkl'.format(day)), 'rb') as file:
            frame_names, transform_list, frame_stride, pointmap = pickle.load(file)

        # Keep the original data that we want to annotate
        N = pointmap.points.shape[0]
        classifs.append(np.zeros((N,), dtype=np.int32))
        short_term_probs.append(np.zeros((N,), dtype=np.float32))
        long_term_probs.append(np.zeros((N,), dtype=np.float32))

        # Extract ground
        ground_mask = extract_map_ground(pointmap, out_folder)
        pointmap.apply_mask(np.logical_not(ground_mask))
        classifs[d][ground_mask] = 1

        # Annotate the rest of the points
        annotated_map = detect_short_term_movables(dataset,
                                                   pointmap,
                                                   frame_names,
                                                   transform_list,
                                                   frame_stride,
                                                   out_folder)

        # Get probabilities
        movable_prob = annotated_map.movable_prob / (annotated_map.movable_count + 1e-6)
        still_prob = annotated_map.still_prob / (annotated_map.movable_count + 1e-6)
        movable_prob[annotated_map.movable_count < 10] = 0
        still_prob[annotated_map.movable_count < 10] = 0

        # Extract short-term movables
        movable_mask = movable_prob > movable_threshold
        movable_points = annotated_map.points[movable_mask]
        annotated_map.apply_mask(np.logical_not(movable_mask))
        write_ply(join(out_folder, 'short_term.ply'),
                  [movable_points],
                  ['x', 'y', 'z'])

        # Reproject probs and class on riginal points
        short_term_probs[d][np.logical_not(ground_mask)] = movable_prob
        movable_fullmask = np.logical_not(ground_mask)
        movable_fullmask[movable_fullmask] = movable_mask
        classifs[d][movable_fullmask] = 3

        day_maps.append(annotated_map)
        day_frame_names.append(frame_names)
        day_transform_lists.append(transform_list)
        day_frame_stride.append(frame_stride)

    ########################################
    # Second path between days for long term
    ########################################

    if dataset.only_day_1:
        return

    # List of the annotated maps of the days
    for d, day in enumerate(dataset.days):

        # Out files
        out_folder = join(dataset.data_path, 'annotation', day)
        if not exists(out_folder):
            makedirs(out_folder)

        # Use frames of all days to annotate the current map
        frame_names = [frame_names0
                       for d0, frame_names0 in enumerate(day_frame_names)
                       if d0 != d]
        transform_list = [np.stack(transform_list0, axis=0)
                          for d0, transform_list0 in enumerate(day_transform_lists)
                          if d0 != d]
        frame_stride = 1

        # Annotate the rest of the points
        longterm_prob = detect_long_term_movables(dataset,
                                                  day_maps[d],
                                                  frame_names,
                                                  transform_list,
                                                  frame_stride,
                                                  out_folder)

        # Use frames of current day for still probabilities
        annotated_map = detect_still_objects_old(dataset,
                                                 day_maps[d],
                                                 day_frame_names[d],
                                                 day_transform_lists[d],
                                                 frame_stride,
                                                 out_folder)

        # Get still probabilities
        still_prob = annotated_map.still_prob / (annotated_map.movable_count + 1e-6)
        still_prob[annotated_map.movable_count < 10] = 0

        # Gather all info and save into one file
        longterm_mask = (classifs[d] == 0)
        longterm_mask[longterm_mask] = longterm_prob > movable_threshold
        still_mask = (classifs[d] == 0)
        still_mask[still_mask] = still_prob > still_threshold
        long_term_probs[d][classifs[d] == 0] = longterm_prob
        classifs[d][still_mask] = 2
        classifs[d][longterm_mask] = 4

        # Load original map_points
        in_folder = join(dataset.data_path, 'slam_gt', day)
        with open(join(in_folder, 'gt_map_{:s}.pkl'.format(day)), 'rb') as file:
            _, _, _, map0 = pickle.load(file)

        write_ply(join(out_folder, 'annotated_map_{:s}.ply'.format(day)),
                  [map0.points, map0.normals, classifs[d], short_term_probs[d], long_term_probs[d]],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz', 'class', 'p_short', 'p_long'])

    ########################
    # Reprojection on frames
    ########################

    # List of the annotated maps of the days
    day_labels = []
    day_classifs = []
    for d, day in enumerate(dataset.days):

        # Out files
        out_folder = join(dataset.data_path, 'annotated_frames', day)
        if not exists(out_folder):
            makedirs(out_folder)

        # Load original map_points
        in_folder = join(dataset.data_path, 'slam_gt', day)
        with open(join(in_folder, 'gt_map_{:s}.pkl'.format(day)), 'rb') as file:
            _, _, _, map0 = pickle.load(file)

        # Create KDTree on the map
        map_tree = KDTree(map0.points)
        N = len(day_frame_names[d])
        all_labels = []
        all_classifs = []
        t0 = time.time()
        print('\nReprojection of day {:s}'.format(day))
        for i, f_name in enumerate(day_frame_names[d]):

            t = [time.time()]

            ply_name = join(out_folder, f_name.split('/')[-1])
            if exists(ply_name):
                data = read_ply(ply_name)
                frame_classif = data['classif']
                frame_labels = data['labels']

            else:

                # Load points
                points, frame_labels = dataset.load_frame_points_labels(f_name)

                # Apply transf
                world_pts = np.hstack((points, np.ones_like(points[:, :1])))
                world_pts = np.matmul(world_pts, day_transform_lists[d][i].T).astype(np.float32)[:, :3]

                # Get closest map points
                neighb_inds = np.squeeze(map_tree.query(world_pts, return_distance=False))
                frame_classif = classifs[d][neighb_inds]
                frame_normals = map0.normals[neighb_inds, :]

                # Save
                write_ply(ply_name,
                          [world_pts, frame_normals, frame_classif, frame_labels],
                          ['x', 'y', 'z', 'nx', 'ny', 'nz', 'classif', 'labels'])

                t += [time.time()]

                dt = 1000 * (t[-1] - t[0])
                print('{:s} {:3d}  --- {:5.1f}%   {:6.1f} ms'.format(day, i + 1, 100 * (i + 1) / N, dt))

            all_labels.append(frame_labels)
            all_classifs.append(frame_classif)

        dt = time.time() - t0
        print('Done in {:6.1f} s'.format(dt))

        all_labels = np.hstack(all_labels)
        all_classifs = np.hstack(all_classifs)
        day_labels.append(all_labels)
        day_classifs.append(all_classifs)

    #################################
    # Semishort-term + regularization
    #################################

    # Idea: For each frame, create a local map with subsequent frames.
    #       For each point of the current frame, compute the number of short-term and long-term in its neighborhood
    #       IF Nshort > 80% => becomes short
    #       IF Nlong > 80% => becomes long
    #       IF

    ####################
    # Confusion matrices
    ####################

    for d, day in enumerate(dataset.days):

        C = fast_confusion(day_labels[d], day_classifs[d])

        label_names = ['ground',
                       'chair',
                       'movingP',
                       'stillP',
                       'tables',
                       'walls']

        pred_names = ['uncertain',
                      'ground',
                      'still',
                      'shortT',
                      'longT']

        C = C[:len(label_names), :len(pred_names)]

        s = ' {:>10s}'.format('')
        for i2, _ in enumerate(C[0, :]):
            s += ' {:^10s}'.format(pred_names[i2])
        s += '\n'
        for i1, cc in enumerate(C):
            s += ' {:>10s}'.format(label_names[i1])
            for i2, c in enumerate(cc):
                s += ' {:^10d}'.format(c)
            s += '\n'

        print('\n')
        print(s)

        print('\n*********************************')

    return


def annotation_process_old(dataset,
                           still_threshold=0.5,
                           short_threshold=0.7,
                           long_threshold=0.6,
                           min_rays=50,
                           on_gt=False):

    #############################
    # STEP 0 - Load data + Ground
    #############################
    #
    #       Ground extracted by ransac on each day independently
    #

    # List of the annotated maps of the days
    full_map = PointMap()
    day_maps = []
    day_frame_names = []
    day_transform_lists = []
    day_frame_stride = []
    long_term_probs = []
    short_term_probs = []
    still_probs = []
    classifs = []
    for d, day in enumerate(dataset.days):

        if dataset.only_day_1 and d > 0:
            break

        # Out files
        out_folder = join(dataset.data_path, 'annotation', day)
        if not exists(out_folder):
            makedirs(out_folder)

        # Load mapping
        if on_gt:
            in_folder = join(dataset.data_path, 'slam_gt', day)
            in_file = join(in_folder, 'gt_map_{:s}.pkl'.format(day))
            with open(in_file, 'rb') as file:
                frame_names, transform_list, frame_stride, pointmap = pickle.load(file)
        else:
            in_folder = join(dataset.data_path, 'slam_offline')
            in_file = join(in_folder, 'slam_{:s}.pkl'.format(day))
            with open(in_file, 'rb') as file:
                frame_names, transform_list, frame_stride, pointmap = pickle.load(file)

            # TODO: IIIIIIIIIIIIIIICCCCCCCCCCCCCCIIIIIIIIIIIII
            #       => The map was initialized with map from previous day so we have to compute again the map
            #       for each single day. We already have the transformation, use c++ code for that
            #       => We can remove short term object from maps to use them as initialization for the other days
            #       =>
            #

        # Keep the original data that we want to annotate
        N = pointmap.points.shape[0]
        classifs.append(np.zeros((N,), dtype=np.int32))
        short_term_probs.append(np.zeros((N,), dtype=np.float32))
        long_term_probs.append(np.zeros((N,), dtype=np.float32))
        still_probs.append(np.zeros((N,), dtype=np.float32))

        # Extract ground
        ground_mask = extract_map_ground(pointmap, out_folder)
        pointmap.apply_mask(np.logical_not(ground_mask))
        classifs[d][ground_mask] = 1

        # Update the map of all days combined
        if pointmap.dl < full_map.dl:
            full_map.dl = pointmap.dl
        full_map.update(pointmap.points, pointmap.normals, np.ones_like(pointmap.points[:, 0]))

        day_maps.append(pointmap)
        day_frame_names.append(frame_names)
        day_transform_lists.append(np.stack(transform_list, axis=0))
        day_frame_stride.append(frame_stride)

    ########################
    # STEP 1 - Still objects
    ########################
    #
    #       Still object by ray casting of all the days combined. Then reproject by nearest neigbors
    #

    # Combine frames of all days
    full_frame_names = np.concatenate(day_frame_names, axis=0)
    full_transform_list = np.concatenate(day_transform_lists, axis=0)

    # Out file in the first day of the dataset
    out_folder = join(dataset.data_path, 'annotation', dataset.days[0])
    if not exists(out_folder):
        makedirs(out_folder)

    # Perform raycasting annotation
    still_map = detect_still_objects(dataset,
                                     full_map,
                                     full_frame_names,
                                     full_transform_list,
                                     day_frame_stride[0],
                                     out_folder)

    # Get still probabilities
    full_still_prob = still_map.still_prob / (still_map.movable_count + 1e-6)
    full_still_prob[still_map.movable_count < min_rays] = 0

    # Tree structure for reprojection
    full_tree = KDTree(still_map.points)

    # Reprojection on each day
    for d, day in enumerate(dataset.days):

        # Out files
        out_folder = join(dataset.data_path, 'annotation', day)
        if not exists(out_folder):
            makedirs(out_folder)

        # Get closest map points and project probability
        neighb_inds = np.squeeze(full_tree.query(day_maps[d].points, return_distance=False))
        still_p = full_still_prob[neighb_inds]
        noclass_mask = classifs[d] == 0
        still_probs[d][noclass_mask] = still_p

        # Save for debug
        write_ply(join(out_folder, 'still_final.ply'),
                  [day_maps[d].points, day_maps[d].normals, still_p],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz', 'still'])

        # Update classification
        still_mask = still_probs[d] > still_threshold
        classifs[d][still_mask] = 2

        # Remove still points from the map
        day_maps[d].apply_mask(still_p <= still_threshold)

    ##########################
    # STEP 2 - Movable objects
    ##########################
    #
    #       In the map remains the points that are movable + uncertain points
    #       We do raycasting on single days to find which objects are short-term or long-term
    #

    # List of the annotated maps of the days
    for d, day in enumerate(dataset.days):

        # Out files
        out_folder = join(dataset.data_path, 'annotation', day)
        if not exists(out_folder):
            makedirs(out_folder)

        annotated_map = detect_short_term_movables(dataset,
                                                   day_maps[d],
                                                   day_frame_names[d],
                                                   day_transform_lists[d],
                                                   day_frame_stride[d],
                                                   out_folder)

        # Get probabilities
        short_term_p = annotated_map.movable_prob / (annotated_map.movable_count + 1e-6)
        long_term_p = annotated_map.still_prob / (annotated_map.movable_count + 1e-6)
        short_term_p[annotated_map.movable_count < min_rays] = 0
        long_term_p[annotated_map.movable_count < min_rays] = 0

        # Probs on all the points
        noclass_mask = classifs[d] == 0
        short_term_probs[d][noclass_mask] = short_term_p
        long_term_probs[d][noclass_mask] = long_term_p

        # Update classification
        short_term_mask = short_term_probs[d] > short_threshold
        long_term_mask = long_term_probs[d] > long_threshold
        classifs[d][short_term_mask] = 4
        classifs[d][long_term_mask] = 3

        # Save whole annotated map
        if on_gt:
            in_folder = join(dataset.data_path, 'slam_gt', day)
            in_file = join(in_folder, 'gt_map_{:s}.pkl'.format(day))
        else:
            in_folder = join(dataset.data_path, 'slam_offline')
            in_file = join(in_folder, 'slam_{:s}.pkl'.format(day))
        with open(in_file, 'rb') as file:
            _, _, _, map0 = pickle.load(file)

        write_ply(join(out_folder, 'annotated_map_{:s}.ply'.format(day)),
                  [map0.points, map0.normals, classifs[d], still_probs[d], short_term_probs[d], long_term_probs[d]],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz', 'class', 'p_still', 'p_short', 'p_long'])

    ##############################
    # STEP 3 - Reproject on frames
    ##############################
    #
    #       Just do a nearest interpolation from the map
    #

    # List of the annotated maps of the days
    day_labels = []
    day_classifs = []
    for d, day in enumerate(dataset.days):

        # Out files
        out_folder = join(dataset.data_path, 'annotated_frames', day)
        if not exists(out_folder):
            makedirs(out_folder)

        # Load original map_points
        if on_gt:
            in_folder = join(dataset.data_path, 'slam_gt', day)
            in_file = join(in_folder, 'gt_map_{:s}.pkl'.format(day))
        else:
            in_folder = join(dataset.data_path, 'slam_offline')
            in_file = join(in_folder, 'slam_{:s}.pkl'.format(day))
        with open(in_file, 'rb') as file:
            _, _, _, map0 = pickle.load(file)

        # Create KDTree on the map
        map_tree = KDTree(map0.points)
        N = len(day_frame_names[d])
        all_labels = []
        all_classifs = []
        t0 = time.time()
        print('\nReprojection of day {:s}'.format(day))
        for i, f_name in enumerate(day_frame_names[d]):

            t = [time.time()]

            ply_name = join(out_folder, f_name.split('/')[-1])
            if exists(ply_name):
                data = read_ply(ply_name)
                frame_classif = data['classif']
                frame_labels = data['labels']

            else:

                # Load points
                points, frame_labels = dataset.load_frame_points_labels(f_name)

                # Apply transf
                world_pts = np.hstack((points, np.ones_like(points[:, :1])))
                world_pts = np.matmul(world_pts, day_transform_lists[d][i].T).astype(np.float32)[:, :3]

                # Get closest map points
                neighb_inds = np.squeeze(map_tree.query(world_pts, return_distance=False))
                frame_classif = classifs[d][neighb_inds]
                frame_normals = map0.normals[neighb_inds, :]

                # Save
                write_ply(ply_name,
                          [world_pts, frame_normals, frame_classif, frame_labels],
                          ['x', 'y', 'z', 'nx', 'ny', 'nz', 'classif', 'labels'])

                t += [time.time()]

                dt = 1000 * (t[-1] - t[0])
                print('{:s} {:3d}  --- {:5.1f}%   {:6.1f} ms'.format(day, i + 1, 100 * (i + 1) / N, dt))

            all_labels.append(frame_labels)
            all_classifs.append(frame_classif)

        dt = time.time() - t0
        print('Done in {:6.1f} s'.format(dt))

        all_labels = np.hstack(all_labels)
        all_classifs = np.hstack(all_classifs)
        day_labels.append(all_labels)
        day_classifs.append(all_classifs)

    #################################
    # Semishort-term + regularization
    #################################

    # Idea: For each frame, create a local map with subsequent frames.
    #       For each point of the current frame, compute the number of short-term and long-term in its neighborhood
    #       IF Nshort > 80% => becomes short
    #       IF Nlong > 80% => becomes long
    #       IF

    ####################
    # Confusion matrices
    ####################

    for d, day in enumerate(dataset.days):

        C = fast_confusion(day_labels[d], day_classifs[d])

        label_names = ['ground',
                       'chair',
                       'movingP',
                       'stillP',
                       'tables',
                       'walls']

        pred_names = ['uncertain',
                      'ground',
                      'still',
                      'longT',
                      'shortT']

        C = C[:len(label_names), :len(pred_names)]

        s = ' {:>10s}'.format('')
        for i2, _ in enumerate(C[0, :]):
            s += ' {:^10s}'.format(pred_names[i2])
        s += '\n'
        for i1, cc in enumerate(C):
            s += ' {:>10s}'.format(label_names[i1])
            for i2, c in enumerate(cc):
                s += ' {:^10d}'.format(c)
            s += '\n'

        print('\n')
        print(s)

        print('\n*********************************')

    return


def annotation_process(dataset,
                       map_dl=0.03,
                       still_threshold=0.5,
                       short_threshold=0.7,
                       long_threshold=0.6,
                       min_rays=10,
                       on_gt=False):

    ###############
    # STEP 0 - Init
    ###############

    print('Start Annotation')

    # Folder where the incrementally updated map is stored
    map_folder = join(dataset.data_path, 'slam_offline', dataset.map_day)

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

    # TODO: HAndle height everywhere
    height_mask = np.logical_and(map_points[:, 2] < 1.8, map_points[:, 2] > -0.4)
    map_points = map_points[height_mask]
    map_normals = map_normals[height_mask]
    map_scores = map_scores[height_mask]
    map_counts = map_counts[height_mask]

    ##############
    # LOOP ON DAYS
    ##############

    # Get remove point form each day independently
    # Otherwise if a table is there in only one day, it will not be removed.
    for d, day in enumerate(dataset.days):

        print('--- Movable detection day {:s}'.format(day))

        ####################
        # Step 1: Load stuff
        ####################

        # Out folder
        out_folder = join(dataset.data_path, 'annotation', day)
        if not exists(out_folder):
            makedirs(out_folder)

        # Frame names
        f_names = dataset.day_f_names[d]

        #################
        # Initial mapping
        #################

        try:
            # Get map_poses
            map_t, map_H = dataset.load_map_poses(day)

        except FileNotFoundError as e:

            # If not available perfrom an initial mapping
            map_t = np.array([np.float64(f.split('/')[-1][:-4]) for f in f_names], dtype=np.float64)
            map_H = None

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
            while f_name_i < len(f_names) and not (f_names[f_name_i].endswith(f_name)):
                # print(f_names[f_name_i], ' skipped for ', f_name)
                f_name_i += 1

            if f_name_i >= len(f_names):
                break

            frame_names.append(f_names[f_name_i])
            f_name_i += 1

        # Remove the double inds form map_t and map_H
        map_t = np.delete(map_t, remove_inds, axis=0)
        if map_H is not None:
            map_H = np.delete(map_H, remove_inds, axis=0)

        #########################
        # Step 2: Frame alignment
        #########################

        # Align frames on the map
        cpp_map_name = join(out_folder, 'map_{:s}.ply'.format(day))
        cpp_traj_name = join(out_folder, 'correct_traj_{:s}.pkl'.format(day))

        # Always redo because map might have changed
        if not exists(cpp_traj_name):

            # Align on map and add the points of this day

            if map_H is None:
                icp_max_iter = 100

                if day == '2021-11-04_10-03-09':
                    init_H = np.array([[-1.0, 0.0, 0.0, 7.33],
                                       [0.0, -1.0, 0.0, 0.00],
                                       [0.0, 0.0, 1.0, 0.70],
                                       [0.0, 0.0, 0.0, 1.00]], dtype=np.float64)

                elif day == '2021-11-04_10-06-45':
                    init_H = np.array([[0.989726383710, 0.14297442209, 0.00000000, 1.2],
                                       [-0.14297442209, 0.98972638371, 0.00000000, 0.0],
                                       [0.000000000000, 0.00000000000, 1.00000000, 0.9],
                                       [0.000000000000, 0.00000000000, 0.00000000, 1.0]], dtype=np.float64)


                elif day.startswith('2021-11-1'):
                    init_H = np.array([[1.0, 0.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0, 0.7],
                                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

                else:

                    raise ValueError('Initial alignment not implemented')
                    init_H = None

                odom_H = [np.linalg.inv(init_H) for _ in map_t]
                odom_H = np.stack(odom_H, 0)

            else:

                # (Do not perform ICP to correct pose as it is already supposed to be good)
                icp_max_iter = 0
                odom_H = [np.linalg.inv(odoH) for odoH in map_H]
                odom_H = np.stack(odom_H, 0)

            correct_H = slam_on_real_sequence(frame_names,
                                              map_t,
                                              out_folder,
                                              init_points=map_points,
                                              init_normals=map_normals,
                                              init_scores=map_scores,
                                              map_voxel_size=map_dl,
                                              frame_voxel_size=3 * map_dl,
                                              motion_distortion=True,
                                              filtering=False,
                                              verbose_time=5.0,
                                              icp_samples=600,
                                              icp_pairing_dist=2.0,
                                              icp_planar_dist=0.08,
                                              icp_max_iter=icp_max_iter,
                                              icp_avg_steps=3,
                                              odom_H=odom_H)

            # Verify that there was no error
            test = np.sum(np.abs(correct_H), axis=(1, 2)) > 1e-6
            if not np.all(test):
                num_computed = np.sum(test.astype(np.int32))
                raise ValueError('PointSlam returned without only {:d} poses computed out of {:d}'.format(num_computed, test.shape[0]))

                                              
            # Save traj
            with open(cpp_traj_name, 'wb') as file:
                pickle.dump(correct_H, file)

        else:

            # Load traj
            with open(cpp_traj_name, 'rb') as f:
                correct_H = pickle.load(f)

        # Save a ply file for traj visu
        save_trajectory(join(out_folder, 'visu_traj_{:s}.ply'.format(day)), correct_H)

        # Load c++ map
        data = read_ply(cpp_map_name)
        day_points = np.vstack((data['x'], data['y'], data['z'])).T
        day_normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
        day_counts = data['f0']
        day_scores = data['f1']
            
        height_mask = np.logical_and(day_points[:, 2] < 1.8, day_points[:, 2] > -0.4)
        day_points = day_points[height_mask]
        day_normals = day_normals[height_mask]
        day_counts = day_counts[height_mask]

        ######################
        # Step 3: Get Movables
        ######################

        old_movable_name = join(out_folder, 'debug_movable.ply')
        movable_name = join(out_folder, 'movables_{:s}.ply'.format(day))
        if exists(old_movable_name):
            rename(old_movable_name, movable_name)

        if not exists(movable_name):

            # Get short term movables
            movable_prob, movable_count = ray_casting_annot(frame_names,
                                                            day_points,
                                                            day_normals,
                                                            correct_H,
                                                            theta_dl=0.33 * np.pi / 180,
                                                            phi_dl=0.5 * np.pi / 180,
                                                            map_dl=map_dl,
                                                            verbose_time=5.0,
                                                            motion_distortion_slices=16)
            movable_prob = movable_prob / (movable_count + 1e-6)
            long_prob = 1.0 - movable_prob
            movable_prob[movable_count < min_rays] -= 2
            long_prob[movable_count < min_rays] -= 2

            # Optionnal debug
            write_ply(movable_name,
                      [day_points, day_normals, movable_prob, long_prob, movable_count],
                      ['x', 'y', 'z', 'nx', 'ny', 'nz', 'shortT', 'longT', 'counts'])

        else:
            data = read_ply(movable_name)
            movable_prob = data['shortT']
            long_prob = data['longT']

        ######################
        # Step 4: Annotate map
        ######################
        #   0 : "uncertain"
        #   1 : "ground"
        #   2 : "still"
        #   3 : "longT"
        #   4 : "shortT"

        annot_name = join(out_folder, 'annotated_{:s}.ply'.format(day))
        if not exists(annot_name):

            categories = np.zeros(movable_prob.shape, dtype=np.int32)
            categories[movable_prob > short_threshold] = 4
            categories[long_prob > long_threshold] = 3

            # Still points come from map
            still_mask = double_still_reproj(day_points, day_counts)
            categories[still_mask] = 2

            # Ground overwrite every other class
            ground_mask = extract_ground(day_points, day_normals,
                                         out_folder,
                                         vertical_thresh=10.0,
                                         dist_thresh=0.12,
                                         remove_dist=0.01)
            categories[ground_mask] = 1

            # Save annotated day_map
            write_ply(annot_name,
                      [day_points, categories],
                      ['x', 'y', 'z', 'classif'])

        else:
            data = read_ply(annot_name)
            categories = data['classif']

        #############################
        # Step 5: Reproject on frames
        #############################

        # Folder where we save the first annotated_frames
        annot_folder = join(dataset.data_path, 'annotated_frames', day)
        if not exists(annot_folder):
            makedirs(annot_folder)

        print(annot_folder)

        # Create KDTree on the map
        print('Reprojection of map day {:s}'.format(day))
        map_tree = None
        N = len(frame_names)
        last_t = time.time()
        fps = 0
        fps_regu = 0.9
        for i, f_name in enumerate(frame_names):

            # Check if we already did reprojection (Always redo because map might have change)
            ply_name = join(annot_folder, f_name.split('/')[-1])
            if exists(ply_name):
                continue
            elif map_tree is None:
                map_tree = KDTree(day_points)

            t = [time.time()]

            # Load points
            f_points, f_labels = dataset.load_frame_points_labels(f_name)

            # Apply transf
            world_pts = np.hstack((f_points, np.ones_like(f_points[:, :1])))
            world_pts = np.matmul(world_pts, correct_H[i].T).astype(np.float32)[:, :3]

            # Get closest map points
            neighb_inds = np.squeeze(map_tree.query(world_pts, return_distance=False))
            frame_classif = categories[neighb_inds]

            # Get normals (useless for now)
            #frame_normals = day_normals[neighb_inds, :]
            #frame_normals = np.matmul(frame_normals, correct_H[i][:3, :3]).astype(np.float32)

            # Save (in original frame coordinates)
            write_ply(ply_name,
                      [f_points, frame_classif, f_labels],
                      ['x', 'y', 'z', 'classif', 'labels'])

            t += [time.time()]
            fps = fps_regu * fps + (1.0 - fps_regu) / (t[-1] - t[0])
            if (t[-1] - last_t > 5.0):
                print('Reproj {:s} {:5d} --- {:5.1f}%% at {:.1f} fps'.format(day,
                                                                             i + 1,
                                                                             100 * (i + 1) / N,
                                                                             fps))
        print('OK')

    return


def extract_map_ground(points, normals, out_folder,
                       vertical_thresh=10.0,
                       dist_thresh=0.15,
                       remove_dist=0.15):

    ground_filename = join(out_folder, 'ground_mask.pkl')
    if exists(ground_filename):
        with open(ground_filename, 'rb') as f:
            plane_mask = pickle.load(f)

    else:

        plane_mask = extract_ground(points, normals,
                                    out_folder,
                                    vertical_thresh=vertical_thresh,
                                    dist_thresh=dist_thresh,
                                    remove_dist=remove_dist)

        # Save ground
        with open(ground_filename, 'wb') as file:
            pickle.dump(plane_mask, file)

    return plane_mask


def double_still_reproj(points, counts, margin=0.1):

    # Points of the original map have a count == -1
    old_map_mask = counts < -0.5
    new_pts_mask = np.logical_not(old_map_mask)
    old_map_pts = points[old_map_mask]

    # Get points within range of old map
    old_tree = KDTree(old_map_pts)
    dists, inds = old_tree.query(points[new_pts_mask], 1)
    still_mask = old_map_mask
    still_mask[new_pts_mask] = np.squeeze(dists) < margin

    # Of these candidates remove the one that are still in range of other points
    candidates = points[still_mask]
    others = points[np.logical_not(still_mask)]
    dists, inds = KDTree(others).query(candidates, 1)
    still_mask[still_mask] = np.squeeze(dists) > margin

    return still_mask


def detect_short_term_movables(dataset, pointmap, frame_names, transform_list, frame_stride, out_folder, verbose=1):

    annot_filename = join(out_folder, 'movable_map.pkl')
    if exists(annot_filename):
        # Read pkl
        with open(annot_filename, 'rb') as f:
            pointmap = pickle.load(f)

    else:

        pointmap = ray_casting_annot(dataset, pointmap, frame_names, transform_list, frame_stride, out_folder)

        # Save last map
        movable_prob = pointmap.movable_prob / (pointmap.movable_count + 1e-6)
        still_prob = pointmap.still_prob / (pointmap.movable_count + 1e-6)
        movable_prob[pointmap.movable_count < 1e-6] = -1
        still_prob[pointmap.movable_count < 1e-6] = -1
        write_ply(join(out_folder, 'movable_final.ply'),
                  [pointmap.points, pointmap.normals, movable_prob, still_prob, pointmap.movable_count],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz', 'movable', 'still', 'counts'])

        # Save alignments
        with open(annot_filename, 'wb') as file:
            pickle.dump(pointmap, file)

    return pointmap


def detect_long_term_movables(dataset, pointmap, all_frame_names, all_transform_list, frame_stride, out_folder, verbose=1):

    annot_filename = join(out_folder, 'longterm_probs.pkl')
    if exists(annot_filename):
        # Read pkl
        with open(annot_filename, 'rb') as f:
            combined_longterm_prob = pickle.load(f)

    else:

        combined_longterm_prob = np.zeros((1,), dtype=np.float32)

        for frame_names, transform_list in zip(all_frame_names, all_transform_list):

            # Reset movable probabilities
            pointmap.movable_prob *= 0
            pointmap.still_prob *= 0
            pointmap.movable_count *= 0

            # Do ray casting
            pointmap = ray_casting_annot(dataset, pointmap, frame_names, transform_list, frame_stride, out_folder)

            # Get probabilities
            longterm_prob = pointmap.movable_prob / (pointmap.movable_count + 1e-6)
            longterm_prob[pointmap.movable_count < 10] = 0

            # Combine al lthe days probabilities
            combined_longterm_prob = np.maximum(longterm_prob, combined_longterm_prob)

        # Save last map
        write_ply(join(out_folder, 'longterm_final.ply'),
                  [pointmap.points, pointmap.normals, combined_longterm_prob],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz', 'longterm'])

        # Save alignments
        with open(annot_filename, 'wb') as file:
            pickle.dump(combined_longterm_prob, file)

    return combined_longterm_prob


def detect_still_objects(dataset, pointmap, frame_names, transform_list, frame_stride, out_folder, verbose=1):

    annot_filename = join(out_folder, 'still_object_map.pkl')
    if exists(annot_filename):
        # Read pkl
        with open(annot_filename, 'rb') as f:
            pointmap = pickle.load(f)

    else:

        pointmap = ray_casting_annot(dataset, pointmap, frame_names, transform_list, frame_stride, out_folder)

        # Save map
        still_prob = pointmap.still_prob / (pointmap.movable_count + 1e-6)
        still_prob[pointmap.movable_count < 10] = 0
        write_ply(join(out_folder, 'full_still.ply'),
                  [pointmap.points, pointmap.normals, still_prob, pointmap.movable_count],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz', 'still', 'counts'])

        # Save alignments
        with open(annot_filename, 'wb') as file:
            pickle.dump(pointmap, file)

    return pointmap


def ray_casting_annot_old(dataset, pointmap, frame_names, transform_list, frame_stride, out_folder, verbose=1):

    # Parameters
    if dataset.motion_distortion:
        frames_slice_n = 12
    else:
        frames_slice_n = 1

    # For each frame, project frame in its polar coordinates and do raycasting
    phi0 = 0
    N = len(transform_list)
    for i, world_H in enumerate(transform_list):

        t = [time.time()]

        # Load ply format points
        points = dataset.load_frame_points(frame_names[i])

        t += [time.time()]

        if dataset.motion_distortion:
            slices_points, slices_R, slices_T, phi2 = get_frame_slices(points, phi0,
                                                                       transform_list[i - 1],
                                                                       world_H,
                                                                       frames_slice_n)
        else:
            slices_points = [points]
            slices_R = [world_H[:3, :3]]
            slices_T = [world_H[:3, 3]]
            phi2 = 0

        # Save phi for next frame
        phi0 = phi2 - frame_stride * 2 * np.pi

        debug_frame_slices = False
        if debug_frame_slices:
            slices_concat = []
            for s, s_points in enumerate(slices_points):
                s_points = np.matmul(s_points, slices_R[s].T) + slices_T[s]
                s_points = np.hstack((s_points, np.ones_like(s_points[:, :1]) * s)).astype(np.float32)
                slices_concat.append(s_points)
            slices_concat = np.vstack(slices_concat)
            write_ply('tts_slices.ply',
                      [slices_concat],
                      ['x', 'y', 'z', 's'])

        t += [time.time()]

        for s, s_points in enumerate(slices_points):
            pointmap.update_movable(s_points, slices_R[s], slices_T[s])

        t += [time.time()]

        if verbose == 1:
            dt = 1000 * (t[-1] - t[0])
            print('Annotating {:3d}  --- {:5.1f}%   {:6.1f} ms'.format(i + 1, 100 * (i + 1) / N, dt))

        if verbose > 1:
            ti = 0
            print('Annotating {:3d}  --- {:5.1f}%'.format(i + 1, 100 * (i + 1) / N))
            print('Load ....... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print('Slices ..... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print('Update .... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            print('******************************')

            if i % 100 == 0:
                movable_prob = pointmap.movable_prob / (pointmap.movable_count + 1e-6)
                still_prob = pointmap.still_prob / (pointmap.movable_count + 1e-6)
                movable_prob[pointmap.movable_count < 1e-6] = -1
                still_prob[pointmap.movable_count < 1e-6] = -1
                write_ply(join(out_folder, 'annot_{:03d}.ply'.format(i)),
                          [pointmap.points, pointmap.normals, movable_prob, still_prob, pointmap.movable_count],
                          ['x', 'y', 'z', 'nx', 'ny', 'nz', 'movable', 'still', 'counts'])

    return pointmap


def test_loop_closure(dataset):

    pointmaps = []
    for d, day in enumerate(dataset.days):

        ############################
        # Load map and detect planes
        ############################

        # slam results
        in_file = join(dataset.data_path, 'slam_offline', day, 'PointSLAM_{:s}.pkl'.format(day))
        with open(in_file, 'rb') as file:
            frame_names, transform_list, frame_stride, _ = pickle.load(file)

        # Load annotation
        in_file = join(dataset.data_path, 'annotation', day, 'annotated_map.pkl')
        with open(in_file, 'rb') as file:
            pointmap = pickle.load(file)

        print('Search planes in map')
        pointmap.planes, pointmap.plane_inds = map_plane_growing(pointmap.points,
                                                                 pointmap.normals,
                                                                 norm_thresh=20 * np.pi / 180,
                                                                 dist_thresh=0.1,
                                                                 min_points=100,
                                                                 max_planes=1000,
                                                                 map_dl=pointmap.dl)
        print('Found: ', pointmap.planes.shape[0], 'planes')

        # Compute area of convex hull of each plane
        print('Get areas')
        pointmap.plane_areas = []
        features = np.zeros(pointmap.plane_inds.shape, np.float32)
        for plane_i, plane in enumerate(pointmap.planes):
            plane_points = pointmap.points[pointmap.plane_inds == plane_i]
            pointmap.plane_areas += [plane_area(plane_points, plane[:3])]
            features[pointmap.plane_inds == plane_i] = pointmap.plane_areas[-1]
        print('Done')

        write_ply('cc_region_growing_map.ply',
                  [pointmap.points, pointmap.normals, pointmap.plane_inds, features],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz', 'planes', 'area'])

        pointmaps.append(pointmap)

        ############
        # Align maps
        ############

        if d > 1:

            # Start loop closure
            pointmap.frame_localization(points, normals)

            t += [time.time()]
            ti = 0
            print('Loop closure {:3d}  --- {:5.1f}%'.format(i + 1, 100 * (i + 1) / N))
            print('Load ....... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print('Normals .... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print('Closing .... {:7.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            print('******************************')

            a = 1 / 0

        break
