#
#
#      0====================0
#      |    PointMapSLAM    |
#      0====================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Python calls for the c++ wrapper functions
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

import cpp_wrappers.cpp_lidar_utils.lidar_utils as cpp_lidar_utils
import cpp_wrappers.cpp_slam.pointmap_slam as cpp_pointmap_slam
import cpp_wrappers.cpp_pointmap.pointmap as cpp_pointmap
import cpp_wrappers.cpp_icp.icp as cpp_icp
import cpp_wrappers.cpp_polar_normals.polar_processing as cpp_polar_processing
import numpy as np
import os
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Python function
#       \*********************/
#


def update_pointmap(points, normals, scores,
                    map_points=None,
                    map_normals=None,
                    map_scores=None,
                    map_counts=None,
                    map_dl=1.0):

    if map_points is None:
        return cpp_pointmap.update_map(points, normals, scores,
                                       map_dl=map_dl)
    else:
        return cpp_pointmap.update_map(points, normals, scores,
                                       map_points=map_points,
                                       map_normals=map_normals,
                                       map_scores=map_scores,
                                       map_counts=map_counts,
                                       map_dl=map_dl)


def point_to_map_icp(points, weights,
                     map_points, map_normals, map_weights,
                     init_H=np.eye(4),
                     init_phi=0.0,
                     motion_distortion=True,
                     n_samples=1000,
                     max_pairing_dist=0.2,
                     max_iter=50,
                     rotDiffThresh=0.004,
                     transDiffThresh=0.02,
                     avg_steps=5):

    all_H, rms, planar_rms = cpp_icp.map_pt2pl(points, weights,
                                               map_points, map_normals, map_weights,
                                               init_H=init_H,
                                               init_phi=init_phi,
                                               n_samples=n_samples,
                                               max_pairing_dist=max_pairing_dist,
                                               max_iter=max_iter,
                                               rotDiffThresh=rotDiffThresh,
                                               transDiffThresh=transDiffThresh,
                                               avg_steps=avg_steps,
                                               motion_distortion=motion_distortion)

    all_H = all_H.T
    all_H = all_H.reshape(-1, 4, 4)

    return all_H, rms, planar_rms


def bundle_pt2pl_icp(frames, normals, weights,
                     n_samples=1000,
                     max_pairing_dist=0.2,
                     max_iter=50,
                     rotDiffThresh=0.004,
                     transDiffThresh=0.02,
                     avg_steps=5):

    # Stack everything
    lengths = np.array([frame.shape[0] for frame in frames], dtype=np.int32)
    frames = np.vstack(frames)
    normals = np.vstack(normals)
    weights = np.vstack(weights)

    H, rms, all_H = cpp_icp.bundle_pt2pl(frames, normals, weights, lengths,
                                         n_samples=n_samples,
                                         max_pairing_dist=max_pairing_dist,
                                         max_iter=max_iter,
                                         rotDiffThresh=rotDiffThresh,
                                         transDiffThresh=transDiffThresh,
                                         avg_steps=avg_steps)

    all_H = all_H.T
    all_H = all_H.reshape(-1, 4, len(lengths), 4)
    all_H = all_H.transpose((0, 2, 1, 3))

    return H.transpose(0, 2, 1), rms, all_H


def polar_normals(points,
                  radius=1.5,
                  lidar_n_lines=32,
                  h_scale=0.5,
                  r_scale=4.0,
                  verbose=0):
    """
    :param points: (N, 3) the points of the frame
    :param radius: radius in angular resolution unit.
    :param lidar_angle_res: angular resolution of the lidar
    :param h_scale: horizontal scale
    :param r_scale: range scale
    :param verbose: display option
    :return: normals and scores
    """

    return cpp_polar_processing.polar_normals(points,
                                              radius=radius,
                                              lidar_n_lines=lidar_n_lines,
                                              h_scale=h_scale,
                                              r_scale=r_scale,
                                              verbose=verbose)


def ray_casting_annot(frame_names,
                      map_points,
                      map_normals,
                      H_frames,
                      theta_dl=0.5 * 1.29 * np.pi / 180,
                      phi_dl=0.2 * 1.29 * np.pi / 180,
                      map_dl=0.1,
                      verbose_time=5.0,
                      motion_distortion_slices=1,
                      lidar_n_lines=32):
    """
    :param frame_points: (N1, 3) points of the slice or frame
    :param map_points: (N2, 3) the points of the map
    :param map_normals: (N2, 3) the normals of the map
    :param R_frame_to_map: rotation to align frame points
    :param T_frame_to_map: translation to align frame points
    :param polar_dl: angular resolution of the polar grid used for comparison
    :param phi_theta_ratio: ratio between resolution in theta and phi direction
    :param map_dl: map cell size
    :return: movable probs and valid boolean
    """

    # Stack frame names in one gig string
    stacked_f_names = '\n'.join([f for f in frame_names])

    movable_prob, movable_count = cpp_polar_processing.map_frame_comp(frame_names=stacked_f_names,
                                                                      map_points=map_points,
                                                                      map_normals=map_normals,
                                                                      H_frames=H_frames,
                                                                      map_dl=map_dl,
                                                                      theta_dl=theta_dl,
                                                                      phi_dl=phi_dl,
                                                                      verbose_time=verbose_time,
                                                                      n_slices=motion_distortion_slices,
                                                                      lidar_n_lines=lidar_n_lines)
    #motion_distortion=motion_distortion)

    return movable_prob, movable_count


def slam_on_sim_sequence(f_names,
                         f_times,
                         gt_poses,
                         gt_times,
                         save_path,
                         init_points=None,
                         init_normals=None,
                         init_scores=None,
                         map_voxel_size=0.03,
                         frame_voxel_size=0.1,
                         motion_distortion=False,
                         filtering=False,
                         verbose_time=5.0,
                         icp_samples=400,
                         icp_pairing_dist=2.0,
                         icp_planar_dist=0.08,
                         icp_avg_steps=3,
                         icp_max_iter=50,
                         H_velo_base=None,
                         odom_H=None):

    # Stack frame names in one gig string
    stacked_f_names = "\n".join([f for f in f_names])

    if (init_points is None) or (init_normals is None) or (init_scores is None):
        init_points = np.zeros((0, 3), dtype=np.float32)
        init_normals = np.zeros((0, 3), dtype=np.float32)
        init_scores = np.zeros((0,), dtype=np.float32)

    if H_velo_base is None:
        H_velo_base = np.eye(4, dtype=np.float64)

    if odom_H is None:
        odom_H = np.zeros((len(f_names), 4, 4), dtype=np.float64)

    H = cpp_pointmap_slam.map_sim_sequence(stacked_f_names,
                                           f_times,
                                           gt_poses,
                                           gt_times,
                                           save_path,
                                           init_points,
                                           init_normals,
                                           init_scores,
                                           map_voxel_size=map_voxel_size,
                                           frame_voxel_size=frame_voxel_size,
                                           motion_distortion=motion_distortion,
                                           filtering=filtering,
                                           verbose_time=verbose_time,
                                           icp_samples=icp_samples,
                                           icp_pairing_dist=icp_pairing_dist,
                                           icp_planar_dist=icp_planar_dist,
                                           icp_avg_steps=icp_avg_steps,
                                           icp_max_iter=icp_max_iter,
                                           H_velo_base=H_velo_base,
                                           odom_H=odom_H)

    H = H.T
    H = np.stack([H[i * 4:(i + 1) * 4, :] for i in range(H.shape[0] // 4)])

    return H


def slam_on_real_sequence(f_names,
                          f_times,
                          save_path,
                          init_points=None,
                          init_normals=None,
                          init_scores=None,
                          map_voxel_size=0.03,
                          frame_voxel_size=0.1,
                          motion_distortion=False,
                          filtering=False,
                          verbose_time=5.0,
                          icp_samples=400,
                          icp_pairing_dist=2.0,
                          icp_planar_dist=0.08,
                          icp_avg_steps=3,
                          icp_max_iter=50,
                          H_velo_base=None,
                          odom_H=None):

    # Stack frame names in one gig string
    stacked_f_names = "\n".join([f for f in f_names])

    if (init_points is None) or (init_normals is None) or (init_scores is None):
        init_points = np.zeros((0, 3), dtype=np.float32)
        init_normals = np.zeros((0, 3), dtype=np.float32)
        init_scores = np.zeros((0,), dtype=np.float32)

    if H_velo_base is None:
        H_velo_base = np.eye(4, dtype=np.float64)

    if odom_H is None:
        odom_H = np.zeros((len(f_names), 4, 4), dtype=np.float64)

    print('Starting slam')

    H = cpp_pointmap_slam.map_real_sequence(stacked_f_names,
                                            f_times,
                                            save_path,
                                            init_points,
                                            init_normals,
                                            init_scores,
                                            map_voxel_size=map_voxel_size,
                                            frame_voxel_size=frame_voxel_size,
                                            motion_distortion=motion_distortion,
                                            filtering=filtering,
                                            verbose_time=verbose_time,
                                            icp_samples=icp_samples,
                                            icp_pairing_dist=icp_pairing_dist,
                                            icp_planar_dist=icp_planar_dist,
                                            icp_avg_steps=icp_avg_steps,
                                            icp_max_iter=icp_max_iter,
                                            H_velo_base=H_velo_base,
                                            odom_H=odom_H)

    H = H.T
    H = np.stack([H[i * 4:(i + 1) * 4, :] for i in range(H.shape[0] // 4)])

    return H


def get_lidar_visibility(points,
                         center,
                         ground_plane,
                         n_angles=720,
                         z_min=0.4,
                         z_max=1.5,
                         dl_2D=-1.0):
    """
    :param points: (N, 3) the points of the frame
    :param radius: radius in angular resolution unit.
    :param lidar_angle_res: angular resolution of the lidar
    :param h_scale: horizontal scale
    :param r_scale: range scale
    :param verbose: display option
    :return: normals and scores
    """

    "points", "center", "ground", "n_angles", "z_min", "z_max"

    return cpp_lidar_utils.get_visibility(points,
                                          center,
                                          ground_plane,
                                          n_angles=n_angles,
                                          z_min=z_min,
                                          z_max=z_max,
                                          dl_2D=dl_2D)
