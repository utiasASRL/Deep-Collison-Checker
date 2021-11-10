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
import struct
import scipy
import time
import numpy as np
import pickle
import torch
import yaml
#from mayavi import mlab
from multiprocessing import Lock
import open3d
from scipy.spatial.transform import Rotation as scipyR
from scipy.spatial.transform import Slerp

import matplotlib.pyplot as plt


# OS functions
from os import listdir
from os.path import exists, join, isdir, getsize

# Dataset parent class
from utils.mayavi_visu import *
from sklearn.neighbors import KDTree
from slam.cpp_slam import polar_normals, bundle_pt2pl_icp

from datasets.common import grid_subsampling
import open3d as o3d
import copy
import re

from utils.mayavi_visu import show_point_cloud



def compute_plane(points):
    ref_point = points[0]
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    normal = normal / np.sqrt(np.sum(np.power(normal, 2)))
    return ref_point, normal


def in_plane(points, ref_pt, normal, threshold_in=0.1):
    return np.abs(np.dot((points - ref_pt), normal)) < threshold_in


def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):

    best_mask = None
    best_vote = 3
    best_ref_pt, best_normal = compute_plane(points[:3])
    N = len(points)

    for i in range(NB_RANDOM_DRAWS):

        # Random selection of points
        random_inds = np.zeros((0,), dtype=np.int32)
        while random_inds.shape[0] < 3:
            new_inds = np.random.randint(0, N, size=3, dtype=np.int32)
            random_inds = np.unique(np.hstack((random_inds, new_inds)))
        random_inds = random_inds[:3]

        # Corresponding plane
        ref_pt, normal = compute_plane(points[random_inds])

        # Number of votes
        mask = in_plane(points, ref_pt, normal, threshold_in)
        vote = np.sum(mask)

        # Save
        if vote > best_vote:
            best_ref_pt = ref_pt
            best_normal = normal
            best_vote = vote
            best_mask = mask

    return best_ref_pt, best_normal, best_mask


def extract_ground(points, normals, out_folder,
                   vertical_thresh=10.0,
                   dist_thresh=0.15,
                   remove_dist=0.15,
                   saving=True):
    # Get points with vertical normal
    vertical_angle = np.arccos(np.abs(np.clip(normals[:, 2], -1.0, 1.0)))

    # Use the thresold on the vertical angle in degree
    plane_mask = vertical_angle < vertical_thresh * np.pi / 180

    # Get the ground plane with RANSAC
    plane_P, plane_N, _ = RANSAC(points[plane_mask], threshold_in=dist_thresh)

    # Get mask on all the points
    plane_mask = in_plane(points, plane_P, plane_N, dist_thresh)
    mask0 = np.copy(plane_mask)

    # Get better ground/objects boundary
    candidates = points[plane_mask]
    others = points[np.logical_not(plane_mask)]
    dists, inds = KDTree(others).query(candidates, 1)
    plane_mask[plane_mask] = np.squeeze(dists) > remove_dist

    if saving:
        ground_points = points[plane_mask]
        ground_normals = normals[plane_mask]
        write_ply(join(out_folder, 'ground.ply'),
                  [ground_points, ground_normals],
                  ['x', 'y', 'z', 'nx', 'ny', 'nz'])

    return plane_mask


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def write_pgm(filename, image):


    # open in text mode to write the header
    with open(filename, 'w') as pgm_file:

        # First magical word
        header = ['P5']
        header.append('{:d} {:d}'.format(image.shape[1], image.shape[0]))
        header.append('255')
        for line in header:
            pgm_file.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as pgm_file:
        image.tofile(pgm_file)



def pointmap_for_AMCL():

    # -----------------------------------------------------------------------------------------
    # Load original map for comparison

    path = '../../Myhal_Simulation/Simulator/JackalTourGuide/src/jackal_velodyne/maps'
    pgm_file = 'myhal_map_V3.pgm'
    yml_file = 'myhal_map_V3.yaml'

    with open(join(path, yml_file), 'r') as stream:
        doc = yaml.safe_load(stream)

    print('-----------------------------')
    print('image:', doc['image'])
    print('resolution:', doc['resolution'])
    print('origin:', doc['origin'])
    print('negate:', doc['negate'])
    print('occupied_thresh:', doc['occupied_thresh'])
    print('free_thresh:', doc['free_thresh'])
    print('-----------------------------')

    image = read_pgm(join(path, pgm_file), byteorder='<')
    # plt.imshow(image)
    # plt.show()

    # -----------------------------------------------------------------------------------------
    # Load point map

    path = '../../Myhal_Simulation/slam_offline/2020-10-02-13-39-05'
    ply_file = 'map_update_0002.ply'

    data = read_ply(join(path, ply_file))

    points = np.vstack((data['x'], data['y'], data['z'])).T

    heights = points[:, 2]
    min_z = np.min(heights)
    heights = heights[heights < min_z + 0.09]
    ground_z = np.median(heights)

    z1 = ground_z + 0.3
    z2 = ground_z + 1.0

    mask_2D = np.logical_and(points[:, 2] < z2, points[:, 2] > z1)
    points_2D = points[mask_2D, :2]

    # -----------------------------------------------------------------------------------------
    # Fill map_image

    origin_2D = np.array(doc['origin'][:2], dtype=np.float32)

    # Compute voxel indice for each frame point
    grid_indices = (np.floor((points_2D - origin_2D) / doc['resolution'])).astype(int)

    # Flip first axis it is an image
    grid_indices[:, 1] = image.shape[0] - grid_indices[:, 1]

    # Scalar equivalent to grid indices
    scalar_indices = grid_indices[:, 0] + grid_indices[:, 1] * image.shape[0]
    vec_img = np.reshape(image * 0 + 255, (-1,))
    vec_img[np.unique(scalar_indices)] = 0
    image2 = np.reshape(vec_img, image.shape)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image)
    axarr[1].imshow(image2)
    plt.show()

    # -----------------------------------------------------------------------------------------
    # Save and check saved

    path = '../../Myhal_Simulation/Simulator/JackalTourGuide/src/jackal_velodyne/maps'
    pgm_file = 'myhal_map_V4.pgm'
    yml_file = 'myhal_map_V4.yaml'
    doc['image'] = pgm_file

    if False and exists(join(path, pgm_file)):

        imagetest = read_pgm(join(path, pgm_file), byteorder='<')
        plt.imshow(imagetest)
        plt.show()

    else:
        with open(join(path, yml_file), 'w') as outfile:
            yaml.dump(doc, outfile)
        write_pgm(join(path, pgm_file), image2)

    # -----------------------------------------------------------------------------------------
    # change map parameters

    doc['image'] = pgm_file
    doc['resolution'] = 0.05
    doc['origin'] = [-22, -22, 0]

    limit_2D = np.array([22, 22], dtype=np.float32)
    origin_2D = np.array(doc['origin'][:2], dtype=np.float32)
    image_w, image_h = (np.ceil((limit_2D - origin_2D) / doc['resolution'])).astype(int)

    # Compute voxel indice for each frame point
    grid_indices = (np.floor((points_2D - origin_2D) / doc['resolution'])).astype(int)

    # Flip first axis it is an image
    grid_indices[:, 1] = image_h - grid_indices[:, 1]

    # Scalar equivalent to grid indices
    scalar_indices = grid_indices[:, 0] + grid_indices[:, 1] * image_w

    vec_img = np.zeros(image_w * image_h, dtype='u1') + 255
    vec_img[np.unique(scalar_indices)] = 0
    image3 = np.reshape(vec_img, (image_w, image_h))

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image2)
    axarr[1].imshow(image3)
    plt.show()

    path = '../../Myhal_Simulation/Simulator/JackalTourGuide/src/jackal_velodyne/maps'
    pgm_file = 'myhal_map_V5.pgm'
    yml_file = 'myhal_map_V5.yaml'
    doc['image'] = pgm_file

    if False and exists(join(path, pgm_file)):

        imagetest = read_pgm(join(path, pgm_file), byteorder='<')
        plt.imshow(imagetest)
        plt.show()

    else:
        with open(join(path, yml_file), 'w') as outfile:
            yaml.dump(doc, outfile)
        write_pgm(join(path, pgm_file), image3)

    return




def normals_orientation(normals):

    # Discretise the sphere in carthesian coordiantes to avoid the resolution problem at poles
    voxel_size = 0.05

    # Compute voxel indice for each point
    grid_indices = (np.floor(normals / voxel_size)).astype(int)

    # Limits of the grid
    min_grid_indices = np.amin(grid_indices, axis=0)
    max_grid_indices = np.amax(grid_indices, axis=0)

    # Number of cells in each direction
    deltaX, deltaY, deltaZ = max_grid_indices - min_grid_indices + 1

    # Relocate indices
    grid_indices -= min_grid_indices

    # Scalar equivalent to grid indices
    scalar_indices = grid_indices[:, 0] + grid_indices[:, 1] * deltaX + grid_indices[:, 2] * deltaX * deltaY
    unique_inds, inverse, counts = np.unique(scalar_indices, return_counts=True, return_inverse=True)

    # Get counts in a 3D matrix
    unique_z = unique_inds // (deltaX * deltaY)
    unique_inds -= unique_z * deltaX * deltaY
    unique_y = unique_inds // deltaX
    unique_x = unique_inds - unique_y * deltaX
    count_matrix = np.zeros((deltaX, deltaY, deltaZ), dtype=np.float32)
    count_matrix[unique_x, unique_y, unique_z] += counts

    # Smooth them with a gaussian filter convolution
    torch_conv = torch.nn.Conv3d(1, 1, kernel_size=5, stride=1, bias=False)
    torch_conv.weight.requires_grad_(False)
    torch_conv.weight *= 0
    torch_conv.weight += gaussian_conv_filter(3, 5)
    torch_conv.weight *= torch.sum(torch_conv.weight) ** -1
    count_matrix = np.expand_dims(count_matrix, 0)
    count_matrix = np.expand_dims(count_matrix, 0)
    torch_count = torch.from_numpy(count_matrix)
    torch_count = torch.nn.functional.pad(torch_count, [2, 2, 2, 2, 2, 2])
    smooth_counts = torch.squeeze(torch_conv(torch_count))
    smooth_counts = smooth_counts.numpy()[unique_x, unique_y, unique_z]

    #################################################
    # Create weight according to the normal direction
    #################################################


    # Show histogram in a spherical point cloud
    n_cloud = np.vstack((unique_x, unique_y, unique_z)).astype(np.float32).T
    n_cloud = (n_cloud + min_grid_indices.astype(np.float32) + 0.5) * voxel_size

    # Only 20% of the normals bins are kept For the rest, we use weights based on ditances
    mask = (smooth_counts > np.percentile(smooth_counts, 80))

    # Align with weighted PCA
    # weighted_cloud = n_cloud[mask] * np.expand_dims(smooth_counts[mask], 1)
    weighted_cloud = n_cloud[mask]

    # mean_P = np.sum(weighted_cloud, axis=0) / np.sum(smooth_counts)
    # print(mean_P.shape)
    # cloud_0 = n_cloud - mean_P


    # TODO: Covarariance not robust, do a ICP???


    cov_mat = np.matmul(weighted_cloud.T, weighted_cloud) / n_cloud[mask].shape[0] #np.sum(smooth_counts[mask])
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Correct eigenvectors orientation with centroid
    # mean_P = np.sum(weighted_cloud, axis=0) / np.sum(smooth_counts)
    # rotated_centroids = np.matmul(mean_P, eigen_vectors.T)
    # corrections = (rotated_centroids > 0).astype(eigen_vectors.dtype) * 2 - 1

    return n_cloud, smooth_counts, eigen_vectors


def rot_trans_diffs(all_H):

    all_R = all_H[:, :3, :3]
    all_R_T = np.transpose(all_R, (0, 2, 1))

    dR = np.matmul(all_R[1:], all_R_T[:-1])
    dR = np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0))

    dT = all_H[1:, :3, 3] - all_H[:-1, :3, 3]
    dT = np.linalg.norm(dT, axis=1)

    return dT, dR


def interp_pose(t, H0, H1):
    """
    Interpolate pose at time t (between 0 and 1) between the pose H(t=0) and H(t=1)
    :param t: interpolation time
    :param H0: first pose
    :param H1: second pose
    :return: interpolated pose
    """


    # Create a slerp interpolation function for the rotation part of the transform
    R1 = H0[:3, :3]
    R2 = H1[:3, :3]
    key_rots = scipyR.from_matrix(np.stack((R1, R2), axis=0))
    slerp = Slerp([0, 1], key_rots)
    interp_R = slerp(t).as_matrix()

    # Create linear interpolation for translation
    interp_H = (1 - t) * H0 + t * H1

    # Update rotation part of the transform
    interp_H[:3, :3] = interp_R
    return interp_H


def frame_H_to_points(H_f, size=1.0):

    # Create artificial frames
    x = np.linspace(0, size, 50, dtype=np.float32)
    points = np.hstack((np.vstack((x, x * 0, x * 0)), np.vstack((x * 0, x, x * 0)), np.vstack((x * 0, x * 0, x)))).T
    colors = ((points > 0.1 * size).astype(np.float32) * 255).astype(np.uint8)
    hpoints = np.hstack((points, np.ones_like(points[:, :1])))
    hpoints = np.matmul(hpoints, H_f.T)
    return hpoints[:, :3], colors


def save_trajectory(filename, all_traj_H):

    # Save full trajectory
    all_traj_pts = []
    all_traj_clrs = []
    for save_i, save_H in enumerate(all_traj_H):
        # Save trajectory
        traj_pts, traj_clrs = frame_H_to_points(save_H, size=0.1)
        traj_pts = np.hstack((traj_pts, np.ones_like(traj_pts[:, :1]) * save_i))
        all_traj_pts.append(traj_pts.astype(np.float32))
        all_traj_clrs.append(traj_clrs)

    write_ply(filename,
              [np.vstack(all_traj_pts), np.vstack(all_traj_clrs)],
              ['x', 'y', 'z', 't', 'red', 'green', 'blue'])


def cart2pol(xyz):
    """
    Convertion from 3D carthesian coordinates xyz to 3D polar coordinates rtp
    :param xyz: [N,3] matrix of x, y, z coordinates
    :return: [N,3] matrix of rho, theta, phi coordinates
    """
    rho = np.linalg.norm(xyz, axis=1)
    phi = (3 * np.pi / 2 - np.arctan2(xyz[:, 1], xyz[:, 0])) % (2 * np.pi)
    theta = np.arctan2(np.linalg.norm(xyz[:, :2], axis=1), xyz[:, 2])
    return np.vstack((rho, theta, phi)).T


def pol2cart(rtp):
    """
    Convertion from 3D polar coordinates rtp to 3D carthesian coordinates xyz
    :param rtp: [N,3] matrix of rho, theta, phi coordinates
    :return: [N,3] matrix of x, y, z coordinates
    """
    x = rtp[:, 0] * np.sin(rtp[:, 1]) * np.cos(rtp[:, 2])
    y = rtp[:, 0] * np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])
    z = rtp[:, 0] * np.cos(rtp[:, 1])
    return np.vstack((x, y, z)).T


def ssc_to_homo(ssc, ssc_in_radians=True):

    # Convert 6-DOF ssc coordinate transformation to 4x4 homogeneous matrix
    # transformation

    if ssc.ndim == 1:
        reduce = True
        ssc = np.expand_dims(ssc, 0)
    else:
        reduce = False

    if not ssc_in_radians:
        ssc[:, 3:] = np.pi / 180.0 * ssc[:, 3:]

    sr = np.sin(ssc[:, 3])
    cr = np.cos(ssc[:, 3])

    sp = np.sin(ssc[:, 4])
    cp = np.cos(ssc[:, 4])

    sh = np.sin(ssc[:, 5])
    ch = np.cos(ssc[:, 5])

    H = np.zeros((ssc.shape[0], 4, 4))

    H[:, 0, 0] = ch*cp
    H[:, 0, 1] = -sh*cr + ch*sp*sr
    H[:, 0, 2] = sh*sr + ch*sp*cr
    H[:, 1, 0] = sh*cp
    H[:, 1, 1] = ch*cr + sh*sp*sr
    H[:, 1, 2] = -ch*sr + sh*sp*cr
    H[:, 2, 0] = -sp
    H[:, 2, 1] = cp*sr
    H[:, 2, 2] = cp*cr

    H[:, 0, 3] = ssc[:, 0]
    H[:, 1, 3] = ssc[:, 1]
    H[:, 2, 3] = ssc[:, 2]

    H[:, 3, 3] = 1

    if reduce:
        H = np.squeeze(H)

    return H


def verify_magic(s):

    magic = 44444

    m = struct.unpack('<HHHH', s)

    return len(m)>=4 and m[0] == magic and m[1] == magic and m[2] == magic and m[3] == magic


def test_read_hits():

    data_path = '../../Data/NCLT'
    velo_folder = 'velodyne_data'
    day = '2012-01-08'

    hits_path = join(data_path, velo_folder, day, 'velodyne_hits.bin')

    all_utimes = []
    all_hits = []
    all_ints = []

    num_bytes = getsize(hits_path)
    current_bytes = 0

    with open(hits_path, 'rb') as f_bin:

        total_hits = 0
        first_utime = -1
        last_utime = -1

        while True:

            magic = f_bin.read(8)
            if magic == b'':
                break

            if not verify_magic(magic):
                print('Could not verify magic')

            num_hits = struct.unpack('<I', f_bin.read(4))[0]
            utime = struct.unpack('<Q', f_bin.read(8))[0]

            # Do not convert padding (it is an int always equal to zero)
            padding = f_bin.read(4)

            total_hits += num_hits
            if first_utime == -1:
                first_utime = utime
            last_utime = utime

            hits = []
            ints = []

            for i in range(num_hits):

                x = struct.unpack('<H', f_bin.read(2))[0]
                y = struct.unpack('<H', f_bin.read(2))[0]
                z = struct.unpack('<H', f_bin.read(2))[0]
                i = struct.unpack('B', f_bin.read(1))[0]
                l = struct.unpack('B', f_bin.read(1))[0]

                hits += [[x, y, z]]
                ints += [i]

            utimes = np.full((num_hits,), utime - first_utime, dtype=np.int32)
            ints = np.array(ints, dtype=np.uint8)
            hits = np.array(hits, dtype=np.float32)
            hits *= 0.005
            hits += -100.0

            all_utimes.append(utimes)
            all_hits.append(hits)
            all_ints.append(ints)

            if 100 * current_bytes / num_bytes > 0.1:
                break

            current_bytes += 24 + 8 * num_hits

            print('{:d}/{:d}  =>  {:.1f}%'.format(current_bytes, num_bytes, 100 * current_bytes / num_bytes))

        all_utimes = np.hstack(all_utimes)
        all_hits = np.vstack(all_hits)
        all_ints = np.hstack(all_ints)

        write_ply('test_hits',
                  [all_hits, all_ints, all_utimes],
                  ['x', 'y', 'z', 'intensity', 'utime'])

    print("Read %d total hits from %ld to %ld" % (total_hits, first_utime, last_utime))

    return 0


def raw_frames_ply():

    # In files
    data_path = '../../Data/NCLT'
    velo_folder = 'velodyne_data'

    # Out folder
    out_folder = join(data_path, 'raw_ply')
    if not exists(out_folder):
        makedirs(out_folder)

    # Transformation from body to velodyne frame (from NCLT paper)
    x_body_velo = np.array([0.002, -0.004, -0.957, 0.807, 0.166, -90.703])
    H_body_velo = ssc_to_homo(x_body_velo, ssc_in_radians=False)
    H_velo_body = np.linalg.inv(H_body_velo)
    x_body_lb3 = np.array([0.035, 0.002, -1.23, -179.93, -0.23, 0.50])
    H_body_lb3 = ssc_to_homo(x_body_lb3, ssc_in_radians=False)
    H_lb3_body = np.linalg.inv(H_body_lb3)

    # properties list for binary file reading
    properties = [('x', '<u2'),
                  ('y', '<u2'),
                  ('z', '<u2'),
                  ('i', '<u1'),
                  ('l', '<u1')]

    # Get gt files and days
    days = np.sort([v_f for v_f in listdir(join(data_path, velo_folder))])

    for d, day in enumerate(days):

        # Out folder
        day_out_folder = join(out_folder, day)
        if not exists(day_out_folder):
            makedirs(day_out_folder)

        # Day binary file
        hits_path = join(data_path, velo_folder, day, 'velodyne_hits.bin')

        # Init variables
        all_hits = []
        num_bytes = getsize(hits_path)
        current_bytes = 0
        frame_i = 0
        last_phi = -1
        t0 = time.time()
        last_display = t0

        with open(hits_path, 'rb') as f_bin:

            while True:

                ####################
                # Read packet header
                ####################

                # Verify packet
                magic = f_bin.read(8)
                if magic == b'':
                    break
                if not verify_magic(magic):
                    print('Could not verify magic')

                # Get header info
                num_hits = struct.unpack('<I', f_bin.read(4))[0]
                utime = struct.unpack('<Q', f_bin.read(8))[0]
                padding = f_bin.read(4) # Do not convert padding (it is an int always equal to zero)

                ##################
                # Read binary hits
                ##################

                # Get face data
                packet_data = np.fromfile(f_bin, dtype=properties, count=num_hits)

                # Rescale point coordinates
                hits = np.vstack((packet_data['x'], packet_data['y'], packet_data['z'])).astype(np.float32).T
                hits *= 0.005
                hits += -100.0

                ##########################
                # Gather frame if complete
                ##########################

                phi = (np.arctan2(- hits[-1, 1], hits[-1, 0]) - np.pi / 2) % (2 * np.pi)

                if phi < last_phi:

                    # Stack all frame points
                    f_hits = np.vstack(all_hits)

                    # Save frame
                    frame_name = join(day_out_folder, '{:.0f}.ply'.format(last_utime))
                    write_ply(frame_name,
                              [f_hits],
                              ['x', 'y', 'z'])

                    # Display
                    t = time.time()
                    if (t - last_display) > 1.0:
                        last_display = t
                        message = '{:s}: frame {:7d} ({:6d} points)'
                        message += ' => {:5.1f}%  and  {:02d}:{:02d}:{:02d} remaining)'

                        # Predict remaining time
                        elapsed = t - t0
                        remaining = int(elapsed * num_bytes / current_bytes - elapsed)

                        hours = remaining // 3600
                        remaining = remaining - 3600 * hours
                        minutes = remaining // 60
                        seconds = remaining - 60 * minutes

                        print(message.format(day,
                                             frame_i,
                                             f_hits.shape[0],
                                             100 * current_bytes / num_bytes,
                                             hours, minutes, seconds))

                    # Update variables
                    frame_i += 1
                    all_hits = []

                ##############################
                # Append hits to current frame
                ##############################

                # Update last phi
                last_phi = phi
                last_utime = utime

                # Append new data
                all_hits.append(hits)

                # Count bytes already read
                current_bytes += 24 + 8 * num_hits

    return 0


def frames_to_ply(show_frames=False):

    # In files
    data_path = '../../Data/NCLT'
    velo_folder = 'velodyne_data'

    days = np.sort([d for d in listdir(join(data_path, velo_folder))])

    for day in days:

        # Out files
        ply_folder = join(data_path, 'frames_ply', day)
        if not exists(ply_folder):
            makedirs(ply_folder)

        day_path = join(data_path, velo_folder, day, 'velodyne_sync')
        f_names = np.sort([f for f in listdir(day_path) if f[-4:] == '.bin'])

        N = len(f_names)
        print('Reading', N, 'files')

        for f_i, f_name in enumerate(f_names):

            ply_name = join(ply_folder, f_name[:-4] + '.ply')
            if exists(ply_name):
                continue


            t1 = time.time()

            hits = []
            ints = []

            with open(join(day_path, f_name), 'rb') as f_bin:

                while True:
                    x_str = f_bin.read(2)

                    # End of file
                    if x_str == b'':
                        break

                    x = struct.unpack('<H', x_str)[0]
                    y = struct.unpack('<H', f_bin.read(2))[0]
                    z = struct.unpack('<H', f_bin.read(2))[0]
                    intensity = struct.unpack('B', f_bin.read(1))[0]
                    l = struct.unpack('B', f_bin.read(1))[0]

                    hits += [[x, y, z]]
                    ints += [intensity]

            ints = np.array(ints, dtype=np.uint8)
            hits = np.array(hits, dtype=np.float32)
            hits *= 0.005
            hits += -100.0

            write_ply(ply_name,
                      [hits, ints],
                      ['x', 'y', 'z', 'intensity'])

            t2 = time.time()
            print('File {:s} {:d}/{:d} Done in {:.1f}s'.format(f_name, f_i, N, t2 - t1))

            if show_frames:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(hits[:, 0], hits[:, 1], -hits[:, 2], c=-hits[:, 2], s=5, linewidths=0)
                plt.show()

    return 0


def merge_day_pointclouds(show_day_trajectory=False, only_SLAM_nodes=False):
    """
    Recreate the whole day point cloud thks to gt pose
    Generate gt_annotation of mobile objects
    """

    # In files
    data_path = '../../Data/NCLT'
    gt_folder = 'ground_truth'
    cov_folder = 'ground_truth_cov'

    # Transformation from body to velodyne frame (from NCLT paper)
    x_body_velo = np.array([0.002, -0.004, -0.957, 0.807, 0.166, -90.703])
    H_body_velo = ssc_to_homo(x_body_velo, ssc_in_radians=False)
    H_velo_body = np.linalg.inv(H_body_velo)
    x_body_lb3 = np.array([0.035, 0.002, -1.23, -179.93, -0.23, 0.50])
    H_body_lb3 = ssc_to_homo(x_body_lb3, ssc_in_radians=False)
    H_lb3_body = np.linalg.inv(H_body_lb3)

    # Get gt files and days
    gt_files = np.sort([gt_f for gt_f in listdir(join(data_path, gt_folder)) if gt_f[-4:] == '.csv'])
    cov_files = np.sort([cov_f for cov_f in listdir(join(data_path, cov_folder)) if cov_f[-4:] == '.csv'])
    days = [d[:-4].split('_')[1] for d in gt_files]

    # Load all gt poses
    print('\nLoading days groundtruth poses...')
    t0 = time.time()
    gt_H = []
    gt_t = []
    for d, gt_f in enumerate(gt_files):

        t1 = time.time()

        gt_pkl_file = join(data_path, gt_folder, gt_f[:-4] + '.pkl')
        if exists(gt_pkl_file):
            # Read pkl
            with open(gt_pkl_file, 'rb') as f:
                day_gt_t, day_gt_H = pickle.load(f)

        else:
            # File paths
            gt_csv = join(data_path, gt_folder, gt_f)

            # Load gt
            gt = np.loadtxt(gt_csv, delimiter=',')

            # Convert gt to homogenous rotation/translation matrix
            day_gt_t = gt[:, 0]
            day_gt_H = ssc_to_homo(gt[:, 1:])

            # Save pickle
            with open(gt_pkl_file, 'wb') as f:
                pickle.dump([day_gt_t, day_gt_H], f)

            t2 = time.time()
            print('{:s} {:d}/{:d} Done in {:.1f}s'.format(gt_f, d, gt_files.shape[0], t2 - t1))

        gt_t += [day_gt_t]
        gt_H += [day_gt_H]

        if show_day_trajectory:

            cov_csv = join(data_path, cov_folder, cov_files[d])
            cov = np.loadtxt(cov_csv, delimiter=',')
            t_cov = cov[:, 0]
            t_cov_bool = np.logical_and(t_cov > np.min(day_gt_t), t_cov < np.max(day_gt_t))
            t_cov = t_cov[t_cov_bool]

            # Note: Interpolation is not needed, this is done as a convinience
            interp = scipy.interpolate.interp1d(day_gt_t, day_gt_H[:, :3, 3], kind='nearest', axis=0)
            node_poses = interp(t_cov)

            plt.figure()
            plt.scatter(day_gt_H[:, 1, 3], day_gt_H[:, 0, 3], 1, c=-day_gt_H[:, 2, 3], linewidth=0)
            plt.scatter(node_poses[:, 1], node_poses[:, 0], 1, c=-node_poses[:, 2], linewidth=5)
            plt.axis('equal')
            plt.title('Ground Truth Position of Nodes in SLAM Graph')
            plt.xlabel('East (m)')
            plt.ylabel('North (m)')
            plt.colorbar()

            plt.show()

    t2 = time.time()
    print('Done in {:.1f}s\n'.format(t2 - t0))

    # Out files
    out_folder = join(data_path, 'day_ply')
    if not exists(out_folder):
        makedirs(out_folder)

    # Focus on a particular point
    p0 = np.array([-220, -527, 12])
    center_radius = 10.0
    point_radius = 50.0

    # Loop on days
    for d, day in enumerate(days):

        #if day != '2012-02-05':
        #    continue
        day_min_t = gt_t[d][0]
        day_max_t = gt_t[d][-1]

        frames_folder = join(data_path, 'frames_ply', day)
        f_times = np.sort([float(f[:-4]) for f in listdir(frames_folder) if f[-4:] == '.ply'])

        # If we want, load only SLAM nodes
        if only_SLAM_nodes:

            # Load node timestamps
            cov_csv = join(data_path, cov_folder, cov_files[d])
            cov = np.loadtxt(cov_csv, delimiter=',')
            t_cov = cov[:, 0]
            t_cov_bool = np.logical_and(t_cov > day_min_t, t_cov < day_max_t)
            t_cov = t_cov[t_cov_bool]

            # Find closest lidar frames
            t_cov = np.expand_dims(t_cov, 1)
            diffs = np.abs(t_cov - f_times)
            inds = np.argmin(diffs, axis=1)
            f_times = f_times[inds]

        # Is this frame in gt
        f_t_bool = np.logical_and(f_times > day_min_t, f_times < day_max_t)
        f_times = f_times[f_t_bool]

        # Interpolation gt poses to frame timestamps
        interp = scipy.interpolate.interp1d(gt_t[d], gt_H[d], kind='nearest', axis=0)
        frame_poses = interp(f_times)

        N = len(f_times)
        world_points = []
        world_frames = []
        world_frames_c = []
        print('Reading', day, ' => ', N, 'files')
        for f_i, f_t in enumerate(f_times):

            t1 = time.time()

            #########
            # GT pose
            #########

            H = frame_poses[f_i].astype(np.float32)
            # s = '\n'
            # for cc in H:
            #     for c in cc:
            #         s += '{:5.2f} '.format(c)
            #     s += '\n'
            # print(s)

            #############
            # Focus check
            #############

            if np.linalg.norm(H[:3, 3] - p0) > center_radius:
                continue

            ###################################
            # Local frame coordinates for debug
            ###################################

            # Create artificial frames
            x = np.linspace(0, 1, 50, dtype=np.float32)
            points = np.hstack((np.vstack((x, x*0, x*0)), np.vstack((x*0, x, x*0)), np.vstack((x*0, x*0, x)))).T
            colors = ((points > 0.1).astype(np.float32) * 255).astype(np.uint8)

            hpoints = np.hstack((points, np.ones_like(points[:, :1])))
            hpoints = np.matmul(hpoints, H.T)
            hpoints[:, 3] *= 0
            world_frames += [hpoints[:, :3]]
            world_frames_c += [colors]

            #######################
            # Load velo point cloud
            #######################

            # Load frame ply file
            f_name = '{:.0f}.ply'.format(f_t)
            data = read_ply(join(frames_folder, f_name))
            points = np.vstack((data['x'], data['y'], data['z'])).T
            #intensity = data['intensity']

            hpoints = np.hstack((points, np.ones_like(points[:, :1])))
            hpoints = np.matmul(hpoints, H.T)
            hpoints[:, 3] *= 0
            hpoints[:, 3] += np.sqrt(f_t - f_times[0])

            # focus check
            focus_bool = np.linalg.norm(hpoints[:, :3] - p0, axis=1) < point_radius
            hpoints = hpoints[focus_bool, :]

            world_points += [hpoints]

            t2 = time.time()
            print('File {:s} {:d}/{:d} Done in {:.1f}s'.format(f_name, f_i, N, t2 - t1))

        if len(world_points) < 2:
            continue

        world_points = np.vstack(world_points)


        ###### DEBUG
        world_frames = np.vstack(world_frames)
        world_frames_c = np.vstack(world_frames_c)
        write_ply('testf.ply',
                  [world_frames, world_frames_c],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        ###### DEBUG

        print(world_points.shape, world_points.dtype)

        # Subsample merged frames
        # world_points, features = grid_subsampling(world_points[:, :3],
        #                                           features=world_points[:, 3:],
        #                                           sampleDl=0.1)
        features = world_points[:, 3:]
        world_points = world_points[:, :3]

        print(world_points.shape, world_points.dtype)

        write_ply('test' + day + '.ply',
                  [world_points, features],
                  ['x', 'y', 'z', 't'])


        # Generate gt annotations

        # Subsample day ply (for visualization)

        # Save day ply

        # a = 1/0


def local_PCA(points):

    # Compute the barycenter
    center = np.mean(points, axis=0)

    # Centered clouds
    points_c = points - center

    # Covariance matrix
    C = (points_c.T).dot(points_c) / points.shape[0]

    # Eigenvalues
    return np.linalg.eigh(C)


def estimate_normals_planarity_debug(cloud):
    """
    Custom function that estimates normals and planarity of lidar frames, using polar coordinates neighborhoods.
    :param cloud: Open3D PointCloud.
    :return: planarities (Normals are modified in place)
    """

    # Rescale for numerical stability
    #

    t = [time.time()]

    # Get point cloud
    points = cloud.astype(np.float32)
    normals0, planarity, linearity = polar_normals(points.astype(np.float32), verbose=1)
    scores0 = planarity + linearity

    t += [time.time()]

    print(normals0.dtype, normals0.shape)
    print(scores0.dtype, scores0.shape)



    # Transform to polar coordinates
    polar_points = cart2pol(points)

    t += [time.time()]

    # Define search radius in l1 metric. Vertical angular resolution of HDL32 is 1.29
    angular_res = 1.29 * np.pi / 180
    polar_r = 1.5 * angular_res

    # Define horizontal scale (smaller distance for the neighbor in horizontal direction)
    horizontal_scale = 0.5

    # Use log of range so that neighbor radius is proportional to the range.
    range_scale = 4.0
    polar_points[:, 0] = np.log(polar_points[:, 0]) * polar_r / (np.log((1 + polar_r) / (1 - polar_r)) * range_scale)

    # Apply horizontal scale
    polar_points[:, 2] *= 1 / horizontal_scale

    t += [time.time()]

    # Create 2d KDTree to search lidar neighborhoods
    polar_tree = KDTree(polar_points, metric='l2')

    t += [time.time()]

    # Find neighbors
    all_neighb_inds = polar_tree.query_radius(polar_points, polar_r)

    t += [time.time()]

    # Rescale everything
    # polar_points[:, 2] *= horizontal_scale
    # polar_points[:, 0] = np.exp(polar_points[:, 0] * np.log((1 + polar_r) / (1 - polar_r)) * range_scale  / polar_r)

    # Compute covariance matrices
    all_eigenvalues = np.empty(polar_points.shape, dtype=np.float32)
    all_eigenvectors = np.empty((polar_points.shape[0], 3, 3), dtype=np.float32)
    for i, neighb_inds in enumerate(all_neighb_inds):
        all_eigenvalues[i, :], all_eigenvectors[i, :, :] = local_PCA(points[neighb_inds, :])

    t += [time.time()]

    # Compute normals and planarity
    normals = all_eigenvectors[:, :, 0]
    sphericity = 1 -all_eigenvalues[:, 0] / (all_eigenvalues[:, 2] + 1e-9)

    t += [time.time()]

    # Choose random point for showing
    rand_inds = np.random.randint(polar_points.shape[0], size=100)

    features = np.zeros_like(polar_points[:, 2])
    for ri, rand_id in enumerate(rand_inds):
        features[all_neighb_inds[rand_id]] = ri
        features[rand_id] = 2 * len(rand_inds)


    write_ply('ttt_xyz.ply',
              [points, normals, features, sphericity, scores0],
              ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f', 'score', 'cpp_score'])

    # polar_points[:, 1] *= 180 / np.pi
    # polar_points[:, 2] *= 180 / np.pi

    #polar_points[:, 0] = np.exp(polar_points[:, 0] * np.log((1 + polar_r) / (1 - polar_r)) * range_scale  / polar_r)

    polar_points = polar_points[:, [2, 1, 0]]
    write_ply('ttt_rtp.ply',
              [polar_points, polar_points[:, 1] * 0, features],
              ['x', 'y', 'z', 'i', 'f'])



    # Filter outlier from ray/edges










    # Assign normals to pointcloud structure
    #cloud.normals = o3d.utility.Vector3dVector(normals)

    t += [time.time()]

    # Display timings
    print('\n*****************\n')
    print('Validation timings:')
    i = 0
    print('C++ ....... {:.1f}s'.format(1000 * (t[i + 1] - t[i])))
    i += 1
    print('polar ..... {:.1f}s'.format(1000 * (t[i + 1] - t[i])))
    i += 1
    print('scale ... {:.1f}s'.format(1000 * (t[i + 1] - t[i])))
    i += 1
    print('Tree ...... {:.1f}s'.format(1000 * (t[i + 1] - t[i])))
    i += 1
    print('neighb .... {:.1f}s'.format(1000 * (t[i + 1] - t[i])))
    i += 1
    print('PCA ...... {:.1f}s'.format(1000 * (t[i + 1] - t[i])))
    i += 1
    print('features . {:.1f}s'.format(1000 * (t[i + 1] - t[i])))
    i += 1
    print('Assign ... {:.1f}s'.format(1000 * (t[i + 1] - t[i])))
    print('\n*****************\n')




    return sphericity


def estimate_normals_planarity(cloud):
    """
    Custom function that estimates normals and planarity of lidar frames, using polar coordinates neighborhoods.
    :param cloud: Open3D PointCloud.
    :return: planarities (Normals are modified in place)
    """

    # Get point cloud
    points = np.asarray(cloud.points)
    normals, planarity, linearity = polar_normals(points.astype(np.float32), verbose=1)

    # Assign normals to pointcloud structure
    cloud.normals = o3d.utility.Vector3dVector(normals)

    return scores


def gaussian_conv_filter(dimension=3, size=5):

    # Sigma according to size
    sig = (size/2 - 0.5) / 2
    eps = 1e-6

    # Get coordinates
    coords = np.arange(-size/2 + 0.5, size/2, 1.0)
    if dimension == 2:
        x, y = np.meshgrid(coords, coords)
        sq_r = x ** 2 + y ** 2
    elif dimension == 3:
        x, y, z = np.meshgrid(coords, coords, coords)
        sq_r = x ** 2 + y ** 2 + z ** 2
    elif dimension == 4:
        x, y, z, t = np.meshgrid(coords, coords, coords, coords)
        sq_r = x ** 2 + y ** 2 + z ** 2 + t ** 2
    else:
        raise ValueError('Unsupported dimension (max is 4)')

    return torch.exp(-torch.from_numpy(sq_r.astype(np.float32)) / (2 * sig ** 2 + eps))


def normal_filtering(normals, debug=False):

    # Discretise the sphere in carthesian coordiantes to avoid the pole reolution problem
    voxel_size = 0.05

    # Compute voxel indice for each point
    grid_indices = (np.floor(normals / voxel_size)).astype(int)

    # Limits of the grid
    min_grid_indices = np.amin(grid_indices, axis=0)
    max_grid_indices = np.amax(grid_indices, axis=0)

    # Number of cells in each direction
    deltaX, deltaY, deltaZ = max_grid_indices - min_grid_indices + 1

    # Relocate indices
    grid_indices -= min_grid_indices

    # Scalar equivalent to grid indices
    scalar_indices = grid_indices[:, 0] + grid_indices[:, 1] * deltaX + grid_indices[:, 2] * deltaX * deltaY
    unique_inds, inverse, counts = np.unique(scalar_indices, return_counts=True, return_inverse=True)

    # Get counts in a 3D matrix
    unique_z = unique_inds // (deltaX * deltaY)
    unique_inds -= unique_z * deltaX * deltaY
    unique_y = unique_inds // deltaX
    unique_x = unique_inds - unique_y * deltaX
    count_matrix = np.zeros((deltaX, deltaY, deltaZ), dtype=np.float32)
    count_matrix[unique_x, unique_y, unique_z] += counts

    # Smooth them with a gaussian filter convolution
    torch_conv = torch.nn.Conv3d(1, 1, kernel_size=5, stride=1, bias=False)
    torch_conv.weight.requires_grad_(False)
    torch_conv.weight *= 0
    torch_conv.weight += gaussian_conv_filter(3, 5)
    torch_conv.weight *= torch.sum(torch_conv.weight) ** -1
    count_matrix = np.expand_dims(count_matrix, 0)
    count_matrix = np.expand_dims(count_matrix, 0)
    torch_count = torch.from_numpy(count_matrix)
    torch_count = torch.nn.functional.pad(torch_count, [2, 2, 2, 2, 2, 2])
    smooth_counts = torch.squeeze(torch_conv(torch_count))
    smooth_counts = smooth_counts.numpy()[unique_x, unique_y, unique_z]

    #################################################
    # Create weight according to the normal direction
    #################################################

    # Only 20% of the normals bins are kept For the rest, we use weights based on ditances
    weights = (smooth_counts > np.percentile(smooth_counts, 80)).astype(np.float32)

    # Show histogram in a spherical point cloud
    if debug:
        n_cloud = np.vstack((unique_x, unique_y, unique_z)).astype(np.float32).T
        n_cloud = (n_cloud + min_grid_indices.astype(np.float32) + 0.5) * voxel_size
        #n_cloud = n_cloud / np.linalg.norm(n_cloud, axis=1, keepdims=True)
        write_ply('nnn_NORMAL_HIST.ply',
                  [n_cloud, smooth_counts],
                  ['x', 'y', 'z', 'counts'])

        a = 1/0

    return weights[inverse]


def load_gt_poses(gt_path, only_day_1=False):

    gt_files = np.sort([gt_f for gt_f in listdir(gt_path) if gt_f[-4:] == '.csv'])
    gt_H = []
    gt_t = []
    for d, gt_f in enumerate(gt_files):

        t1 = time.time()
        gt_pkl_file = join(gt_path, gt_f[:-4] + '.pkl')
        if exists(gt_pkl_file):
            # Read pkl
            with open(gt_pkl_file, 'rb') as f:
                day_gt_t, day_gt_H = pickle.load(f)

        else:
            # File paths
            gt_csv = join(gt_path, gt_f)

            # Load gt
            gt = np.loadtxt(gt_csv, delimiter=',')

            # Convert gt to homogenous rotation/translation matrix
            day_gt_t = gt[:, 0]
            day_gt_H = ssc_to_homo(gt[:, 1:])

            # Save pickle
            with open(gt_pkl_file, 'wb') as f:
                pickle.dump([day_gt_t, day_gt_H], f)

            t2 = time.time()
            print('{:s} {:d}/{:d} Done in {:.1f}s'.format(gt_f, d, gt_files.shape[0], t2 - t1))

        gt_t += [day_gt_t]
        gt_H += [day_gt_H]

        if only_day_1 and d > -1:
            break

    return gt_t, gt_H


def get_area_frames(days, gt_t, gt_H, raw_path, area_center, area_radius, only_day_1=False):

    # Loop on days
    day_f_times = []
    for d, day in enumerate(days):

        # Get frame timestamps
        frames_folder = join(raw_path, day)
        f_times = np.sort([float(f[:-4]) for f in listdir(frames_folder) if f[-4:] == '.ply'])

        # Ground truth does not cover all frames
        day_min_t = gt_t[d][0]
        day_max_t = gt_t[d][-1]
        f_t_bool = np.logical_and(f_times > day_min_t, f_times < day_max_t)
        f_times = f_times[f_t_bool]

        # Interpolation gt poses to frame timestamps
        interp = scipy.interpolate.interp1d(gt_t[d], gt_H[d], kind='nearest', axis=0)
        frame_poses = interp(f_times)

        # Closest frame to picked point
        closest_i = 0
        closest_d = 1e6

        new_f_times = []
        for f_i, f_t in enumerate(f_times):

            # GT pose
            H = frame_poses[f_i].astype(np.float32)

            # Focus check
            f_dist = np.linalg.norm(H[:3, 3] - area_center)
            if f_dist > area_radius:
                continue

            # Save closest frame
            if (f_dist < closest_d):
                closest_d = f_dist
                closest_i = len(new_f_times)

            # Append frame to candidates
            new_f_times.append(f_t)

        # Filter to only get subsequent frames
        new_f_times = np.array(new_f_times, dtype=np.float64)
        gaps = new_f_times[1:] - new_f_times[:-1]
        med_gap = np.median(gaps[:50])
        jumps = np.sort(np.where(gaps > 5 * med_gap)[0])
        i0 = 0
        i1 = len(new_f_times)
        for j in jumps:
            if j + 1 < closest_i:
                i0 = j + 1
        for j in jumps[::-1]:
            if j + 1 > closest_i:
                i1 = j + 1
        day_f_times.append(new_f_times[i0:i1])

        if only_day_1 and d > -1:
            break

    return day_f_times


def test_icp_registration():
    """
    Test ICP registration Use GT to extract a small interesting region.
    """

    ############
    # Parameters
    ############

    # In files
    data_path = '../../Data/NCLT'
    gt_folder = 'ground_truth'
    cov_folder = 'ground_truth_cov'

    # Transformation from body to velodyne frame (from NCLT paper)
    x_body_velo = np.array([0.002, -0.004, -0.957, 0.807, 0.166, -90.703])
    H_body_velo = ssc_to_homo(x_body_velo, ssc_in_radians=False)
    H_velo_body = np.linalg.inv(H_body_velo)
    x_body_lb3 = np.array([0.035, 0.002, -1.23, -179.93, -0.23, 0.50])
    H_body_lb3 = ssc_to_homo(x_body_lb3, ssc_in_radians=False)
    H_lb3_body = np.linalg.inv(H_body_lb3)

    # Out files
    out_folder = join(data_path, 'day_ply')
    if not exists(out_folder):
        makedirs(out_folder)

    # Get gt files and days
    gt_files = np.sort([gt_f for gt_f in listdir(join(data_path, gt_folder)) if gt_f[-4:] == '.csv'])
    cov_files = np.sort([cov_f for cov_f in listdir(join(data_path, cov_folder)) if cov_f[-4:] == '.csv'])
    days = [d[:-4].split('_')[1] for d in gt_files]

    ###############
    # Load GT poses
    ###############

    print('\nLoading days groundtruth poses...')
    t0 = time.time()
    gt_H = []
    gt_t = []
    for d, gt_f in enumerate(gt_files):

        t1 = time.time()

        gt_pkl_file = join(data_path, gt_folder, gt_f[:-4] + '.pkl')
        if exists(gt_pkl_file):
            # Read pkl
            with open(gt_pkl_file, 'rb') as f:
                day_gt_t, day_gt_H = pickle.load(f)

        else:
            # File paths
            gt_csv = join(data_path, gt_folder, gt_f)

            # Load gt
            gt = np.loadtxt(gt_csv, delimiter=',')

            # Convert gt to homogenous rotation/translation matrix
            day_gt_t = gt[:, 0]
            day_gt_H = ssc_to_homo(gt[:, 1:])

            # Save pickle
            with open(gt_pkl_file, 'wb') as f:
                pickle.dump([day_gt_t, day_gt_H], f)

            t2 = time.time()
            print('{:s} {:d}/{:d} Done in {:.1f}s'.format(gt_f, d, gt_files.shape[0], t2 - t1))

        gt_t += [day_gt_t]
        gt_H += [day_gt_H]

        if d > -1:
            break

    t2 = time.time()
    print('Done in {:.1f}s\n'.format(t2 - t0))

    ########################
    # Get lidar frames times
    ########################

    # Focus on a particular point
    p0 = np.array([-220, -527, 12])
    center_radius = 10.0
    point_radius = 50.0

    print('\nGet timestamps in focused area...')
    t0 = time.time()

    # Loop on days
    day_f_times = []
    for d, day in enumerate(days):

        day_min_t = gt_t[d][0]
        day_max_t = gt_t[d][-1]

        frames_folder = join(data_path, 'raw_ply', day)
        f_times = np.sort([float(f[:-4]) for f in listdir(frames_folder) if f[-4:] == '.ply'])

        # Is this frame in gt
        f_t_bool = np.logical_and(f_times > day_min_t, f_times < day_max_t)
        f_times = f_times[f_t_bool]

        # Interpolation gt poses to frame timestamps
        interp = scipy.interpolate.interp1d(gt_t[d], gt_H[d], kind='nearest', axis=0)
        frame_poses = interp(f_times)

        N = len(f_times)
        new_f_times = []
        for f_i, f_t in enumerate(f_times):

            t1 = time.time()

            # GT pose
            H = frame_poses[f_i].astype(np.float32)

            # Focus check
            if np.linalg.norm(H[:3, 3] - p0) > center_radius:
                continue

            new_f_times.append(f_t)

        # DEBUGGGGGG
        new_f_times = new_f_times[5:-5]

        day_f_times.append(np.array(new_f_times, dtype=np.float64))

        if d > -1:
            break

    t2 = time.time()
    print('Done in {:.1f}s\n'.format(t2 - t0))

    ###########################
    # coarse map with pt2pt icp
    ###########################

    for d, day in enumerate(days):

        frames_folder = join(data_path, 'raw_ply', day)

        N = len(day_f_times[d])
        print('Reading', day, ' => ', N, 'files')

        # Load first frame as map
        last_transform = np.eye(4)
        last_cloud = None
        threshold = 0.3
        score_thresh = 0.99
        voxel_size = 0.1
        transform_list = []
        cloud_list = []
        cloud_map = None
        full_map = None
        full_map_t = None
        verbose = 1

        t = [time.time()]

        for f_i, f_t in enumerate(day_f_times[d]):

            #######################
            # Load velo point cloud
            #######################

            t = [time.time()]

            # Load frame ply file
            f_name = '{:.0f}.ply'.format(f_t)
            cloud = o3d.io.read_point_cloud(join(frames_folder, f_name))

            t += [time.time()]

            # Cloud normals and planarity
            scores = estimate_normals_planarity(cloud)

            if f_i < 1:
                last_cloud = cloud
                cloud_map = cloud
                continue
            t += [time.time()]

            # Remove low score for fitting
            cloud_down = o3d.geometry.PointCloud()
            cloud_down.points = o3d.utility.Vector3dVector(np.asarray(cloud.points)[scores > score_thresh, :])
            cloud_down.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals)[scores > score_thresh, :])

            # Downsample target
            cloud_down = cloud_down.voxel_down_sample(voxel_size)

            # if f_i > 2:
            #
            #     np.asarray(last_cloud.normals).astype(np.float32)
            #     new_scores = np.ones_like(np.asarray(cloud_down.points).astype(np.float32))[:, 0]
            #     H, rms = pt2pl_icp(np.asarray(cloud_down.points).astype(np.float32),
            #                              np.asarray(last_cloud.points).astype(np.float32),
            #                              np.asarray(last_cloud.normals).astype(np.float32),
            #                              new_scores,
            #                              n_samples=1000,
            #                              max_pairing_dist=0.2,
            #                              max_iter=10,
            #                              minDiffRMS=0.001)
            #
            #     print(H)
            #     print(rms)
            #     a = 1 / 0

            t += [time.time()]

            # Measure initial ICP metrics
            if verbose == 2:
                reg_init = o3d.registration.evaluate_registration(cloud_down, last_cloud,
                                                                  threshold, last_transform)
                t += [time.time()]
            else:
                reg_init = None

            # Apply ICP
            reg_pt2pl = o3d.registration.registration_icp(
                cloud_down, last_cloud, threshold, last_transform,
                o3d.registration.TransformationEstimationPointToPlane())
            t += [time.time()]

            # Print results
            if verbose == 2:
                print('ICP convergence:')
                print('fitness ...... {:7.4f} => {:7.4f}'.format(reg_init.fitness,
                                                                 reg_pt2pl.fitness))
                print('inlier_rmse .. {:7.4f} => {:7.4f}'.format(reg_init.inlier_rmse,
                                                                 reg_pt2pl.inlier_rmse))
                print('corresp_n .... {:7d} => {:7d}'.format(np.asarray(reg_init.correspondence_set).shape[0],
                                                             np.asarray(reg_pt2pl.correspondence_set).shape[0]))

            # Apply transformation for the init of next step
            cloud_down.transform(reg_pt2pl.transformation)

            if verbose == 2:

                # Save init cloud
                # cloud_init = copy.deepcopy(cloud)
                # cloud_init.transform(last_transform)
                # write_ply('ttt_{:d}_init.ply'.format(f_i),
                #           [np.asarray(cloud_init.points)],
                #           ['x', 'y', 'z'])

                # Save result cloud
                cloud.transform(reg_pt2pl.transformation)
                write_ply('ttt_{:d}_reg.ply'.format(f_i),
                          [np.asarray(cloud.points)],
                          ['x', 'y', 'z'])

                t += [time.time()]

                # Update sub map
                cloud_map.points.extend(cloud.points)
                cloud_map = cloud_map.voxel_down_sample(voxel_size=voxel_size)
                write_ply('tt_sub_map.ply'.format(f_i),
                          [np.asarray(cloud_map.points)],
                          ['x', 'y', 'z'])

                # Update full map
                if full_map is None:
                    full_map = copy.deepcopy(cloud_down)
                    full_map_t = np.full(shape=(np.asarray(cloud_down.points).shape[0],),
                                         fill_value=f_t - day_f_times[d][0],
                                         dtype=np.float64)
                else:
                    full_map.points.extend(cloud_down.points)
                    full_map_t = np.hstack((full_map_t, np.full(shape=(np.asarray(cloud_down.points).shape[0],),
                                                                fill_value=f_t - day_f_times[d][0],
                                                                dtype=np.float64)))
                write_ply('tt_full_map.ply'.format(f_i),
                          [np.asarray(full_map.points), full_map_t],
                          ['x', 'y', 'z', 't'])
                t += [time.time()]

            # Update variables
            last_cloud = cloud_down
            last_transform = reg_pt2pl.transformation
            transform_list += [reg_pt2pl.transformation]
            cloud_list += [np.asarray(cloud_down.points).astype(np.float32)]

            t += [time.time()]

            if verbose > 0:
                print('{:.0f} registered on {:.0f} in {:.1f}ms ({:d}/{:d})'.format(f_t,
                                                                                   day_f_times[d][f_i - 1],
                                                                                   1000 * (t[-1] - t[0]),
                                                                                   f_i,
                                                                                   N))
            # Display timings
            if verbose == 2:
                print('\n*********************')
                i = 0
                print('Load ...... {:.1f}ms'.format(1000 * (t[i + 1] - t[i])))
                i += 1
                print('Normals ... {:.1f}ms'.format(1000 * (t[i + 1] - t[i])))
                i += 1
                print('Filter .... {:.1f}ms'.format(1000 * (t[i + 1] - t[i])))
                i += 1
                print('Eval ...... {:.1f}ms'.format(1000 * (t[i + 1] - t[i])))
                i += 1
                print('ICP ....... {:.1f}ms'.format(1000 * (t[i + 1] - t[i])))
                i += 1
                print('Transform . {:.1f}ms'.format(1000 * (t[i + 1] - t[i])))
                i += 1
                print('Save ...... {:.1f}ms'.format(1000 * (t[i + 1] - t[i])))
                i += 1
                print('Update .... {:.1f}ms'.format(1000 * (t[i + 1] - t[i])))
                print('*********************\n')
                print('\n********************************************\n')

        # Save results

        full_map = np.vstack(cloud_list)
        times_list = [f_t - day_f_times[d][0] for f_t in day_f_times[d][1:]]
        full_map_t = np.vstack([np.full((cld.shape[0], 1), f_t, dtype=np.float64)
                                for f_t, cld in zip(times_list, cloud_list)])
        write_ply('tt_full_map.ply',
                  [full_map, full_map_t],
                  ['x', 'y', 'z', 't'])

        # TODO:
        #  > Multithread this first path at a python level (use Pytorch?). No need for multitherad cpp wrapper
        #  > Second path (refinement) with normals re-estimnated on the map
        #  > Take motion distortion into account (in second path).
        #  > Use graph optimization for loop closure and day merging

        a = 1 / 0


def bundle_icp_debug(verbose=2):
    """
    Test ICP registration Use GT to extract a small interesting region.
    """

    ############
    # Parameters
    ############

    # Path to data
    data_path = '../../Data/NCLT'
    gt_folder = 'ground_truth'
    raw_folder = 'raw_ply'
    days = np.sort([d for d in listdir(join(data_path, raw_folder))])

    # Out files
    out_folder = join(data_path, 'day_ply')
    if not exists(out_folder):
        makedirs(out_folder)

    # Stride (nb of frames skipped for transformations)
    frame_stride = 2

    # Bundle size (number of frames jointly optimized) and stride (nb of frames between each bundle start)
    bundle_size = 7
    bundle_stride = bundle_size - 1

    # Normal estimation parameters
    score_thresh = 0.99

    # Pointcloud filtering parameters
    map_voxel_size = 0.05
    frame_voxel_size = -0.05

    # Group of frames saved together
    save_group = 100

    ###############
    # Load GT poses
    ###############

    print('\nLoading days groundtruth poses...')
    t0 = time.time()
    gt_t, gt_H = load_gt_poses(join(data_path, gt_folder), only_day_1=True)
    t2 = time.time()
    print('Done in {:.1f}s\n'.format(t2 - t0))

    #######################
    # Get lidar frame times
    #######################

    # Focus on a particular point
    p0 = np.array([-220, -527, 12])
    R0 = 10.0

    print('\nGet timestamps in focused area...')
    t0 = time.time()
    day_f_times = get_area_frames(days, gt_t, gt_H, join(data_path, raw_folder), p0, R0, only_day_1=True)
    t2 = time.time()
    print('Done in {:.1f}s\n'.format(t2 - t0))

    ###########################
    # coarse map with pt2pl icp
    ###########################

    for d, day in enumerate(days):

        # List of transformation we are trying to optimize
        frames_folder = join(data_path, 'raw_ply', day)
        f_times = [f_t for f_t in day_f_times[d][::frame_stride]]
        transform_list = [np.eye(4) for _ in f_times]
        last_saved_frames = 0
        FPS = 0
        N = len(f_times)

        for b_i, bundle_i0 in enumerate(np.arange(0, len(f_times), bundle_stride)):

            ####################
            # Load bundle frames
            ####################

            t = [time.time()]

            if (bundle_i0 + bundle_size > len(f_times)):
                bundle_i0 = len(f_times) - bundle_size

            frame_pts = []
            frame_norms = []
            frame_w = []
            for f_t in f_times[bundle_i0:bundle_i0+bundle_size]:

                # Load ply format points
                f_name = '{:.0f}.ply'.format(f_t)
                data = read_ply(join(frames_folder, f_name))
                points = np.vstack((data['x'], data['y'], data['z'])).T

                t += [time.time()]

                # Get normals
                normals, planarity, linearity = polar_normals(points, radius=1.5, h_scale=0.5)

                # Remove low quality normals for fitting
                points = points[norm_scores > score_thresh]
                normals = normals[norm_scores > score_thresh]
                norm_scores = (norm_scores[norm_scores > score_thresh] - score_thresh) / (1 - score_thresh)

                t += [time.time()]

                # Subsample to reduce number of points
                if frame_voxel_size > 0:

                    # grid supsampling
                    points, normals = grid_subsampling(points, features=normals, sampleDl=map_voxel_size)

                    # Renormalize normals
                    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

                # Filter out points according to main normal directions (NOt necessary if normals are better computed)
                bool_filter = normal_filtering(normals) > 0.5
                points = points[bool_filter]
                normals = normals[bool_filter]
                norm_scores = norm_scores[bool_filter]

                t += [time.time()]

                # Compute score for each component of rotations / translation
                # Weights according to distance the futher, the higher (square rule because points lies on surfaces)
                #rot_scores = np.expand_dims(norm_scores, 1) * np.cross(points, normals, axis=1)
                #weights = np.hstack((rot_scores, -rot_scores))

                weights = np.expand_dims(norm_scores, 1)

                # Gather frames data
                frame_pts.append(points)
                frame_norms.append(normals)
                frame_w.append(weights)

                t += [time.time()]

            if verbose == 3:
                dt = np.array(t[1:]) - np.array(t[:-1])
                dt = dt.reshape(bundle_size, -1)
                timing_names = ['Load', 'Normals', 'Filter', 'Append']
                s = ''
                for t_name in timing_names:
                    s += '{:^10s} '.format(t_name)
                s += '\n'
                for b in range(bundle_size):
                    for t_i in range(len(timing_names)):
                        s += '{:^10.1f} '.format(1000 * dt[b, t_i])
                    s += '\n'
                print(s)

            t = t[:1]
            t += [time.time()]

            ##################
            # Apply bundle ICP
            ##################

            # for b in range(bundle_size):
            #     w_names = ['w{:d}'.format(i) for i in range(frame_w[b].shape[1])]
            #     write_ply('bb_init_{:02d}.ply'.format(b),
            #               [frame_pts[b], frame_w[b]],
            #               ['x', 'y', 'z'] + w_names)

            bundle_H, bundle_rms, all_H = bundle_pt2pl_icp(frame_pts,
                                                                frame_norms,
                                                                frame_w,
                                                                n_samples=1000,
                                                                max_pairing_dist=0.2,
                                                                max_iter=200,
                                                                avg_steps=5)


            t += [time.time()]

            save_debug_frames = False
            if save_debug_frames:
                print(all_H.shape)
                bundle_inds = []
                steps = []
                all_pts = []
                for s, HH in enumerate(all_H):
                    for b, H in enumerate(HH):
                        if b == 0:
                            world_H = np.linalg.inv(H)
                        else:
                            world_H = np.eye(4)
                            for bb in range(b, 0, -1):
                                world_H = np.matmul(HH[bb], world_H)
                        pts, clrs = frame_H_to_points(world_H, size=0.1)
                        pts = pts.astype(np.float32)
                        all_pts.append(pts)
                        bundle_inds.append(pts[:, 0]*0+b)
                        steps.append(pts[:, 0]*0+s)
                write_ply('bb_frames.ply',
                          [np.vstack(all_pts), np.hstack(steps), np.hstack(bundle_inds)],
                          ['x', 'y', 'z', 's', 'b'])

            debug_rms = False
            if debug_rms:


                fig = plt.figure('RMS')
                for b, b_rms in enumerate(bundle_rms):
                    if b == 0:
                        plt.plot(b_rms, '-', linewidth=2, label='{:d}-0'.format(bundle_size - 1))
                    else:
                        plt.plot(b_rms, '-', linewidth=1, label='{:d}-{:d}'.format(b, b - 1))
                plt.xlabel('steps')
                plt.ylabel('rms')
                #plt.legend(loc=1)
                plt.ylim(0, 0.3)


                all_H_inv = np.copy(all_H.transpose((0, 1, 3, 2)))
                all_H_inv[:, :, 3, :3] = 0
                all_H_inv[:, :, :3, 3:] = -np.matmul(all_H_inv[:, :, :3, :3], all_H[:, :, :3, 3:])
                dH = np.matmul(all_H[1:], all_H_inv[:-1])
                dH = dH.transpose((1, 0, 2, 3))

                plt.figure('dT')
                for b, b_dH in enumerate(dH):
                    b_dT = np.linalg.norm(b_dH[:, :3, 3], axis=1)
                    b_dT = running_mean(b_dT, 4)
                    if b == 0:
                        plt.plot(b_dT, '-', linewidth=2, label='{:d}-0'.format(bundle_size - 1))
                    else:
                        plt.plot(b_dT, '-', linewidth=1, label='{:d}-{:d}'.format(b, b - 1))
                plt.xlabel('steps')
                plt.ylabel('rms')
                #plt.legend(loc=1)
                plt.ylim(0, 0.05)

                plt.figure('dR')
                for b, b_dH in enumerate(dH):
                    b_dR = np.arccos((np.trace(b_dH[:, :3, :3], axis1=1, axis2=2) - 1) / 2)
                    b_dR = running_mean(b_dR, 4)
                    if b == 0:
                        plt.plot(b_dR, '-', linewidth=2, label='{:d}-0'.format(bundle_size - 1))
                    else:
                        plt.plot(b_dR, '-', linewidth=1, label='{:d}-{:d}'.format(b, b - 1))
                plt.xlabel('steps')
                plt.ylabel('rms')
                #plt.legend(loc=1)
                plt.ylim(0, 0.01)
                plt.show()

                a = 1/0


            # Update transformations to world coordinates
            for b in range(bundle_size):
                world_H = np.eye(4)
                for bb in range(b, 0, -1):
                    world_H = np.matmul(bundle_H[bb], world_H)
                world_H = np.matmul(transform_list[bundle_i0], world_H)
                transform_list[bundle_i0 + b] = world_H

            t += [time.time()]

            if verbose == 2:
                print('Bundle {:9.1f}ms / ICP {:.1f}ms => {:.1f} FPS'.format(1000 * (t[1] - t[0]),
                                                                             1000 * (t[2] - t[1]),
                                                                             bundle_size / (t[2] - t[1])))
            if verbose == 1:
                fmt_str = 'Bundle [{:3d},{:3d}]  --- {:5.1f}% or {:02d}:{:02d}:{:02d} remaining at {:.1f}fps'
                if bundle_i0 == 0:
                    FPS = bundle_size / (t[-1] - t[0])
                else:
                    FPS += (bundle_size / (t[-1] - t[0]) - FPS) / 10
                remaining = int((N - (bundle_i0 + bundle_size)) / FPS)
                hours = remaining // 3600
                remaining = remaining - 3600 * hours
                minutes = remaining // 60
                seconds = remaining - 60 * minutes
                print(fmt_str.format(bundle_i0,
                                     bundle_i0 + bundle_size - 1,
                                     100 * (bundle_i0 + bundle_size) / N,
                                     hours, minutes, seconds,
                                     FPS))

            # Save groups of 100 frames together
            if (bundle_i0 > last_saved_frames + save_group + 1):
                all_points = []
                all_traj_pts = []
                all_traj_clrs = []
                i0 = last_saved_frames
                i1 = i0 + save_group
                for i, world_H in enumerate(transform_list[i0: i1]):
                    # Load ply format points
                    f_name = '{:.0f}.ply'.format(f_times[i0 + i])
                    data = read_ply(join(frames_folder, f_name))
                    points = np.vstack((data['x'], data['y'], data['z'])).T

                    # Apply transf
                    world_pts = np.hstack((points, np.ones_like(points[:, :1])))
                    world_pts = np.matmul(world_pts, world_H.T)

                    # Save frame
                    world_pts[:, 3] = i0 + i
                    all_points.append(world_pts)

                    # also save trajectory
                    traj_pts, traj_clrs = frame_H_to_points(world_H, size=0.1)
                    traj_pts = np.hstack((traj_pts, np.ones_like(traj_pts[:, :1]) * (i0 + i)))
                    all_traj_pts.append(traj_pts.astype(np.float32))
                    all_traj_clrs.append(traj_clrs)

                last_saved_frames += save_group
                filename = join(out_folder, 'd_{:s}_{:05d}.ply'.format(day, i0))
                write_ply(filename,
                          [np.vstack(all_points)],
                          ['x', 'y', 'z', 't'])
                filename = join(out_folder, 'd_{:s}_{:05d}_traj.ply'.format(day, i0))
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
        for i, world_H in enumerate(transform_list[i0:]):
            # Load ply format points
            f_name = '{:.0f}.ply'.format(f_times[i0 + i])
            data = read_ply(join(frames_folder, f_name))
            points = np.vstack((data['x'], data['y'], data['z'])).T

            # Apply transf
            world_pts = np.hstack((points, np.ones_like(points[:, :1])))
            world_pts = np.matmul(world_pts, world_H.T)

            # Save frame
            world_pts[:, 3] = i0 + i
            all_points.append(world_pts)

            # also save trajectory
            traj_pts, traj_clrs = frame_H_to_points(world_H, size=0.1)
            traj_pts = np.hstack((traj_pts, np.ones_like(traj_pts[:, :1]) * (i0 + i)))
            all_traj_pts.append(traj_pts.astype(np.float32))
            all_traj_clrs.append(traj_clrs)

        last_saved_frames += save_group
        filename = join(out_folder, 'd_{:s}_{:05d}.ply'.format(day, i0))
        write_ply(filename,
                  [np.vstack(all_points)],
                  ['x', 'y', 'z', 't'])
        filename = join(out_folder, 'd_{:s}_{:05d}_traj.ply'.format(day, i0))
        write_ply(filename,
                  [np.vstack(all_traj_pts), np.vstack(all_traj_clrs)],
                  ['x', 'y', 'z', 't', 'red', 'green', 'blue'])


def bundle_icp(frame_names,
               bundle_size=5,
               score_thresh=0.99,
               frame_voxel_size=-1,
               verbose=2):
    """
    Test ICP registration Use GT to extract a small interesting region.
    """

    ############
    # Parameters
    ############

    # Bundle stride (nb of frames between each bundle start)
    bundle_stride = bundle_size - 1

    # Group of frames saved together
    save_group = 100

    # List of transformation we are trying to optimize
    transform_list = [np.eye(4) for _ in frame_names]
    last_saved_frames = 0
    FPS = 0
    N = len(frame_names)

    for b_i, bundle_i0 in enumerate(np.arange(0, len(frame_names), bundle_stride)):

        ####################
        # Load bundle frames
        ####################

        t = [time.time()]

        if (bundle_i0 + bundle_size > N):
            bundle_i0 = N - bundle_size

        frame_pts = []
        frame_norms = []
        frame_w = []
        for f_name in frame_names[bundle_i0:bundle_i0+bundle_size]:

            # Load ply format points
            data = read_ply(f_name)
            points = np.vstack((data['x'], data['y'], data['z'])).T

            # Get normals
            normals, planarity, linearity = polar_normals(points, radius=1.5, h_scale=0.5)
            norm_scores = planarity + linearity

            # Remove low quality normals for fitting
            points = points[norm_scores > score_thresh]
            normals = normals[norm_scores > score_thresh]
            norm_scores = (norm_scores[norm_scores > score_thresh] - score_thresh) / (1 - score_thresh)

            # Subsample to reduce number of points
            if frame_voxel_size > 0:

                # grid supsampling
                points, normals = grid_subsampling(points, features=normals, sampleDl=frame_voxel_size)

                # Renormalize normals
                normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

            # Filter out points according to main normal directions (NOt necessary if normals are better computed)
            bool_filter = normal_filtering(normals) > 0.5
            points = points[bool_filter]
            normals = normals[bool_filter]
            norm_scores = norm_scores[bool_filter]

            # Compute score for each component of rotations / translation
            # Weights according to distance the futher, the higher (square rule because points lies on surfaces)
            #rot_scores = np.expand_dims(norm_scores, 1) * np.cross(points, normals, axis=1)
            #weights = np.hstack((rot_scores, -rot_scores))

            weights = np.expand_dims(norm_scores, 1)

            # Gather frames data
            frame_pts.append(points)
            frame_norms.append(normals)
            frame_w.append(weights)

        t += [time.time()]

        ##################
        # Apply bundle ICP
        ##################

        bundle_H, bundle_rms, all_H = bundle_pt2pl_icp(frame_pts,
                                                            frame_norms,
                                                            frame_w,
                                                            n_samples=1000,
                                                            max_pairing_dist=0.2,
                                                            max_iter=200,
                                                            avg_steps=5)


        t += [time.time()]

        # Update transformations to world coordinates
        for b in range(bundle_size):
            world_H = np.eye(4)
            for bb in range(b, 0, -1):
                world_H = np.matmul(bundle_H[bb], world_H)
            world_H = np.matmul(transform_list[bundle_i0], world_H)
            transform_list[bundle_i0 + b] = world_H

        t += [time.time()]

        if verbose > 0:
            fmt_str = 'Bundle [{:3d},{:3d}]  --- {:5.1f}% or {:02d}:{:02d}:{:02d} remaining at {:.1f}fps'
            if bundle_i0 == 0:
                FPS = bundle_size / (t[-1] - t[0])
            else:
                FPS += (bundle_size / (t[-1] - t[0]) - FPS) / 10
            remaining = int((N - (bundle_i0 + bundle_size)) / FPS)
            hours = remaining // 3600
            remaining = remaining - 3600 * hours
            minutes = remaining // 60
            seconds = remaining - 60 * minutes
            print(fmt_str.format(bundle_i0,
                                 bundle_i0 + bundle_size - 1,
                                 100 * (bundle_i0 + bundle_size) / N,
                                 hours, minutes, seconds,
                                 FPS))

        # Save groups of 100 frames together
        if (bundle_i0 > last_saved_frames + save_group + 1):
            all_points = []
            all_traj_pts = []
            all_traj_clrs = []
            i0 = last_saved_frames
            i1 = i0 + save_group
            for i, world_H in enumerate(transform_list[i0: i1]):
                # Load ply format points
                data = read_ply(frame_names[i0 + i])
                points = np.vstack((data['x'], data['y'], data['z'])).T

                # Apply transf
                world_pts = np.hstack((points, np.ones_like(points[:, :1])))
                world_pts = np.matmul(world_pts, world_H.T)

                # Save frame
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
    for i, world_H in enumerate(transform_list[i0:]):

        # Load ply format points
        data = read_ply(frame_names[i0 + i])
        points = np.vstack((data['x'], data['y'], data['z'])).T

        # Apply transf
        world_pts = np.hstack((points, np.ones_like(points[:, :1])))
        world_pts = np.matmul(world_pts, world_H.T)

        # Save frame
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

    return transform_list


def bundle_slam(verbose=1):

    ############
    # Parameters
    ############

    # Path to data
    data_path = '../../Data/NCLT'
    gt_folder = 'ground_truth'
    raw_folder = 'raw_ply'
    days = np.sort([d for d in listdir(join(data_path, raw_folder))])

    # Out files
    out_folder = join(data_path, 'day_ply')
    if not exists(out_folder):
        makedirs(out_folder)

    # Stride (nb of frames skipped for transformations)
    frame_stride = 2

    # Bundle size (number of frames jointly optimized) and stride (nb of frames between each bundle start)
    bundle_size = 7
    bundle_stride = bundle_size - 1

    # Normal estimation parameters
    score_thresh = 0.99

    # Pointcloud filtering parameters
    map_voxel_size = 0.05
    frame_voxel_size = 0.05

    # Group of frames saved together
    save_group = 100

    ###############
    # Load GT poses
    ###############

    print('\nLoading days groundtruth poses...')
    t0 = time.time()
    gt_t, gt_H = load_gt_poses(join(data_path, gt_folder), only_day_1=True)
    t2 = time.time()
    print('Done in {:.1f}s\n'.format(t2 - t0))

    #######################
    # Get lidar frame times
    #######################

    # Focus on a particular point
    p0 = np.array([-220, -527, 12])
    R0 = 20.0

    print('\nGet timestamps in focused area...')
    t0 = time.time()
    day_f_times = get_area_frames(days, gt_t, gt_H, join(data_path, raw_folder), p0, R0, only_day_1=True)
    t2 = time.time()
    print('Done in {:.1f}s\n'.format(t2 - t0))

    ###########################
    # coarse map with pt2pl icp
    ###########################

    for d, day in enumerate(days):

        # List of transformation we are trying to optimize
        frames_folder = join(data_path, 'raw_ply', day)
        f_times = [f_t for f_t in day_f_times[d][::frame_stride]]
        transform_list = [np.eye(4) for _ in f_times]
        last_saved_frames = 0
        FPS = 0
        N = len(f_times)

        for b_i, bundle_i0 in enumerate(np.arange(0, len(f_times), bundle_stride)):

            ####################
            # Load bundle frames
            ####################

            t = [time.time()]

            if (bundle_i0 + bundle_size > len(f_times)):
                bundle_i0 = len(f_times) - bundle_size

            frame_pts = []
            frame_norms = []
            frame_w = []
            for f_t in f_times[bundle_i0:bundle_i0+bundle_size]:

                # Load ply format points
                f_name = '{:.0f}.ply'.format(f_t)
                data = read_ply(join(frames_folder, f_name))
                points = np.vstack((data['x'], data['y'], data['z'])).T

                estimate_normals_planarity_debug(points)

                a = 1/0

                t += [time.time()]

                # Get normals
                normals, planarity, linearity = polar_normals(points, radius=1.5, h_scale=0.5)

                # Remove low quality normals for fitting
                points = points[norm_scores > score_thresh]
                normals = normals[norm_scores > score_thresh]
                norm_scores = (norm_scores[norm_scores > score_thresh] - score_thresh) / (1 - score_thresh)

                t += [time.time()]

                # Subsample to reduce number of points
                if frame_voxel_size > 0:

                    # grid supsampling
                    points, normals = grid_subsampling(points, features=normals, sampleDl=map_voxel_size)

                    # Renormalize normals
                    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

                # Filter out points according to main normal directions (NOt necessary if normals are better computed)
                bool_filter = normal_filtering(normals) > 0.5
                points = points[bool_filter]
                normals = normals[bool_filter]
                norm_scores = norm_scores[bool_filter]

                t += [time.time()]

                # Compute score for each component of rotations / translation
                # Weights according to distance the futher, the higher (square rule because points lies on surfaces)
                #rot_scores = np.expand_dims(norm_scores, 1) * np.cross(points, normals, axis=1)
                #weights = np.hstack((rot_scores, -rot_scores))

                weights = np.expand_dims(norm_scores, 1)

                # Gather frames data
                frame_pts.append(points)
                frame_norms.append(normals)
                frame_w.append(weights)

                t += [time.time()]

            if verbose == 3:
                dt = np.array(t[1:]) - np.array(t[:-1])
                dt = dt.reshape(bundle_size, -1)
                timing_names = ['Load', 'Normals', 'Filter', 'Append']
                s = ''
                for t_name in timing_names:
                    s += '{:^10s} '.format(t_name)
                s += '\n'
                for b in range(bundle_size):
                    for t_i in range(len(timing_names)):
                        s += '{:^10.1f} '.format(1000 * dt[b, t_i])
                    s += '\n'
                print(s)

            t = t[:1]
            t += [time.time()]

            ##################
            # Apply bundle ICP
            ##################

            # for b in range(bundle_size):
            #     w_names = ['w{:d}'.format(i) for i in range(frame_w[b].shape[1])]
            #     write_ply('bb_init_{:02d}.ply'.format(b),
            #               [frame_pts[b], frame_w[b]],
            #               ['x', 'y', 'z'] + w_names)

            bundle_H, bundle_rms, all_H = bundle_pt2pl_icp(frame_pts,
                                                                frame_norms,
                                                                frame_w,
                                                                n_samples=1000,
                                                                max_pairing_dist=0.2,
                                                                max_iter=200,
                                                                avg_steps=5)



    # TODO: Lidar scan cleanup. AFTER THE MAPPING
    #   > In polar coordinate: retrieve each line of scan. like 1D grid subs, index in a 1D grid, adjust grid
    #     by min max and nb of scan lines. r = log(r) pour la suite
    #   > In each line, order points by phi. find jumps in r direction. get dr0 = r(j)-(j-1) and dr1 = r(j+1)-r(j)
    #       IF dr0 = dr1 THEN we probably are on a plane keep the point.
    #       IF abs(dr0-dr1) > Thresh THEN outlier, remove the point
    #
    #

    # TODO: Mapping
    #   > Start with a frame to frame bundle adjustement (do 20 frames, between 5 and 10 meters))
    #   > Create map from these 20 frames (USe our smart spherical grid subs)
    #   > ICP on the map
    #   > Choose, update map or compute it again from 20 new frames (Better to update if possible)

    # TODO: motion distortion, use phi angle to get points timestamps, remembre frame stime stamp is the one of the
    #  last points => unperiodicize phi, create linear interp function with last points and their angle Interpolate
    #  pose based on points angle


def get_odometry(sensor_path, day, t0, t1):

    odom_name = join(sensor_path, day, 'odometry_mu_100hz.csv')

    odom = np.loadtxt(odom_name, delimiter=",", dtype=np.float64)

    mask = np.logical_and(odom[:, 0] > t0, odom[:, 0] < t1)
    ssc = odom[mask, 1:]
    t = odom[mask, 0]

    H = ssc_to_homo(ssc, ssc_in_radians=True)

    return t, H



