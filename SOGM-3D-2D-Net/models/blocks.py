#
#
#      0==============================0
#      |    Deep Collision Checker    |
#      0==============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network blocks
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


import time
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
from kernels.kernel_points import load_kernels, get_random_rotations, get_identity_lrfs

from utils.ply import write_ply

import warnings
warnings.filterwarnings("ignore", message="Using a target size ")


# ----------------------------------------------------------------------------------------------------------------------
#
#           Simple functions
#       \**********************/
#


def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))


def closest_pool(x, inds):
    """
    Pools features from the closest neighbors. WARNING: this function assumes the neighbors are ordered.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] Only the first column is used for pooling
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get features for each pooling location [n2, d]
    return gather(x, inds[..., 0])


def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features, _ = torch.max(pool_features, dim=-2)
    return max_features


def avg_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [X, X, ..., X, max_num] pooling indices
    :return: [X, X, ..., X, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [X, X, ..., X, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [X, X, ..., X, d]
    return torch.mean(pool_features, dim=-2)


def global_average(x, batch_lengths):
    """
    Block performing a global average over batch pooling
    :param x: [N, D] input features
    :param batch_lengths: [B] list of batch lengths
    :return: [B, D] averaged features
    """

    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):

        # Average features for each batch cloud
        averaged_features.append(torch.mean(x[i0:i0 + length], dim=0))

        # Increment for next cloud
        i0 += length

    # Average features in each batch
    return torch.stack(averaged_features)


def global_PCA_alignment(stacked_points, batch_lengths):

    i0 = 0
    covariances = []
    centroids = []
    for b_i, length in enumerate(batch_lengths):

        # Extract point cloud from batch
        points = stacked_points[i0:i0 + length]

        # Center points on centroid [n_points, dim]
        centroid = torch.mean(points, dim=0, keepdim=True)
        points0 = points - centroid

        # Get covariance matrix [dim, n_points] x [n_points, dim]
        covariances.append(torch.matmul(points0.t(), points0))
        centroids.append(centroid)

        # Increment for next cloud
        i0 += length

    covariances = torch.stack(covariances, dim=0).cpu()
    centroids = torch.cat(centroids, dim=0)

    # Get eigenvalues and eigenvectors of covariances matrices [n_points, dim] and [n_points, dim, dim]
    eigen_values, eigen_vectors = torch.symeig(covariances, eigenvectors=True)
    eigen_vectors = eigen_vectors.to(stacked_points.device)

    # Correct rotations with centroids
    rotated_centroids = torch.matmul(centroids.unsqueeze(1), eigen_vectors)
    corrections = (rotated_centroids < 0).type(torch.float32) * 2 - 1

    # Get rotation matrices. if X is neighbors, then R.T x X is the aligned neighbors [n_points, dim, dim]
    rotations = eigen_vectors * corrections

    # Correct the orientation to avoid symmetries
    rotations[:, :, 2] = torch.cross(rotations[:, :, 0], rotations[:, :, 1])

    return rotations


def align_lrf_global(stacked_lrf, R, batch_lengths):

    i0 = 0
    aligned_lrf = []
    for b_i, length in enumerate(batch_lengths):

        # Extract lrf from batch
        lrf = stacked_lrf[i0:i0 + length]

        # Align it with rotation: aligned_lrf = R.T x lrf
        aligned_lrf.append(torch.matmul(lrf.transpose(-2, -1), R[b_i]).transpose(-2, -1))

        # Increment for next cloud
        i0 += length

    return torch.cat(aligned_lrf, dim=0)


# ----------------------------------------------------------------------------------------------------------------------
#
#           KPConv class
#       \******************/
#


class KPConv(nn.Module):

    def __init__(self, kernel_size, p_dim, in_channels, out_channels, KP_extent, radius,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum',
                 deformable=False, modulated=False):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param deformable: choose deformable or not
        :param modulated: choose if kernel weights are modulated in addition to deformed
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated
        self.diff_op = torch.nn.MSELoss(reduction='none')

        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # Initiate weights for offsets
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(self.K,
                                      self.p_dim,
                                      self.in_channels,
                                      self.offset_dim,
                                      KP_extent,
                                      radius,
                                      fixed_kernel_points=fixed_kernel_points,
                                      KP_influence=KP_influence,
                                      aggregation_mode=aggregation_mode)
            self.offset_bias = Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)

        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, q_pts, s_pts, neighb_inds, x):

        ###################
        # Offset generation
        ###################

        if self.deformable:

            # Get offsets with a KPConv that only takes part of the features
            self.offset_features = self.offset_conv(q_pts, s_pts, neighb_inds, x) + self.offset_bias

            if self.modulated:

                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features[:, :self.p_dim * self.K]
                unscaled_offsets = unscaled_offsets.view(-1, self.K, self.p_dim)

                # Get modulations
                modulations = 2 * torch.sigmoid(self.offset_features[:, self.p_dim * self.K:])

            else:

                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features.view(-1, self.K, self.p_dim)

                # No modulations
                modulations = None

            # Rescale offset for this layer
            offsets = unscaled_offsets * self.KP_extent

        else:
            offsets = None
            modulations = None

        ######################
        # Deformed convolution
        ######################

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points
            deformed_K_points = self.deformed_KP.unsqueeze(1)
        else:
            deformed_K_points = self.kernel_points

        # Get the square distances [n_points, n_neighbors, n_kpoints]
        neighbors.unsqueeze_(2)
        sq_distances = torch.sum(self.diff_op(neighbors, deformed_K_points), dim=-1)

        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:

            # Save distances for loss
            self.min_d2, _ = torch.min(sq_distances, dim=1)

            # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
            in_range = torch.any(sq_distances < self.KP_extent ** 2, dim=2).type(torch.int32)

            # New value of max neighbors
            new_max_neighb = torch.max(torch.sum(in_range, dim=1))

            # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
            neighb_row_bool, neighb_row_inds = torch.topk(in_range, new_max_neighb.item(), dim=1)

            # Gather new neighbor indices [n_points, new_max_neighb]
            new_neighb_inds = neighb_inds.gather(1, neighb_row_inds, sparse_grad=False)

            # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
            neighb_row_inds.unsqueeze_(2)
            neighb_row_inds = neighb_row_inds.expand(-1, -1, self.K)
            sq_distances = sq_distances.gather(1, neighb_row_inds, sparse_grad=False)

            # New shadow neighbors have to point to the last shadow point
            new_neighb_inds *= neighb_row_bool
            new_neighb_inds -= (neighb_row_bool.type(torch.int64) - 1) * int(s_pts.shape[0] - 1)
        else:
            new_neighb_inds = neighb_inds

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 1, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, new_neighb_inds)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = torch.matmul(all_weights, neighb_x)

        # Apply modulations
        if self.deformable and self.modulated:
            weighted_features *= modulations.unsqueeze(2)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute((1, 0, 2))
        kernel_outputs = torch.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        return torch.sum(kernel_outputs, dim=0)

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)



class EKPConv_v1(nn.Module):

    def __init__(self, kernel_size, p_dim, in_channels, out_channels, n_LRF, KP_extent, radius,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum'):
        """
        Initialize parameters for Equivaraint KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param n_LRF: Number of local alignment reference frames (rotation matrices).
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        """
        super(EKPConv_v1, self).__init__()

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.n_LRF = n_LRF
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.diff_op = torch.nn.MSELoss(reduction='none')

        # Number of feature per lrf
        self.lrf_channels = in_channels // n_LRF
        if in_channels % n_LRF != 0:
            raise ValueError('Input feature dimension of an equivariant convolution '
                             'is not divisible by the number of lrf')

        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, q_pts, s_pts, neighb_inds, x, lrf):

        ######################
        # Deformed convolution
        ######################

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)

        # Align all neighbors [n_points, 1, n_neighbors, dim] x [n_points, n_LRF, dim, dim]
        # Beware of transpose operations. Each point is multiplied like this:
        #
        #   aligned_X = lrf.T * X
        #
        neighbors.unsqueeze_(1)
        #aligned_neighbors = torch.matmul(neighbors, lrf)
        aligned_neighbors = torch.matmul(neighbors, lrf.detach())

        # Get the square distances [n_points, n_LRF, n_neighbors, n_kpoints]
        aligned_neighbors.unsqueeze_(3)
        sq_distances = torch.sum(self.diff_op(aligned_neighbors, self.kernel_points), dim=-1)

        # Get Kernel point influences [n_points, n_LRF, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 2, 3)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 2, 3)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 2, 3)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=3)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 2, 3)

        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Reshape features for each specific lrf form [n_points, in_fdim] to [n_points, n_LRF, lrf_fdim]
        x = x.reshape((-1, self.n_LRF, self.lrf_channels))

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, n_LRF, lrf_fdim]
        neighb_x = gather(x, neighb_inds)

        # Apply distance weights [n_points, n_LRF, n_kpoints, lrf_fdim]
        neighb_x.transpose_(1, 2)
        weighted_features = torch.matmul(all_weights, neighb_x)

        # [n_points, n_LRF, n_kpoints, lrf_fdim] => [n_kpoints, n_points, n_LRF, lrf_fdim]
        weighted_features = weighted_features.permute((2, 0, 1, 3))

        # [n_kpoints, n_points, n_LRF, lrf_fdim] => [n_kpoints, n_points, n_LRF * lrf_fdim]
        weighted_features = weighted_features.reshape((self.K, -1, self.in_channels))

        # Apply network weights [n_kpoints, n_points, out_fdim]
        kernel_outputs = torch.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        return torch.sum(kernel_outputs, dim=0)

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)


class EKPConv_v2(nn.Module):

    def __init__(self, kernel_size, p_dim, in_channels, out_channels, ns_LRF, nq_LRF, KP_extent, radius,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum', all_detached=False):
        """
        Initialize parameters for Equivaraint KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param n_LRF: Number of local alignment reference frames (rotation matrices).
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        """
        super(EKPConv_v2, self).__init__()

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.ns_LRF = ns_LRF
        self.nq_LRF = nq_LRF
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.all_detached = all_detached
        self.diff_op = torch.nn.MSELoss(reduction='none')

        # Number of feature per lrf
        if in_channels > 4:
            self.lrf_channels = in_channels // nq_LRF
            fdim = 2 * self.in_channels
            if in_channels % nq_LRF != 0:
                raise ValueError('Input feature dimension of an equivariant convolution '
                                 'is not divisible by the number of lrf')
        else:
            self.lrf_channels = out_channels // nq_LRF
            fdim = (self.in_channels + self.lrf_channels) * self.nq_LRF

        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, fdim, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # MLP to transform aligned lrf into features
        self.lrf_mlp = nn.Linear(ns_LRF * (p_dim ** 2), self.lrf_channels, bias=True)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, q_pts, s_pts, neighb_inds, x, q_lrf, s_lrf):

        ##################
        # Points alignment
        ##################

        # Parameter
        n_neighb = int(neighb_inds.shape[1])

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)

        # Align all neighbors [n_points, 1, n_neighbors, dim] x [n_points, nq_LRF, dim, dim]
        # Beware of transpose operations. Each point is multiplied like this:
        #
        #   aligned_X = lrf.T * X
        #
        neighbors.unsqueeze_(1)
        if self.all_detached:
            aligned_neighbors = torch.matmul(neighbors, q_lrf.detach())
        else:
            #aligned_neighbors = torch.matmul(neighbors, q_lrf)
            aligned_neighbors = torch.matmul(neighbors, q_lrf.detach())

        # Get the square distances [n_points, nq_LRF, n_neighbors, n_kpoints]
        aligned_neighbors.unsqueeze_(3)
        sq_distances = torch.sum(self.diff_op(aligned_neighbors, self.kernel_points), dim=-1)

        ###############
        # lrf alignment
        ###############

        # Add a fake lrf in the last row for shadow neighbors
        if self.all_detached:
            s_lrf = s_lrf.detach()
        s_lrf = torch.cat((s_lrf, torch.zeros_like(s_lrf[:1])), 0)


        # Get neighbor points [n_points, n_neighbors, ns_LRF, dim, dim]
        neighbors_lrf = gather(s_lrf, neighb_inds)

        # Align all lrf [n_points, 1, nq_LRF, 1, dim, dim] x [n_points, n_neighbors, 1, ns_LRF, dim, dim]
        # Beware of transpose operations. Each rotation is multiplied like this:
        #
        #   aligned_R = lrf.T * R
        #
        neighbors_lrf.unsqueeze_(2)
        tmp_lrf = q_lrf.detach().unsqueeze(1).unsqueeze(3)

        # Matmul is too slow. Use naive mul+sum
        x_ = tmp_lrf.transpose(-2, -1).unsqueeze(-1)
        y_ = neighbors_lrf.unsqueeze(-3)
        aligned_lrf = (x_ * y_).sum(-2)

        # Convert aligned lrf to features
        # [n_points, n_neighbors, nq_LRF, ns_LRF, dim, dim] => [n_points, n_neighbors, nq_LRF, lrf_fdim]
        aligned_lrf = aligned_lrf.reshape((-1, n_neighb, self.nq_LRF, self.ns_LRF * (self.p_dim ** 2)))
        lrf_features = self.lrf_mlp(aligned_lrf)

        # Reapply neighbors mask (because of bias in MLP)
        neighb_mask = (neighb_inds < (int(s_pts.shape[0]) - 1)).type(torch.float32)
        lrf_features = lrf_features * neighb_mask.unsqueeze(-1).unsqueeze(-1)

        ##################################
        # kernel point correlation weights
        ##################################

        # Get Kernel point influences [n_points, nq_LRF, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 2, 3)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 2, 3)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 2, 3)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=3)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 2, 3)

        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        ######################
        # Features combination
        ######################

        # Reshape features for each specific lrf form [n_points, in_fdim] to [n_points, nq_LRF, lrf_fdim]
        if self.in_channels > 4:
            x = x.reshape((-1, self.nq_LRF, self.lrf_channels))
        else:
            x = x.reshape((-1, 1, self.in_channels))
            x = x.expand((-1, self.nq_LRF, -1))

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, nq_LRF, lrf_fdim]
        neighb_x = gather(x, neighb_inds)

        # Concat with the features from aligned lrf [n_points, n_neighbors, nq_LRF, 2lrf_fdim]
        neighb_x = torch.cat((neighb_x, lrf_features), dim=-1)

        # Apply distance weights [n_points, nq_LRF, n_kpoints, 2lrf_fdim]
        neighb_x.transpose_(1, 2)
        weighted_features = torch.matmul(all_weights, neighb_x)

        # [n_points, nq_LRF, n_kpoints, 2lrf_fdim] => [n_kpoints, n_points, nq_LRF, 2lrf_fdim]
        weighted_features = weighted_features.permute((2, 0, 1, 3))

        # [n_kpoints, n_points, nq_LRF, 2lrf_fdim] => [n_kpoints, n_points, 2 * nq_LRF * lrf_fdim]
        weighted_features = weighted_features.reshape((self.K, -1, self.weights.shape[1]))

        # Apply network weights [n_kpoints, n_points, out_fdim]
        kernel_outputs = torch.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        return torch.sum(kernel_outputs, dim=0)

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)


class EKPConv_v3(nn.Module):

    def __init__(self, kernel_size, p_dim, in_channels, out_channels, ns_LRF, nq_LRF, KP_extent, radius,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum'):
        """
        Initialize parameters for Equivaraint KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param n_LRF: Number of local alignment reference frames (rotation matrices).
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        """
        super(EKPConv_v3, self).__init__()

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.ns_LRF = ns_LRF
        self.nq_LRF = nq_LRF
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.diff_op = torch.nn.MSELoss(reduction='none')

        # Number of feature per lrf
        self.lrf_channels = in_channels // nq_LRF
        if in_channels % nq_LRF != 0:
            raise ValueError('Input feature dimension of an equivariant convolution '
                             'is not divisible by the number of lrf')

        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, 2 * in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # MLP to transform aligned lrf into features
        self.lrf_mlp = nn.Linear(ns_LRF * (p_dim ** 2), self.lrf_channels, bias=True)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, q_pts, s_pts, neighb_inds, x, q_lrf, s_lrf):

        ##################
        # Points alignment
        ##################

        # Parameter
        n_neighb = int(neighb_inds.shape[1])

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)

        # Align all neighbors [n_points, 1, n_neighbors, dim] x [n_points, nq_LRF, dim, dim]
        # Beware of transpose operations. Each point is multiplied like this:
        #
        #   aligned_X = lrf.T * X
        #
        neighbors.unsqueeze_(1)
        #aligned_neighbors = torch.matmul(neighbors, q_lrf)
        aligned_neighbors = torch.matmul(neighbors, q_lrf.detach())

        # Get the square distances [n_points, nq_LRF, n_neighbors, n_kpoints]
        aligned_neighbors.unsqueeze_(3)
        sq_distances = torch.sum(self.diff_op(aligned_neighbors, self.kernel_points), dim=-1)

        ###############
        # lrf alignment
        ###############

        # Add a fake lrf in the last row for shadow neighbors
        s_lrf = torch.cat((s_lrf, torch.zeros_like(s_lrf[:1])), 0)

        # Get neighbor points [n_points, n_neighbors, ns_LRF, dim, dim]
        neighbors_lrf = gather(s_lrf, neighb_inds)

        # Align all lrf [n_points, 1, nq_LRF, 1, dim, dim] x [n_points, n_neighbors, 1, ns_LRF, dim, dim]
        # Beware of transpose operations. Each rotation is multiplied like this:
        #
        #   aligned_R = lrf.T * R
        #
        neighbors_lrf.unsqueeze_(2)
        tmp_lrf = q_lrf.detach().unsqueeze(1).unsqueeze(3)

        # Matmul is too slow. Use naive mul+sum
        x_ = tmp_lrf.transpose(-2, -1).unsqueeze(-1)
        y_ = neighbors_lrf.unsqueeze(-3)
        aligned_lrf = (x_ * y_).sum(-2)

        # Convert aligned lrf to features
        # [n_points, n_neighbors, nq_LRF, ns_LRF, dim, dim] => [n_points, n_neighbors, nq_LRF, lrf_fdim]
        aligned_lrf = aligned_lrf.reshape((-1, n_neighb, self.nq_LRF, self.nq_LRF * (self.p_dim ** 2)))
        lrf_features = self.lrf_mlp(aligned_lrf)

        # Reapply neighbors mask (because of bias in MLP)
        neighb_mask = (neighb_inds < (int(s_pts.shape[0]) - 1)).type(torch.float32)
        lrf_features = lrf_features * neighb_mask.unsqueeze(-1).unsqueeze(-1)

        ##################################
        # kernel point correlation weights
        ##################################

        # Get Kernel point influences [n_points, nq_LRF, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 2, 3)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 2, 3)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 2, 3)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=3)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 2, 3)

        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        ######################
        # Features combination
        ######################

        # Reshape features for each specific lrf form [n_points, in_fdim] to [n_points, nq_LRF, lrf_fdim]
        x = x.reshape((-1, self.nq_LRF, self.lrf_channels))

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, nq_LRF, lrf_fdim]
        neighb_x = gather(x, neighb_inds)

        # Concat with the features from aligned lrf [n_points, n_neighbors, nq_LRF, 2lrf_fdim]
        neighb_x = torch.cat((neighb_x, lrf_features), dim=-1)

        # Apply distance weights [n_points, nq_LRF, n_kpoints, 2lrf_fdim]
        neighb_x.transpose_(1, 2)
        weighted_features = torch.matmul(all_weights, neighb_x)

        # [n_points, nq_LRF, n_kpoints, 2lrf_fdim] => [n_kpoints, n_points, nq_LRF, 2lrf_fdim]
        weighted_features = weighted_features.permute((2, 0, 1, 3))

        # [n_kpoints, n_points, nq_LRF, 2lrf_fdim] => [n_kpoints, n_points, 2 * nq_LRF * lrf_fdim]
        weighted_features = weighted_features.reshape((self.K, -1, 2 * self.in_channels))

        # Apply network weights [n_kpoints, n_points, out_fdim]
        kernel_outputs = torch.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        return torch.sum(kernel_outputs, dim=0)

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Complex blocks
#       \********************/
#

def block_decider(block_name,
                  radius,
                  in_dim,
                  out_dim,
                  layer_ind,
                  config,
                  n_lrf=0,
                  up_lrf=1):

    if block_name == 'unary':
        return UnaryBlock(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum)

    elif block_name in ['simple',
                        'simple_deformable',
                        'simple_invariant',
                        'simple_equivariant',
                        'simple_strided',
                        'simple_deformable_strided',
                        'simple_invariant_strided',
                        'simple_equivariant_strided']:
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config, n_lrf, up_lrf)

    elif block_name in ['resnetb',
                        'resnetb_invariant',
                        'resnetb_equivariant',
                        'resnetb_deformable',
                        'resnetb_strided',
                        'resnetb_deformable_strided',
                        'resnetb_equivariant_strided',
                        'resnetb_invariant_strided']:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config, n_lrf, up_lrf)

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)

    elif block_name in ['global_average',
                        'global_average_equivariant']:
        return GlobalAverageBlock(block_name, n_lrf, out_dim, config)

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)


class BatchNormBlock(nn.Module):

    def __init__(self, in_dim, use_bn, bn_momentum):
        """
        Initialize a batch normalization block. If network does not use batch normalization, replace with biases.
        :param in_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        self.batch_norm = nn.BatchNorm1d(in_dim, momentum=bn_momentum)
        #self.batch_norm = nn.InstanceNorm1d(in_dim, momentum=bn_momentum)
        self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.use_bn:

            x = x.unsqueeze(2)
            x = x.transpose(0, 2)
            x = self.batch_norm(x)
            x = x.transpose(0, 2)
            return x.squeeze()
        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlock(in_feat: {:d}, momentum: {:.3f}, only_bias: {:s}, training: {})'.format(self.in_dim,
                                                                                                       self.bn_momentum,
                                                                                                       str(not self.use_bn),
                                                                                                       self.training)


class UnaryBlock(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """

        super(UnaryBlock, self).__init__()
        self.block_name = 'unary'
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        return

    def forward(self, x, batch=None):
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(self.in_dim,
                                                                                        self.out_dim,
                                                                                        str(self.use_bn),
                                                                                        str(not self.no_relu))


class LRFBlock(nn.Module):

    def __init__(self, in_dim, in_n_LRF, up_factor, p_dim=3):
        """
        Initialize a local reference frame predictor block.
        :param in_dim:
        :param in_n_LRF:
        :param up_factor: We can multiply the number of lrf by this factor to have more than we had first
        """

        super(LRFBlock, self).__init__()
        self.in_dim = in_dim
        self.in_n_LRF = in_n_LRF
        self.up_factor = up_factor
        self.p_dim = p_dim
        self.out_dim = in_n_LRF * up_factor * (p_dim ** 2)
        #self.mlp = nn.Linear(in_dim, self.out_dim, bias=False)
        self.weights = Parameter(torch.zeros([in_dim, self.out_dim], dtype=torch.float32),
                                 requires_grad=True)
        self.bias = Parameter(torch.zeros([up_factor, in_n_LRF, p_dim, p_dim], dtype=torch.float32),
                              requires_grad=True)

        # Reset parameters
        self.reset_parameters()

        # Running variable with the predicted lrf to apply rotations
        self.pred_rots = None

        return

    def reset_parameters(self):

        # Reset values
        #nn.init.zeros_(self.bias)
        nn.init.uniform_(self.bias, a=-1.0, b=1.0)
        nn.init.uniform_(self.weights, a=-1e-3, b=1e-3)
        #nn.init.uniform_(self.weights, a=-1e-6, b=1e-6)

        # Init biases from a random rotation so that we start for a good point with the ortho loss
        with torch.no_grad():
            self.bias *= 0
            self.bias += torch.from_numpy(get_random_rotations(tuple(self.bias.shape[:-2])))
            #self.bias += torch.from_numpy(get_identity_lrfs(tuple(self.bias.shape[:-2])))

        return

    def forward(self, x, lrf):

        # Predict new lrf (actually the rotation applied to old lrf so that we keep equivariance)
        #new_lrf = self.mlp(x)
        new_lrf = torch.matmul(x, self.weights)
        new_lrf = new_lrf.reshape((-1, self.up_factor, self.in_n_LRF, self.p_dim, self.p_dim))

        # To predict a rotation matrix start from identity
        self.pred_rots = new_lrf + self.bias  # + torch.eye(self.p_dim, dtype=torch.float32, device=new_lrf.device)

        # Reshape old lrf for the up_factor [n_points, in_n_LRF, dim, dim]
        lrf = lrf.detach().unsqueeze(1)

        # Realign new lrf with old lrf [n_points, up_factor, in_n_LRF, dim, dim]
        # Here we want each new lrf called X to be multiplied like this
        #
        #   new_lrf = lrf * pred_rot
        #

        new_lrf = torch.matmul(lrf, self.pred_rots)

        # Reshape new lrf [n_points, in_n_LRF * up_factor, dim, dim]
        new_lrf = new_lrf.reshape((-1, self.up_factor * self.in_n_LRF, self.p_dim, self.p_dim))

        return new_lrf

    def __repr__(self):
        return 'LRFBlock(in_feat: {:d}, in_n_LRF: {:d}, up_factor: {:d})'.format(self.in_dim,
                                                                                 self.in_n_LRF,
                                                                                 self.up_factor)


class SimpleBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config, n_lrf, up_lrf):
        """
        Initialize a simple convolution block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(SimpleBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_LRF = n_lrf
        self.lrf_up_factor = up_lrf

        if 'equivariant' in block_name:

            # Define the EKPConv class
            self.KPConv = EKPConv_v2(config.num_kernel_points,
                                     config.in_points_dim,
                                     in_dim,
                                     out_dim // 2,
                                     self.n_LRF,
                                     self.n_LRF,
                                     current_extent,
                                     radius,
                                     fixed_kernel_points=config.fixed_kernel_points,
                                     KP_influence=config.KP_influence,
                                     aggregation_mode=config.aggregation_mode)

            # Define the lrf prediction block
            self.lrf_block = LRFBlock(out_dim // 2,
                                      n_lrf,
                                      up_lrf,
                                      p_dim=config.in_points_dim)

        else:

            # Define the normal KPConv class
            self.KPConv = KPConv(config.num_kernel_points,
                                 config.in_points_dim,
                                 in_dim,
                                 out_dim // 2,
                                 current_extent,
                                 radius,
                                 fixed_kernel_points=config.fixed_kernel_points,
                                 KP_influence=config.KP_influence,
                                 aggregation_mode=config.aggregation_mode,
                                 deformable='deform' in block_name,
                                 modulated=config.modulated)

        # Other opperations
        self.batch_norm = BatchNormBlock(out_dim // 2, self.use_bn, self.bn_momentum)
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, x, batch, lrf=None):

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
            if self.n_LRF > 0:
                q_lrf = gather(lrf, neighb_inds[:, 0])
            else:
                q_lrf = None

        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]
            q_lrf = lrf

        if self.n_LRF > 0:
            x = self.KPConv(q_pts, s_pts, neighb_inds, x, q_lrf, lrf)
            x = self.leaky_relu(self.batch_norm(x))
            new_lrf = self.lrf_block(x, q_lrf)
            return x, new_lrf
        else:
            x = self.KPConv(q_pts, s_pts, neighb_inds, x)
            return self.leaky_relu(self.batch_norm(x))


class ResnetBottleneckBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config, n_lrf, up_lrf):
        """
        Initialize a resnet bottleneck block.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(ResnetBottleneckBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_LRF = n_lrf
        self.lrf_up_factor = up_lrf

        # First downscaling mlp
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlock(in_dim, out_dim // 4, self.use_bn, self.bn_momentum)
        else:
            self.unary1 = nn.Identity()

        if 'equivariant' in block_name:

            # Define the EKPConv class
            self.KPConv = EKPConv_v2(config.num_kernel_points,
                                     config.in_points_dim,
                                     out_dim // 4,
                                     out_dim // 4,
                                     self.n_LRF,
                                     self.n_LRF,
                                     current_extent,
                                     radius,
                                     fixed_kernel_points=config.fixed_kernel_points,
                                     KP_influence=config.KP_influence,
                                     aggregation_mode=config.aggregation_mode)

            # Define the lrf prediction block
            self.lrf_block = LRFBlock(out_dim,
                                      n_lrf,
                                      up_lrf,
                                      p_dim=config.in_points_dim)

        else:
            # KPConv block
            self.KPConv = KPConv(config.num_kernel_points,
                                 config.in_points_dim,
                                 out_dim // 4,
                                 out_dim // 4,
                                 current_extent,
                                 radius,
                                 fixed_kernel_points=config.fixed_kernel_points,
                                 KP_influence=config.KP_influence,
                                 aggregation_mode=config.aggregation_mode,
                                 deformable='deform' in block_name,
                                 modulated=config.modulated)
        self.batch_norm_conv = BatchNormBlock(out_dim // 4, self.use_bn, self.bn_momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(out_dim // 4, out_dim, self.use_bn, self.bn_momentum, no_relu=True)

        # Shortcut optional mpl
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, features, batch, lrf=None):

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
            if self.n_LRF > 0:
                q_lrf = gather(lrf, neighb_inds[:, 0])
            else:
                q_lrf = None
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]
            q_lrf = lrf

        # First downscaling mlp
        x = self.unary1(features)

        # Convolution
        if self.n_LRF > 0:
            x = self.KPConv(q_pts, s_pts, neighb_inds, x, q_lrf, lrf)
            x = self.leaky_relu(self.batch_norm_conv(x))
        else:
            x = self.KPConv(q_pts, s_pts, neighb_inds, x)
            x = self.leaky_relu(self.batch_norm_conv(x))

        # Second upscaling mlp
        x = self.unary2(x)

        # Shortcut
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighb_inds)
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut)

        # Realign lrf
        if self.n_LRF > 0:
            x = self.leaky_relu(x + shortcut)
            new_lrf = self.lrf_block(x, q_lrf)
            return x, new_lrf
        else:
            return self.leaky_relu(x + shortcut)


class GlobalAverageBlock(nn.Module):

    def __init__(self, block_name, n_lrf, out_dim, config):
        """
        Initialize a global average block with its ReLU and BatchNorm.
        """
        super(GlobalAverageBlock, self).__init__()
        self.n_LRF = n_lrf
        self.block_name = block_name
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.p_dim = config.in_points_dim

        if 'equivariant' in block_name:

            self.lrf_mlp = UnaryBlock(n_lrf * (self.p_dim ** 2), out_dim, self.use_bn, self.bn_momentum)

        return

    def forward(self, x, batch, lrf=None):

        if self.n_LRF > 0:

            # First realign all lrf so that output is invariant global_PCA_alignment
            rotations = global_PCA_alignment(batch.points[0], batch.lengths[0])
            aligned_lrf = align_lrf_global(lrf, rotations, batch.lengths[-1])

            # Convert to features and merge with network features
            lrf_features = self.lrf_mlp(aligned_lrf.reshape((-1, self.n_LRF * (self.p_dim ** 2))))
            x = torch.cat([x, lrf_features], dim=1)
            return global_average(x, batch.lengths[-1]), None

        else:
            return global_average(x, batch.lengths[-1])


class NearestUpsampleBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind
        self.block_name = 'nearest_upsample'
        return

    def forward(self, x, batch):
        return closest_pool(x, batch.upsamples[self.layer_ind - 1])

    def __repr__(self):
        return 'NearestUpsampleBlock(layer: {:d} -> {:d})'.format(self.layer_ind,
                                                                  self.layer_ind - 1)


class MaxPoolBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a max pooling block with its ReLU and BatchNorm.
        """
        super(MaxPoolBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return max_pool(x, batch.pools[self.layer_ind + 1])


class ProjectorBlock(nn.Module):

    def __init__(self, detached, pooling='avg'):
        """
        Initialize a max pooling block with its ReLU and BatchNorm.
        """
        super(ProjectorBlock, self).__init__()
        self.detached = detached
        self.pooling = pooling
        return

    def forward(self, x, batch):

        if self.pooling == 'avg':
            pooling_fn = avg_pool
        elif self.pooling == 'max':
            pooling_fn = max_pool
        elif self.pooling == 'closest':
            pooling_fn = closest_pool
        else:
            raise ValueError('Unknown pooling function name in ProjectorBlock: ' + self.pooling)

        # Get flat pooling indices or gradient is bugged
        inds = batch.pools_2D.detach()
        B = int(inds.shape[0])
        L = int(inds.shape[1])
        inds_flat = torch.reshape(inds, (B*L*L, -1))        

        # Get pooled features with shape [B*L*L, D]
        if self.detached:
            x_2D = pooling_fn(x.detach(), inds_flat)
        else:
            x_2D = pooling_fn(x, inds_flat)

        # Reshape into [B, L, L, D]
        x_2D = torch.reshape(x_2D, (B, L, L, -1))


        # import numpy as np
        # import matplotlib.pyplot as plt
        # imgs = x_2D.detach().cpu().numpy()
        # for i, img in enumerate(imgs):
        #     img = np.sum(np.abs(img), axis=-1)
        #     plt.subplots()
        #     imgplot = plt.imshow(img)
        #     print(batch.future_2D.shape)
        #     gt_im = batch.future_2D.detach().cpu().numpy()[i, self.n_frames - 1, :, :]
        #     plt.subplots()
        #     imgplot = plt.imshow(gt_im)
        #     plt.show()



        # Permute into [B, D, L, L]
        return x_2D.permute(0, 3, 1, 2)


class Unary2DBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Unary2DBlock, self).__init__()

        # Get other parameters
        self.in_dim = in_dim
        self.out_dim = out_dim

        # First downscaling mlp
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, x):
        return self.leaky_relu(self.bn1(self.conv1(x)))
        
    def __repr__(self):
        return 'Unary2DBlock({:d}, {:d})'.format(self.in_dim, self.out_dim)


class Resnet2DBlock(nn.Module):

    def __init__(self, in_dim, out_dim, stride=1):
        super(Resnet2DBlock, self).__init__()

        # Get other parameters
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride


        # First downscaling mlp
        self.conv1 = nn.Conv2d(in_dim, out_dim // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim // 4)
        
        # Conv block
        self.conv2 = nn.Conv2d(out_dim // 4, out_dim // 4, kernel_size=3, bias=False, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_dim // 4)
        
        # Second upscaling mlp
        self.conv3 = nn.Conv2d(out_dim // 4, out_dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim)

        # Shortcut optional mpl
        self.shortcut = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_dim))

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return


    def forward(self, x):
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.leaky_relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.leaky_relu(out)
        return out


    def __repr__(self):
        return 'Resnet2DBlock({:d}, {:d}, k=3, stride={:d})'.format(self.in_dim, self.out_dim, self.stride)


class ResNet2DLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_dim, out_dim, downsampling=False, n=1):
        super(ResNet2DLayer, self).__init__()

        if downsampling:
            stride0 = 2
        else:
            stride0 = 1

        self.blocks = nn.Sequential(
            Resnet2DBlock(in_dim, out_dim, stride=stride0),
            *[Resnet2DBlock(out_dim, out_dim, stride=1) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class Initial2DBlock(nn.Module):

    def __init__(self, in_dim, out_dim, levels=3, resnet_per_level=3):
        super(Initial2DBlock, self).__init__()

        # Get other parameters
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.levels = levels

        # Define encoder
        self.resnet_layers = nn.ModuleList()
        current_dim = out_dim
        self.resnet_layers.append(ResNet2DLayer(in_dim, current_dim, downsampling=False, n=resnet_per_level))
        for _ in range(levels - 1):
            self.resnet_layers.append(ResNet2DLayer(current_dim, current_dim*2, downsampling=True, n=resnet_per_level))
            current_dim *= 2

        # Define decoder
        self.up_layers = nn.ModuleList()
        for _ in range(levels - 1):
            current_dim = current_dim // 2
            self.up_layers.append(Unary2DBlock(current_dim + current_dim*2, current_dim))

        return


    def forward(self, x):

        # Encoder 128 -> 128 -> 256 -> 512  (50, 50, 25, 12)
        skips = []
        out_sizes = []
        for resnet_layer in self.resnet_layers:
            x = resnet_layer(x)
            skips.append(x)
            out_sizes.append(x[0, 0, :, :].shape)

        # Decoder
        out_sizes.pop()
        skips.pop()
        for up_layer in self.up_layers:
            x = nn.functional.interpolate(x, size=out_sizes.pop(), mode='bilinear', align_corners=True)
            x = torch.cat([x, skips.pop()], dim=1)
            x = up_layer(x)

        return x


class Propagation2DBlock(nn.Module):


    def __init__(self, in_dim, out_dim, stride=1, n_blocks=2):
        """
        Initialize a resnet bottleneck block.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(Propagation2DBlock, self).__init__()

        # Get other parameters
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Perform a few resnet convolution
        self.resnet_convs = nn.ModuleList()
        self.resnet_convs.append(Resnet2DBlock(in_dim, out_dim))
        for i in range(n_blocks - 1):
            self.resnet_convs.append(Resnet2DBlock(out_dim, out_dim))

        return


    def forward(self, x):
        for resnet_convs in self.resnet_convs:
            x = resnet_convs(x)
        return x





