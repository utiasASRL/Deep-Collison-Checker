#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
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
import os
import torch
import pickle
import time
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np
from os.path import exists

# Useful classes
from utils.config import Config
from utils.ply import read_ply, write_ply
from models.architectures import KPFCNN
from kernels.kernel_points import create_3D_rotations

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)


def batch_grid_subsampling(points, batches_len, features=None, labels=None,
                           sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=False):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Subsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                                batches_len,
                                                                                features=features,
                                                                                classes=labels,
                                                                                sampleDl=sampleDl,
                                                                                max_p=max_p,
                                                                                verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels


def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Online tester class
#       \*************************/
#


class OnlineData:

    def __init__(self, config):

        # Dict from labels to names
        self.label_to_names = {0: 'uncertain',
                               1: 'ground',
                               2: 'still',
                               3: 'longT',
                               4: 'shortT'}

        # Initiate a bunch of variables concerning class labels
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.sort([0])

        # Load neighb_limits dictionary
        neighb_lim_file = '../../Data/MyhalSim/neighbors_limits.pkl'
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            raise ValueError('No neighbors limit file found')

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(config.num_layers):

            dl = config.first_subsampling_dl * (2**layer_ind)
            if config.deform_layers[layer_ind]:
                r = dl * config.deform_radius
            else:
                r = dl * config.conv_radius

            key = '{:s}_{:d}_{:d}_{:.3f}_{:.3f}'.format('random',
                                                        config.n_frames,
                                                        config.max_val_points,
                                                        dl,
                                                        r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if len(neighb_limits) == config.num_layers:
            self.neighborhood_limits = neighb_limits
        else:
            raise ValueError('The neighbors limits were not initialized')

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, :self.neighborhood_limits[layer]]
        else:
            return neighbors

    def build_segmentation_input_list(self, config, stacked_points, stack_lengths):

        # Starting radius of convolutions
        r_normal = config.first_subsampling_dl * config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = config.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * config.deform_radius / config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * config.deform_radius / config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors(stacked_points, pool_p, stack_lengths, pool_b, 2 * r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 3), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)
                up_i = np.zeros((0, 1), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            if up_i.shape[0] > 0:
                up_i = self.big_neighborhood_filter(up_i, len(input_points) + 1)

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_upsamples += [up_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths

        return li


class OnlineBatch:
    """Custom batch definition for online frame processing"""

    def __init__(self, frame_points, config, data_handler):
        """
        Function creating the batch structure from frame points.
        :param frame_points: matrix of the frame points
        :param config: Configuration class
        :param data_handler: Data handling class
        """

        # TODO: Speed up this CPU preprocessing
        #           > Use OMP for neighbors processing
        #           > Use the polar coordinates to get neighbors???? (avoiding tree building time)

        # First subsample points
        in_pts = grid_subsampling(frame_points, sampleDl=config.first_subsampling_dl)

        # Randomly drop some points (safety for GPU memory consumption)
        if in_pts.shape[0] > config.max_val_points:
            input_inds = np.random.choice(in_pts.shape[0], size=config.max_val_points, replace=False)
            in_pts = in_pts[input_inds, :]

        # Length of the point list (here not really useful but the network need that value)
        in_lengths = np.array([in_pts.shape[0]], dtype=np.int32)

        # Features the network was trained with
        in_features = np.ones_like(in_pts[:, :1], dtype=np.float32)
        if config.in_features_dim == 1:
            pass
        elif config.in_features_dim == 2:
            # Use height coordinate
            in_features = np.hstack((in_features, in_pts[:, 2:3]))
        elif config.in_features_dim == 4:
            # Use all coordinates
            in_features = np.hstack((in_features, in_pts[:3]))

        # Get the whole input list
        input_list = data_handler.build_segmentation_input_list(config, in_pts, in_lengths)

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+config.num_layers]]
        ind += config.num_layers
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+config.num_layers]]
        ind += config.num_layers
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+config.num_layers]]
        ind += config.num_layers
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+config.num_layers]]
        ind += config.num_layers
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+config.num_layers]]
        ind += config.num_layers
        self.features = torch.from_numpy(in_features)

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)

        return self


class OnlineTester:

    def __init__(self, in_topic, out_topic, training_path):

        ####################
        # Init environment #
        ####################

        # Set which gpu is going to be used
        GPU_ID = '0'

        # Set GPU visible device
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

        # Get the GPU for PyTorch
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        ######################
        # Load trained model #
        ######################

        print('\nModel Preparation')
        print('*****************')
        t1 = time.time()

        # Choose which training checkpoints to use
        chkp_path = os.path.join(training_path, 'checkpoints', 'current_chkp.tar')

        # Load configuration class used at training
        self.config = Config()
        self.config.load(training_path)

        # Init data class
        self.data_handler = OnlineData(self.config)

        # Define network model
        self.net = KPFCNN(self.config, self.data_handler.label_values, self.data_handler.ignored_labels)
        self.net.to(self.device)
        self.softmax = torch.nn.Softmax(1)

        # Load the pretrained weigths
        checkpoint = torch.load(chkp_path)
        #checkpoint = torch.load(chkp_path, map_location={'cuda:0': 'cpu'})
        self.net.load_state_dict(checkpoint['model_state_dict'])

        # Switch network from training to evaluation mode
        self.net.eval()

        print("\nModel and training state restored from " + chkp_path)
        print('Done in {:.1f}s\n'.format(time.time() - t1))

        ############
        # Init ROS #
        ############

        # self.pub = rospy.Publisher(out_topic, PointCloud2, queue_size=10)
        # rospy.init_node('classifier', anonymous=True)
        # rospy.Subscriber(in_topic, PointCloud2, self.LidarCallback)
        # rospy.spin()

    def network_inference(self, points):
        """
        Function simulating a network inference.
        :param points: The input list of points as a numpy array (type float32, size [N,3])
        :return: predictions : The output of the network. Class for each point as a numpy array (type int32, size [N])
        """

        #####################
        # Input preparation #
        #####################

        t = [time.time()]

        # Create batch from the frame points
        batch = OnlineBatch(points, self.config, self.data_handler)

        t += [time.time()]

        # Convert batch to a cuda
        batch.to(self.device)
        t += [time.time()]
        torch.cuda.synchronize(self.device)

        #####################
        # Network inference #
        #####################

        # Forward pass
        outputs = self.net(batch, self.config)
        torch.cuda.synchronize(self.device)
        t += [time.time()]

        # Get probs and labels
        predicted_probs = self.softmax(outputs).cpu().detach().numpy()
        torch.cuda.synchronize(self.device)
        t += [time.time()]

        # Insert false columns for ignored labels
        for l_ind, label_value in enumerate(self.data_handler.label_values):
            if label_value in self.data_handler.ignored_labels:
                predicted_probs = np.insert(predicted_probs, l_ind, 0, axis=1)

        # Get predicted labels
        predictions = self.data_handler.label_values[np.argmax(predicted_probs, axis=1)].astype(np.int32)
        t += [time.time()]

        print('\n************************\n')
        print('Timings:')
        i = 0
        print('Batch ...... {:7.1f} ms'.format(1000*(t[i+1] - t[i])))
        i += 1
        print('ToGPU ...... {:7.1f} ms'.format(1000*(t[i+1] - t[i])))
        i += 1
        print('Forward .... {:7.1f} ms'.format(1000*(t[i+1] - t[i])))
        i += 1
        print('Softmax .... {:7.1f} ms'.format(1000*(t[i+1] - t[i])))
        i += 1
        print('Preds ...... {:7.1f} ms'.format(1000*(t[i+1] - t[i])))
        print('-----------------------')
        print('TOTAL  ..... {:7.1f} ms'.format(1000*(t[-1] - t[0])))
        print('\n************************\n')

        return predictions, batch.points[0].cpu().numpy()


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ########
    # Init #
    ########

    chosen_log = 'results/Log_2020-08-14_10-02-36'  # Chose the log you want to test here

    # Choose which session you want to test the model on on
    day = '2021-05-15-23-15-09'
    frames_folder = '../../Myhal_Simulation/simulated_runs/{:s}/sim_frames'.format(day)

    # Get frame names and timestamps
    f_names = [f for f in os.listdir(frames_folder) if f[-4:] == '.ply']
    f_times = np.array([float(f[:-4]) for f in f_names], dtype=np.float64)
    f_names = np.array([os.path.join(frames_folder, f) for f in f_names])
    ordering = np.argsort(f_times)
    f_names = f_names[ordering]
    f_times = f_times[ordering]

    # Online tester
    tester = OnlineTester(0, 0, chosen_log)

    ######### 
    # Start #
    #########

    saving_folder = '../../Myhal_Simulation/predicted_frames/{:s}'.format(day)

    if not exists(saving_folder):
        os.makedirs(saving_folder)

    for f_name in f_names:

        # Load points
        data = read_ply(f_name)
        points = np.vstack((data['x'], data['y'], data['z'])).T

        predictions, new_points = tester.network_inference(points)

        new_name = os.path.join(saving_folder, f_name.split('/')[-1])
        write_ply(new_name,
                  [new_points, predictions],
                  ['x', 'y', 'z', 'pred'])




