#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on MyhalCollision dataset
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
import sys
import time
import signal
import os
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np
import torch


# Dataset
from slam.PointMapSLAM import pointmap_slam, detect_short_term_movables, annotation_process
from slam.dev_slam import bundle_slam, pointmap_for_AMCL
from torch.utils.data import DataLoader
from datasets.MyhalCollision import MyhalCollisionDataset, MyhalCollisionSlam, MyhalCollisionSampler, \
    MyhalCollisionCollate

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPCollider

from os.path import exists, join
from os import makedirs


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class MyhalCollisionConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'MyhalCollision'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 16

    #########################
    # Architecture definition
    #########################

    # Define layers (only concerning the 3D architecture)
    architecture = ['simple',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    ######################
    # Collision parameters
    ######################

    # Number of propagating layer
    n_2D_layers = 30

    # Total time propagated
    T_2D = 3.0

    # Size of 2D convolution grid
    dl_2D = 0.12

    # Power of the loss for the 2d predictions (use smaller prop loss when shared weights)
    power_2D_init_loss = 1.0
    power_2D_prop_loss = 50.0
    neg_pos_ratio = 1.0
    loss2D_version = 2

    # Specification of the 2D networks composition
    init_2D_levels = 4      # 3
    init_2D_resnets = 3     # 2
    prop_2D_resnets = 3     # 2

    # Path to a pretrained 3D network. if empty, ignore, if 'todo', then only train 3D part of the network.
    #pretrained_3D = 'Log_2021-01-27_18-53-05'
    pretrained_3D = ''

    # Detach the 2D network from the 3D network when backpropagating gradient
    detach_2D = False

    # Share weights for 2D network TODO: see if not sharing makes a difference
    shared_2D = False

    # Trainable backend 3D network
    apply_3D_loss = True
    #frozen_layers = ['encoder_blocks', 'decoder_blocks', 'head_mlp', 'head_softmax']

    # Use visibility mask for training
    use_visibility = False

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 8.0
    val_radius = 8.0
    n_frames = 3
    in_features_dim = n_frames
    max_in_points = -1
    max_val_points = -1

    # Choice of input features
    first_features_dim = 100

    # Number of batch
    batch_num = 6
    val_batch_num = 6

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.06

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 1000

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 120) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 30

    # Number of epoch between each checkpoint
    checkpoint_gap = 20

    # Augmentations
    augment_scale_anisotropic = False
    augment_symmetries = [False, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.99
    augment_scale_max = 1.01
    augment_noise = 0.001
    augment_color = 1.0

    # Do we nee to save convergence
    saving = True
    saving_path = None


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    # NOT_NOW_TODO: Optimize online predictions
    #           > Try to parallelise the batch preprocessing for a single input frame.
    #           > Use OMP for neighbors processing
    #           > Use the polar coordinates to get neighbors???? (avoiding tree building time)
    #           > cpp extension for conversion into a 2D lidar_range_scan
    #

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used (auto for automatic choice)
    GPU_ID = 'auto'

    # Automatic choice (need pynvml to be installed)
    if GPU_ID == 'auto':
        print('\nSearching a free GPU:')
        for i in range(torch.cuda.device_count()):
            a = torch.cuda.list_gpu_processes(i)
            print(torch.cuda.list_gpu_processes(i))
            a = a.split()
            if a[1] == 'no':
                GPU_ID = a[0][-1:]

    # Safe check no free GPU
    if GPU_ID == 'auto':
        print('\nNo free GPU found!\n')
        a = 1/0

    else:
        print('\nUsing GPU:', GPU_ID, '\n')

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    chosen_gpu = int(GPU_ID)

    ###################
    # Training sessions
    ###################

    # Day used as map
    map_day = '2020-10-02-13-39-05'

    train_days_RandBounce = ['2021-05-15-23-15-09',
                             '2021-05-15-23-33-25',
                             '2021-05-15-23-54-50',
                             '2021-05-16-00-44-53',
                             '2021-05-16-01-09-43',
                             '2021-05-16-20-37-47',
                             '2021-05-16-20-59-49',
                             '2021-05-16-21-22-30',
                             '2021-05-16-22-26-45',
                             '2021-05-16-22-51-06',
                             '2021-05-16-23-34-15',
                             '2021-05-17-01-21-44',
                             '2021-05-17-01-37-09',
                             '2021-05-17-01-58-57',
                             '2021-05-17-02-34-27',
                             '2021-05-17-02-56-02',
                             '2021-05-17-03-54-39',
                             '2021-05-17-05-26-10',
                             '2021-05-17-05-41-45']


    ######################
    # Automatic Annotation
    ######################

    # Choose the dataset
    train_days = np.array(train_days_RandBounce)
    val_inds = [0, 1, 2]
    train_inds = [i for i in range(len(train_days)) if i not in val_inds]

    # Check if we need to redo annotation (only if there is no collison folder)
    redo_annot = False
    for day in train_days:
        annot_path = join('../Data/Simulation/collisions', day)
        if not exists(annot_path):
            redo_annot = True
            break

    # train_days = ['2020-10-20-16-30-49']
    # redo_annot = True
    if redo_annot:

        # Initiate dataset
        slam_dataset = MyhalCollisionSlam(day_list=train_days, map_day=map_day)

        # Create a refined map from the map_day.
        # UNCOMMENT THIS LINE if you are using your own data for the first time
        # COMMENT THIS LINE if you already have a nice clean map of the environment as a point cloud 
        # like this one: Data/Simulation/slam_offline/2020-10-02-13-39-05/map_update_0001.ply

        # slam_dataset.refine_map()  

        # Groundtruth annotation
        annotation_process(slam_dataset, on_gt=False)

        # TODO: Loop closure for aligning days together when not simulation

        # Annotation of preprocessed 2D+T point clouds for SOGM generation
        slam_dataset.collision_annotation()

        print('annotation finished')

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = MyhalCollisionConfig()
  
    # Override with configuration from previous 3D network if given
    if config.pretrained_3D and config.pretrained_3D != 'todo':

        # Check if path exists
        previous_path = os.path.join('results', config.pretrained_3D)
        if not exists(previous_path):
            raise ValueError('Given path for previous 3D network does not exist')
        
        # Load config
        prev_config = MyhalCollisionConfig()
        prev_config.load(previous_path)

        # List of params we should not overwrite:
        kept_params = ['n_2D_layers',
                       'T_2D',
                       'dl_2D',
                       'power_2D_init_loss',
                       'power_2D_prop_loss',
                       'neg_pos_ratio',
                       'init_2D_levels',
                       'init_2D_resnets',
                       'prop_2D_resnets',
                       'pretrained_3D',
                       'detach_2D',
                       'shared_2D',
                       'apply_3D_loss',
                       'frozen_layers',
                       'max_epoch',
                       'learning_rate',
                       'momentum',
                       'lr_decays',
                       'grad_clip_norm',
                       'epoch_steps',
                       'validation_size',
                       'checkpoint_gap',
                       'saving',
                       'saving_path',
                       'input_threads']
        
        for attr_name, attr_value in vars(config).items():
            if attr_name not in kept_params:
                setattr(config, attr_name, getattr(prev_config, attr_name))


    # Get path from argument if given
    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]

    ###############
    # Previous chkp
    ###############
    # Choose here if you want to start training from a previous snapshot (None for new training)

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None

    chosen_chkp = None
    if config.pretrained_3D and config.pretrained_3D != 'todo':

        # Check if path exists
        chkp_path = os.path.join('results', config.pretrained_3D, 'checkpoints')
        if not exists(chkp_path):
            raise ValueError('Given path for previous 3D network does contain any checkpoints')

        # Find all snapshot in the chosen training folder
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', config.pretrained_3D, 'checkpoints', chosen_chkp)

    #####################
    # Init input pipeline
    #####################

    # Initialize datasets (dummy validation)
    training_dataset = MyhalCollisionDataset(config, train_days[train_inds], chosen_set='training', balance_classes=True)
    test_dataset = MyhalCollisionDataset(config, train_days[val_inds], chosen_set='validation', balance_classes=False)

    # Initialize samplers
    training_sampler = MyhalCollisionSampler(training_dataset)
    test_sampler = MyhalCollisionSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=MyhalCollisionCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=MyhalCollisionCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate max_in_point value
    if config.max_in_points < 0:
        config.max_in_points = 1e9
        training_loader.dataset.max_in_p = 1e9
        training_sampler.calib_max_in(config, training_loader, untouched_ratio=0.9, verbose=True)
    if config.max_val_points < 0:
        config.max_val_points = 1e9
        test_loader.dataset.max_in_p = 1e9
        test_sampler.calib_max_in(config, test_loader, untouched_ratio=0.95, verbose=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_class_w(training_dataset, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPCollider(config, training_dataset.label_values, training_dataset.ignored_labels)

    debug = False
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    # Freeze layers if necessary
    if config.frozen_layers:
        for name, child in net.named_children():
            if name in config.frozen_layers:
                for param in child.parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                child.eval()


    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp, gpu_id=chosen_gpu)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    trainer.train(net, training_loader, test_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)
