#
#
#      0==============================0
#      |    Deep Collision Checker    |
#      0==============================0
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
from torch.utils.data import DataLoader
from datasets.MyhalCollision import MyhalCollisionDataset, MyhalCollisionSampler, MyhalCollisionCollate

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPCollider

from os.path import exists, join
from os import makedirs

from MyhalCollision_sessions import UTIn3D_H_sessions, UTIn3D_A_sessions, UTIn3D_A_sessions_v2


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
    n_2D_layers = 40

    # Total time propagated
    T_2D = 4.0

    # Size of 2D convolution grid
    dl_2D = 0.12

    # Radius of the considered arear in 2D
    radius_2D = 8.0

    # Power of the loss for the 2d predictions (use smaller prop loss when shared weights)
    power_2D_init_loss = 1.0
    power_2D_prop_loss = 50.0
    neg_pos_ratio = 0.5
    loss2D_version = 2

    # Power of the 2d future predictions loss for each class [permanent, movable, dynamic]
    power_2D_class_loss = [1.0, 1.0, 2.0]

    # Mutliplying factor between loss on the last and the first layer of future prediction
    # factor is interpolated linearly between (1.0 and factor_2D_prop_loss) / sum_total at each layer
    factor_2D_prop_loss = 2.0
    
    # Balance class in sampler, using custom proportions
    # It can have an additionnal value (one more than num_classes), to encode the proportion of simulated data we use for training
    balance_proportions = [0, 0, 1, 1, 20, 1.0]

    # Specification of the 2D networks composition
    init_2D_levels = 3      # 3
    init_2D_resnets = 2     # 2
    prop_2D_resnets = 2     # 2

    # Path to a pretrained 3D network. if empty, ignore, if 'todo', then only train 3D part of the network.
    #pretrained_3D = 'Log_2021-01-27_18-53-05'
    pretrained_3D = ''

    # Detach the 2D network from the 3D network when backpropagating gradient
    detach_2D = False

    # Share weights for 2D network TODO: see if not sharing makes a difference
    shared_2D = False
    skipcut_2D = False

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
    first_features_dim = 128

    # Number of batch
    batch_num = 6
    val_batch_num = 6

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.12

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
    max_epoch = 250

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 60) for i in range(1, max_epoch)}
    #lr_decays = {150: 0.1, 200: 0.1, 250: 0.1}
    grad_clip_norm = 100.0

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 15

    # Number of epoch between each checkpoint
    checkpoint_gap = 40

    # Augmentations
    augment_scale_anisotropic = False
    augment_symmetries = [True, False, False]
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

    #############################
    # Additionnal Simulation data
    #############################

    sim_path = '../Data/Simulation'

    # train_days_RandBounce = ['2021-05-15-23-15-09',
    #                          '2021-05-15-23-33-25',
    #                          '2021-05-15-23-54-50',
    #                          '2021-05-16-00-44-53',
    #                          '2021-05-16-01-09-43',
    #                          '2021-05-16-20-37-47',
    #                          '2021-05-16-20-59-49',
    #                          '2021-05-16-21-22-30',
    #                          '2021-05-16-22-26-45',
    #                          '2021-05-16-22-51-06',
    #                          '2021-05-16-23-34-15',
    #                          '2021-05-17-01-21-44',
    #                          '2021-05-17-01-37-09',
    #                          '2021-05-17-01-58-57',
    #                          '2021-05-17-02-34-27',
    #                          '2021-05-17-02-56-02',
    #                          '2021-05-17-03-54-39',
    #                          '2021-05-17-05-26-10',
    #                          '2021-05-17-05-41-45']

    # train_days_RandWand = ['2021-05-17-14-04-52',
    #                        '2021-05-17-14-21-56',
    #                        '2021-05-17-14-44-46',
    #                        '2021-05-17-15-26-04',
    #                        '2021-05-17-15-50-45',
    #                        '2021-05-17-16-14-26',
    #                        '2021-05-17-17-02-17',
    #                        '2021-05-17-17-27-02',
    #                        '2021-05-17-17-53-42',
    #                        '2021-05-17-18-46-44',
    #                        '2021-05-17-19-02-37',
    #                        '2021-05-17-19-39-19',
    #                        '2021-05-17-20-14-57',
    #                        '2021-05-17-20-48-53',
    #                        '2021-05-17-21-36-22',
    #                        '2021-05-17-22-16-13',
    #                        '2021-05-17-22-40-46',
    #                        '2021-05-17-23-08-01',
    #                        '2021-05-17-23-48-22',
    #                        '2021-05-18-00-07-26',
    #                        '2021-05-18-00-23-15',
    #                        '2021-05-18-00-44-33',
    #                        '2021-05-18-01-24-07']

    train_days_RandFlow = ['2021-06-02-19-55-16',
                           '2021-06-02-20-33-09',
                           '2021-06-02-21-09-48',
                           '2021-06-02-22-05-23',
                           '2021-06-02-22-31-49',
                           '2021-06-03-03-51-03',
                           '2021-06-03-14-30-25',
                           '2021-06-03-14-59-20',
                           '2021-06-03-15-43-06',
                           '2021-06-03-16-48-18',
                           '2021-06-03-18-00-33',
                           '2021-06-03-19-07-19',
                           '2021-06-03-19-52-45',
                           '2021-06-03-20-28-22',
                           '2021-06-03-21-32-44',
                           '2021-06-03-21-57-08']

    
    # Additional train and validation  from simulation
    sim_train_days = np.array(train_days_RandFlow)
    sim_val_inds = [0, 1, 2]
    sim_train_inds = [i for i in range(len(sim_train_days)) if i not in sim_val_inds]

    # Disable simulation HERE
    # sim_path = ''


    ###################
    # Training sessions
    ###################

    # Get sessions from the annotation script
    dataset_path, map_day, refine_sessions, train_days, train_comments = UTIn3D_A_sessions_v2()

    # Get training and validation sets
    val_inds = np.array([i for i, c in enumerate(train_comments) if 'val' in c.split('>')[0]])

    ######################
    # Automatic Annotation
    ######################

    # See annotate_MyhalCollision.py

    # # Check if we need to redo annotation (only if there is no collison folder)
    # redo_annot = False
    # for day in train_days:
    #     annot_path = join(dataset_path, 'collisions', day)
    #     if not exists(annot_path):
    #         redo_annot = True
    #         break

    # # To perform annnotation use the annotate_MyhalCollisions.py script
    # redo_annot = False

    # if redo_annot:
    #
    #     # Initiate dataset
    #     slam_dataset = MyhalCollisionSlam(day_list=train_days, map_day=map_day, dataset_path=dataset_path)
    #
    #     # Create a refined map from the map_day.
    #     # UNCOMMENT THIS LINE if you are using your own data for the first time
    #     # COMMENT THIS LINE if you already have a nice clean map of the environment as a point cloud
    #     # like this one: Data/Simulation/slam_offline/2020-10-02-13-39-05/map_update_0001.ply
    #
    #     slam_dataset.refine_map()
    #
    #     # Groundtruth annotation
    #     annotation_process(slam_dataset, on_gt=False)
    #
    #     # Annotation of preprocessed 2D+T point clouds for SOGM generation
    #     slam_dataset.collision_annotation()
    #
    #     print('annotation finished')

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Validation sessions
    train_inds = [i for i in range(len(train_days)) if i not in val_inds]
    

    # TMP, lifelong learning exp. Use trains inds up to :7, :17, :25, :all
    # train_inds = train_inds[:25]
    # train_inds = train_inds[:17]
    # train_inds = train_inds[:7]

    # Then, add simulation data


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
    training_dataset = MyhalCollisionDataset(config,
                                             train_days[train_inds],
                                             chosen_set='training',
                                             dataset_path=dataset_path,
                                             balance_classes=True,
                                             add_sim_path=sim_path,
                                             add_sim_days=sim_train_days[sim_train_inds])
    test_dataset = MyhalCollisionDataset(config,
                                         train_days[val_inds],
                                         chosen_set='validation',
                                         dataset_path=dataset_path,
                                         balance_classes=False,
                                         add_sim_path=sim_path,
                                         add_sim_days=sim_train_days[sim_val_inds])

    # Initialize samplers
    training_sampler = MyhalCollisionSampler(training_dataset, manual_training_frames=True)
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
        training_sampler.calib_max_in(config, training_loader, untouched_ratio=0.9, verbose=True, force_redo=False)
    if config.max_val_points < 0:
        config.max_val_points = 1e9
        test_loader.dataset.max_in_p = 1e9
        test_sampler.calib_max_in(config, test_loader, untouched_ratio=0.95, verbose=True, force_redo=False)


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
