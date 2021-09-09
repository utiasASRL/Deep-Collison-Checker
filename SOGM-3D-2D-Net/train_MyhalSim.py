#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on MyhalSim dataset
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
import signal
import os
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np
import sys
import torch
import time


# Dataset
from slam.PointMapSLAM import pointmap_slam, detect_short_term_movables, annotation_process
from slam.dev_slam import bundle_slam, pointmap_for_AMCL
from torch.utils.data import DataLoader
from datasets.MyhalSim import MyhalSimDataset, MyhalSimSlam, MyhalSimSampler, MyhalSimCollate

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN

from os.path import exists, join
from os import makedirs


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class MyhalSimConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'MyhalSim'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 20

    #########################
    # Architecture definition
    #########################

    # Define layers
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

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 4.0
    val_radius = 51.0
    n_frames = 1
    max_in_points = -1
    max_val_points = -1

    # Number of batch
    batch_num = 10
    val_batch_num = 1

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.03

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

    # Choice of input features
    first_features_dim = 128
    in_features_dim = 1

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
    max_epoch = 800

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 100

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_color = 0.8

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

    ###################
    # Training sessions
    ###################

    # Day used as map
    map_day = '2020-10-02-13-39-05'

    # Fisrt dataset: successful tours without filtering Initiate dataset. Remember last day is used as validation for the training
    train_days_0 = ['2020-10-12-22-06-54',
                    '2020-10-12-22-14-48',
                    '2020-10-12-22-28-15']

    # Second dataset
    train_days_1 = ['2020-10-12-22-06-54',
                    '2020-10-12-22-14-48',
                    '2020-10-12-22-28-15',
                    '2020-10-16-12-29-11',
                    '2020-10-16-12-37-53',
                    '2020-10-16-12-50-41',
                    '2020-10-16-13-06-53',
                    '2020-10-16-13-20-04',
                    '2020-10-16-13-38-50',
                    '2020-10-16-14-01-49',
                    '2020-10-16-14-36-12',
                    '2020-10-16-14-56-40']

    # Third dataset
    train_days_2 = ['2020-10-12-22-06-54',
                    '2020-10-12-22-14-48',
                    '2020-10-12-22-28-15',
                    '2020-10-16-12-29-11',
                    '2020-10-16-12-37-53',
                    '2020-10-16-12-50-41',
                    '2020-10-16-13-06-53',
                    '2020-10-16-13-20-04',
                    '2020-10-16-13-38-50',
                    '2020-10-16-14-01-49',
                    '2020-10-16-14-36-12',
                    '2020-10-16-14-56-40',
                    '2020-10-19-17-25-50',
                    '2020-10-19-17-34-13',
                    '2020-10-19-17-47-10',
                    '2020-10-19-18-03-33',
                    '2020-10-19-18-14-42',
                    '2020-10-19-18-33-04',
                    '2020-10-19-18-55-15',
                    '2020-10-19-20-04-41',
                    '2020-10-19-20-23-25',
                    '2020-10-22-21-17-35',
                    '2020-10-22-21-37-50',
                    '2020-10-22-22-04-14']


    ######################
    # Automatic Annotation
    ######################

    # Choose the dataset
    train_days = train_days_2

    # Check if we need to redo annotation (only if there is no video)
    redo_annot = False
    for day in train_days:
        annot_path = join('../../Myhal_Simulation/annotated_frames', day)
        if not exists(annot_path):
            redo_annot = True
            break

    # train_days = ['2020-10-20-16-30-49']
    redo_annot = True
    if redo_annot:

        # Initiate dataset
        slam_dataset = MyhalSimSlam(day_list=train_days, map_day=map_day)
        #slam_dataset = MyhalSimDataset(first_day='2020-06-24-14-36-49', last_day='2020-06-24-14-40-33')

        # Create a refined map from the map_day
        slam_dataset.refine_map()

        # Groundtruth mapping
        #slam_dataset.debug_angular_velocity()
        #slam_dataset.gt_mapping() # can you add all frames at onec in this function?

        # Groundtruth annotation
        #annotation_process(slam_dataset, on_gt=True)

        # SLAM mapping
        #slam_dataset.pointmap_slam()

        # Groundtruth annotation
        annotation_process(slam_dataset, on_gt=False)

        # TODO: Loop closure for aligning days together when niot simulation

        a = 1/0

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '1'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Choose here if you want to start training from a previous snapshot (None for new training)
    # previous_training_path = 'Log_2020-03-19_19-53-27'
    previous_training_path = ''

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = MyhalSimConfig()
    if previous_training_path:
        config.load(os.path.join('results', previous_training_path))
        config.saving_path = None

    # Get path from argument if given
    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]

    # Initialize datasets (dummy validation)
    train_days += [train_days[-1]]
    training_dataset = MyhalSimDataset(config, train_days, set='training', balance_classes=True)
    test_dataset = MyhalSimDataset(config, train_days, set='validation', balance_classes=False)

    # Initialize samplers
    training_sampler = MyhalSimSampler(training_dataset)
    test_sampler = MyhalSimSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=MyhalSimCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=MyhalSimCollate,
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
    net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)

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

    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    trainer.train(net, training_loader, test_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)
