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
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from datasets.ModelNet40 import *
from datasets.ModelNet40_normals import *
from datasets.S3DIS import *
from datasets.ShapeNetPart import *
from datasets.SemanticKitti import *
from datasets.SemanticKitti2 import *
from datasets.MyhalSim import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN, KPFCNN_regress


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    #chosen_log = 'results/Log_2020-06-23_00-57-37'  # => ModelNet40
    #chosen_log = 'results/Log_2020-04-28_12-31-37'  # +> SemanticKitti2

    #chosen_log = 'results/Log_2020-06-23_00-57-37'  # => NR normal
    #chosen_log = 'results/Log_2020-06-24_00-01-45'  # => NR invar
    #chosen_log = 'results/Log_2020-06-24_18-55-08'  # => NR invar stronger

    #chosen_log = 'results/Log_2020-06-26_01-07-40'  # => Equivar_v0
    #chosen_log = 'results/Log_2020-06-26_22-17-46'  # => Equivar_v1 ortho 0.1

    #chosen_log = 'results/Log_2020-06-28_15-20-28'  # => DetachLRF rot+sym/Identity
    #chosen_log = 'results/Log_2020-06-28_19-07-36'  # => DetachLRF no augm/Identity deeper


    ### New equivaraint head ###

    # chosen_log = 'results/Log_2020-07-03_09-27-07' # => equi_head/IdLRF/AR
    #chosen_log = 'results/Log_2020-07-03_09-29-01' # => equi_head/IdLRF/NR
    #chosen_log = 'results/Log_2020-07-03_13-41-22' # => equi_head/det_4*2^n/o=0.1/AR

    #chosen_log = 'results/Log_2020-07-06_10-39-23' # =>  v2_det_2*2^n/o=0.5/aligned_head/NR

    #############################################

    #chosen_log = 'results/Log_2020-07-29_12-00-10'  # ShapeNetPart 1-global
    #chosen_log = 'results/Log_2020-07-29_20-10-28'  # ShapeNetPart 1-local
    #chosen_log = 'results/Log_2020-07-16_11-31-10'  # ShapeNetPart 4-local

    #############################################

    #chosen_log = 'results/Log_2020-07-15_07-54-21'  # ShapeNetPart normal/in_f1/NR
    #chosen_log = 'results/Log_2020-07-15_07-55-23'  # ShapeNetPart normal/in_f1/AR

    #chosen_log = 'results/Log_2020-06-23_00-57-37'  # ModelNet40 normal/in_f1/NR
    #chosen_log = 'results/Log_2020-06-23_09-05-14'  # ModelNet40 normal/in_f1/AR

    # Myhal_Sim, triggers a special test function
    #chosen_log = 'results/Log_2020-10-13_15-43-31'  # Round 2
    #chosen_log = 'results/Log_2020-10-16_23-40-33'  # Round 3
    chosen_log = 'results/Log_2020-10-27_15-02-20'  # Round 4

    GT_days = ['2020-11-05-22-09-11']

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = 7

    # Choose to test on validation or test split
    on_val = False

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

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

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    config.augment_noise = 0.0001
    config.augment_symmetries = [False, False, False]
    #config.batch_num = 3
    #config.in_radius = 4
    config.validation_size = 680
    config.input_threads = 20
    config.augment_rotation = 'none'

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'ModelNet40':
        test_dataset = ModelNet40Dataset(config, train=False)
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = ModelNet40Collate
    elif config.dataset == 'ModelNet40N':
        test_dataset = ModelNet40NDataset(config, train=False)
        test_sampler = ModelNet40NSampler(test_dataset)
        collate_fn = ModelNet40NCollate
    elif config.dataset == 'S3DIS':
        test_dataset = S3DISDataset(config, set='validation', use_potentials=True)
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    elif config.dataset == 'ShapeNetPart_multi':
        test_dataset = ShapeNetPartDataset(config, train=False)
        test_sampler = ShapeNetPartSampler(test_dataset, use_potential=True, balance_labels=False)
        collate_fn = ShapeNetPartCollate
    elif config.dataset == 'SemanticKitti':
        test_dataset = SemanticKittiDataset(config, set=set, balance_classes=False)
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate
    elif config.dataset == 'SemanticKitti2':
        test_dataset = SemanticKitti2Dataset(config, set=set, balance_classes=False)
        test_sampler = SemanticKitti2Sampler(test_dataset)
        collate_fn = SemanticKitti2Collate
    elif config.dataset == 'MyhalSim':
        test_dataset = MyhalSimDataset(config, GT_days, set=set, balance_classes=False)
        test_sampler = MyhalSimSampler(test_dataset)
        collate_fn = MyhalSimCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    if config.dataset_task == 'classification':
        net = KPCNN(config)
    elif config.dataset_task in ['normals_regression']:
        net = KPFCNN_regress(config)
    elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    elif config.dataset_task in ['multi_part_segmentation']:
        net = KPFCNN(config,
                     test_dataset.label_values,
                     test_dataset.ignored_labels,
                     num_parts=test_dataset.num_parts)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')



    if config.dataset == 'MyhalSim':
        tester.slam_MyhalSim_test(net, test_loader, config)
    elif config.dataset_task == 'classification':
        tester.classification_test(net, test_loader, config)
    elif config.dataset_task == 'normals_regress':
        tester.normals_regress_test(net, test_loader, config)
    elif config.dataset_task == 'cloud_segmentation':
        tester.cloud_segmentation_test(net, test_loader, config)
    elif config.dataset_task == 'slam_segmentation':
        tester.slam_segmentation_test(net, test_loader, config)
    elif config.dataset_task == 'multi_part_segmentation':
        tester.part_segmentation_test(net, test_loader, config)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)


    # TODO: For test and also for training. When changing epoch do not restart the worker initiation. Keep workers
    #  active with a while loop instead of using for loops.
    #  For training and validation, keep two sets of worker active in parallel? is it possible?

    # TODO: We have to verify if training on smaller spheres and testing on whole frame changes the score because
    #  batchnorm may not have the same result as distribution of points will be different.

    # Et test








