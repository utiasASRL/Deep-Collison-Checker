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
import signal
import os
import numpy as np
import sys
import torch
import time

# Dataset
from slam.PointMapSLAM import pointmap_slam, detect_short_term_movables, annotation_process
from slam.dev_slam import bundle_slam, pointmap_for_AMCL
from torch.utils.data import DataLoader
from datasets.MyhalCollision import MyhalCollisionDataset, MyhalCollisionSlam, MyhalCollisionSampler, MyhalCollisionCollate

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN

from os.path import exists, join
from os import makedirs


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###################
    # Training sessions
    ###################

    # Day used as map
    map_day = '2020-10-02-13-39-05'

    # Third dataset
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

    ######################
    # Automatic Annotation
    ######################

    # Choose the dataset
    train_days = train_days_1

    # Check if we need to redo annotation (only if there is no video)
    redo_annot = False
    for day in train_days:
        annot_path = join('../../../Myhal_Simulation/annotated_frames', day)
        if not exists(annot_path):
            redo_annot = True
            break


    redo_annot = True
    if redo_annot:

        # Initiate dataset
        slam_dataset = MyhalCollisionSlam(day_list=train_days, map_day=map_day)
        #slam_dataset = MyhalCollisionDataset(first_day='2020-06-24-14-36-49', last_day='2020-06-24-14-40-33')

        # Create a refined map from the map_day
        slam_dataset.refine_map()

        # Groundtruth mapping
        # slam_dataset.debug_angular_velocity()
        # slam_dataset.gt_mapping() #TODO: can you add all frames at onec in this function?

        # Groundtruth annotation
        #annotation_process(slam_dataset, on_gt=True)

        # SLAM mapping
        # slam_dataset.pointmap_slam()

        # Groundtruth annotation
        annotation_process(slam_dataset, on_gt=False)

        slam_dataset.collision_annotation()

        a = 1/0


