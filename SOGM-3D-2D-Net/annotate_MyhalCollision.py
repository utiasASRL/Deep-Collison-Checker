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
#           Main Function
#       \***************/
#


def main(dataset_path, map_day, refine_days, train_days, force_redo=True):

    # Check if we need to redo annotation (only if there is no collison folder)
    redo_annot = False
    for day in train_days:
        annot_path = join(dataset_path, 'collisions', day)
        if not exists(annot_path):
            redo_annot = True
            break

    # Forced redo        
    redo_annot = redo_annot or force_redo

    if redo_annot:

        # Initiate dataset
        slam_dataset = MyhalCollisionSlam(day_list=train_days, map_day=map_day, dataset_path=dataset_path)


        ####################
        # Map initialization
        ####################

        slam_dataset.init_map()

        ################
        # Map refinement
        ################

        slam_dataset.refine_map(refine_days)

        ######################
        # Automatic Annotation
        ######################

        # Groundtruth annotation
        annotation_process(slam_dataset, on_gt=False)

        #################
        # SOGM Generation
        #################

        # Annotation of preprocessed 2D+T point clouds for SOGM generation
        slam_dataset.collision_annotation()

        print('annotation finished')

    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###################
    # Training sessions
    ###################

    # Exp 0 First tries
    # Day used as map
    # map_day = '2021-11-04_09-56-16'
    # train_days = ['2021-11-04_10-03-09',
    #               '2021-11-04_10-06-45']

    ################################################################################################################################################
    # Exp 2
    #
    # # Notes for myself
    # #   > 1: We deleted a lot of frames in the map 2021-11-16_19-42-45 because they were all in the same place
    # #   > 2: We perform the loop closure manually by chossing which frames to align
    #
    # map_day = '2021-11-16_19-42-45'
    # train_days = ['2021-11-04_10-03-09',
    #               '2021-11-04_10-06-45',
    #               '2021-11-16_20-08-59',
    #               '2021-11-17_10-21-56',
    #               '2021-11-17_10-45-12']
    ################################################################################################################################################
          
    ################################################################################################################################################
    # Exp 3
    #
    # Notes for myself
    #   > 1: New runs with aniket further away
    #   > 2: TODO We perform the loop closure manually by chossing which frames to align
    #   > 3: TODO We implemented the map refinement to remove doors

    # map_day = '2021-11-30_12-05-32'
    # # train_days = ['2021-11-30_12-22-23',
    # #               '2021-11-30_12-33-09',
    # #               '2021-12-04_13-27-32',
    # #               '2021-12-04_13-44-05',
    # #               '2021-12-04_13-59-29']
    # dataset_path = '../Data/Real'

    ################################################################################################################################################
    
          
    ################################################################################################################################################
    # TMP Just my house

    # map_day = '2021-12-05_18-04-51'
    # train_days = ['2021-12-05_18-04-51']
    # dataset_path = '../Data/RealAlbany'
    ################################################################################################################################################
          
    ################################################################################################################################################
    # Exp 4
    #
    # In Myhal
    # Notes for myself
    #   > 1:

    dataset_path = '../Data/RealMyhal'
    train_days = ['2021-12-06_08-12-39',    # - \
                  '2021-12-06_08-38-16',    # -  \
                  '2021-12-06_08-44-07',    # -   > First runs with controller for mapping of the environment
                  '2021-12-06_08-51-29',    # -  /
                  '2021-12-06_08-54-58',    # - /
                  '2021-12-10_13-32-10',    # - \
                  '2021-12-10_13-26-07',    # -  \
                  '2021-12-10_13-17-29',    # -   > Session with normal TEB planner
                  '2021-12-10_13-06-09',    # -  /
                  '2021-12-10_12-53-37']    # - /
    map_i = 3
    refine_i = np.array([0, 6, 7, 8])
    train_i = np.array([5, 6, 7, 8, 9])
    val_inds = [0]
         
    map_day = train_days[map_i]
    refine_days = np.array(train_days)[refine_i]
    train_days = np.array(train_days)[train_i]

    ################################################################################################################################################


    main(dataset_path, map_day, refine_days, train_days)



