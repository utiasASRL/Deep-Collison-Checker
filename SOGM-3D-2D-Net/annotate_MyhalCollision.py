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
import os
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np

# Dataset
from slam.PointMapSLAM import annotation_process
from datasets.MyhalCollision import MyhalCollisionSlam

from gt_annotation_video import get_videos, show_2D_SOGMS

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

    # dataset_path = '../Data/RealMyhal'
    # train_days = ['2021-12-06_08-12-39',    # - \
    #               '2021-12-06_08-38-16',    # -  \
    #               '2021-12-06_08-44-07',    # -   > First runs with controller for mapping of the environment
    #               '2021-12-06_08-51-29',    # -  /
    #               '2021-12-06_08-54-58',    # - /
    #               '2021-12-10_13-32-10',    # 5 \
    #               '2021-12-10_13-26-07',    # 6  \
    #               '2021-12-10_13-17-29',    # 7   > Session with normal TEB planner
    #               '2021-12-10_13-06-09',    # 8  /
    #               '2021-12-10_12-53-37',    # - /
    #               '2021-12-13_18-16-27',    # - \
    #               '2021-12-13_18-22-11',    # -  \
    #               '2021-12-15_19-09-57',    # -   > Session with normal TEB planner Tour A and B
    #               '2021-12-15_19-13-03']    # -  /

    # train_days += ['2022-01-18_10-38-28',   # 14 \
    #                '2022-01-18_10-42-54',   # -   \
    #                '2022-01-18_10-47-07',   # -    \
    #                '2022-01-18_10-48-42',   # -     \
    #                '2022-01-18_10-53-28',   # -      > Sessions with normal TEB planner on loop_3
    #                '2022-01-18_10-58-05',   # -      > Simple scenarios for experiment
    #                '2022-01-18_11-02-28',   # 20    /
    #                '2022-01-18_11-11-03',   # -    /
    #                '2022-01-18_11-15-40',   # -   /
    #                '2022-01-18_11-20-21']   # -  /
                   

    # train_days += ['2022-02-25_18-19-12',   # 24 \
    #                '2022-02-25_18-24-30',   # -   > Face to face scenario on (loop_2)
    #                '2022-02-25_18-29-18']   # -  /

    # train_days += ['2022-03-01_22-01-13',   # 27 \
    #                '2022-03-01_22-06-28',   # -   > More data (loop_2inv and loop8)
    #                '2022-03-01_22-19-41',   # -   > face to face and crossings
    #                '2022-03-01_22-25-19']   # -  /

    # map_i = 3
    # refine_i = np.array([0, 6, 7, 8, 14, 20, 24, 27])
    # train_i = np.arange(len(train_days))[5:]
         
    # map_day = train_days[map_i]
    # refine_days = np.array(train_days)[refine_i]
    # train_days = np.sort(np.array(train_days)[train_i])

    ################################################################################################################################################
    
    ################################################################################################################################################
    # Exp 5
    #
    # In Myhal first floor
    #

    # Mapping sessions
    dataset_path = '../Data/Myhal1'
    train_days = ['2022-03-08_12-34-12',  # - \
                  '2022-03-08_12-51-26',  # -  \  First runs with controller for mapping of the environment  
                  '2022-03-08_12-52-56',  # -  /  Include refinement runs for table and elevator doors
                  '2022-03-08_14-24-09']  # - /

    # Actual sessions for training
    train_days += ['2022-03-08_21-02-28',   # - \
                   '2022-03-08_21-08-04',   # -  > ff1, Tuesday 4pm. Quite a lot of people
                   '2022-03-08_21-14-04']   # - /

    train_days += ['2022-03-08_22-19-08',   # - \
                   '2022-03-08_22-24-22',   # -  > ff2/ff1/ff2, Tuesday 5pm. Some of these are empty, maybe delete them?
                   '2022-03-08_22-28-23']   # - /

    train_days += ['2022-03-09_15-55-10',   # - \
                   '2022-03-09_15-58-56',   # -  \ ff1/ff2/ff1/ff1, Wednesday 11am.
                   '2022-03-09_16-03-21',   # -  /
                   '2022-03-09_16-07-11']   # - /

    # TODO: 
    #       > Check each run:
    #           a. Mapping problem?
    #           b. Annotation good?
    #           a. Interesting stuff happening
    #
    #   OK: PB 1: OK
    #       PB 2: OK
    #       PB 3: OK solved with robust init
    #       PB 4: ATTENTION sur Idefix, changer le run sogm car record met du temps a s'initialiser
    #
    #
    # TODO: Other
    #       > Simulation thing Tim talked about?
    #       > ICRA video (Vendredi matin?)

    map_i = 0
    refine_i = np.arange(len(train_days))[1:4]
    train_i = np.arange(len(train_days))[4:]
         
    map_day = train_days[map_i]
    refine_days = np.array(train_days)[refine_i]
    train_days = np.sort(np.array(train_days)[train_i])

    ################################################################################################################################################


    main(dataset_path, map_day, refine_days, train_days)

    # show_2D_SOGMS(dataset_path, train_days, map_day=map_day)

    # get_videos(dataset_path, train_days, map_day=map_day)


