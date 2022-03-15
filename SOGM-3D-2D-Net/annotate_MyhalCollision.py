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
import shutil
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np

# Dataset
from slam.PointMapSLAM import annotation_process
from datasets.MyhalCollision import MyhalCollisionSlam

from gt_annotation_video import get_videos, inspect_sogm_sessions, show_seq_slider

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

    
def erase_runs(dataset_path, runs_to_erase):
    
    erased_runs = False

    for run in runs_to_erase:

        folders = [join(dataset_path, 'annotated_frames', run),
                   join(dataset_path, 'annotation', run),
                   join(dataset_path, 'collisions', run),
                   join(dataset_path, 'noisy_collisions', run),
                   join(dataset_path, 'runs', run),
                   join(dataset_path, 'annotated_frames', run)]

        erase_this_runs = False
        for folder in folders:
            if (os.path.isdir(folder)):
                erase_this_runs = True
        
        if erase_this_runs:
            print('Are you sure you want to erase the following run:', run, '? [y/n]')
            confirm = input()
            if (confirm == 'y'):
                erased_runs = True
                for folder in folders:
                    if (os.path.isdir(folder)):
                        shutil.rmtree(folder)
        
    return erased_runs


# ----------------------------------------------------------------------------------------------------------------------
#
#           Listing Training sessions
#       \*******************************/
#


def Myhal5_sessions():

    # Mapping sessions
    # ****************

    dataset_path = '../Data/RealMyhal'
    train_days = ['2021-12-06_08-12-39',    # - \
                  '2021-12-06_08-38-16',    # -  \
                  '2021-12-06_08-44-07',    # -   > First runs with controller for mapping of the environment
                  '2021-12-06_08-51-29',    # -  /
                  '2021-12-06_08-54-58',    # - /
                  '2021-12-10_13-32-10',    # 5 \
                  '2021-12-10_13-26-07',    # 6  \
                  '2021-12-10_13-17-29',    # 7   > Session with normal TEB planner
                  '2021-12-10_13-06-09',    # 8  /
                  '2021-12-10_12-53-37',    # - /
                  '2021-12-13_18-16-27',    # - \
                  '2021-12-13_18-22-11',    # -  \
                  '2021-12-15_19-09-57',    # -   > Session with normal TEB planner Tour A and B
                  '2021-12-15_19-13-03']    # -  /

    train_days += ['2022-01-18_10-38-28',   # 14 \
                   '2022-01-18_10-42-54',   # -   \
                   '2022-01-18_10-47-07',   # -    \
                   '2022-01-18_10-48-42',   # -     \
                   '2022-01-18_10-53-28',   # -      > Sessions with normal TEB planner on loop_3
                   '2022-01-18_10-58-05',   # -      > Simple scenarios for experiment
                   '2022-01-18_11-02-28',   # 20    /
                   '2022-01-18_11-11-03',   # -    /
                   '2022-01-18_11-15-40',   # -   /
                   '2022-01-18_11-20-21']   # -  /
                   

    train_days += ['2022-02-25_18-19-12',   # 24 \
                   '2022-02-25_18-24-30',   # -   > Face to face scenario on (loop_2)
                   '2022-02-25_18-29-18']   # -  /

    train_days += ['2022-03-01_22-01-13',   # 27 \
                   '2022-03-01_22-06-28',   # -   > More data (loop_2inv and loop8)
                   '2022-03-01_22-19-41',   # -   > face to face and crossings
                   '2022-03-01_22-25-19']   # -  /

    map_i = 3
    refine_i = np.array([0, 6, 7, 8, 14, 20, 24, 27])
    train_i = np.arange(len(train_days))[5:]
         
    map_day = train_days[map_i]
    refine_days = np.array(train_days)[refine_i]
    train_days = np.sort(np.array(train_days)[train_i])

    return dataset_path, map_day, refine_days, train_days


def Myhal1_sessions():

    # Mapping sessions
    # ****************

    dataset_path = '../Data/Myhal1'
    train_days = ['2022-03-08_12-34-12',  # - \
                  '2022-03-08_12-51-26',  # -  \  First runs with controller for mapping of the environment
                  '2022-03-08_12-52-56',  # -  /  Include refinement runs for table and elevator doors
                  '2022-03-08_14-24-09']  # - /

    # Actual sessions for training
    # ****************************

    # Tuesday 4pm
    train_days += ['2022-03-08_21-02-28',   # ff1 >    Good    (lots of people moving)
                   '2022-03-08_21-08-04',   # ff1 >   Medium   (Some people just not moving)
                   '2022-03-08_21-14-04']   # ff1 >    Bad     (Even fewer people and not moving either)

    # Tuesday 5pm.
    train_days += ['2022-03-08_22-19-08',   # ff2 >    Good    (Fair amount of people, some around the robot in the beginning)
                   '2022-03-08_22-24-22',   # ff1 >    Good    (Not much, but a group of people getting out of the elevator)
                   '2022-03-08_22-28-23']   # ff2 >    Bad     (No one in this run)

    # Wednesday 11am.
    train_days += ['2022-03-09_15-55-10',   # ff1 >    Good    (Fair amount of people, all walking)
                   '2022-03-09_15-58-56',   # ff2 > Borderline (Some people but not many)
                   '2022-03-09_16-03-21',   # ff1 >    Good    (Many people, moving still or getting to see the robot)
                   '2022-03-09_16-07-11']   # ff1 >    Good    (Many people)

    # TODO:
    #       > Inspect each new run:
    #           a. Mapping problem?
    #           b. Annotation good?
    #           c. Rate (Interesting stuff happening)
    #
    # TODO: PB 1: Epaisseur du sol corrigee
    #               > Verif alignemnt du sol a cause des relections
    #               > Notamment detection du sol dans les frames lidar a cause des reflections
    #
    #       PB 2: ATTENTION sur Idefix, changer le run sogm car record met du temps a s'initialiser
    #
    #
    # TODO: Other
    #       > Simulation thing Tim talked about?
    #       > ICRA video
    #       > sur Idefix, larger global map extension radius, higher refresh rate, and use classified frame for global map
    #       > TEB convergence speed is not very fast... Maybe work on this
    #       > Reduce turning radius

    map_i = 0
    refine_i = np.arange(len(train_days))[1:4]
    train_i = np.arange(len(train_days))[4:]
         
    map_day = train_days[map_i]
    refine_days = np.array(train_days)[refine_i]
    train_days = np.sort(np.array(train_days)[train_i])

    return dataset_path, map_day, refine_days, train_days


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':
        

    # Choose sessions
    # ***************

    # # Sessions from Myhal fifth floor
    # dataset_path, map_day, refine_days, train_days = Myhal5_sessions()

    # Sessions from Myhal first floor
    dataset_path, map_day, refine_days, train_days = Myhal1_sessions()


    # Start Annotation
    # ****************

    # Main function to annotate everything
    t0 = time.time()
    main(dataset_path, map_day, refine_days, train_days)
    t1 = time.time()


    # Inspection
    # **********

    # Do it only if we are not in a nohup mode and annotation has already been done before
    if len(sys.argv) < 2 and (t1 - t0 < 60):
            
        # Feel free to delete bad runs after inspection
        runs_to_erase = ['XXXX-XX-XX_XX-XX-XX',
                         'XXXX-XX-XX_XX-XX-XX']
        erased_runs = erase_runs(dataset_path, runs_to_erase)

        # Inspect runs
        if not erased_runs:
            inspect_sogm_sessions(dataset_path, map_day, train_days)
