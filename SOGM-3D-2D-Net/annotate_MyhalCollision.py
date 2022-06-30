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
from os import makedirs
from os.path import exists, join
from gt_annotation_video import get_videos, inspect_sogm_sessions, show_seq_slider
from datasets.MyhalCollision import MyhalCollisionSlam
from slam.PointMapSLAM import annotation_process
import numpy as np
import sys
import time
import os
import shutil
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)

# Dataset
from MyhalCollision_sessions import Myhal1_sessions, Myhal5_sessions, Myhal55_sessions, Myhal5_sessions_v2


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Function
#       \***************/
#


def main(dataset_path, map_day, refine_sessions, train_sessions, force_redo=True):

    # Check if we need to redo annotation (only if there is no collison folder)
    redo_annot = False
    for day in train_sessions:
        annot_path = join(dataset_path, 'collisions', day)
        if not exists(annot_path):
            redo_annot = True
            break

    # Forced redo
    redo_annot = redo_annot or force_redo

    if redo_annot:

        # Initiate dataset
        slam_dataset = MyhalCollisionSlam(day_list=train_sessions, map_day=map_day, dataset_path=dataset_path)

        ####################
        # Map initialization
        ####################

        slam_dataset.init_map()

        ################
        # Map refinement
        ################

        slam_dataset.refine_map(refine_sessions)

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
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    # Choose sessions
    # ***************

    # # Sessions from Myhal fifth floor
    # dataset_path, map_day, refine_sessions, train_sessions = Myhal5_sessions()

    # Sessions from Myhal first floor
    # dataset_path, map_day, refine_sessions, train_sessions, train_comments = Myhal5_sessions_v2()
    dataset_path, map_day, refine_sessions, train_sessions, train_comments = Myhal1_sessions()

    # Start Annotation
    # ****************

    # Main function to annotate everything
    t0 = time.time()
    main(dataset_path, map_day, refine_sessions, train_sessions)
    t1 = time.time()

    # Inspection
    # **********

    # Do it only if we are not in a nohup mode and annotation has already been done before
    if len(sys.argv) < 2 and (t1 - t0 < 60):

        # Feel free to delete bad runs after inspection
        runs_to_erase = ['XXXXXXXXXXXXXXXXXXX',
                         'XXXXXXXXXXXXXXXXXXX',
                         'XXXXXXXXXXXXXXXXXXX']
        erased_runs = erase_runs(dataset_path, runs_to_erase)

        # Inspect runs
        if not erased_runs:
            # Only inspect sessions with empty comments

            inspect_mask = np.array([tc.endswith('()') for tc in train_comments], dtype=bool)
            inspect_sessions = train_sessions[inspect_mask]
            inspect_comments = train_comments[inspect_mask]

            if len(inspect_sessions) > 0:
                inspect_sogm_sessions(dataset_path, map_day, inspect_sessions, inspect_comments)
            else:
                inspect_sogm_sessions(dataset_path, map_day, train_sessions, train_comments)
