#
#
#      0==============================0
#      |    Deep Collision Checker    |
#      0==============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define the sessions and dataset of MyhalCollision here
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# Common libs
import numpy as np


def Myhal5_sessions():

    # Mapping sessions
    # ****************

    dataset_path = '../Data/RealMyhal'
    train_sessions = ['2021-12-06_08-12-39',    # - \
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

    train_sessions += ['2022-01-18_10-38-28',   # 14 \
                       '2022-01-18_10-42-54',   # -   \
                       '2022-01-18_10-47-07',   # -    \
                       '2022-01-18_10-48-42',   # -     \
                       '2022-01-18_10-53-28',   # -      > Sessions with normal TEB planner on loop_3
                       '2022-01-18_10-58-05',   # -      > Simple scenarios for experiment
                       '2022-01-18_11-02-28',   # 20    /
                       '2022-01-18_11-11-03',   # -    /
                       '2022-01-18_11-15-40',   # -   /
                       '2022-01-18_11-20-21']   # -  /

    train_sessions += ['2022-02-25_18-19-12',   # 24 \
                       '2022-02-25_18-24-30',   # -   > Face to face scenario on (loop_2)
                       '2022-02-25_18-29-18']   # -  /

    train_sessions += ['2022-03-01_22-01-13',   # 27 \
                       '2022-03-01_22-06-28',   # -   > More data (loop_2inv and loop8)
                       '2022-03-01_22-19-41',   # -   > face to face and crossings
                       '2022-03-01_22-25-19']   # -  /

    # Notes for myself: number of dynamic people emcoutered in each run
    #
    # '2021-12-10_12-53-37',    1 driver / 3 people
    # '2021-12-10_13-06-09',    1 person
    # '2021-12-10_13-17-29',    Nobody
    # '2021-12-10_13-26-07',    1 follower / 9 people or more
    # '2021-12-10_13-32-10',    9 people or more (groups)
    # '2021-12-13_18-16-27',    1 blocker
    # '2021-12-13_18-22-11',    1 blocker / 3 people
    # '2021-12-15_19-09-57',    4 people
    # '2021-12-15_19-13-03']    3 people


    map_i = 3
    refine_i = np.array([0, 6, 7, 8, 14, 20, 24, 27])
    train_i = np.arange(len(train_sessions))[5:]

    map_day = train_sessions[map_i]
    refine_sessions = np.array(train_sessions)[refine_i]
    train_sessions = np.sort(np.array(train_sessions)[train_i])

    return dataset_path, map_day, refine_sessions, train_sessions


def Myhal1_sessions():

    # Mapping sessions
    # ****************

    dataset_path = '../Data/Myhal1'
    sessions_and_comments = [['2022-03-08_12-34-12', ''],  # - \
                             ['2022-03-08_12-51-26', ''],  # -  \  First runs with controller for mapping of the environment
                             ['2022-03-08_12-52-56', ''],  # -  /  Include refinement runs for table and elevator doors
                             ['2022-03-08_14-24-09', '']]  # - /

    # Actual sessions for training
    # ****************************

    # The list contains tuple (name, comments), so that we do not reinspect things we already did

    # Tuesday 4pm
    sessions_and_comments += [['2022-03-08_21-02-28', 'ff1 train >    Good    (lots of people moving)'],
                              ['2022-03-08_21-08-04', 'ff1 train >   Medium   (Some people just not moving)']]

    # Tuesday 5pm.
    sessions_and_comments += [['2022-03-08_22-19-08', 'ff2 train >    Good    (Fair amount of people, some around the robot in the beginning)'],
                              ['2022-03-08_22-24-22', 'ff1 train >    Good    (Not much, but a group of people getting out of the elevator)']]

    # Wednesday 11am.
    sessions_and_comments += [['2022-03-09_15-55-10', 'ff1 train >    Good    (Fair amount of people, all walking)'],
                              ['2022-03-09_15-58-56', 'ff2  val  > Borderline (Some people but not many)'],
                              ['2022-03-09_16-03-21', 'ff1  val  >    Good    (Many people, moving still or getting to see the robot)'],
                              ['2022-03-09_16-07-11', 'ff1 train >    Good    (Many people)']]

    # Wednesday 10h/12h/14h/15h.
    sessions_and_comments += [['2022-03-16_16-05-29', 'ff1 train >    Good    (Huge crowd in front of the entrance and the elevator)'],
                              ['2022-03-16_20-05-22', 'ff1 train >    Good    (Huge number of people)'],
                              ['2022-03-16_20-13-08', 'ff2 train >    Good    (Small tour but fair amount of people)'],
                              ['2022-03-16_21-21-35', 'ff2 train >    Good    (Few people but full ff2 tour)'],
                              ['2022-03-16_21-28-09', 'ff1 train > Borderline (Some people but ok)']]

    # Tuesday 9h/10h/11h.
    sessions_and_comments += [['2022-03-22_14-04-53', 'ff1 train >    Good    (Session ended early, but overal nice)'],
                              ['2022-03-22_14-07-26', 'ff1 train >    Good    (Fair amount of people)'],
                              ['2022-03-22_14-12-20', 'ff1  val  >    Good    (not many people)'],
                              ['2022-03-22_15-05-20', 'ff1 train >    Good    (Many people)'],
                              ['2022-03-22_15-09-02', 'ff1 train >    Good    (Fair amount of people)'],
                              ['2022-03-22_15-12-23', 'ff1 train >    Good    (A few people)'],
                              ['2022-03-22_16-04-06', 'ff1 train >    Good    (Mega Busy)'],
                              ['2022-03-22_16-08-09', 'ff1  val  >    Good    (Many people)']]

    # Monday 10h/12h/17h.
    sessions_and_comments += [['2022-03-28_14-53-33', 'ff1 train >    Good    ()'],
                              ['2022-03-28_14-57-17', 'ff1 train >    Good    ()'],
                              ['2022-03-28_15-00-42', 'ff1 train >    Good    ()'],
                              ['2022-03-28_15-04-24', 'ff1 train >    Good    ()'],
                              ['2022-03-28_16-56-52', 'ff1 train >    Good    ()'],
                              ['2022-03-28_16-59-33', 'ff1 train >    Good    ()'],
                              ['2022-03-28_17-03-29', 'ff1 train >    Good    ()'],
                              ['2022-03-28_17-07-19', 'ff1 train >    Good    ()'],
                              ['2022-03-28_17-10-13', 'ff1 train >    Good    ()'],
                              ['2022-03-28_21-57-36', 'ff1 train >    Good    ()'],
                              ['2022-03-28_22-02-15', 'ff1  val  >    Good    ()']]

    # Monday 10h/12h/17h.
    sessions_and_comments += [['2022-04-01_13-04-29', 'ff1 train >    Good    ()'],
                              ['2022-04-01_13-10-27', 'ff1 train >    Good    ()'],
                              ['2022-04-01_14-00-06', 'ff1 train >    Good    ()'],
                              ['2022-04-01_14-03-50', 'ff1 train >    Good    ()'],
                              ['2022-04-01_14-08-19', 'ff1 train >    Good    ()'],
                              ['2022-04-01_14-53-42', 'ff1 train >    Good    ()'],
                              ['2022-04-01_14-57-35', 'ff1 train >    Good    ()'],
                              ['2022-04-01_15-01-18', 'ff1 train >    Good    ()'],
                              ['2022-04-01_15-06-55', 'ff1 train >    Good    ()'],
                              ['2022-04-01_15-11-29', 'ff1 train >    Good    ()']]
                              

    # Procesing
    # *********
    
    sessions_and_comments = [(s, c) for s, c in sessions_and_comments if 'ERASED' not in c]

    sessions = [s_c[0] for s_c in sessions_and_comments]
    comments = [s_c[1] for s_c in sessions_and_comments]

    map_i = 0
    refine_i = np.arange(len(sessions))[1:4]
    train_i = np.arange(len(sessions))[4:]

    map_day = sessions[map_i]
    refine_sessions = np.array(sessions)[refine_i]
    train_sessions = np.array(sessions)[train_i]
    train_comments = np.array(comments)[train_i]

    train_order = np.argsort(train_sessions)
    train_sessions = train_sessions[train_order]
    train_comments = train_comments[train_order]

    # Instuctions to add new data
    # ---------------------------
    #
    #   1. place rosbag is the rosbag folder **Data/Real/rosbags**
    #
    #   2. Process them:
    #       > `./run_in_pytorch.sh -c "./process_rosbags.sh"`
    #
    #   3. Add their name to the session list (use empty model below)
    #
    #        sessions_and_comments += [['XXXXXXXXXXXXXXXXXXX', 'ff1 train >    Good    ()'],
    #                                  ['XXXXXXXXXXXXXXXXXXX', 'ff1 train >    Good    ()'],
    #                                  ['XXXXXXXXXXXXXXXXXXX', 'ff1 train >    Good    ()']]
    #
    #   4. Start annotation:
    #       > `./run_in_pytorch.sh -d -c "python3 annotate_MyhalCollision.py"` (in detach mode)
    #
    #   5. When finished inspect each run by reruning the annotation script:
    #       > `./run_in_pytorch.sh -c "python3 annotate_MyhalCollision.py"`
    #
    #   6. During inspection, add comment and feel free to create video with hotkey *g*
    #
    #   7. When finished, delete runs you did not like with the runs_to_erase variable
    #
    #   8. Delete the rosbags in their folders
    #
    #   9. Add train or val word in the comment (before >) for the training
    #
    #   10. You can now start training:
    #       > `./run_in_pytorch.sh -c "python3 train_MyhalCollision.py"`
    #

    # TODO: Other
    #       > sur Idefix, larger global map extension radius, higher refresh rate, and use classified frame for global map
    #       > TEB convergence speed is not very fast... Maybe work on this
    #
    #       > TODO Install PyVista and use it instead of open3d for videos because it has EDL
    #
    return dataset_path, map_day, refine_sessions, train_sessions, train_comments

