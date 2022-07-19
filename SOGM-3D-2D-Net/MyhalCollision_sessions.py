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


def old_A_sessions():

    # Mapping sessions
    # ****************
    
    # Notes for myself: number of dynamic people emcoutered in each run
    #
    # '2021-12-10_12-53-37',    1 driver / 3 people
    # '2021-12-10_13-06-09',    1 person
    # '2021-12-10_13-17-29',    Nobody
    # '2021-12-10_13-26-07',    1 follower / 9 people or more
    # '2021-12-10_13-32-10',    9 people or more (groups), '']
    # '2021-12-13_18-16-27',    1 blocker
    # '2021-12-13_18-22-11',    1 blocker / 3 people
    # '2021-12-15_19-09-57',    4 people
    # '2021-12-15_19-13-03']    3 people
    #
    
    dataset_path = '../Data/UTIn3D_A'
    sessions_and_comments = [['2021-12-06_08-12-39', ' mapping > '],    # - \
                             ['2021-12-06_08-38-16', ' mapping > '],    # -  \
                             ['2021-12-06_08-44-07', ' mapping > '],    # -   > First runs with controller for mapping of the environment
                             ['2021-12-06_08-51-29', ' mapping > '],    # -  /
                             ['2021-12-06_08-54-58', ' mapping > '],    # - /
                             ['2021-12-10_13-32-10', ' train > '],      # 5 \
                             ['2021-12-10_13-26-07', ' train > '],      # 6  \
                             ['2021-12-10_13-17-29', '  val  > '],      # 7   > Session with normal TEB planner
                             ['2021-12-10_13-06-09', '  val  > '],      # 8  /
                             ['2021-12-10_12-53-37', ' train > '],      # - /
                             ['2021-12-13_18-16-27', '  val  > '],      # - \
                             ['2021-12-13_18-22-11', ' train > '],      # -  \
                             ['2021-12-15_19-09-57', '  val  > '],      # -   > Session with normal TEB planner Tour A and B
                             ['2021-12-15_19-13-03', ' train > ']]      # -  /

    sessions_and_comments += [['2022-01-18_10-38-28', ' train > '],     # 14 \
                              ['2022-01-18_10-42-54', ' train > '],     # -   \
                              ['2022-01-18_10-47-07', '  val  > '],     # -    \
                              ['2022-01-18_10-48-42', ' train > '],     # -     \
                              ['2022-01-18_10-53-28', ' train > '],     # -      > Sessions with normal TEB planner on loop_3
                              ['2022-01-18_10-58-05', ' train > '],     # -      > Simple scenarios for experiment
                              ['2022-01-18_11-02-28', ' train > '],     # 20    /
                              ['2022-01-18_11-11-03', ' train > '],     # -    /
                              ['2022-01-18_11-15-40', '  val  > '],     # -   /
                              ['2022-01-18_11-20-21', '  val  > ']]     # -  /

    sessions_and_comments += [['2022-02-25_18-19-12', ' train > '],     # 24 \
                              ['2022-02-25_18-24-30', ' train > '],     # -   > Face to face scenario on (loop_2)
                              ['2022-02-25_18-29-18', ' train > ']]     # -  /

    sessions = [s_c[0] for s_c in sessions_and_comments]
    comments = [s_c[1] for s_c in sessions_and_comments]

    map_i = 3
    refine_i = np.array([0, 6, 7, 8])
    train_i = np.arange(len(sessions))[5:]

    map_day = sessions[map_i]
    refine_sessions = np.array(sessions)[refine_i]
    train_sessions = np.array(sessions)[train_i]
    train_comments = np.array(comments)[train_i]

    train_order = np.argsort(train_sessions)
    train_sessions = train_sessions[train_order]
    train_comments = train_comments[train_order]

    return dataset_path, map_day, refine_sessions, train_sessions, train_comments


def old_A_sessions_v2():

    # Mapping sessions
    # ****************
    
    # Notes for myself: number of dynamic people emcoutered in each run
    #
    # '2021-12-10_12-53-37',    1 driver / 3 people
    # '2021-12-10_13-06-09',    1 person
    # '2021-12-10_13-17-29',    Nobody
    # '2021-12-10_13-26-07',    1 follower / 9 people or more
    # '2021-12-10_13-32-10',    9 people or more (groups), '']
    # '2021-12-13_18-16-27',    1 blocker
    # '2021-12-13_18-22-11',    1 blocker / 3 people
    # '2021-12-15_19-09-57',    4 people
    # '2021-12-15_19-13-03']    3 people
    #
    
    dataset_path = '../Data/UTIn3D_A'
    sessions_and_comments = [['2021-12-06_08-12-39', ' mapping > '],    # - \
                             ['2021-12-06_08-38-16', ' mapping > '],    # -  \
                             ['2021-12-06_08-44-07', ' mapping > '],    # -   > First runs with controller for mapping of the environment
                             ['2021-12-06_08-51-29', ' mapping > '],    # -  /
                             ['2021-12-06_08-54-58', ' mapping > '],    # - /
                             ['2021-12-10_13-32-10', ' train > '],      # 5 \
                             ['2021-12-10_13-26-07', ' train > '],      # 6  \
                             ['2021-12-10_13-17-29', '  val  > '],      # 7   > Session with normal TEB planner
                             ['2021-12-10_13-06-09', '  val  > '],      # 8  /
                             ['2021-12-10_12-53-37', ' train > '],      # - /
                             ['2021-12-13_18-16-27', '  val  > '],      # - \
                             ['2021-12-13_18-22-11', ' train > '],      # -  \
                             ['2021-12-15_19-09-57', '  val  > '],      # -   > Session with normal TEB planner Tour A and B
                             ['2021-12-15_19-13-03', ' train > ']]      # -  /

    sessions_and_comments += [['2022-01-18_10-38-28', ' train > '],     # 14 \
                              ['2022-01-18_10-42-54', ' train > '],     # -   \
                              ['2022-01-18_10-47-07', '  val  > '],     # -    \
                              ['2022-01-18_10-48-42', ' train > '],     # -     \
                              ['2022-01-18_10-53-28', ' train > '],     # -      > Sessions with normal TEB planner on loop_3
                              ['2022-01-18_10-58-05', ' train > '],     # -      > Simple scenarios for experiment
                              ['2022-01-18_11-02-28', ' train > '],     # 20    /
                              ['2022-01-18_11-11-03', ' train > '],     # -    /
                              ['2022-01-18_11-15-40', '  val  > '],     # -   /
                              ['2022-01-18_11-20-21', '  val  > ']]     # -  /

    sessions_and_comments += [['2022-02-25_18-19-12', ' train > '],     # 24 \
                              ['2022-02-25_18-24-30', ' train > '],     # -   > Face to face scenario on (loop_2)
                              ['2022-02-25_18-29-18', ' train > ']]     # -  /

    sessions_and_comments += [['2022-05-31_14-45-53', ' train > '],
                              ['2022-05-31_16-25-23', ' train > '],
                              ['2022-05-31_16-29-56', ' train > '],
                              ['2022-05-31_16-35-32', ' train > '],
                              ['2022-05-31_16-38-34', ' train > '],
                              ['2022-05-31_16-56-04', ' train > '],
                              ['2022-05-31_18-33-02', ' train > '],
                              ['2022-05-31_18-36-31', ' train > '],
                              ['2022-05-31_19-34-18', ' train > '],
                              ['2022-05-31_19-37-08', ' train > '],
                              ['2022-05-31_19-40-52', ' train > '],
                              ['2022-05-31_19-44-52', ' train > '],
                              ['2022-05-31_19-47-52', ' train > '],
                              ['2022-05-31_19-51-14', ' train > '], ]

    sessions = [s_c[0] for s_c in sessions_and_comments]
    comments = [s_c[1] for s_c in sessions_and_comments]

    map_i = 3
    refine_i = np.array([0, 6, 7, 8])
    train_i = np.arange(len(sessions))[5:]

    map_day = sessions[map_i]
    refine_sessions = np.array(sessions)[refine_i]
    train_sessions = np.array(sessions)[train_i]
    train_comments = np.array(comments)[train_i]

    train_order = np.argsort(train_sessions)
    train_sessions = train_sessions[train_order]
    train_comments = train_comments[train_order]

    return dataset_path, map_day, refine_sessions, train_sessions, train_comments


def UTIn3D_A_sessions():

    # Mapping sessions
    # ****************
    
    # Notes for myself: number of dynamic people emcoutered in each run
    #
    # '2021-12-10_12-53-37',    1 driver / 3 people
    # '2021-12-10_13-06-09',    1 person
    # '2021-12-10_13-17-29',    Nobody
    # '2021-12-10_13-26-07',    1 follower / 9 people or more
    # '2021-12-10_13-32-10',    9 people or more (groups), '']
    # '2021-12-13_18-16-27',    1 blocker
    # '2021-12-13_18-22-11',    1 blocker / 3 people
    # '2021-12-15_19-09-57',    4 people
    # '2021-12-15_19-13-03']    3 people
    
    dataset_path = '../Data/UTIn3D_A'
    sessions_and_comments = [['2021-12-06_08-12-39', ' mapping > '],    # - \
                             ['2021-12-06_08-38-16', ' mapping > '],    # -  \
                             ['2021-12-06_08-44-07', ' mapping > '],    # -   > First runs with controller for mapping of the environment
                             ['2021-12-06_08-51-29', ' mapping > '],    # -  /
                             ['2021-12-06_08-54-58', ' mapping > '],    # - /
                             ['2021-12-10_13-32-10', ' train > '],      # 5 \
                             ['2021-12-10_13-26-07', ' train > '],      # 6  \
                             ['2021-12-10_13-17-29', '  val  > '],      # 7   > Session with normal TEB planner
                             ['2021-12-10_13-06-09', '  val  > '],      # 8  /
                             ['2021-12-10_12-53-37', ' train > '],      # - /
                             ['2021-12-13_18-16-27', ' train > '],      # - \
                             ['2021-12-13_18-22-11', ' train > '],      # -  \
                             ['2021-12-15_19-09-57', ' train > '],      # -   > Session with normal TEB planner Tour A and B
                             ['2021-12-15_19-13-03', ' train > ']]      # -  /

    sessions_and_comments += [['2022-01-18_10-38-28', ' train > '],     # 14 \
                              ['2022-01-18_10-42-54', ' train > '],     # -   \
                              ['2022-01-18_10-47-07', ' train > '],     # -    \
                              ['2022-01-18_10-48-42', ' train > '],     # -     \
                              ['2022-01-18_10-53-28', ' train > '],     # -      > Sessions with normal TEB planner on loop_3
                              ['2022-01-18_10-58-05', ' train > '],     # -      > Simple scenarios for experiment
                              ['2022-01-18_11-02-28', ' train > '],     # 20    /
                              ['2022-01-18_11-11-03', ' train > '],     # -    /
                              ['2022-01-18_11-15-40', ' train > '],     # -   /
                              ['2022-01-18_11-20-21', ' train > ']]     # -  /

    sessions_and_comments += [['2022-02-25_18-19-12', ' train > '],     # 24 \
                              ['2022-02-25_18-24-30', ' train > '],     # -   > Face to face scenario on (loop_2)
                              ['2022-02-25_18-29-18', ' train > ']]     # -  /

    sessions_and_comments += [['2022-03-01_22-01-13', ' train > '],     # 27 \
                              ['2022-03-01_22-06-28', ' train > '],     # -   > More data (loop_2inv and loop8)
                              ['2022-03-01_22-25-19', ' train > ']]     # -  /

    sessions_and_comments += [['2022-05-20_11-46-11', ' mapping > '],   # 30    > Refinement run
                              ['2022-05-20_12-47-48', '  val  > '],     # -     > movers moving tables
                              ['2022-05-20_12-54-23', ' train > '],     # -     > movers moving tables
                              ['2022-05-20_12-58-26', ' train > '],     # -     > movers moving tables
                              ['2022-05-20_13-04-19', ' mapping > ']]   # -     > Refinement run

    sessions = [s_c[0] for s_c in sessions_and_comments]
    comments = [s_c[1] for s_c in sessions_and_comments]

    map_i = 3
    refine_i = np.array([0, 6, 7, 8, 30, 34])
    train_i = np.arange(len(sessions))[5:-5]
    train_i = np.hstack((train_i, np.array([31, 32, 33])))

    map_day = sessions[map_i]
    refine_sessions = np.array(sessions)[refine_i]
    train_sessions = np.array(sessions)[train_i]
    train_comments = np.array(comments)[train_i]

    train_order = np.argsort(train_sessions)
    train_sessions = train_sessions[train_order]
    train_comments = train_comments[train_order]




    return dataset_path, map_day, refine_sessions, train_sessions, train_comments


def UTIn3D_A_sessions_v2():

    # Mapping sessions
    # ****************
    
    # Notes for myself: number of dynamic people emcoutered in each run
    #
    # '2021-12-10_12-53-37',    1 driver / 3 people
    # '2021-12-10_13-06-09',    1 person
    # '2021-12-10_13-17-29',    Nobody
    # '2021-12-10_13-26-07',    1 follower / 9 people or more
    # '2021-12-10_13-32-10',    9 people or more (groups), '']
    # '2021-12-13_18-16-27',    1 blocker
    # '2021-12-13_18-22-11',    1 blocker / 3 people
    # '2021-12-15_19-09-57',    4 people
    # '2021-12-15_19-13-03']    3 people

    # batch 1: 9 sessions
    dataset_path = '../Data/UTIn3D_A'
    sessions_and_comments = [['2021-12-06_08-12-39', ' mapping > '],    # - \
                             ['2021-12-06_08-38-16', ' mapping > '],    # -  \
                             ['2021-12-06_08-44-07', ' mapping > '],    # -   > First runs with controller for mapping of the environment
                             ['2021-12-06_08-51-29', ' mapping > '],    # -  /
                             ['2021-12-06_08-54-58', ' mapping > '],    # - /
                             ['2021-12-10_13-32-10', ' train > '],      # 5 \
                             ['2021-12-10_13-26-07', ' train > '],      # 6  \
                             ['2021-12-10_13-17-29', '  val  > '],      # 7   > Session with normal TEB planner
                             ['2021-12-10_13-06-09', '  val  > '],      # 8  /
                             ['2021-12-10_12-53-37', ' train > '],      # - /
                             ['2021-12-13_18-16-27', ' train > '],      # - \
                             ['2021-12-13_18-22-11', ' train > '],      # -  \
                             ['2021-12-15_19-09-57', ' train > '],      # -   > Session with normal TEB planner Tour A and B
                             ['2021-12-15_19-13-03', ' train > ']]      # -  /

    # batch 2: 10 sessions
    sessions_and_comments += [['2022-01-18_10-38-28', ' train > '],     # 14 \
                              ['2022-01-18_10-42-54', ' train > '],     # -   \
                              ['2022-01-18_10-47-07', ' train > '],     # -    \
                              ['2022-01-18_10-48-42', ' train > '],     # -     \
                              ['2022-01-18_10-53-28', ' train > '],     # -      > Sessions with normal TEB planner on loop_3
                              ['2022-01-18_10-58-05', ' train > '],     # -      > Simple scenarios for experiment
                              ['2022-01-18_11-02-28', ' train > '],     # 20    /
                              ['2022-01-18_11-11-03', ' train > '],     # -    /
                              ['2022-01-18_11-15-40', ' train > '],     # -   /
                              ['2022-01-18_11-20-21', ' train > ']]     # -  /

    # batch 3: 6 sessions
    sessions_and_comments += [['2022-02-25_18-19-12', ' train > '],     # 24 \
                              ['2022-02-25_18-24-30', ' train > '],     # -   > Face to face scenario on (loop_2)
                              ['2022-02-25_18-29-18', ' train > ']]     # -  /

    sessions_and_comments += [['2022-03-01_22-01-13', ' train > '],     # 27 \
                              ['2022-03-01_22-06-28', ' train > '],     # -   > More data (loop_2inv and loop8)
                              ['2022-03-01_22-25-19', ' train > ']]     # -  /

    # batch 4: 15 sessions
    sessions_and_comments += [['2022-05-20_11-46-11', ' mapping > '],   # 30    > Refinement run
                              ['2022-05-20_12-47-48', '  val  > '],     # -     > movers moving tables
                              ['2022-05-20_12-54-23', ' train > '],     # -     > movers moving tables
                              ['2022-05-20_12-58-26', ' train > '],     # -     > movers moving tables
                              ['2022-05-20_13-04-19', ' mapping > ']]   # -     > Refinement run

    sessions_and_comments += [['2022-05-31_14-45-53', ' train > '],     #
                              ['2022-05-31_16-25-23', ' train > '],     #
                              ['2022-05-31_16-29-56', ' train > '],     #
                              ['2022-05-31_16-35-32', ' train > '],     #
                              ['2022-05-31_16-38-34', ' train > '],     #
                              ['2022-05-31_18-33-02', ' train > '],     # Conference collected data
                              ['2022-05-31_19-34-18', '  val  > '],     # 
                              ['2022-05-31_19-37-08', ' train > '],     #
                              ['2022-05-31_19-40-52', '  val  > '],     #
                              ['2022-05-31_19-44-52', ' train > '],     #
                              ['2022-05-31_19-47-52', ' train > '],     #
                              ['2022-05-31_19-51-14', ' train > '], ]   #

    # # Test sessions for controleld exp only comment this otherwise
    # sessions_and_comments += [['2022-06-01_16-28-54', '  val  > Run made by Catherine, not much happening'],        #
    #                           ['2022-06-01_18-11-37', '  val  > Controlled comparison:  AI-CRV-4 / SOGM ( ok )'],   #
    #                           ['2022-06-01_18-15-07', '  val  > Controlled comparison:  AI-CRV-4 / TEB  ( ok )'],   #
    #                           ['2022-06-01_18-20-40', '  val  > Controlled comparison:  AI-CRV-4 / SOGM ( ok )'],   # Controlled comparison
    #                           ['2022-06-01_18-23-28', '  val  > Controlled comparison:  AI-CRV-4 / TEB  ( ok )'],   # Conference collected data
    #                           ['2022-06-01_20-36-03', '  val  > Controlled comparison:  AI-CRV-4 / no_t ( ok )'],   # 
    #                           ['2022-06-01_20-42-11', '  val  > Controlled comparison:  AI-CRV-4 / no_t ( ok )']]   #


    sessions = [s_c[0] for s_c in sessions_and_comments]
    comments = [s_c[1] for s_c in sessions_and_comments]

    map_i = 3
    refine_i = np.array([0, 6, 7, 8, 30, 34])
    train_i = np.arange(len(sessions))[5:30]
    train_i = np.hstack((train_i, np.array([31, 32, 33])))
    train_i = np.hstack((train_i, np.arange(len(sessions))[35:]))

    map_day = sessions[map_i]
    refine_sessions = np.array(sessions)[refine_i]
    train_sessions = np.array(sessions)[train_i]
    train_comments = np.array(comments)[train_i]

    train_order = np.argsort(train_sessions)
    train_sessions = train_sessions[train_order]
    train_comments = train_comments[train_order]

    return dataset_path, map_day, refine_sessions, train_sessions, train_comments


def UTIn3D_H_sessions():

    # Mapping sessions
    # ****************

    dataset_path = '../Data/UTIn3D_H'
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
    sessions_and_comments += [['2022-03-28_14-53-33', 'ff1  val  > Borderline (Only two or three persons)'],
                              ['2022-03-28_14-57-17', 'ff1 train >    Good    (Some people ok)'],
                              ['2022-03-28_15-00-42', 'ff1 train >    Good    (Good amount of people)'],
                              ['2022-03-28_15-04-24', 'ff1 train >    Good    (Good amount of people)'],
                              ['2022-03-28_16-56-52', 'ff1  val  >     Bad    (no movement)'],
                              ['2022-03-28_17-03-29', 'ff1 train >    Good    (Mega crowd)'],
                              ['2022-03-28_17-07-19', 'ff1 train >    Good    (Some people)'],
                              ['2022-03-28_17-10-13', 'ff1 train >    Good    (Good amount of people)'],
                              ['2022-03-28_21-57-36', 'ff1 train >   Medium   (Some people + dancers)'],
                              ['2022-03-28_22-02-15', 'ff1 train >    Good    (Some people)']]

    # Monday 10h/12h/17h.
    sessions_and_comments += [['2022-04-01_14-00-06', 'ff1  val  >    Good    (Some people)'],
                              ['2022-04-01_14-03-50', 'ff1 train >    Good    (Many people)'],
                              ['2022-04-01_14-53-42', 'ff1 train > Borderline (Some people and me passing in front)'],
                              ['2022-04-01_14-57-35', 'ff1  val  > Borderline (just a group of 4)'],
                              ['2022-04-01_15-01-18', 'ff1 train >    Good    (Mega crowd)'],
                              ['2022-04-01_15-06-55', 'ff1  val  >    Good    (Mega crowd)'],
                              ['2022-04-01_15-11-29', 'ff1  val  >    Good    (Many people)']]
                              

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


def Test_A_sessions():

    # Mapping sessions
    # ****************

    dataset_path = '../Data/new_UTIn3D_A'
    sessions_and_comments = [['2022-05-17_19-05-27', 'mapping']]  # - /

    # Actual sessions for training
    # ****************************

    # The list contains tuple (name, comments), so that we do not reinspect things we already did

    # # Tuesday 4pm
    sessions_and_comments += [['2022-03-01_22-06-28', 'old run for debug purposes'],
                              ['2022-03-01_22-25-19', 'old run for debug purposes']]
                          


    # Procesing
    # *********
    
    sessions_and_comments = [(s, c) for s, c in sessions_and_comments if 'ERASED' not in c]

    sessions = [s_c[0] for s_c in sessions_and_comments]
    comments = [s_c[1] for s_c in sessions_and_comments]


    # Setup your sessions here
    map_i = 0
    refine_i = np.arange(len(sessions))[1:2]
    train_i = np.arange(len(sessions))[2:]


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

