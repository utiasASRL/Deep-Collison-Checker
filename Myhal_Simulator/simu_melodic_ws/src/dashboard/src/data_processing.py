#!/usr/bin/env python

import sys
import os
import numpy as np
import time as RealTime
import pickle
import json
import subprocess
import rosbag
import plyfile as ply
import shutil
from utilities import bag_tools as bt
from utilities import math_utilities as mu
from utilities import plot_utilities as pu


if __name__ == "__main__":
    
    ######
    # Init
    ######

    start_time = RealTime.time()
    
    username = os.environ['USER']
    if (len(sys.argv)-1 < 2):
        print "ERROR: must input filename and filter status"
        exit()
    filename = sys.argv[1]
    filter_status = True if (sys.argv[2] == "true") else False
    print "Processing data for file", filename

    path = "/home/" + username + "/Myhal_Simulation/simulated_runs/" + filename + "/"
    if (not os.path.isdir(path)):
        print 'File ' + path + ' has been deleted, aborting data processing'
        exit()

    logs_path = path + "logs-" + filename + "/"

    # load in meta data

    if (not os.path.exists(logs_path + "meta.json")):
        print 'meta.json is missing, aborting data processing'
        exit()

    file = open(logs_path + "meta.json", "r")
    meta_data = json.load(file)

    localization_test = True if (meta_data['localization_test'] == 'true') else False

    try:
        bag = rosbag.Bag(path + "raw_data.bag")
    except:
        try:
            bag = rosbag.Bag(path + "localization_test.bag")
            localization_test = True
        except:
            print "ERROR: invalid filename"
            exit()
        
    pickle_dict = {}
    
    ##############
    # Lidar frames
    ##############

    print "Reading lidar frames"

    # read in lidar frames
    frames = bt.read_pointcloud_frames("/velodyne_points", bag)
    dir_name = "sim_frames"
    if(len(frames)):
        pickle_dict['lidar_frames'] = frames
        if (not os.path.isdir(path + dir_name)):
            try:
                os.mkdir(path + dir_name)
            except OSError:
                print ("Creation of the classifed_frames directory failed")
                exit()

        print "Writing",len(frames),"lidar frames"

    # write lidar frames to .ply files 
    for frame in frames:
        time, points = frame
        dtype = [("x", np.float32), ("y",np.float32), ("z",np.float32), ("cat", np.int32)]
        cat = points['intensity'] if (filter_status) else ([-1]*len(points['intensity']))
        arr = np.array([points['x'], points['y'], points['z'], points['intensity']])
        arr = np.core.records.fromarrays(arr, dtype = dtype)
        el = ply.PlyElement.describe(arr, "vertex")
        time = time.to_sec()
        time = "{:.6f}".format(time)
        
        ply.PlyData([el]).write(path +  dir_name + "/" + time + ".ply")

    
    # read in classified frames
    frames_bis = bt.read_pointcloud_frames("/classified_points", bag)
    dir_name = "classified_frames"
    if(len(frames_bis)):
        pickle_dict['classified_frames'] = frames_bis
        if (not os.path.isdir(path + dir_name)):
            try:
                os.mkdir(path + dir_name)
            except OSError:
                print ("Creation of the classifed_frames directory failed")
                exit()

        print "Writing",len(frames_bis),"lidar frames"

    # write classified frames to .ply files 
    for frame in frames_bis:
        time, points = frame
        dtype = [("x", np.float32), ("y",np.float32), ("z",np.float32), ("cat", np.int32)]
        cat = points['intensity'] if (filter_status) else ([-1]*len(points['intensity']))
        arr = np.array([points['x'], points['y'], points['z'], points['intensity']])
        arr = np.core.records.fromarrays(arr, dtype = dtype)
        el = ply.PlyElement.describe(arr, "vertex")
        time = time.to_sec()
        time = "{:.6f}".format(time)
        
        ply.PlyData([el]).write(path +  dir_name + "/" + time + ".ply")
    
    ####################
    # Collider / planner
    ####################

    collider_preds = bt.read_collider_preds("/plan_costmap_3D", bag)

    print "Saving collider to", logs_path + "collider_data.pickle"
    with open(logs_path + 'collider_data.pickle', 'wb') as handle:
        pickle.dump(collider_preds, handle)
 
    teb_local_plans = bt.read_local_plans("/move_base/TebLocalPlannerROS/local_plan", bag)

    print "Saving teb plans to", logs_path + "teb_local_plans.pickle"
    with open(logs_path + 'teb_local_plans.pickle', 'wb') as handle:
        pickle.dump(teb_local_plans, handle)

    ##############
    # Trajectories
    ##############

    print "Reading trajectories"

    # read in ground truth pose
    gt_traj = bt.read_nav_odometry("/ground_truth/state",bag)
    
    #read in optimal traj
    optimal_traj = bt.read_nav_odometry("/optimal_path",bag, False)
    if (len(optimal_traj)):
        pickle_dict['optimal_traj'] = bt.trajectory_to_array(optimal_traj)

    #read in tour waypoints
    waypoints = bt.read_nav_odometry("/tour_data", bag, False)
    if (len(waypoints)):
        pickle_dict['waypoints'] = bt.trajectory_to_array(waypoints)

    #read in move_base results
    results = bt.read_action_result('/move_base/result', bag)
    if (len(results)):
        pickle_dict['action_results'] = results

    # output ground truth pose to .ply file
    el = ply.PlyElement.describe(bt.trajectory_to_array(gt_traj), "trajectory")
    ply.PlyData([el]).write(path + "/gt_pose.ply")

    # read in amcl poses if they exist
    amcl_status = bool(bt.num_messages("/amcl_pose", bag))
    
    map_frame = "hugues_map" if (localization_test) else "map"

    odom_to_base = bt.read_tf_transform("odom","base_link", bag)
    map_to_odom = bt.read_tf_transform(map_frame,"odom", bag)
    odom_to_base = bt.transforms_to_trajectory(odom_to_base)
    tf_traj = mu.transform_trajectory(odom_to_base, map_to_odom)

    # interplote tf_traj to the times of gt_traj
    
    tf_traj = mu.get_interpolations(gt_traj, tf_traj, False)
    
    #gt_traj = mu.get_interpolations(tf_traj, gt_traj, False)
    
    pickle_dict['gt_traj'] = bt.trajectory_to_array(gt_traj)
    
    if (amcl_status):
        print "Saving amcl_traj"
        pickle_dict['amcl_traj'] = bt.trajectory_to_array(tf_traj)
        
    else:
        print "Saving gmapping_traj"
        pickle_dict['gmapping_traj'] = bt.trajectory_to_array(tf_traj)

    #pickle_dict['loc_traj'] = bt.trajectory_to_array(tf_traj)
    
    # output loc pose to .ply file
    el = ply.PlyElement.describe(bt.trajectory_to_array(tf_traj), "trajectory")
    ply.PlyData([el]).write(path + "/loc_pose.ply")
    

    print "Dumping data to", logs_path + "processed_data.pickle"
    with open(logs_path + 'processed_data.pickle', 'wb') as handle:
        pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #######
    # Video
    #######

    # TODO: convert image files to .mp4 and save 

    duration = bt.bag_metadata(bag)['duration']

    vid_path = "/home/" + username + "/Myhal_Simulation/simulated_runs/" + filename + "/logs-" + filename + "/videos/"
    vid_dirs = os.listdir(vid_path) if os.path.isdir(vid_path) else []

    count = 0

    for dir in vid_dirs:

        try:
            num_pics = len(os.listdir(vid_path + dir + "/"))
        except:
            continue

        fps = int(num_pics/duration) 

        s_name = dir.split("_")
        mode = ["sentry","hoverer", "stalker"]
        mode = mode[int(s_name[1])] + "_" + str(count) + "_" + str(fps)

        print "Converting " + str(num_pics) +  " .jpg files at " + str(fps) + " fps to create " + mode + ".mp4 that is: {:.2f}s long".format(num_pics/float(fps))
        FNULL = open(os.devnull, 'w')
        

        command = 'ffmpeg -r ' + str(fps) + ' -pattern_type glob -i ' + '"'+ vid_path + dir + '/' + dir + '-*.jpg" -c:v libx264 ' + '"' +  vid_path + mode + '.mp4"'

        retcode = subprocess.call(command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
        if (retcode == 0): # a success
            shutil.rmtree(vid_path + dir) 
        else:
            print 'Video creation failed'
        FNULL.close()

        count+=1

    bag.close()
        
    duration = RealTime.time() - start_time
    
    print "Data processed in", "{:.2f}".format(duration) ,"seconds"




