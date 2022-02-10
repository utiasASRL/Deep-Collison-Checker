#!/usr/bin/env python

import matplotlib.pyplot as plt

import sys
import os
import numpy as np
import pickle

import bag_tools as bt
import math_utilities as mu
#import plot_utilities as pu
from scipy.spatial.transform import Rotation as R

def list_distances(traj, min_time = -1, max_time = -1):
	dist_list = [0]
	dist = 0
	prev = traj[0]
	for pose in traj[1:]:
		if (min_time >= 0 and pose['time'] < min_time):
			prev = pose
			continue
		if (max_time >= 0 and pose['time'] > max_time):
			break
		dx = pose['pos_x'] - prev['pos_x']
		dy = pose['pos_y'] - prev['pos_y']
		dist += np.sqrt(dx**2 + dy**2)
		dist_list.append(dist)
		prev = pose

	return (dist_list,dist)

def translation_error(traj, base_traj):
	x_error = traj['pos_x'] - base_traj['pos_x']
	y_error = traj['pos_y'] - base_traj['pos_y']
	error = np.sqrt(x_error*x_error + y_error*y_error)
	return error


def yaw_error(traj, base_traj):

	traj_rots = R.from_quat(np.vstack([traj['rot_x'], traj['rot_y'], traj['rot_z'], traj['rot_w']]).T)
	base_rots = R.from_quat(np.vstack([base_traj['rot_x'], base_traj['rot_y'], base_traj['rot_z'], base_traj['rot_w']]).T)
	#error = np.abs(traj_rots.as_euler('xyz')[:, 2] - base_rots.as_euler('xyz')[:, 2])

	traj_rots = traj_rots.as_dcm()
	base_rots = base_rots.as_dcm()
	base_rots_T = np.transpose(base_rots, (0, 2, 1))

	dR = np.matmul(traj_rots, base_rots_T)
	error = np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0))
	
	# error = []
	# for i in range(len(traj)):
	# 	r1 = R.from_quat([traj[i]['rot_x'], traj[i]['rot_y'], traj[i]['rot_z'], traj[i]['rot_w']])
	# 	r2 = R.from_quat([base_traj[i]['rot_x'], base_traj[i]['rot_y'], base_traj[i]['rot_z'], base_traj[i]['rot_w']])
	# 	e = np.abs(r1.as_euler('xyz')[2] - r2.as_euler('xyz')[2])
	# 	if (e > np.pi):
	# 		e = error[-1]
	# 	error.append(e)


	return error


def percent_path_difference(gt_traj, optimal_traj, results, waypoints):

	# compute optimal traj path lengths for each waypoint

	optimal_dists = []

	for i in range(len(waypoints)):
		optimal_dists.append(max(list_distances(optimal_traj,i-1, i)[1]-0.25,0))


	# compute actual path lengths

	actual_dists = []
	
	prev_time = -1

	for i in range(len(results)):
		# if the jackal succeed in reaching this target
		if (results['status'] == 3):
			actual_dists.append(list_distances(gt_traj,prev_time,results['time'])[1])
			prev_time = results['time']


	res = []

	for i in range(len(optimal_dists)):
		if (i >= len(actual_dists)):
			res.append(0)
		else:
			res.append((actual_dists[i]- optimal_dists[i])/optimal_dists[i])

	return res 

if __name__ == "__main__":

	cwd = os.getcwd()
	if ((len(sys.argv)-1) == 0):
		print "ERROR: must input filename"
		exit()
	filename = sys.argv[1] 
	print "Processing data for file", filename

	path = cwd + "/../Data/Simulation_v2/simulated_runs/" + filename + "/"
	logs_path = path + "logs-" + filename + "/"

	with open(logs_path + 'processed_data.pickle', 'rb') as handle:
		data = pickle.load(handle)

	keys = []
	for key in data:
		keys.append(key)

	gt_traj = data['gt_traj']
	waypoints = data['waypoints']
	results = data['action_results']
	optimal_traj = data['optimal_traj']


	if ('amcl_traj' in keys):
		loc_name = 'amcl_traj'
		loc_traj = data['amcl_traj']
	if ('gmapping_traj' in keys):
		loc_name = 'gmapping_traj'
		loc_traj = data['gmapping_traj']

	print(percent_path_difference(gt_traj, optimal_traj, results, waypoints))

	fig, axs = plt.subplots(2,2)
	axs[0][0].plot(list_distances(gt_traj)[0], translation_error(loc_traj,gt_traj))
	axs[1][0].plot(gt_traj['pos_x'], gt_traj['pos_y'])
	axs[1][0].plot(loc_traj['pos_x'], loc_traj['pos_y'])
	axs[1][0].plot(optimal_traj['pos_x'], optimal_traj['pos_y'])
	axs[0][1].plot(list_distances(gt_traj)[0], yaw_error(loc_traj,gt_traj))

	plt.show()
	
