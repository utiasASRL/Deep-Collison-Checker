#!/usr/bin/env python

import json
import os
import numpy as np
import rospy
import subprocess
import signal
import sys
import shutil
import logging
import time

from std_msgs.msg import Bool
from move_base_msgs.msg import MoveBaseActionResult
from nav_msgs.msg import Odometry


class MetaHandler:

    def __init__(self):
        
        rospy.init_node('meta_data')
        
        self.home_path = os.getenv("HOME")

        self.read_params()
        self.file = self.home_path + "/Deep-Collison-Checker/Data/Simulation_v2/simulated_runs/" + self.start_time
        self.path = self.home_path + "/Deep-Collison-Checker/Data/Simulation_v2/simulated_runs/" + self.start_time + "/logs-" + self.start_time + "/"
        self.meta_json = open(self.path + "meta.json", 'w')
        self.start_subscribers()


        self.shutting_down = False
        signal.signal(signal.SIGINT, self.unix_term)
        signal.signal(signal.SIGTERM, self.unix_term)
        rospy.on_shutdown(self.ros_term)

        rospy.spin()

    def create_table(self):
        self.table = {}
        self.table['tour_names'] = self.tour_name
        self.table['filter_status'] = self.filter_status

        self.loc_method = rospy.get_param("/loc_method")
        if self.loc_method == 0:
            tech = "gmapping"
        elif self.loc_method == 1:
            tech = "amcl"
        else:
            tech = "pointslam"
        self.table['class_method'] = self.class_method
        self.table['localization_test'] = self.localization_test
        self.table['load_world'] = self.load_world
        self.table['localization_technique'] = tech
        self.table['success_status'] = self.successful
        self.table['scenarios'] = []
        for name in self.room_params:
            self.table['scenarios'].append(self.room_params[name]['scenario'])

    def ros_term(self):
        ''' If we recieve a ros shutdown signal, delete the log file '''
        if (self.shutting_down):
            return
        if (os.path.isdir(self.file)):
            shutil.rmtree(self.file)
            logging.warning('Deleting ' + self.file + ' due to early shutdown')

        self.meta_json.close()  
        shutdown_script = self.home_path + "/Deep-Collison-Checker/Myhal_Simulator/simu_melodic_ws/scripts/shutdown.sh"
        subprocess.call(shutdown_script, shell = True)

    def unix_term(self, sig, frame):
        ''' If we recieve a signal that the program has terminated early, delete the log file '''
        if (self.shutting_down):
            return
        if (os.path.isdir(self.file)):
            shutil.rmtree(self.file)
            logging.warning('Deleting ' + self.file + ' due to early shutdown')

        self.meta_json.close()  
        shutdown_script = self.home_path + "/Deep-Collison-Checker/Myhal_Simulator/simu_melodic_ws/scripts/shutdown.sh"
        subprocess.call(shutdown_script, shell = True)

    def on_shutdown(self, msg):
        self.shutting_down = True  
        if (not msg.data):
            self.successful = 'false'
        self.create_table()
        json.dump(self.table, self.meta_json, indent = 4, sort_keys=True)
        self.meta_json.close()  
        shutdown_script = self.home_path + "/Deep-Collison-Checker/Myhal_Simulator/simu_melodic_ws/scripts/shutdown.sh"
        subprocess.call(shutdown_script, shell = True)
        sys.exit()

    def read_params(self):
        self.localization_test = 'true' if rospy.get_param("/localization_test") else 'false'
        self.class_method = rospy.get_param("/class_method")
        self.load_world = rospy.get_param("/load_world")
        self.start_time = rospy.get_param("/start_time")
        self.tour_name = rospy.get_param("/tour_name")
        self.filter_status = 'true' if rospy.get_param("/filter_status") else 'false'
        # self.loc_method = rospy.get_param("/loc_method")

        room_names = rospy.get_param("/room_names")
        scenario_names = rospy.get_param("/room_names")

        self.scenario_params = {}

        for name in scenario_names:
            self.scenario_params[name] = rospy.get_param("/" + name)

        self.room_params = {}

        for name in room_names:
            self.room_params[name] = rospy.get_param("/" + name)


    def start_subscribers(self):
        rospy.Subscriber("/shutdown_signal", Bool, self.on_shutdown)
        rospy.Subscriber("/ground_truth/state", Odometry, self.dummy)
        self.tour_results = []
        self.successful = 'true'
        rospy.Subscriber("/move_base/result", MoveBaseActionResult, self.on_result)

    def dummy(self, msg):
        pass

    def on_result(self, msg):
        self.tour_results.append(msg)
        if msg.status.status != 3:
            self.successful = 'false'


if __name__ == "__main__":
    '''
    create subscribers to relevant topics 
    read in relevant parameters 
    open global json file and modify it based on the read parameters 

    once shutdown message is recieved, dump to json and call data processing
    '''
    M = MetaHandler()

  
    
    
