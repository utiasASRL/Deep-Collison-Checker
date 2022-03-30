#!/usr/bin/env python
''' A module which actively assesses the Jackals Preformance during the run and determines when
it needs to be shut down'''
import os
import numpy as np
import rospy
import time
import tf2_ros 
import tf2_geometry_msgs
import tf.transformations
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from move_base_msgs.msg import MoveBaseActionResult
from std_msgs.msg import Int32


class Assessor(object):
    ''' A class which actively assesses the Jackals Preformance during the run and determines
    it needs to be shut down'''

    def __init__(self):
        ''' read ROS/system params and spin ros subscriber'''

        self.home_path = os.getenv("HOME")
        self.start_time = rospy.get_param("/start_time", None)
        self.mapping_status = rospy.get_param("/gmapping_mapping", False)
        if self.start_time is None:
            print "Error setting start time"
        try:
            self.log_file = open(self.home_path + "/Deep-Collison-Checker/Data/Simulation_v2/simulated_runs/" + self.start_time + "/logs-" + self.start_time + "/log.txt", "a")
        except IOError:
            print "Could not find/open log file"
            exit()
        self.avg_speed = 0
        self.first_stuck_t = -1
        self.stuck_time_limit = 20.0
        timeout = 1.0
        self.max_samples = int(timeout/0.1)
        self.last_msg = np.array((0, 0, 0, 0), dtype=[("x", np.float), ("y", np.float),
                                                      ("z", np.float), ("t", np.float)])
        self.shutdown_pub = rospy.Publisher("shutdown_signal", Bool, queue_size=1)
        self.odom_to_base = None
        self.map_to_odom = None
        self.tour_length = None
        self.curr_t = 1 
        self.last_time = time.time()
        rospy.init_node("assessor")
        rospy.Subscriber("ground_truth/state", Odometry, self.ground_truth_callback)
        rospy.Subscriber("/tf", TFMessage, self.tf_callback)
        rospy.Subscriber("/move_base/result", MoveBaseActionResult, self.on_result)
        rospy.Subscriber("/tour_length", Int32, self.tour_length_callback)
        
        rospy.spin()


    def ground_truth_callback(self, msg):
        ''' called whenever a ground truth pose message is recieved '''
        pos = np.array((msg.pose.pose.position.x, msg.pose.pose.position.y,
                        msg.pose.pose.position.z, msg.header.stamp.to_sec()),
                       dtype=[("x", np.float), ("y", np.float), ("z", np.float),
                              ("t", np.float)])
   
        if (self.tour_length is not None):
            if self.curr_t != -1:
                print "t = {:.2f}s => target {}/{}".format(pos["t"], self.curr_t, self.tour_length)
            else:
                print "Tour failed, Seeking target {}/{}".format(self.curr_t, self.tour_length) 

        # Print gazebo simulation time compared to real time
        current_time = time.time()
        dt_simu = pos["t"] - self.last_msg["t"]  
        dt_real = current_time - self.last_time
        self.last_time = current_time
        
        print "    Time: Simu running {:.1f} times slower than reality".format(dt_real / dt_simu)


        # Two ways to get the instateneuous speed of the robot
        velocity = np.array((msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z))
        inst_speed = np.hypot(self.last_msg['x'] - pos['x'], self.last_msg['y'] - pos["y"]) / (pos["t"] - self.last_msg["t"])
        self.running_average(inst_speed)
        self.last_msg = pos
        if not self.mapping_status:
            print "     Pos: [{:.2f}, {:.2f}]".format(pos["x"], pos["y"])
        drift = 0
        if (self.odom_to_base is not None and self.map_to_odom is not None):

            otob = PoseStamped()
            otob.pose.position.x = self.odom_to_base.transform.translation.x
            otob.pose.position.y = self.odom_to_base.transform.translation.y
            otob.pose.position.z = self.odom_to_base.transform.translation.z
            otob.pose.orientation.x = self.odom_to_base.transform.rotation.x
            otob.pose.orientation.y = self.odom_to_base.transform.rotation.y
            otob.pose.orientation.z = self.odom_to_base.transform.rotation.z
            otob.pose.orientation.w = self.odom_to_base.transform.rotation.w

            est_pose = tf2_geometry_msgs.do_transform_pose(otob, self.map_to_odom)
            drift = np.hypot(est_pose.pose.position.x - pos['x'], est_pose.pose.position.y - pos['y'])
            if not self.mapping_status:
                print "   Estim: [{:.2f}, {:.2f}] => drift = {:.2f} m".format(est_pose.pose.position.x, est_pose.pose.position.y, drift)
                print "   Speed: {:.2f} m/s  (average = {:.2f})".format(np.linalg.norm(velocity), self.avg_speed)

        lower_lim_speed = 0.03
        upper_lim_drift = 2


        if (pos["t"] > self.stuck_time_limit):

            if self.first_stuck_t < 0:
                if (self.avg_speed < lower_lim_speed) or (drift > upper_lim_drift):
                    self.first_stuck_t = pos["t"]

            else:
                stuck_time = pos["t"] - self.first_stuck_t
                if (self.avg_speed > lower_lim_speed) and (drift < upper_lim_drift):
                    print "Robot recovered"
                    self.first_stuck_t = -1

                elif stuck_time < self.stuck_time_limit:
                    print "Robot has been stuck for {:.1f}s, aborting run in {:.1f}s".format(stuck_time, self.stuck_time_limit - stuck_time)

                else:
                    print "Robot stuck, aborting run"
                    self.log_file.write("Tour failed: robot got stuck\n")
                    self.log_file.close()
                    shutdown = Bool()
                    shutdown.data = False
                    self.shutdown_pub.publish(shutdown.data)

        print ""

    def tf_callback(self, msg):
        for transform in msg.transforms:
            if (transform.header.frame_id == "odom" and transform.child_frame_id == "base_link"):
                self.odom_to_base = transform
            if (transform.header.frame_id == "map" and transform.child_frame_id == "odom"):
                self.map_to_odom = transform


    def running_average(self, new_sample):
        ''' compute running average speed across max_samples '''
        self.avg_speed += (new_sample - self.avg_speed) / self.max_samples

    def on_result(self, msg):
        if msg.status.status != 3:
            self.curr_t = -1
        if self.curr_t != -1:
            self.curr_t +=1

    def tour_length_callback(self, msg):
        self.tour_length = msg.data

if __name__ == "__main__":
    A = Assessor()
