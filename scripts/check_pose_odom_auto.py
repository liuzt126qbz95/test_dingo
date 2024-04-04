#!/usr/bin/python3.8
import rospy
import tf
import numba
import numpy as np
import time
import sys, select, tty, termios
from copy import deepcopy
from numpy import linalg
from math import hypot, floor, atan2, cos, sin
from std_msgs.msg import ColorRGBA, String
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseWithCovarianceStamped, Twist
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



FREQUENCY = 10 # [hz]



class CheckCostMapValue():
    def __init__(self):
        # 准备
        rospy.init_node("CheckPoseOdom")
        self.pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.status_sub = rospy.Subscriber('/driving_status', String, self.status_callback)
        
        self.cur_X, self.cur_Y, self.cur_Theta = None, None, None
        self.cur_v_x, self.cur_v_y, self.cur_w_z, self.cur_speed = None, None, None, None
        self.driving_status = None

        self.start_time = None
        v_x_set = [0]
        v_y_set = [0]
        speed_set = [0]
        time_set = [0]


        # navtiagting开始，开始记录数据
        while not rospy.is_shutdown():
            rospy.sleep(1/FREQUENCY)
            
            if self.cur_speed == None:
                continue 

            if self.driving_status == "near_goal":
                break
            
            # record
            if self.start_time != None:
                print(self.cur_speed, rospy.Time.now().to_sec() - self.start_time)
                v_x_set.append(abs(self.cur_v_x))
                v_y_set.append(abs(self.cur_v_y))
                speed_set.append(self.cur_speed)
                time_set.append(rospy.Time.now().to_sec() - self.start_time)

        v_x_set.append(0)
        v_y_set.append(0)
        speed_set.append(0)
        time_set.append(rospy.Time.now().to_sec() - self.start_time + 1/FREQUENCY)


        # plot
        print(f"average speed:{sum(speed_set)/len(speed_set)}")
        plt.figure(figsize=(10, 4), dpi=200, constrained_layout=True) # 画板大小(宽，高)
        plt.plot(time_set, speed_set, color='r', linestyle='-', linewidth=2, label='speed')
        plt.plot(time_set, v_x_set, color='y', linestyle='--', label='speed(x direction)')
        plt.plot(time_set, v_y_set, color='b', linestyle='--', label='speed(y direction)')
        plt.legend(fontsize=10, loc='upper right') # 线段名称字体大小(默认10？)
        plt.ylabel('speed(m/s)', fontsize=15) # 轴名称大小(默认10？)
        plt.xlabel('time(s)', fontsize=15)
        plt.ylim(0, 0.75)
        plt.xlim(time_set[0], time_set[-1])
        plt.xticks(fontsize=10) # 刻度字体大小(默认10？)
        plt.grid(True)
        plt.show()
        


    ### callbacks
    # status
    def status_callback(self, status_msg):
        driving_status = status_msg.data

        if driving_status != None:
            self.driving_status = driving_status
            if self.start_time == None:
                self.start_time = rospy.Time.now().to_sec()

    
    # pose
    def pose_callback(self, pose_msg):
        # X
        cur_X = pose_msg.pose.pose.position.x
        # Y
        cur_Y = pose_msg.pose.pose.position.y
        # Theta
        quaternion = np.array([pose_msg.pose.pose.orientation.x, 
                               pose_msg.pose.pose.orientation.y, 
                               pose_msg.pose.pose.orientation.z, 
                               pose_msg.pose.pose.orientation.w])
        euler = tf.transformations.euler_from_quaternion(quaternion)
        cur_Theta = euler[2]

        self.cur_X, self.cur_Y, self.cur_Theta = cur_X, cur_Y, cur_Theta


    # odom
    def odom_callback(self, odom_msg):
        # speed
        cur_v_x = odom_msg.twist.twist.linear.x
        cur_v_y = odom_msg.twist.twist.linear.y
        # cur_speed_angle = atan2(odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.x)
        # if cur_speed_angle != 0:
        #     cur_speed *= -1
        # angular speed: it is 0 when the robot moving backward???
        cur_w_z = odom_msg.twist.twist.angular.z
        
        self.cur_v_x, self.cur_v_y, self.cur_w_z = cur_v_x, cur_v_y, cur_w_z
        self.cur_speed = hypot(cur_v_x, cur_v_y)


if __name__ == "__main__":
    CheckCostMapValue()
    rospy.spin()