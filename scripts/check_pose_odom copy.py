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
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        
        self.cur_X, self.cur_Y, self.cur_Theta = None, None, None
        self.cur_v_x, self.cur_v_y, self.cur_w_z, self.cur_speed = None, None, None, None

        self.start_time = None
        v_x_set = [0]
        v_y_set = [0]
        speed_set = [0]
        time_set = [0]

        #获取键盘敲击，修改终端属性
        key = None
        old_attr = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        # 输入k开始记录，输入c记录结束
        while not rospy.is_shutdown():
            if select.select([sys.stdin], [], [], 0)[0] == [sys.stdin]:
                key = sys.stdin.read(1)

            rospy.sleep(1/FREQUENCY)

            if key == "k":
                if self.start_time == None:
                    self.start_time = rospy.Time.now().to_sec()
                # record
                print(self.cur_speed, rospy.Time.now().to_sec() - self.start_time)
                v_x_set.append(abs(self.cur_v_x))
                v_y_set.append(abs(self.cur_v_y))
                speed_set.append(self.cur_speed)
                time_set.append(rospy.Time.now().to_sec() - self.start_time)
            elif key == "c":
                break
        
        # quit keyboard
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)

        v_x_set.append(0)
        v_y_set.append(0)
        speed_set.append(0)
        time_set.append(rospy.Time.now().to_sec() - self.start_time + 1/FREQUENCY)

        # plot
        plt.figure(figsize=(10, 5)) # dpi = 80?
        plt.plot(time_set, speed_set, color='r', linestyle='-', linewidth=2, label='speed')
        plt.plot(time_set, v_x_set, color='y', linestyle='--', label='speed (x direction)')
        plt.plot(time_set, v_y_set, color='b', linestyle='--', label='speed (y direction)')
        plt.legend()
        plt.ylabel('speed')
        plt.xlabel('time')
        plt.ylim(0, 1.0)
        plt.xlim(time_set[0], time_set[-1])
        plt.grid(True)
        plt.show()


    ### callbacks
    # goal as a trigger of recording
    

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