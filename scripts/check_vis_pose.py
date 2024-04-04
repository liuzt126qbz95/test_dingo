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
from std_msgs.msg import ColorRGBA, String, Header
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseWithCovarianceStamped, Twist, PoseArray, Pose
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



FREQUENCY = 2 # [hz]



class CheckCostMapValue():
    def __init__(self):
        # 准备
        rospy.init_node("CheckPoseOdom")
        self.pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        self.pose_array_pub = rospy.Publisher('/pose_record', PoseArray, queue_size=1)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        

        self.cur_X, self.cur_Y, self.cur_Theta = None, None, None

        self.start_record = False
        pose_set = []


        #获取键盘敲击，修改终端属性
        key = None
        old_attr = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        # 输入k开始记录，输入c记录结束
        while not rospy.is_shutdown():
            if select.select([sys.stdin], [], [], 0)[0] == [sys.stdin]:
                key = sys.stdin.read(1)

            rospy.sleep(1/FREQUENCY)

            if self.start_record:
                # vis
                print("publish pose array!")

                # cur_pose add to poses
                cur_pose = Pose()
                cur_pose.position.x = self.cur_X
                cur_pose.position.y = self.cur_Y
                qtn = tf.transformations.quaternion_from_euler(0, 0, self.cur_Theta)
                cur_pose.orientation.x = qtn[0]
                cur_pose.orientation.y = qtn[1]
                cur_pose.orientation.z = qtn[2]
                cur_pose.orientation.w = qtn[3]
                
                pose_set.append(cur_pose)

                # pub pose_array
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'map'

                pose_array = PoseArray()
                pose_array.header = header
                pose_array.poses = pose_set
                self.pose_array_pub.publish(pose_array)
                
            if key == "c":
                break
        
        # quit keyboard
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)


    ### callbacks
    # goal as a trigger of recording
    def goal_callback(self, goal_msg):
        self.start_record = True


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

if __name__ == "__main__":
    CheckCostMapValue()
    rospy.spin()