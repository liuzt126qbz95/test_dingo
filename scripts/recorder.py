#!/usr/bin/python3.8
import rospy
import tf
import numpy as np
import atexit
import numba
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
from os.path import expanduser
from time import gmtime, strftime
from numpy import linalg as LA



### settings
RECORD_FREQUENCY = 10 # [hz]
SPEED_ROUND_NUM = 3 # 速度四舍五入至多少位
PLOT = True
SAVE = True
home = expanduser('~')
file_position = open(strftime(home + '/catkin_ws/src/test_dingo/records/path-%Y-%m-%d-%H-%M-%S',gmtime()) + '.csv', 'w')
file_speed = open(strftime(home + '/catkin_ws/src/test_dingo/records/speed-%Y-%m-%d-%H-%M-%S',gmtime()) + '.csv', 'w')
idx = -2



### in the recorder we log position and speed data and save them to .csv files
class Recorder():
    def __init__(self):
        rospy.init_node('data_recorder', anonymous=True)
        self.pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.status_sub = rospy.Subscriber('/driving_status', String, self.status_callback)
        
        self.record_start = False
        self.cur_X, self.cur_Y, self.cur_Theta = None, None, None
        self.cur_v_x, self.cur_v_y, self.cur_w_z, self.cur_v = None, None, None, None
        self.driving_status = None
        self.start_time = None
        
        self.X_set = []
        self.Y_set = []
        self.Theta_set = []
        self.v_x_set = []
        self.v_y_set = []
        self.w_z_set = []
        self.v_set = []
        self.time_set = []

        # navtiagting开始，开始记录数据
        while not rospy.is_shutdown():
            rospy.sleep(1/RECORD_FREQUENCY)
            # record
            if self.record_start:
                self.record()
                # record finish if near goal
                if self.driving_status == "near_goal" or self.driving_status == "reach_goal":                
                    self.X_set.append(self.cur_X)
                    self.Y_set.append(self.cur_Y)
                    self.Theta_set.append(self.cur_Theta)
                    self.v_x_set.append(0)
                    self.v_y_set.append(0)
                    self.w_z_set.append(0)
                    self.v_set.append(0)
                    self.time_set.append(rospy.Time.now().to_sec() - self.start_time + 1/RECORD_FREQUENCY)
                    print(f"recording finish!   SAVE:{SAVE}     PLOT:{PLOT}")
                    break


        if SAVE:
            num_of_record = len(self.time_set)
            print(len(self.time_set), len(self.X_set), len(self.v_x_set))
            for idx in range(num_of_record):
                file_position.write('%f, %f, %f, %f, %f\n' %(idx,
                                                             self.X_set[idx],
                                                             self.Y_set[idx],
                                                             self.Theta_set[idx],
                                                             self.time_set[idx]))
                file_speed.write('%f, %f, %f, %f, %f, %f\n' %(idx,
                                                             self.v_x_set[idx],
                                                             self.v_y_set[idx],
                                                             self.v_set[idx],
                                                             self.w_z_set[idx],
                                                             self.time_set[idx]))
            file_position.close()
            file_speed.close()  

        if PLOT:
            # position
            gap_X = abs(min(self.X_set) - max(self.X_set))
            gap_Y = abs(min(self.Y_set) - max(self.Y_set))
            plt.figure(1, figsize=(5, 5)) # dpi = 80?
            plt.plot(self.X_set, self.Y_set, color='r', linewidth=2, label='real_trajectory')
            plt.legend()
            plt.xlabel('X(m)')
            plt.ylabel('Y(m)')
            if gap_X >= gap_Y:
                plt.xlim(min(self.X_set), min(self.X_set)+gap_X)
                plt.ylim(min(self.Y_set), min(self.Y_set)+gap_X)
            else:
                plt.xlim(min(self.X_set), min(self.X_set)+gap_Y)
                plt.ylim(min(self.Y_set), min(self.Y_set)+gap_Y)
            plt.grid(False)

            # speed
            plt.figure(2, figsize=(10, 5)) # dpi = 80?
            plt.plot(self.time_set, self.v_set, color='r', linestyle='-', linewidth=2, label='speed')
            plt.plot(self.time_set, self.v_x_set, color='y', linestyle='--', label='speed (x direction)')
            plt.plot(self.time_set, self.v_y_set, color='b', linestyle='--', label='speed (y direction)')
            plt.legend()
            plt.ylabel('speed')
            plt.xlabel('time')
            plt.ylim(0, 1.0)
            plt.xlim(self.time_set[0], self.time_set[-1])
            plt.grid(True)
            plt.show()


    def record(self):
        self.X_set.append(self.cur_X)
        self.Y_set.append(self.cur_Y)
        self.Theta_set.append(self.cur_Theta)
        self.v_x_set.append(abs(self.cur_v_x))
        self.v_y_set.append(abs(self.cur_v_y))
        self.w_z_set.append(self.cur_w_z)
        self.v_set.append(abs(self.cur_v))
        self.time_set.append(rospy.Time.now().to_sec() - self.start_time)
        print(f"X:{self.cur_X}  Y:{self.cur_Y}  Theta:{self.cur_Theta}  v:{self.cur_v}")
        # print(f"vx:{self.cur_v_x}  vy:{self.cur_v_y}  wz:{self.cur_w_z}  v:{self.cur_v}")



    ### callbacks
    # status
    def status_callback(self, status_msg):
        self.driving_status = status_msg.data

    
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
        # vx
        cur_v_x = round(odom_msg.twist.twist.linear.x, SPEED_ROUND_NUM)
        # vy
        cur_v_y = round(odom_msg.twist.twist.linear.y, SPEED_ROUND_NUM)
        # cur_v_angle = atan2(odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.x)
        # if cur_v_angle != 0:
        #     cur_v *= -1
        # angular speed: it is 0 when the robot moving backward???
        # wz
        cur_w_z = round(odom_msg.twist.twist.angular.z, SPEED_ROUND_NUM)
        
        self.cur_v_x, self.cur_v_y, self.cur_w_z = cur_v_x, cur_v_y, cur_w_z
        self.cur_v = round(hypot(cur_v_x, cur_v_y), SPEED_ROUND_NUM)


        # record starts when robot starts to move and robot hasnt start to record yet
        if self.record_start == False:
            if not ((cur_v_x == 0) and (cur_v_y == 0) and (cur_w_z == 0) and self.start_time == None):
                if self.record_start == False:
                    print(f"recording starts at {self.start_time}!")
                    self.start_time = rospy.Time.now().to_sec()
                    self.record_start = True
                    # 第一次记录位置、速度(=0)
                    self.X_set.append(self.cur_X)
                    self.Y_set.append(self.cur_Y)
                    self.Theta_set.append(self.cur_Theta)
                    self.v_x_set.append(0)
                    self.v_y_set.append(0)
                    self.w_z_set.append(0)
                    self.v_set.append(0)
                    self.time_set.append(0)


# shutdown
def shutdown():
    print("bye bye")


if __name__ == '__main__':
    try:
        Recorder()
    except rospy.ROSInterruptException:
        pass
