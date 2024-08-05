#!/usr/bin/python3.8
import rospy
import tf
import numba
import numpy as np
import time
from copy import deepcopy
from numpy import linalg
from math import hypot, floor, atan2, cos, sin
from std_msgs.msg import ColorRGBA, String
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseWithCovarianceStamped, Twist
from visualization_msgs.msg import Marker
from planner_aStar import *
from planner_aStar_bi import *
from planner_jps import *
from smoother_floyd import *
from controller_dwa_diff import *


# parameters
RATE = 5
SLEEP_TIME = 1/RATE
PATH_RESOLUTION = 0.05
TRACK = 0.260 * 2
WHEEL_BASE = 0.345 * 2
DIAGINAL_DIST = sqrt(TRACK**2 + WHEEL_BASE**2)


# planner class
class global_planner():
    def __init__(self):
        rospy.init_node("global_planner_node")
        self.pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback) # X Y Theata 
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback) # v w
        self.gcmap_sub = rospy.Subscriber('/costmap_global/costmap/costmap', OccupancyGrid, self.gcmap_callback)
        self.gcmap_update_sub = rospy.Subscriber('/costmap_global/costmap/costmap_updates', OccupancyGridUpdate, self.gcmap_update_callback)
        self.lcmap_sub = rospy.Subscriber('/costmap_local/costmap/costmap', OccupancyGrid, self.lcmap_callback)
        self.gcmap_update_sub = rospy.Subscriber('/costmap_local/costmap/costmap_updates', OccupancyGridUpdate, self.lcmap_update_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.path_pub = rospy.Publisher('/global_path', Path, queue_size=1)
        self.twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.status_pub = rospy.Publisher('/driving_status', String, queue_size=1)
        
        self.start_ready = False
        self.goal_ready = False
        self.gcmap_ready = False
        self.lcmap_ready = False
        self.status_show = True
        motion_start_time = None
        motion_duration = None

        while not rospy.is_shutdown():
            cycle_start_time = time.perf_counter()
            # check start, goal, map status
            if self.gcmap_ready:
                self.start_X, self.start_Y = self.XY_to_cmap_XY(self.cur_X, self.cur_Y)
                self.start_ready = True

            if not (self.start_ready and self.goal_ready and self.gcmap_ready and self.lcmap_ready):
                # if self.status_show:
                #     print(f"waiting for     start:{self.start_ready}    goal:{self.goal_ready}    map:{self.map_ready}...")
                #     self.status_show = False
                print(f"start:{self.start_ready}    goal:{self.goal_ready}    gcmap:{self.gcmap_ready}   lcmap:{self.lcmap_ready}...")
                continue


            # 1st step
            # node_XYs = run_aStarBiCore(self.map, self.map_width, self.map_height, self.start_X, self.start_Y, self.goal_X, self.goal_Y)
            # node_XYs  = run_aStarCore(self.map, self.map_width, self.map_height, self.start_X, self.start_Y, self.goal_X, self.goal_Y)
            node_XYs, open_set, closed_set = run_jpsCore(self.gcmap, self.gcmap_width, self.gcmap_height, self.start_X, self.start_Y, self.goal_X, self.goal_Y)
            # node_XYs, visited_XYs, closed_XYs = run_jpsTunningCore(self.map, self.map_width, self.map_height, self.start_X, self.start_Y, self.goal_X, self.goal_Y)
            
            # vis
            # open_set = [[self.cmap_XY_to_XY(XY[0], XY[1])] for XY in open_set]
            # open_set = [[XY[0][0], XY[0][1]] for XY in open_set]
            # self.vis_points(open_set, topic_name="vis_open_set", rgba=ColorRGBA(1, 0, 0, 1), scale=0.06)

            # closed_set = [[self.cmap_XY_to_XY(XY[0], XY[1])] for XY in closed_set]
            # closed_set = [[XY[0][0], XY[0][1]] for XY in closed_set]
            # self.vis_points(closed_set, topic_name="vis_closed_set", rgba=ColorRGBA(1, 0, 0, 1), scale=0.06)

            # visited_XYs = [[self.cmap_XY_to_XY(XY[0], XY[1])] for XY in visited_XYs]
            # visited_XYs = [[XY[0][0], XY[0][1]] for XY in visited_XYs]
            # self.vis_points(visited_XYs, topic_name="vis_visited_XYs")

            # results(info, publish path, print time)
            if node_XYs == None:
                planner_info = "FAILED.."
                controller_info = "FAILED.."
            else:
                planner_info = "SUCCESS!"


                # 2nd step
                # vis_XYs = [self.cmap_XY_to_XY(XY[0], XY[1]) for XY in node_XYs]
                # self.vis_points(vis_XYs, rgba=ColorRGBA(1, 0, 0, 1), scale=0.05, topic_name="vis_unsmoothed_node_XYs")
                node_XYs = floyd_smooting(np.array(node_XYs), self.gcmap, self.gcmap_width, self.gcmap_height)

                # 转换
                path_XYThetas, path_length = self.nodes_transfer(node_XYs)
                path_XYThetas[-1][2] = self.goal_Theta

                # 发布path_msg
                path_msg = Path()
                path_msg.header.frame_id = "map"
                path_msg.header.stamp = rospy.Time.now() ### maybe other choice?
                for XYTheta in path_XYThetas:
                    pose = PoseStamped()
                    pose.pose.position.x = XYTheta[0]
                    pose.pose.position.y = XYTheta[1]
                    qtn = tf.transformations.quaternion_from_euler(0,0,XYTheta[2])
                    pose.pose.orientation.x = qtn[0]
                    pose.pose.orientation.y = qtn[1]
                    pose.pose.orientation.z = qtn[2]
                    pose.pose.orientation.w = qtn[3]
                    path_msg.poses.append(pose)
                self.path_pub.publish(path_msg)
                self.vis_points(path_XYThetas, topic_name="vis_path_XYs")


                # 3rd step
                trajectory = None

                ### 
                object_X, object_Y = 7, 3
                self.vis_points([[object_X, object_Y]], rgba=ColorRGBA(1, 1, 0, 1), scale=0.2, topic_name="vis_object_point")
                control, trajectory, tracking_point, driving_status = run_dwaCore(self.lcmap, self.lcmap_width, self.lcmap_height, self.lcmap_origin_X, self.lcmap_origin_Y, self.lcmap_resolution, 
                                                                                                np.array(path_XYThetas),
                                                                                                self.cur_X, self.cur_Y, self.cur_Theta, self.cur_v_x, self.cur_w_z,
                                                                                                object_X, object_Y)
                if trajectory == None:
                    controller_info = "FAILED.."
                else:
                    controller_info = "SUCCESS!"
                    # vis trajectory, tracking point, corners
                    self.vis_points(trajectory, rgba=ColorRGBA(1, 0, 0, 1), scale=0.05, topic_name="vis_trajectory_XYs")
                    self.vis_points([tracking_point], rgba=ColorRGBA(0, 0, 1, 1), scale=0.1, topic_name="vis_tracking_XY")
                    corners = self.get_corners(self.cur_X, self.cur_Y, self.cur_Theta)
                    self.vis_points(corners, rgba=ColorRGBA(0, 0, 1, 1), scale=0.05, topic_name="vis_corners_XY")
                    # print(control)

                # 发布control
                twist_msg = Twist()
                twist_msg.linear.x = control[0] # 
                twist_msg.linear.y = 0
                twist_msg.angular.z = control[1]
                self.twist_pub.publish(twist_msg)
            

            # time # completion time
            # computauion time
            cycle_end_time = time.perf_counter()
            duration = cycle_end_time - cycle_start_time
            

            # print(f"input   vx:{control[0]}     vy:{control[1]}     wz:{control[2]}")
            # print(f"path_length:{path_length}   driving_status:{driving_status}     motion_duration:{motion_duration}")
            # print(f"cycle duration:{duration}   planner:{planner_info}  controller:{controller_info}")
            # print(f"cycle duration:{duration}   planner:{planner_info}  controller:{controller_info}    status:{driving_status}")
            # print(f"planner:{planner_info}  controller:{controller_info}    control:{control}")
            
            # gap_X, gap_Y = - (object_Y - self.cur_Y), (object_X - self.cur_X)
            # if not (gap_X < 0 and gap_Y < 0):
            #     object_Theta = atan2(gap_Y, gap_X) - pi/2
            # else:
            #     object_Theta = pi + atan2(gap_Y, gap_X) + pi/2
            # print(f"cur_XYTheta:{[self.cur_X, self.cur_Y, self.cur_Theta]}     objecty_XY:{[object_X, object_Y]}   object_Theta:{object_Theta}")




    ### callbacks
    # global costmap
    def gcmap_callback(self,gcmap_msg):
        map_width = gcmap_msg.info.width
        map_height = gcmap_msg.info.height
        map_resolution = gcmap_msg.info.resolution
        map_origin_X = gcmap_msg.info.origin.position.x
        map_origin_Y = gcmap_msg.info.origin.position.y
        map = np.array(gcmap_msg.data)
        map = map.reshape(map_height, map_width)
        map = np.transpose(map)
        self.gcmap_width, self.gcmap_height, self.gcmap_resolution, self.gcmap_origin_X, self.gcmap_origin_Y, self.gcmap = \
            map_width, map_height, map_resolution, map_origin_X, map_origin_Y, deepcopy(map)
        self.gcmap_ready = True

    def gcmap_update_callback(self, gcmap_update_msg):
        if not self.gcmap_ready:
            return
        update_width = gcmap_update_msg.width
        update_height = gcmap_update_msg.height
        update_x, update_y = gcmap_update_msg.x, gcmap_update_msg.y
        update_map = np.array(gcmap_update_msg.data)
        update_map = update_map.reshape(update_height, update_width)
        update_map = np.transpose(update_map)
        self.gcmap[update_x:update_x+update_width, update_y:update_y+update_height] = update_map

    def XY_to_cmap_XY(self, X, Y):
        cmap_X = int((X- self.gcmap_origin_X) / self.gcmap_resolution)
        cmap_Y = int((Y- self.gcmap_origin_Y) / self.gcmap_resolution)
        return cmap_X, cmap_Y
    
    def cmap_XY_to_XY(self, cmap_X, cmap_Y):
        X = self.gcmap_resolution * cmap_X + self.gcmap_origin_X
        Y = self.gcmap_resolution * cmap_Y + self.gcmap_origin_Y
        return X, Y


    # local costmap
    def lcmap_callback(self,lcmap_msg):
        map_width = lcmap_msg.info.width
        map_height = lcmap_msg.info.height
        map_resolution = lcmap_msg.info.resolution
        map_origin_X = lcmap_msg.info.origin.position.x
        map_origin_Y = lcmap_msg.info.origin.position.y
        map = np.array(lcmap_msg.data)
        map = map.reshape(map_height, map_width)
        map = np.transpose(map)
        self.lcmap_width, self.lcmap_height, self.lcmap_resolution, self.lcmap_origin_X, self.lcmap_origin_Y, self.lcmap = \
            map_width, map_height, map_resolution, map_origin_X, map_origin_Y, deepcopy(map)
        self.lcmap_ready = True
    
    def lcmap_update_callback(self, lcmap_update_msg):
        if not self.lcmap_ready:
            return
        update_width = lcmap_update_msg.width
        update_height = lcmap_update_msg.height
        update_x, update_y = lcmap_update_msg.x, lcmap_update_msg.y
        update_map = np.array(lcmap_update_msg.data)
        update_map = update_map.reshape(update_height, update_width)
        update_map = np.transpose(update_map)
        self.lcmap[update_x:update_x+update_width, update_y:update_y+update_height] = update_map


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


    # goal
    def goal_callback(self, goal_msg):
        if not self.gcmap_ready:
            print("map not ready...")
            self.goal_ready = False
            return
        # X,Y
        goal_X, goal_Y = self.XY_to_cmap_XY(goal_msg.pose.position.x, goal_msg.pose.position.y)
        # Theta
        quaternion = np.array([goal_msg.pose.orientation.x, 
                               goal_msg.pose.orientation.y, 
                               goal_msg.pose.orientation.z, 
                               goal_msg.pose.orientation.w])
        euler = tf.transformations.euler_from_quaternion(quaternion)
        goal_Theta = euler[2]

        print(goal_msg.pose.position.x, goal_msg.pose.position.y, goal_Theta)


        if not self.is_camp_XY_valid(goal_X, goal_Y):
            self.goal_ready = False
            self.status_show = True
            return
        
        self.goal_X, self.goal_Y, self.goal_Theta = goal_X, goal_Y, goal_Theta
        self.goal_ready = True
        pass



    ### tools
    def is_camp_XY_valid(self, cmap_X, cmap_Y):
        # in range
        if cmap_X < 0 or cmap_X > self.gcmap_width - 1 or cmap_Y < 0 or cmap_Y > self.gcmap_height:
            print(f"invalid goal, goal out of map range...")
            return False
        
        # free
        if self.gcmap[cmap_X][cmap_Y] != 0:
            print(f"invalid goal, goal position occupied:{self.gcmap[cmap_X][cmap_Y]}...")
            return False
        
        print(f"valid goal! planning!")
        return True 


    # use node to generate path with Thetas and implote it:
    def nodes_transfer(self, node_XYs):
        path_length = 0

        # 反转
        node_XYs.reverse()
        
        # 转化
        new_XYs = []
        for node in node_XYs:
            X, Y = self.cmap_XY_to_XY(node[0], node[1])
            new_XYs.append([X, Y])

        # 插补
        path_XYThetas = []
        prev_Theta = 0
        for i in range(len(new_XYs) - 1):
            start_node_XY = new_XYs[i]
            end_node_XY = new_XYs[i+1]
            Theta = atan2(end_node_XY[1] - start_node_XY[1], end_node_XY[0] - start_node_XY[0])
            if Theta != prev_Theta:
                prev_Theta = Theta
            dist = hypot(end_node_XY[1] - start_node_XY[1], end_node_XY[0] - start_node_XY[0])
            path_length += dist
            step_num = floor(dist/PATH_RESOLUTION)

            X, Y = start_node_XY[0], start_node_XY[1]
            path_XYThetas.append([X, Y, Theta])
            for i in range(step_num):
                X += PATH_RESOLUTION * cos(Theta)
                Y += PATH_RESOLUTION * sin(Theta)
                path_XYThetas.append([X, Y, Theta])
        path_XYThetas = np.array(path_XYThetas)
            
        return path_XYThetas, path_length
    

    # visualization 
    def vis_points(self, XYs, rgba=ColorRGBA(0, 0, 1, 1), scale=0.03, topic_name="vis_points"):
        points = [Point(XY[0], XY[1], 0) for XY in XYs]
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.POINTS
        marker.points = points
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.orientation = Quaternion(0, 0, 0, 1)
        marker.scale.x = scale
        marker.scale.y = scale
        marker.color = rgba
        marker.lifetime = rospy.Duration(secs=500)
        vis_pub = rospy.Publisher(topic_name, Marker, queue_size=1)
        vis_pub.publish(marker)

    def vis_occupied_points(self):
        map = self.map
        map_list = []
        for X in range(self.map_width):
            for Y in range(self.map_height):
                if map[X][Y] > 0:
                    map_list.append(self.cmap_XY_to_XY(X, Y))
        self.vis_points(map_list, topic_name="vis_occupied_XYs", rgba=ColorRGBA(0,0,1,1), scale=0.01)


    def get_corners(self, cur_X, cur_Y, cur_Theta):
        sin_Theta, cos_Theta = sin(cur_Theta), cos(cur_Theta)
        corners = [[cur_X + 1/2*WHEEL_BASE*cos_Theta, cur_Y + 1/2*WHEEL_BASE*sin_Theta], 
                [cur_X + 1/2*WHEEL_BASE*cos_Theta - 1/2*TRACK*sin_Theta, cur_Y + 1/2*WHEEL_BASE*sin_Theta + 1/2*TRACK*cos_Theta], 
                [cur_X + 1/2*WHEEL_BASE*cos_Theta + 1/2*TRACK*sin_Theta, cur_Y + 1/2*WHEEL_BASE*sin_Theta - 1/2*TRACK*cos_Theta],
                [cur_X - 1/2*WHEEL_BASE*cos_Theta, cur_Y - 1/2*WHEEL_BASE*sin_Theta],
                [cur_X - 1/2*WHEEL_BASE*cos_Theta - 1/2*TRACK*sin_Theta, cur_Y - 1/2*WHEEL_BASE*sin_Theta + 1/2*TRACK*cos_Theta],
                [cur_X - 1/2*WHEEL_BASE*cos_Theta + 1/2*TRACK*sin_Theta, cur_Y - 1/2*WHEEL_BASE*sin_Theta - 1/2*TRACK*cos_Theta],
                [cur_X - 1/2*TRACK*sin_Theta, cur_Y + 1/2*TRACK*cos_Theta],
                [cur_X + 1/2*TRACK*sin_Theta, cur_Y - 1/2*TRACK*cos_Theta]]
        return corners

# main
if __name__ == "__main__":
    global_planner()
    rospy.spin()
