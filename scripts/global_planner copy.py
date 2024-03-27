#!/usr/bin/python3.8
import rospy
import tf
import numba
import numpy as np
import time
from copy import deepcopy
from numpy import linalg
from math import hypot, floor, atan2, cos, sin
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseWithCovarianceStamped, Twist
from visualization_msgs.msg import Marker
from core_aStar import *
from core_aStar_bi import *
from core_jps import *
from smoother_floyd import *
from smoother_aStar import *
from core_dwa import *



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
        self.pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.gcmap_sub = rospy.Subscriber('/costmap_global/costmap/costmap', OccupancyGrid, self.gcmap_callback)
        self.gcmap_update_sub = rospy.Subscriber('/costmap_global/costmap/costmap_updates', OccupancyGridUpdate, self.gcmap_update_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.path_pub = rospy.Publisher('/global_path', Path, queue_size=1)
        self.twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        self.start_ready = False
        self.goal_ready = False
        self.map_ready = False
        self.status_show = True

        while not rospy.is_shutdown():
            cycle_start_time = time.perf_counter()
            # check start, goal, map status
            if self.map_ready:
                self.start_X, self.start_Y = self.XY_to_cmap_XY(self.cur_X, self.cur_Y)
                self.start_ready = True

            if not (self.start_ready and self.goal_ready and self.map_ready):
                # if self.status_show:
                #     print(f"waiting for     start:{self.start_ready}    goal:{self.goal_ready}    map:{self.map_ready}...")
                #     self.status_show = False
                print(f"status     start:{self.start_ready}    goal:{self.goal_ready}    map:{self.map_ready}...")
                continue


            # 1st step
            # node_XYs = run_aStarBiCore(self.map, self.map_width, self.map_height, self.start_X, self.start_Y, self.goal_X, self.goal_Y)
            # node_XYs  = run_aStarCore(self.map, self.map_width, self.map_height, self.start_X, self.start_Y, self.goal_X, self.goal_Y)
            node_XYs, open_set, closed_set = run_jpsCore(self.map, self.map_width, self.map_height, self.start_X, self.start_Y, self.goal_X, self.goal_Y)
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
                node_XYs = floyd_smooting(np.array(node_XYs), self.map, self.map_width, self.map_height)

                # 转换
                path_XYThetas, path_length = self.nodes_transfer(node_XYs)
                path_XYThetas[-1][2] = self.goal_Theta


                # 3rd step
                control, trajectory, tracking_point = run_dwaCore(self.map, self.map_width, self.map_height, self.map_origin_X, self.map_origin_Y, self.map_resolution, 
                                                                  np.array(path_XYThetas), 
                                                                  self.cur_X, self.cur_Y, self.cur_Theta, self.cur_v_x, self.cur_v_y, self.cur_w_z)
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

                twist_msg = Twist()
                twist_msg.linear.x = control[0]
                twist_msg.linear.y = control[1]
                twist_msg.angular.z = control[2]
                # self.twist_pub.publish(twist_msg)


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


            # time
            cycle_end_time = time.perf_counter()
            duration = cycle_end_time - cycle_start_time
            # print(f"cycle duration:{duration}   planner:{planner_info}  controller:{controller_info}    path length:{path_length}")
            # print(f"input   vx:{control[0]}     vy:{control[1]}     wz:{control[2]}")
            print(f"cycle duration:{duration}   planner:{planner_info}  controller:{controller_info}")



    ### callbacks
    # map
    def gcmap_callback(self,gcmap_msg):
        map_width = gcmap_msg.info.width
        map_height = gcmap_msg.info.height
        map_resolution = gcmap_msg.info.resolution
        map_origin_X = gcmap_msg.info.origin.position.x
        map_origin_Y = gcmap_msg.info.origin.position.y
        map = np.array(gcmap_msg.data)
        map = map.reshape(map_height, map_width)
        map = np.transpose(map)
        self.map_width, self.map_height, self.map_resolution, self.map_origin_X, self.map_origin_Y, self.map = \
            map_width, map_height, map_resolution, map_origin_X, map_origin_Y, deepcopy(map)
        self.map_ready = True
        # print(np.shape(self.map), map_height, map_width) # 2048 2048 # height:1504 width:992 shape:(992, 1504)
        # print(map_origin_X, map_origin_Y, map_resolution) # -50.0 -50.0 0.05000000074505806 # 10.0 -20.24 0.019999999552965164

    def gcmap_update_callback(self, gcmap_update_msg):
        if not self.map_ready:
            return
        update_width = gcmap_update_msg.width
        update_height = gcmap_update_msg.height
        update_x, update_y = gcmap_update_msg.x, gcmap_update_msg.y
        update_map = np.array(gcmap_update_msg.data)
        update_map = update_map.reshape(update_height, update_width)
        update_map = np.transpose(update_map)
        self.map[update_x:update_x+update_width, update_y:update_y+update_height] = update_map

        # if self.map_ready and update_width == self.map_width and update_height == self.map_height: # 因为只有在出现新障碍物时updates才会全面更新，此时才需要更新costmap
        #     update_x = gcmap_update_msg.x
        #     update_y = gcmap_update_msg.y
        #     update_map = np.array(gcmap_update_msg.data)
        #     update_map = update_map.reshape(update_height, update_width)
        #     update_map = np.transpose(update_map)
        #     self.map = update_map
        #     # print("map updated")

    def XY_to_cmap_XY(self, X, Y):
        cmap_X = int((X- self.map_origin_X) / self.map_resolution)
        cmap_Y = int((Y- self.map_origin_Y) / self.map_resolution)
        return cmap_X, cmap_Y
    
    def cmap_XY_to_XY(self, cmap_X, cmap_Y):
        X = self.map_resolution * cmap_X + self.map_origin_X
        Y = self.map_resolution * cmap_Y + self.map_origin_Y
        return X, Y


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
        if not self.map_ready:
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
        if cmap_X < 0 or cmap_X > self.map_width - 1 or cmap_Y < 0 or cmap_Y > self.map_height:
            print(f"invalid goal, goal out of map range...")
            return False
        
        # free
        if self.map[cmap_X][cmap_Y] != 0:
            print(f"invalid goal, goal position occupied:{self.map[cmap_X][cmap_Y]}...")
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
        for i in range(len(new_XYs) - 1):
            start_node_XY = new_XYs[i]
            end_node_XY = new_XYs[i+1]
            Theta = atan2(end_node_XY[1] - start_node_XY[1], end_node_XY[0] - start_node_XY[0])
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
