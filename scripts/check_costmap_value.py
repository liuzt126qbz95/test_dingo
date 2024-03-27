#!/usr/bin/python3.8
import rospy
import numpy as np
import tf
from copy import deepcopy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker



class CheckCostMapValue():
    def __init__(self):
        rospy.init_node("CheckCostMapValue")
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.lcmap_sub = rospy.Subscriber('/costmap_local/costmap/costmap', OccupancyGrid, self.lcmap_callback)
        self.gcmap_update_sub = rospy.Subscriber('/costmap_local/costmap/costmap_updates', OccupancyGridUpdate, self.lcmap_update_callback)

        self.map_ready = False


        while not rospy.is_shutdown():
            if not self.map_ready:
                continue
            
            print("vis points")
            self.vis_occupied_points()





    ### callbacks
    # map
    def lcmap_callback(self,lcmap_msg):
        map_width = lcmap_msg.info.width
        map_height = lcmap_msg.info.height
        map_resolution = lcmap_msg.info.resolution
        map_origin_X = lcmap_msg.info.origin.position.x
        map_origin_Y = lcmap_msg.info.origin.position.y
        map = np.array(lcmap_msg.data)
        map = map.reshape(map_height, map_width)
        map = np.transpose(map)
        self.map_width, self.map_height, self.map_resolution, self.map_origin_X, self.map_origin_Y, self.map = \
            map_width, map_height, map_resolution, map_origin_X, map_origin_Y, deepcopy(map)
        self.map_ready = True
    
    def lcmap_update_callback(self, lcmap_update_msg):
        if not self.map_ready:
            return
        update_width = lcmap_update_msg.width
        update_height = lcmap_update_msg.height
        update_x, update_y = lcmap_update_msg.x, lcmap_update_msg.y
        update_map = np.array(lcmap_update_msg.data)
        update_map = update_map.reshape(update_height, update_width)
        update_map = np.transpose(update_map)
        self.map[update_x:update_x+update_width, update_y:update_y+update_height] = update_map
    
    def XY_to_cmap_XY(self, X, Y):
        cmap_X = int((X- self.map_origin_X) / self.map_resolution)
        cmap_Y = int((Y- self.map_origin_Y) / self.map_resolution)
        return cmap_X, cmap_Y
    
    def cmap_XY_to_XY(self, cmap_X, cmap_Y):
        X = self.map_resolution * cmap_X + self.map_origin_X
        Y = self.map_resolution * cmap_Y + self.map_origin_Y
        return X, Y

    
    # goal
    def goal_callback(self, goal_msg):
        return 
        if not self.map_ready:
            print("map not ready...")
            self.goal_ready = False
            return
        
        # X,Y
        goal_X, goal_Y = self.XY_to_cmap_XY(goal_msg.pose.position.x, goal_msg.pose.position.y)
        print(self.map[goal_X][goal_Y])
    


    # tools
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
        self.vis_points(map_list, topic_name="vis_occupied_XYs", rgba=ColorRGBA(0,0,1,1), scale=0.1)


if __name__ == "__main__":
    CheckCostMapValue()
    rospy.spin()