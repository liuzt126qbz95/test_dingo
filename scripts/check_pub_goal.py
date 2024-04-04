#!/usr/bin/python3.8
import rospy
import tf
import numpy as np
import time
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Quaternion


def pub_goal():
    rospy.init_node("pub_goal")
    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

    goal = PoseStamped()
    header = Header()
    header.stamp = rospy.Time.now()
    goal.header = header
    goal.pose.position.x = 4.0
    goal.pose.position.y = -4.0
    goal.pose.orientation = Quaternion(0, 0, 0, 3.14159)
    goal_pub.publish(goal)
    print("goal published!!!")


if __name__ == "__main__":
    pub_goal()
    rospy.spin()