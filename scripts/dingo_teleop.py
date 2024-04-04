#!/usr/bin/python3.8
import rospy
import sys, select, tty, termios
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry 


LINEAR_SPEED = 0.5
ANGULAR_SPEED = 0.4

class Teleop():
    def __init__(self):
        # node, pub, sub
        rospy.init_node("Dingo_teleop")
        # self.key_sub = rospy.Subscriber('/Key', String, self.key_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback, queue_size=1)
        self.twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        print("node established!")

        # values
        self.cur_v_x, self.cur_v_y, self.cur_w_z = None, None, None

        #获取键盘敲击，修改终端属性
        old_attr = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        while not rospy.is_shutdown():
            # cur status:
            # print(f"velocity x:{self.cur_v_x} \nvelocity y:{self.cur_v_y} \nvelocity z:{self.cur_w_z} \n \n")
            if select.select([sys.stdin], [], [], 0)[0] == [sys.stdin]:
                key = sys.stdin.read(1)
            else:
                continue
            
            if not key in ["w", "a", "s", "d", "q", "e", "z", "c", " ", "1", "2"]:
                # print("invalid key...")
                continue
            print(key)

            twist = Twist()
            if key == " ":
                twist.linear.x = 0
                twist.linear.y = 0
                twist.angular.z = 0
            elif key == "w":
                twist.linear.x = LINEAR_SPEED
            elif key == "s":
                twist.linear.x = -LINEAR_SPEED
            elif key == "a":
                twist.linear.y = LINEAR_SPEED
            elif key == "d":
                twist.linear.y = -LINEAR_SPEED
            elif key == "q":
                twist.linear.x = LINEAR_SPEED
                twist.linear.y = LINEAR_SPEED
            elif key == "e":
                twist.linear.x = LINEAR_SPEED
                twist.linear.y = -LINEAR_SPEED
            elif key == "z":
                twist.linear.x = -LINEAR_SPEED
                twist.linear.y = LINEAR_SPEED
            elif key == "c":
                twist.linear.x = -LINEAR_SPEED
                twist.linear.y = -LINEAR_SPEED
            elif key == "1":
                twist.angular.z = ANGULAR_SPEED
            elif key == "2":
                twist.angular.z = -ANGULAR_SPEED
            self.twist_pub.publish(twist)
            
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)


    # key
    def key_callback(self, key_msg):
        self.key = key_msg.data


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



if __name__ == "__main__":
    Teleop()
