#!/usr/bin/python3.8
import rospy
import sys, select, tty, termios
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry 


class Key():
    def __init__(self):
        # node, pub, sub
        rospy.init_node("keyboard_listen_and_pub")
        self.key_pub = rospy.Publisher('/Key', String, queue_size=10)
        print("node established!")

        #获取键盘敲击，修改终端属性
        old_attr = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        while not rospy.is_shutdown():
            # 检测键盘
            if select.select([sys.stdin], [], [], 0)[0] == [sys.stdin]:
                # 得到键盘输入
                key = sys.stdin.read(1)
                print(key)

                # 根据键盘输入发布速度指令
                self.key_pub.publish(key)

        #将终端还原为标准模式 
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)


if __name__ == "__main__":
    Key()