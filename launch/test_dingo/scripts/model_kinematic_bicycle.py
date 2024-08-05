"""
Bicycle model(rear axle reference)

state: X, Y, Theta, v, delta
    X_t+1 = X_t + v_X*dt
    Y_t+1 = Y_t + v_Y*dt
    Theta_t+1 = Theta_t + w*dt

    where:
        v_X = v*cos(Theta)
        v_Y = v*sin(Theta)
        w = (v*tan(delta)) / L

input: v, delta

"""


from math import cos, sin, tan, atan2, hypot, radians, pi, floor, sqrt
import numpy as np


L = 0.3302


# 根据状态量、控制量计算离散时间后的状态
def get_next_state(state, control, dt):
    """
    根据当前状态、控制量、离散时间得到下一状态
    Args:
        state(状态量): [X, Y, Theta, v, delta]
        control(控制量): [v, delta]
        dt(离散时间): dt
    Returns:
        state: 下一步的状态
    """
    X, Y, Theta = state[0], state[1], state[2]
    v, delta = control[0], control[1]

    X_t = X + v*cos(Theta)*dt
    Y_t = Y + v*sin(Theta)*dt
    Theta_t = Theta + (v*tan(delta)/L)*dt

    next_state = [X_t, Y_t, Theta_t, v, delta]

    return next_state


# 根据角速度、速度反推得到当前前轮转角
def get_steering_angle(angular_speed, speed):
    """
    根据角速度、速度反推得到当前前轮转角: w = (v*tan(delta)) / L
    Args:
        angular_speed: 角速度[radians/s]
    Returns:
        delta: 前轮转角[radians]
    """
    delta = atan2(angular_speed*L, abs(speed))
    if speed < 0:
        delta *= -1
    return delta
