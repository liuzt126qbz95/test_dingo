#!/usr/bin/python3.8
import numpy as np
import numba
import rospy
from random import uniform
from math import cos, sin, atan2, hypot, radians, pi, floor, sqrt
# from model_kinematic_bicycle import *
from model_kinematic_omni import *
from utils import *



INF = 999999999.

TRACK = 0.260 * 2
WHEEL_BASE = 0.345 * 2
DIAGINAL_DIST = sqrt(TRACK**2 + WHEEL_BASE**2)
LINEAR_V = 0.65 # [m/s] default 1.3
LINEAR_A = 0.75 # [m/s^2] default 1
ANGULAR_W = 2 # [rad/s] defaykt 4
ANGULAR_A = 1.5 # [rad/s^2] default 2
MIN_V_X = -LINEAR_V
MAX_V_X = LINEAR_V
MIN_V_Y = -LINEAR_V
MAX_V_Y = LINEAR_V
MIN_W_Z = -ANGULAR_W 
MAX_W_Z = ANGULAR_W 
MIN_A_X = -LINEAR_A 
MAX_A_X = LINEAR_A
MIN_A_Y = -LINEAR_A
MAX_A_Y = LINEAR_A
MIN_A_W_Z = -ANGULAR_A
MAX_A_W_Z = ANGULAR_A

RESOLUTION_V = 0.1
RESOLUTION_W = 0.1
DT = 0.1
PREDICTION_TIME = 2.0 # [s]
PREDICTION_TIME_NEAR_GOAL = 0.5
K_L = 0.75 # lookahead distance = K_L * vs
MIN_L = 0.1 # [m], the minimum lookahead distance
W_DEV = 4 # 评价参考路径的贴合度
W_SAFE = 1 # 评价是否碰撞
W_FAST = 0.25 # 评价运行速度
W_HEADING = 1
W_DIST = 1
REACH_GOAL_DIST = 0.1
REACH_GOAL_THETA_GAP = 0.1
OMNI_DRIVE = True



def run_dwaCore(map, map_width, map_height, map_origin_X, map_origin_Y, map_resolution, reference_XYThetas, cur_X, cur_Y, cur_Theta, cur_v_x, cur_v_y, cur_w_z):
    # 1.得到v，angular_v上下限和tracking_point, tracking_trajectory
    min_v_x, max_v_x, min_v_y, max_v_y, min_w_z, max_w_z = get_speed_and_delta_limits(cur_v_x, cur_v_y, cur_w_z)
    L_by_speed = hypot(max_v_y, max_v_x) * PREDICTION_TIME * 1
    L = L_by_speed if L_by_speed > MIN_L else MIN_L 
    tracking_Point, tracking_trajectory = get_tracking_point_and_tracking_trajectory(cur_X, cur_Y, L, reference_XYThetas)
    # print(f"vx:{cur_v_x}   vy:{cur_v_y}   wz:{cur_w_z}")
    

    # 2. 确定当前轨迹跟踪状态：navigating, tracking_goal, near_goal, reach_goal
    driving_status = "navigating"
    goal_X, goal_Y, goal_Theta = reference_XYThetas[-1][0], reference_XYThetas[-1][1], reference_XYThetas[-1][2]
    if tracking_Point[0] == goal_X and tracking_Point[1] == goal_Y:
        if not hypot(goal_Y-cur_Y, goal_X-cur_X) <= REACH_GOAL_DIST:
            driving_status = "tracking_goal"
        else:
            if not get_Theta_gap(cur_Theta, goal_Theta) <= REACH_GOAL_THETA_GAP:
                driving_status = "near_goal"
            else:
                driving_status = "reach_goal"
    # print(driving_status)
    if driving_status == "reach_goal":
        return [0,0,0], [[cur_X, cur_Y, cur_Theta]], tracking_Point
    

    # 3.得到speed、delta集合
    v_x_set = np.arange(min_v_x, max_v_x, RESOLUTION_V)
    v_x_set = np.append(v_x_set, max_v_x)
    w_z_set = np.arange(min_w_z, max_w_z, RESOLUTION_W)
    w_z_set = np.append(w_z_set, max_w_z)

    if not OMNI_DRIVE:
        v_y_set = [0]
    else:
        v_y_set = np.arange(min_v_y, max_v_y, RESOLUTION_V)
        v_y_set = np.append(v_y_set, max_v_y)

    if driving_status == "near_goal":
        v_x_set, v_y_set = [0], [0]

    # print(speed_set, delta_set)
    

    # 4.得到trajectory set
    prediction_time = PREDICTION_TIME if driving_status != "near_goal" else PREDICTION_TIME_NEAR_GOAL
    trajectory_set = []
    control_set = []
    for control_v_x in v_x_set:
        for control_v_y in v_y_set:
            for control_w_z in w_z_set:
                trajectory = get_simulated_trajectory_and_check_collision(cur_X, cur_Y, cur_Theta, cur_v_x, cur_v_y, cur_w_z,
                                                                          control_v_x, control_v_y, control_w_z,
                                                                          map, map_width, map_height, map_origin_X, map_origin_Y, map_resolution,
                                                                          prediction_time)
                if len(trajectory) == 0:
                    continue
                trajectory_set.append(trajectory)
                control_set.append([control_v_x, control_v_y, control_w_z])

    
    # 5.对trajectory进行评价
    # unable to generate any trajectory, stay still
    if len(trajectory_set) == 0:
        return [0,0,0], None, None

    score_dev_set, score_safe_set, score_fast_set, score_heading_set, score_dist_set, score_set = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for trajectory in trajectory_set:
        trajectory = np.array(trajectory)
        # navigating
        if driving_status == "navigating":
            score_safe_set = np.append(score_safe_set, score_trajectory_by_safe(trajectory, map, map_width, map_height, map_origin_X, map_origin_Y, map_resolution))
            score_dev_set = np.append(score_dev_set, score_trajectory_by_deviation(trajectory, tracking_trajectory, map, map_width, map_height, map_origin_X, map_origin_Y, map_resolution))
            score_fast_set = np.append(score_fast_set, score_trajectory_by_fast(trajectory))
        # tracking_goal
        elif driving_status == "tracking_goal":
            score_safe_set = np.append(score_safe_set, score_trajectory_by_safe(trajectory, map, map_width, map_height, map_origin_X, map_origin_Y, map_resolution))
            score_dev_set = np.append(score_dev_set, score_trajectory_by_deviation(trajectory, tracking_trajectory, map, map_width, map_height, map_origin_X, map_origin_Y, map_resolution))
            score_dist_set = np.append(score_dist_set, score_trajectory_by_final_dist(trajectory, tracking_Point))
        # near_goal
        elif driving_status == "near_goal":
            score_safe_set = np.append(score_safe_set, score_trajectory_by_safe(trajectory, map, map_width, map_height, map_origin_X, map_origin_Y, map_resolution))
            score_heading_set = np.append(score_heading_set, score_trajectory_by_heading(trajectory, tracking_Point))
            score_dist_set = np.append(score_dist_set, score_trajectory_by_final_dist(trajectory, tracking_Point))
            score_fast_set = np.append(score_fast_set, score_trajectory_by_fast(trajectory))

    for idx in range(len(trajectory_set)):
        score = 0
        # navigating
        if driving_status == "navigating":
            score += W_SAFE*(score_safe_set[idx]/np.sum(score_safe_set))
            score += W_DEV*(score_dev_set[idx]/np.sum(score_dev_set))
            score += W_FAST*(score_fast_set[idx]/np.sum(score_fast_set))
        # tracking_goal
        elif driving_status == "tracking_goal":
            score += W_SAFE*(score_safe_set[idx]/np.sum(score_safe_set))
            score += W_DEV*(score_dev_set[idx]/np.sum(score_dev_set))
            score += W_DIST*(score_dist_set[idx]/np.sum(score_dist_set))
        # near_goal
        elif driving_status == "near_goal":
            score += W_SAFE*(score_safe_set[idx]/np.sum(score_safe_set))
            score += W_HEADING*(score_heading_set[idx]/np.sum(score_heading_set))
            score += W_DIST*(score_dist_set[idx]/np.sum(score_dist_set))
            score += W_FAST*(score_fast_set[idx]/np.sum(score_fast_set))*(-1)
        score_set = np.append(score_set, score)

    best_idx = np.argmax(score_set)


    # 6.返回最佳trajectory，控制量，追踪点
    # trajectory = get_simulated_trajectory_and_check_collision(cur_X, cur_Y, cur_Theta, cur_v_x, cur_v_y, cur_w_z,
    #                                                           cur_v_x, cur_v_y, cur_w_z,
    #                                                           map, map_width, map_height, map_origin_X, map_origin_Y, map_resolution)
    # return [0,0,0], trajectory, trajectory[0]
    return control_set[best_idx], trajectory_set[best_idx], tracking_Point



### functions
def get_speed_and_delta_limits(cur_v_x, cur_v_y, cur_w_z):
    '''
    得到v和delta的上下取值范围
    '''
    min_v_x = max([MIN_V_X, cur_v_x + MIN_A_X*DT])
    max_v_x = min([MAX_V_X, cur_v_x + MAX_A_X*DT])
    min_v_y = max([MIN_V_Y, cur_v_y + MIN_A_Y*DT])
    max_v_y = min([MAX_V_Y, cur_v_y + MAX_A_Y*DT])
    min_w_z = max([MIN_W_Z, cur_w_z + MIN_A_W_Z*DT])
    max_w_z = min([MAX_W_Z, cur_w_z + MAX_A_W_Z*DT])
    return min_v_x, max_v_x, min_v_y, max_v_y, min_w_z, max_w_z


def get_tracking_point_and_tracking_trajectory(cur_X, cur_Y, L, reference_trajectory):
    '''
    得到tracking point和tracking trajectory(当前位置到tracking point间的轨迹点)
    '''
    # closest_idx
    closest_idx = np.argmin((reference_trajectory[:,1]-cur_Y)**2 + (reference_trajectory[:,0]-cur_X)**2)
    # tracking_idx
    for idx in range(closest_idx, len(reference_trajectory)):
        X, Y, Theta = reference_trajectory[idx][0], reference_trajectory[idx][1], reference_trajectory[idx][2]
        if hypot(Y-cur_Y, X-cur_X) > L:
            return [X, Y, Theta], reference_trajectory[closest_idx:idx]
    return reference_trajectory[-1][0:3], reference_trajectory[closest_idx:]


def get_simulated_trajectory_and_check_collision(cur_X, cur_Y, cur_Theta, cur_v_x, cur_v_y, cur_w_z, control_v_x, control_v_y, control_w_z, map, map_width, map_height, map_origin_X, map_origin_Y, map_resolution, prediction_time):
    '''
    得到前向模拟出的路径点(如果collision free loose都不满足的直接放弃该路径)
    '''
    cur_state = [cur_X, cur_Y, cur_Theta, cur_v_x, cur_v_y, cur_w_z]
    trajectory = [cur_state]
    n_steps = int(prediction_time/DT)
    for i in range(n_steps):
        cur_state = get_next_state(cur_state, [control_v_x, control_v_y, control_w_z], DT)
        cur_X, cur_Y = cur_state[0], cur_state[1]
        cur_cmap_X, cur_cmap_Y = XY_to_cmap_XY(cur_X, cur_Y, map_origin_X, map_origin_Y, map_resolution)
        if not is_node_free_and_in_map_range(cur_cmap_X, cur_cmap_Y, map, map_width, map_height):
            return []
        trajectory.append(cur_state)
    return trajectory


@numba.njit()
def score_trajectory_by_deviation(trajectory, reference_trajectory, map, map_width, map_height, map_origin_X, map_origin_Y, map_resolution): #[X, Y, Theta, speed, delta]
    '''
    偏移评价
    '''
    total_dist = 0
    for state in trajectory:
        X, Y = state[0], state[1]
        idx = np.argmin((reference_trajectory[:,1]-Y)**2 + (reference_trajectory[:,0]-X)**2)
        total_dist += hypot(reference_trajectory[idx][1]-Y, reference_trajectory[idx][0]-X)
    return 99 - total_dist


def score_trajectory_by_safe(trajectory, map, map_width, map_height, map_origin_X, map_origin_Y, map_resolution):
    '''
    安全评价
    '''
    score = 2
    for XY in trajectory:
        cur_cmap_X, cur_cmap_Y = XY_to_cmap_XY(XY[0], XY[1], map_origin_X, map_origin_Y, map_resolution)
        if not is_node_free_and_in_map_range_strict(cur_cmap_X, cur_cmap_Y, map, map_width, map_height):
            score = 0.01
    return score


def score_trajectory_by_fast(trajectory):
    '''
    速度评价
    '''
    average_speed = abs(np.average(trajectory[1:,3:5])) # [X, Y, Theta, vx, vy ,wz] 考虑vx,vy平均
    return average_speed + 0.01


def score_trajectory_by_heading(trajectory, tracking_point):
    '''
    heading评价
    '''
    tracking_Theta = tracking_point[2]
    end_Theta = trajectory[-1][2]
    
    Theta_gap = get_Theta_gap(end_Theta, tracking_Theta)

    return pi- Theta_gap + 0.01


def score_trajectory_by_final_dist(trajectory, tracking_point):
    '''
    到达程度评价
    '''
    dist = hypot(tracking_point[1]-trajectory[-1][1], tracking_point[0]-trajectory[-1][0])
    return 99 - dist


def get_Theta_gap(Theta, tracking_Theta):
    if abs(tracking_Theta - Theta) > pi:
        return pi - abs(tracking_Theta) + pi - abs(Theta)
    else:
        return abs(tracking_Theta - Theta)


