#!/usr/bin/python3.8
import numpy as np
import numba
import rospy
from random import uniform
from math import cos, sin, atan2, hypot, radians, pi, floor, sqrt, ceil



INF = 999999999.



### aStar Core in numba style
# 在网格下,根据起点格和终点格使用a*算法算出路径
# @numba.njit()
def aStar_smoothing(node_XYs, map, map_width, map_height):
    # 0. start, goal
    if node_XYs is None:
        return None
    
    node_XYs = remove_aligned_node(node_XYs)
    start_node_X, start_node_Y = node_XYs[0][0], node_XYs[0][1]
    goal_node_X, goal_node_Y = node_XYs[-1][0], node_XYs[-1][1]


    # 1. open_set, closed_set, start loop
    open_set = np.array([[start_node_X, start_node_Y, 0., cal_h(start_node_X, start_node_Y, goal_node_X, goal_node_Y), INF, INF]]) # Node的形式[X, Y, g, h, parent_X, parent_Y]
    closed_set = np.array([[INF, INF, INF, INF, INF, INF]]) # 必须这么写来保持array是二维数组,给空或者只给一个列表都会导致后期向其中新增[[X, Y, g, h, pX, pY]]显示array结构不相符
    iteration = 0
    while not rospy.is_shutdown():
        iteration += 1
        if len(open_set) == 0:
            return None
        

    # 2. cur_node: open_set.poop closed_set.add
        cur_node_idx = get_min_cost_node_idx(open_set)
        cur_node = open_set[cur_node_idx]
        cur_X, cur_Y, cur_g = cur_node[0], cur_node[1], cur_node[2]
        closed_set = np.append(closed_set, [cur_node], axis=0)
        open_set = np.delete(open_set, cur_node_idx, axis=0)
        cur_i_array = get_node_idx_array_in_set(cur_X, cur_Y, np.array(node_XYs))
        cur_i = cur_i_array[0]


    # 3. if cur_node == goal node: backtracing
        if cur_X == goal_node_X and cur_Y == goal_node_Y:
            return backtracing(cur_X, cur_Y, closed_set)
            

    # 4. near_node:
        if cur_i < len(node_XYs) - 3:
            best_first_node_i = INF
            furtherest_j = cur_i
            for i in range(cur_i+1, len(node_XYs)):
                # first node
                first_X, first_Y = node_XYs[i]
                if not is_path_free_and_in_map_range(cur_X, cur_Y, first_X, first_Y, map, map_width, map_height):
                    continue
                
                # second node
                for j in range(i+1, len(node_XYs)):
                    second_X, second_Y = node_XYs[j]
                    if not is_path_free_and_in_map_range(first_X, first_Y, second_X, second_Y, map, map_width, map_height):
                        continue
                    if j >= furtherest_j:
                        furtherest_j = j
                        best_first_node_i = i

            best_X, best_Y = node_XYs[best_first_node_i]
            best_g = cur_g + hypot(best_Y-cur_Y, best_Y-cur_X)
            best_h = cal_h(best_X, best_Y, goal_node_X, goal_node_Y)
            open_set = np_append_in_numba(open_set, np.array([[best_X, best_Y, best_g, best_h, cur_X, cur_Y]]))
        else:
            for i in range(cur_i, len(node_XYs)):
                first_X, first_Y = node_XYs[i]
                if not is_path_free_and_in_map_range(cur_X, cur_Y, first_X, first_Y, map, map_width, map_height):
                    continue
                dist = hypot(first_Y - cur_Y, first_X - cur_X)
                open_set = np_append_in_numba(open_set, np.array([[first_X, first_Y, cur_g + dist, cal_h(first_X, first_Y, goal_node_X, goal_node_Y), cur_X, cur_Y]]))



### smooting functions
# 去除所有共线的中间点
def remove_aligned_node(node_XYs):
    remove_XYs = []
    # 去除共线点
    for i in range(len(node_XYs) - 2):
        X1, Y1 = node_XYs[i][0], node_XYs[i][1]
        X2, Y2 = node_XYs[i+1][0], node_XYs[i+1][1]
        X3, Y3 = node_XYs[i+2][0], node_XYs[i+2][1]
        first_vector = atan2(Y2 - Y1, X2 - X1)
        second_vector = atan2(Y3 - Y2, X3 - X2)
        if first_vector == second_vector:
            remove_XYs.append([X2, Y2])
    node_XYs = [XY for XY in node_XYs if XY not in remove_XYs]
    return node_XYs


# 判断两点连线会不会碰到障碍
@numba.njit()
def is_path_free_and_in_map_range(from_X, from_Y, to_X, to_Y, map, map_width, map_height):
    extend_resolution = 1

    dist = hypot(to_Y-from_Y, to_X-from_X)
    theta = atan2(to_Y-from_Y, to_X-from_X)
    n_extend = floor(dist/extend_resolution)

    X, Y = from_X, from_Y
    for _ in range(n_extend):
        X += extend_resolution*cos(theta)
        Y += extend_resolution*sin(theta)
        
        if not is_node_free_and_in_map_range(X, Y, map, map_width, map_height):
            return False
    
    return True


# 在判断是否碰撞时，只有该点上下左右四个点全部occupied的情况下，才认为该点碰撞
@numba.njit()
def is_node_free_and_in_map_range(X, Y, map, map_width, map_height):
    # XY turn to occuped format
    resolution = 1
    newXY = []
    for num in [X,Y]:
        rem = num % resolution
        if rem >= resolution/2:
            newXY.append(round(num - rem + resolution, 2))
        else:
            newXY.append(round(num - rem, 2))
    X, Y = newXY[0], newXY[1]

    # check occupied status
    int_X, int_Y = int(X), int(Y)
    if map[int_X][int_Y] > 0:
        if (map[int_X+1][int_Y] != 0 and map[int_X-1][int_Y] != 0 and map[int_X][int_Y+1] != 0 and map[int_X][int_Y-1] != 0):
            return False
    
    if X < 0 or X > map_width - 1 or Y < 0 or Y > map_height:
        return False
    
    return True



### aStar functions
# get h cost from X Y to goal_X, goal_Y
@numba.njit()
def cal_h(X, Y, goal_X, goal_Y):
    # 欧氏距离：欧几里得距离,即勾股定理中的斜边长度
    # return hypot(goal_Y - Y, goal_X - X)

    # 切比雪夫距离(Chebyshev Distance):
    # return max(abs(goal_Y - Y), abs(goal_X - X))

    # 切比雪夫距离.改 
    dy_abs = abs(goal_Y - Y)
    dx_abs = abs(goal_X - X)
    min_d = min(dy_abs, dx_abs)
    max_d = max(dy_abs, dx_abs)
    expected_dist = (sqrt(2) * min_d) + (max_d - min_d)
    # p = 1.414/expected_dist * 0.5 if expected_dist != 0 else 0 # Tie breaker
    p = 0.001
    return expected_dist * (1 + p)

    # 0
    # return 0


@numba.njit()
def np_append_in_numba(arr, object_arr):
    row, col = np.shape(arr)
    arr = np.append(arr, object_arr)
    arr = np.reshape(arr, (row+1, col))
    return arr


@numba.njit()
def np_delete_in_numba(arr, idx):
    row, col = np.shape(arr)
    del_idxs = list(range(idx*col, idx*col+col))
    arr = np.delete(arr, del_idxs)
    arr = np.reshape(arr, (row-1, col))
    return arr


# get the node with min cost from open_set
@numba.njit()
def get_min_cost_node_idx(open_set): # [X, Y, g, h, parent_X, parent_Y]
    return np.argmin(open_set[:,2] + open_set[:,3])


# get the idx of node in set that match with X, Y: if no math the len(idx) == 0, else the idx valid means there's matching in the set # 注意这里返回None, numba中 None == None 返回的是false,所以建议所有的比较用idx is None
@numba.njit()
def get_node_idx_array_in_set(X, Y, set): # Node的形式[X, Y, g, h, parent_X, parent_Y]
    X, Y = int(X), int(Y)
    condition = ((set[:,0] == X) & (set[:,1] == Y))
    idx_array = np.where(condition) # array([[50]])
    return idx_array[0]


# backtracing
# @numba.njit() # 不使用numba是因为backtracing只会在最后被调用一次,不会在循环在重复调用,不用numba反而更快
def backtracing(goal_X, goal_Y, closed_set):
    path_XYs = []
    cur_X, cur_Y = goal_X, goal_Y

    while not (cur_X == INF or cur_Y == INF):
        path_XYs.append([cur_X, cur_Y])
        cur_node_idx_array = get_node_idx_array_in_set(cur_X, cur_Y, closed_set)
        cur_node = closed_set[cur_node_idx_array[0]]
        cur_X, cur_Y = cur_node[4], cur_node[5]

    return path_XYs