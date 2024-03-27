### 2 loops
import numba
import numpy as np
import rospy
from math import hypot, atan2, cos, sin, floor
from utils import *


INF = 999999999.


### smoothing function
def floyd_smooting(node_XYs, map, map_width, map_height):
    start_XY = node_XYs[-1]
    goal_XY = node_XYs[0]
    # 去除共线点 不能有！否则路径不会最短
    # remove_XYs = []
    # for i in range(len(node_XYs) - 2):
    #     X1, Y1 = node_XYs[i][0], node_XYs[i][1]
    #     X2, Y2 = node_XYs[i+1][0], node_XYs[i+1][1]
    #     X3, Y3 = node_XYs[i+2][0], node_XYs[i+2][1]
    #     first_vector = atan2(Y2 - Y1, X2 - X1)
    #     second_vector = atan2(Y3 - Y2, X3 - X2)
    #     if first_vector == second_vector:
    #         remove_XYs.append([X2, Y2])
    # node_XYs = [XY for XY in node_XYs if XY not in remove_XYs]


    # 在每个节点周围添加距离两格子的节点
    node_XYs = expand_node_XYs(node_XYs, map, map_width, map_height)


    # 连接最大效率拐点
    cur_idx = 0
    reserved_XYs = [node_XYs[cur_idx]]
    while cur_idx < (len(node_XYs)-1):
        best_first_idx = get_best_first_idx(cur_idx, start_XY, node_XYs, map, map_width, map_height)
        cur_idx = best_first_idx
        reserved_XYs.append(node_XYs[cur_idx])

    return reserved_XYs



### smoother functions
@numba.njit()
def get_best_first_idx(cur_idx, start_XY, node_XYs, map, map_width, map_height):
    cur_XY = node_XYs[cur_idx]

    furthest_idx = cur_idx + 1
    min_cur_to_first_to_second_dist = INF
    best_first_idx = cur_idx + 1

    for first_idx in range(cur_idx+1, len(node_XYs)):
        first_XY = node_XYs[first_idx]
        if is_path_free_and_in_map_range(cur_XY, first_XY, map, map_width, map_height):
            if first_XY[0] == start_XY[0] and first_XY[1] == start_XY[1]:
                best_first_idx = first_idx
                break

            for second_idx in range(first_idx+1, len(node_XYs)):
                second_XY = node_XYs[second_idx]
                if is_path_free_and_in_map_range(first_XY, second_XY, map, map_width, map_height):
                    if second_idx > furthest_idx:
                        furthest_idx = second_idx
                        best_first_idx = first_idx
                        min_cur_to_first_to_second_dist = hypot(first_XY[1]-cur_XY[1], first_XY[0]-cur_XY[0]) + hypot(second_XY[1]-first_XY[1], second_XY[0]-first_XY[0])
                    if second_idx == furthest_idx:
                        cur_to_first_to_second_dist = hypot(first_XY[1]-cur_XY[1], first_XY[0]-cur_XY[0]) + hypot(second_XY[1]-first_XY[1], second_XY[0]-first_XY[0])
                        if cur_to_first_to_second_dist < min_cur_to_first_to_second_dist:
                            furthest_idx = second_idx
                            best_first_idx = first_idx
                            min_cur_to_first_to_second_dist = cur_to_first_to_second_dist
    
    return best_first_idx


@numba.njit()
def expand_node_XYs(node_XYs, map, map_width, map_height):
    start_XY = node_XYs[-1]
    goal_XY = node_XYs[0]
    offsets = [[2,0], [-2,0], [0,2], [0,-2]]

    added_node_XYs = np.array([[INF, INF]])
    added_node_XYs = np_append_in_numba(added_node_XYs, goal_XY)

    for idx in range(1, len(node_XYs)-1):
        XY = node_XYs[idx]
        added_node_XYs = np_append_in_numba(added_node_XYs, XY)
        for offset in offsets:
            add_X, add_Y = XY[0]+offset[0], XY[1]+offset[1]
            if is_node_free_and_in_map_range(add_X, add_Y, map, map_width, map_height):
                added_node_XYs = np_append_in_numba(added_node_XYs, np.array([[add_X, add_Y]]))

    added_node_XYs = np_append_in_numba(added_node_XYs, start_XY)
    added_node_XYs = np_delete_in_numba(added_node_XYs, 0)

    return added_node_XYs