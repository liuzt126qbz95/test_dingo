#!/usr/bin/python3.8
import numpy as np
import numba
import rospy
from random import uniform
from math import cos, sin, atan2, hypot, radians, pi, floor, sqrt



INF = 999999999.



### aStar Core in numba style
# 在网格下，根据起点格和终点格使用a*算法算出路径
# @numba.njit()
def run_aStarBiCore(map, map_width, map_height, start_X, start_Y, goal_X, goal_Y):
    '''
    priority queue::open_set, closed_set, open_set.add(start)
    while len(open_set) > 0:
        cur_node = open_set.pop()
        closed_set.add(cur_node)
        if cur_node == goal_node: 
            goal_node.parent_idx = cur_node.parent_idx
            goal_node.cost = cur_node.cost
            break
        else:
            for near_node in near_nodes:
                if near_node occupied or near_node in closed_set:
                    continue
                
                near_cost = g(near_node) + h(near_node)
                if near_node in open_set:
                    if near_cost < open_set(near_node).cost:
                        open_set(near_node).cost = near_cost
                        open_set(near_node).parent_idx = near_node.parent_idx

                if near_node not in open_set:
                    open_set.add(near_node)
    
    if len(open_set) == 0:
        path_XYs = None
    else:
        path_XYs = backtracing(goal_node)
    
    return path_XYs
    '''
    # 0. connect parten
    connects = [[1, 0], # 8 connects
                [1, 1],
                [0, 1],
                [-1, 1],
                [-1, 0],
                [-1, -1],
                [0, -1],
                [1, -1]]


    # 1. open_set, closed_set, start loop
    ### BIDIRECTIONAL
    open_set_1 = np.array([[start_X, start_Y, 0., cal_h(start_X, start_Y, goal_X, goal_Y), INF, INF]]) # Node的形式[X, Y, g, h, parent_X, parent_Y]
    open_set_2 = np.array([[goal_X, goal_Y, 0., cal_h(goal_X, goal_Y, start_X, start_Y), INF, INF]]) # Node的形式[X, Y, g, h, parent_X, parent_Y]
    closed_set_1 = np.array([[INF, INF, INF, INF, INF, INF]]) # 必须这么写来保持array是二维数组，给空或者只给一个列表都会导致后期向其中新增[[X, Y, g, h, pX, pY]]显示array结构不相符
    closed_set_2 = np.array([[INF, INF, INF, INF, INF, INF]]) # 必须这么写来保持array是二维数组，给空或者只给一个列表都会导致后期向其中新增[[X, Y, g, h, pX, pY]]显示array结构不相符
    while not rospy.is_shutdown():
        len_open_set_1 = len(open_set_1)
        len_open_set_2 = len(open_set_2)
        if len_open_set_1 == 0 or len_open_set_2 == 0:
            return None
  

    # 2. cur_node: open_set.poop closed_set.add
        ### BIDIRECTIONAL
        cur_node_idx_1 = get_min_cost_node_idx(open_set_1)
        cur_node_idx_2 = get_min_cost_node_idx(open_set_2)
        cur_node_1 = open_set_1[cur_node_idx_1]
        cur_node_2 = open_set_2[cur_node_idx_2]
        cur_X_1, cur_Y_1, cur_g_1 = cur_node_1[0], cur_node_1[1], cur_node_1[2]
        cur_X_2, cur_Y_2, cur_g_2 = cur_node_2[0], cur_node_2[1], cur_node_2[2]
        closed_set_1 = np.append(closed_set_1, [cur_node_1], axis=0)
        closed_set_2 = np.append(closed_set_2, [cur_node_2], axis=0)
        open_set_1 = np.delete(open_set_1, cur_node_idx_1, axis=0)
        open_set_2 = np.delete(open_set_2, cur_node_idx_2, axis=0)


    # 3. near_node: 
        for connect in connects:
            offset_X, offset_Y = connect[0], connect[1]
            open_set_1 = explore(cur_X_1, cur_Y_1, cur_g_1, offset_X, offset_Y, goal_X, goal_Y, map, map_width, map_height, closed_set_1, open_set_1)
            open_set_2 = explore(cur_X_2, cur_Y_2, cur_g_2, offset_X, offset_Y, start_X, start_Y, map, map_width, map_height, closed_set_2, open_set_2)

    
    # 4. check if interaction from two openset: backtracing
        ### BIDIRECTIONAL
        interact_nodes_idxs = check_intercetion(open_set_1, open_set_2)
        interact_nodes_idxs = np.delete(interact_nodes_idxs, 0, axis=0)
        if len(interact_nodes_idxs) != 0:
            # low total cost idx1 idx2
            best_idx_1, best_idx_2 = get_cheapest_idxs(interact_nodes_idxs, open_set_1, open_set_2)

            # add to closed list
            best_node_1 = open_set_1[best_idx_1]
            best_node_2 = open_set_2[best_idx_2]
            closed_set_1 = np.append(closed_set_1, [best_node_1], axis=0)
            closed_set_2 = np.append(closed_set_2, [best_node_2], axis=0)

            # backtracing from closed list
            path_1 = backtracing(best_node_1[0], best_node_1[1], closed_set_1)
            path_2 = backtracing(best_node_2[0], best_node_2[1], closed_set_2)

            path_1.reverse()
            path_1.extend(path_2)
            path_1.reverse()

            # total_node_num = len(open_set_1) + len(open_set_2) + len(closed_set_1) + len(closed_set_2)
            # print(f"total node:{total_node_num}     start(1):{len(open_set_1)+len(closed_set_1)}      goal(2):{len(open_set_2)+len(closed_set_2)}")
            return path_1



# get h cost from X Y to goal_X, goal_Y
@numba.njit()
def cal_h(X, Y, goal_X, goal_Y):
    # 欧氏距离：欧几里得距离，即勾股定理中的斜边长度
    # return hypot(goal_Y - Y, goal_X - X)

    # 切比雪夫距离(Chebyshev Distance):
    # return max(abs(goal_Y - Y), abs(goal_X - X))

    # 切比雪夫距离.改
    dy_abs = abs(goal_Y - Y)
    dx_abs = abs(goal_X - X)
    min_d = min(dy_abs, dx_abs)
    max_d = max(dy_abs, dx_abs)
    return (sqrt(2) * min_d) + (max_d - min_d)


# get the node with min cost from open_set
@numba.njit()
def get_min_cost_node_idx(open_set): # [X, Y, g, h, parent_X, parent_Y]
    return np.argmin(open_set[:,2] + open_set[:,3])


# get the idx of node in set that match with X, Y: if no math the len(idx) == 0, else the idx valid means there's matching in the set # 注意这里返回None, numba中 None == None 返回的是false，所以建议所有的比较用idx is None
@numba.njit()
def get_node_idx_array_in_set(X, Y, set): # Node的形式[X, Y, g, h, parent_X, parent_Y]
    condition = ((set[:,0] == X) & (set[:,1] == Y))
    idx_array = np.where(condition) # array([[50]])
    return idx_array[0]


@numba.njit()
def is_node_valid(X, Y, map, map_width, map_height, closed_set):
    # not free
    if map[X][Y] > 0:
        return False

    # out of range
    if X < 0 or X > map_width - 1 or Y < 0 or Y > map_height:
        return False

    # alr in closed set
    idx_array = get_node_idx_array_in_set(X, Y, closed_set)
    if len(idx_array) != 0:
        return False

    return True


@numba.njit()
def explore(cur_X, cur_Y, cur_g, offset_X, offset_Y, goal_X, goal_Y, map, map_width, map_height, closed_set, open_set):
    near_X, near_Y = int(cur_X + offset_X), int(cur_Y + offset_Y)
    # 5. free, in range, not in closed_set
    if not is_node_valid(near_X, near_Y, map, map_width, map_height, closed_set):
        return open_set
    
    # 6. in open_set?
    old_node_idx_array = get_node_idx_array_in_set(near_X, near_Y, open_set)
    near_g = cur_g + hypot(offset_Y, offset_X)
    near_h = cal_h(near_X, near_Y, goal_X, goal_Y)

    # 6.1 in open set: compare
    if len(old_node_idx_array) != 0:
        old_node = open_set[old_node_idx_array[0]]
        old_g = old_node[2]
        if near_g < old_g:
            old_node[2] = near_g
            old_node[4] = cur_X
            old_node[5] = cur_Y

    # 6.2 not in open_set: add
    else:
        row, col = np.shape(open_set)
        open_set = np.append(open_set, [[near_X, near_Y, near_g, near_h, cur_X, cur_Y]])
        open_set = np.reshape(open_set, (row+1, col))

    return open_set


@numba.njit()
def check_intercetion(open_set_1, open_set_2):
    interact_nodes = np.array([[INF, INF]])
    for idx_2, node_in_2 in enumerate(open_set_2):
        idx_1_array = np.argwhere((open_set_1[:,0] == node_in_2[0]) & (open_set_1[:,1] == node_in_2[1]))
        if len(idx_1_array) != 0:
            idx_1 = idx_1_array[0][0]
            row, col = np.shape(interact_nodes)
            interact_nodes = np.append(interact_nodes, [[idx_1, idx_2]])
            interact_nodes = np.reshape(interact_nodes, (row+1, col))
    return interact_nodes
            

@numba.njit()
def get_cheapest_idxs(interact_nodes_idxs, open_set_1, open_set_2):
    best_f = INF
    best_idx_1, best_idx_2 = INF, INF
    for idx in interact_nodes_idxs:
        idx_1, idx_2 = int(idx[0]), int(idx[1])
        f = open_set_1[idx_1][2] + open_set_1[idx_1][3] + open_set_2[idx_2][2] + open_set_2[idx_2][3]
        if f < best_f:
            best_idx_1, best_idx_2 = idx_1, idx_2
    return int(best_idx_1), int(best_idx_2)            


# backtracing
# @numba.njit() # 不使用numba是因为backtracing只会在最后被调用一次，不会在循环在重复调用，不用numba反而更快
def backtracing(goal_X, goal_Y, closed_set):
    path_XYs = []
    cur_X, cur_Y = goal_X, goal_Y

    while not (cur_X == INF or cur_Y == INF):
        path_XYs.append([cur_X, cur_Y])
        cur_node_idx_array = get_node_idx_array_in_set(cur_X, cur_Y, closed_set)
        cur_node = closed_set[cur_node_idx_array[0]]
        cur_X, cur_Y = cur_node[4], cur_node[5]

    return path_XYs