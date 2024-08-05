#!/usr/bin/python3.8
import numpy as np
import numba
import rospy
from random import uniform
from math import cos, sin, atan2, hypot, radians, pi, floor, sqrt
from utils import *



INF = 999999999.



### aStar Core in numba style
# 在网格下,根据起点格和终点格使用a*算法算出路径
# @numba.njit()
def run_jpsCore(map, map_width, map_height, start_X, start_Y, goal_X, goal_Y):
    '''
    # JPS
    每一次循环：
    1)上下左右线搜索, 搜索过程中发现跳点(不在closedset), 则该线搜方向结束, 跳点添加至open list, 继续其他方向线搜索
    2)左上斜搜索，发现跳点(不在closedset), 则跳点添加至openset, 左上搜索结束(更改为不结束，也是搜索完为止), 否则左上移动一格记录为新父节点,左和上子线搜索,发现不在closedset的跳点,则新父节点添加至openset,左上搜索结束(更改为不结束，也是搜索完为止),
        否则, 重复斜向搜索+左上+线搜索直至因跳点结束或无法继续左上为止
    3)进行右上、右下、左下
    下一次循环
    补充: 起点S和终点E均可以被视为跳点(但真实检测的时候往往不会检测起点)
    '''
    # 1. open_set, closed_set, start loop
    open_set = np.array([[start_X, start_Y, 0., cal_h(start_X, start_Y, goal_X, goal_Y), INF, INF]]) # Node的形式[X, Y, g, h, parent_X, parent_Y]
    closed_set = np.array([[INF, INF, INF, INF, INF, INF]]) # 必须这么写来保持array是二维数组,给空或者只给一个列表都会导致后期向其中新增[[X, Y, g, h, pX, pY]]显示array结构不相符
    visited_node_set = np.zeros(np.shape(map))
    visited_node_set[start_X][start_Y] = 1

    iteration = 0
    while not rospy.is_shutdown():
        iteration += 1
        # print(f"iteration:{iteration}   len_openset:{len(open_set)}     len_closedset:{len(closed_set)}")

        if len(open_set) == 0:
            return None, open_set, closed_set
            

    # 2. cur_node: open_set.poop closed_set.add
        cur_node_idx = get_min_cost_node_idx(open_set)
        cur_node = open_set[cur_node_idx]
        cur_X, cur_Y, cur_g, father_X, father_Y = int(cur_node[0]), int(cur_node[1]), int(cur_node[2]), int(cur_node[4]), int(cur_node[5])
        closed_set = np.append(closed_set, [cur_node], axis=0)
        open_set = np.delete(open_set, cur_node_idx, axis=0)


    # 3. if cur_node == goal node: backtracing
        if cur_X == goal_X and cur_Y == goal_Y:
            return backtracing(cur_X, cur_Y, closed_set), open_set, closed_set

            
    # 4. expand cur node:
        dir = round(atan2(cur_Y-father_Y, cur_X-father_X), 2)
        # 父节点到cur为垂直/水平 或 当前节点为初始节点(father_XY为INF), 则搜索上下左右对角线8方向，考虑visisted set
        if (abs(dir) in [0, 1.57, 3.14]) or (father_X == INF):
            line_search_directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            angular_search_directions = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
            for direction in line_search_directions:
                visited_node_set, open_set = search_line(cur_X, cur_Y, cur_g, direction[0], direction[1], map, map_width, map_height, goal_X, goal_Y, visited_node_set, open_set, closed_set,
                                                         False,
                                                         INF, INF, INF)
            for direction in angular_search_directions:
                visited_node_set, open_set = search_diagonal(cur_X, cur_Y, cur_g, direction[0], direction[1], map, map_width, map_height, goal_X, goal_Y, visited_node_set, open_set, closed_set,
                                                             False)
        # 父节点到cur为对角线，则搜索对角线方向和其水平与垂直分量3方向, 不考虑visited set
        else:
            # 右上
            if dir == 0.79:
                line_search_directions = [[1, 0], [0, 1]]
                angular_search_directions = [[1, 1]]
            # 右下
            elif dir == -0.79:
                line_search_directions = [[1,0], [0, -1]]
                angular_search_directions = [[1, -1]]
            # 左下
            elif dir == -2.36:
                line_search_directions = [[-1,0], [0, -1]]
                angular_search_directions = [[-1, -1]]
            # 左上
            elif dir == 2.36:
                line_search_directions = [[-1,0], [0, 1]]
                angular_search_directions = [[-1, 1]]
            
            for direction in line_search_directions:
                visited_node_set, open_set = search_line(cur_X, cur_Y, cur_g, direction[0], direction[1], map, map_width, map_height, goal_X, goal_Y, visited_node_set, open_set, closed_set,
                                                         True,
                                                         INF, INF, INF)
            for direction in angular_search_directions:
                visited_node_set, open_set = search_diagonal(cur_X, cur_Y, cur_g, direction[0], direction[1], map, map_width, map_height, goal_X, goal_Y, visited_node_set, open_set, closed_set,
                                                             True)



### JPS functions
'''
1. 方向的定义， open set, closed set, visit_status
#               上      下      左       右
# direction X   1       -1      0       0
# direction Y   0       0       1       -1
        
#               左上    右上     右下     左下
# direction X   1       1       -1       -1
# direction Y   1       -1      -1       1

2. 扩展逻辑:
2.1 直线搜索:
1) 如果current当前方向可走,则沿current当前方向寻找不在closedset的跳点,跳点添加至openset
直线搜索至找到跳点/无路可走为止

2.2 斜向搜索:
1) 如果current当前方向可走,则沿current当前方向寻找不在closedset的跳点,跳点添加至openset
2) 如果current当前方向的水平分量可走(例如current当前为东北方向,则水平分量为东,垂直分量为北),则沿current当前方向的水平分量寻找不在closedset的跳点,current添加至openset
3) 如果current当前方向的垂直分量可走(例如current当前为东北方向,则水平分量为东,垂直分量为北),则沿current当前方向的垂直分量寻找不在closedset的跳点,current添加至openset
循环 1) 2) 3) 至找到跳点/无路可作为止

2.3 总逻辑:
1) 选最低f点cur_node
2) cur_node探索
循环

3. 移动和跳点定义:跳点不能在closedset内
方案1. ***
3.1 直线搜索
    移动: P左为O, 可以走C, 然后判断C周围
        O O O
        O O P
        O O O

        O O O
        O C P
        O O O
    0) 前提: C左O
        O O O
        X C P
        O O O
    1) C为跳点: C左下O, C下X
        O O O
        O C P
        O X O
    2) C为跳点: C左上O, C上X
        O X O
        O C P
        O O O
    3) C为跳点: 1)+2)情况
        O X O 
        O C P
        O X O

3.2 斜向
    移动: P左上为O, 且P左和不同时为X, 可以走C, 然后判断C周围
        O O O        O O O
        O O O   AND  O O X
        O O P        O X P

        O O O
        O C O
        O O P
    1) C为跳点: C左下O, C下X, C左O
        O O O
        O C O
        O X P
    2) C为跳点: C右上O, C右X, C上O
        O O O
        O C X
        O O P
    3) 1)+2)情况: C不为跳点
        O O O
        O C X
        O X P

方案2. 相对方案1, 直线搜索中若C为跳点, 则向C的左侧移动一格, 将这个格当作跳点; 斜搜索中若
3.1 直线搜索
    移动: P左为O, 可以走C, 然后判断C周围
        O O O
        O O P
        O O O

        O O O
        O C P
        O O O
    0) 前提: C左O
        O O O
        X C P
        O O O
    1) T为跳点: C左下O, C下X
        O O O
        T C P
        O X O
    2) T为跳点: C左上O, C上X
        O X O
        T C P
        O O O
    3) T为跳点: 1)+2)情况
        O X O 
        T C P
        O X O

3.2 斜向
    移动: P左上为O, 且P左和不同时为X, 可以走C, 然后判断C周围
        O O O        O O O
        O O O   AND  O O X
        O O P        O X P

        O O O
        O C O
        O O P
    1) C为跳点: C左下O, C下X, C左O
        T O O
        O C O
        O X P
    2) C为跳点: C右上O, C右X, C上O
        T O O
        O C X
        O O P
    3) 1)+2)情况: C不为跳点
        T O O
        O C X
        O X P

3.3 起点、终点
1) C为起点, C为跳点
2) C为终点, C为跳点

4. 细节
1) openset, closed set在搜索过程中是否需要被考虑: 是否需要考虑该点已经在openset里? 已经在closedset里?
    跳点在closed set中: 如果发现的跳点已经在closed set中, 即已经被探索过并确定了其最优路径, 那么可以安全地忽略它
    跳点在open set中: 比较g值, 更新open set中的跳点
    跳点不在open set中, 不在closed set中: 添加进open set
    方案1. 仅考虑跳点是不是在closedset,如果在则忽略，不在则视为有效跳点 ***
    方案2. 考虑跳点是不是在closedset,如果在则忽略, 不在则视为有效跳点, 考虑跳点在不在openset,在则比较更新, 不在则添加跳点或father
    方案3. 考虑跳点是不是在closedset,如果在则忽略, 考虑跳点在不在openset,在则比较更新,不在则视为有效跳点， 添加跳点或father


2) 新节点搜索什么方向？方向顺序是什么?每个方向的终止条件是什么？
    方案1.搜索上下左右对角线8个方向; 顺序可以固定 (也可以每次对节点扩展前, 根据atan2判读终点与节点的位置关系, 比如终点在右上, 则先扩展右上，然后上，右，然后按顺序来)
        每个方向探索到有效跳点、撞墙、撞visitedset结束
        但是这样假如在子线搜索找到跳点添加的是father,在从father扩展会撞visitedset,因为导致该father被放入openset的跳点已经被放在visitedst里了

    方案2.根据父节点方位来:  ***
        1) 如果父节点到当前节点方向为水平或垂直
            则探索上下左右对角线8个方向, 每个方向探索到有效跳点、撞墙、撞到visistedset结束
        2)如果父节点到当前节点方向为对角线
            则探索来自的方向和其水平分量垂直分量3个方向(如右、右上、右), 每个方向探索到有效跳点、撞墙结束

3) 搜索到nonclosed跳点,进行什么操作？
    线搜索: 将跳点加入openset
    斜向走一步: 将跳点加入openset
    斜向走完后子线搜索: 将father加入openset
'''

@numba.njit()
def search_diagonal(cur_X, cur_Y, cur_g, direction_X, direction_Y, map, map_width, map_height, goal_X, goal_Y, visited_node_set, open_set, closed_set,
                    ignore_visited_set):
    cur_X, cur_Y, direction_X, direction_Y = int(cur_X), int(cur_Y), int(direction_X), int(direction_Y)
    father_X, father_Y, father_g = cur_X, cur_Y, cur_g

    # 左上对角线移动
    while True:
        # 移动一格子: 左上O 且 不为左X上X 且 左上unvisited， 否则斜搜索结束
        if is_node_free_and_in_map_range(cur_X+direction_X, cur_Y+direction_Y, map, map_width, map_height) == False:
            return visited_node_set, open_set
        
        if (is_node_free_and_in_map_range(cur_X+direction_X, cur_Y, map, map_width, map_height) == False \
            and is_node_free_and_in_map_range(cur_X, cur_Y+direction_Y, map, map_width, map_height) == False):
            return visited_node_set, open_set
        
        if (not ignore_visited_set) and visited_node_set[cur_X+direction_X][cur_Y+direction_Y] == 1:
            return visited_node_set, open_set
        
        cur_X += direction_X
        cur_Y += direction_Y
        visited_node_set[cur_X][cur_Y] = 1

        # 判断cur是否为终点
        if cur_X == goal_X and cur_Y == goal_Y:
            jump_point_node = create_node(cur_X, cur_Y, father_X, father_Y, father_g, goal_X, goal_Y)
            open_set = np_append_in_numba(open_set, jump_point_node)
            return visited_node_set, open_set        

        # 判断cur是否为有效跳点: 不在closedset 且 C左下O, C下X, C左O 或 C右上O, C右X, C上O 或 为终点， 则添加openset，斜搜索结束, 否则从该点开始子线搜索左和上
        cur_in_closed_set_array = get_node_idx_array_in_set(cur_X, cur_Y, closed_set)
        if len(cur_in_closed_set_array) == 0:
            if (is_node_free_and_in_map_range(cur_X+direction_X, cur_Y-direction_Y, map, map_width, map_height) == True \
            and is_node_free_and_in_map_range(cur_X+direction_X, cur_Y, map, map_width, map_height) == True \
            and is_node_free_and_in_map_range(cur_X, cur_Y-direction_Y, map, map_width, map_height) == False) \
            or \
            (is_node_free_and_in_map_range(cur_X-direction_X, cur_Y+direction_Y, map, map_width, map_height) == True \
            and is_node_free_and_in_map_range(cur_X, cur_Y+direction_Y, map, map_width, map_height) == True \
            and is_node_free_and_in_map_range(cur_X-direction_X, cur_Y, map, map_width, map_height) == False):
                jump_point_node = create_node(cur_X, cur_Y, father_X, father_Y, father_g, goal_X, goal_Y)
                open_set = np_append_in_numba(open_set, jump_point_node)
                return visited_node_set, open_set

        
        # 子线搜索,从cur开始搜索 direction_X,0 和 0,direction_Y 两个方向, 搜索过程中发现有效跳点，则将cur添加至openset，否则继续下一轮循环
        prev_len_open_set = len(open_set)
        for direction in [[direction_X, 0], [0, direction_Y]]:
            visited_node_set, open_set = search_line(cur_X, cur_Y, cur_g, direction[0], direction[1], map, map_width, map_height, goal_X, goal_Y, visited_node_set, open_set, closed_set, 
                                                     ignore_visited_set,
                                                     father_X, father_Y, father_g)
            # 子线搜索过程中找到了跳点, 本斜搜索结束
            if len(open_set) > prev_len_open_set: 
                return visited_node_set, open_set
                

# 线搜索: 找到合适的跳点后将跳点添加openset并停止线搜索 # Node的形式[X, Y, g, h, parent_X, parent_Y]
# 子线搜索: 斜搜索中，找到合适的跳点后将父节点添加openset并停止斜搜索
@numba.njit()
def search_line(cur_X, cur_Y, cur_g, direction_X, direction_Y, map, map_width, map_height, goal_X, goal_Y, visited_node_set, open_set, closed_set,
                ignore_visited_set, 
                grand_X, grand_Y, grand_g):
    # 线搜索
    cur_X, cur_Y, direction_X, direction_Y = int(cur_X), int(cur_Y), int(direction_X), int(direction_Y)
    father_X, father_Y, father_g = cur_X, cur_Y, cur_g
    child_search = False
    # 子线搜索需要补充这些
    if grand_X != INF:
        child_search = True
        grand_X, grand_Y = int(grand_X), int(grand_Y)


    # 判断为水平移动(X方向)
    if abs(direction_X) == 1 and abs(direction_Y) == 0:
        # 向左移动
        while True:
            # 移动一格:如果左侧为free且unvisited，向左一格得到新的curXY， 否则停止搜索
            if (is_node_free_and_in_map_range(cur_X+direction_X, cur_Y, map, map_width, map_height) == False):
                return visited_node_set, open_set
            if (not ignore_visited_set) and (visited_node_set[cur_X+direction_X][cur_Y] == 1):
                return visited_node_set, open_set
            cur_X += direction_X
            visited_node_set[cur_X][cur_Y] = 1

            # 判断cur是否为终点
            if cur_X == goal_X and cur_Y == goal_Y:
                if not child_search:
                    jump_point_node = create_node(cur_X, cur_Y, father_X, father_Y, father_g, goal_X, goal_Y)
                    open_set = np_append_in_numba(open_set, jump_point_node)
                else:
                    father_point_node = create_node(father_X, father_Y, grand_X, grand_Y, grand_g, goal_X, goal_Y)
                    open_set = np_append_in_numba(open_set, father_point_node) 
                return visited_node_set, open_set

            # 判断cur是否为跳点: C左O 且 C左下O，C下X 或 C左上O，C上X 且 cur不在closedset，或为终点, 将cur添加到openset，停止搜索，否则继续搜索
            if (is_node_free_and_in_map_range(cur_X+direction_X, cur_Y, map, map_width, map_height) == True):
                if ((is_node_free_and_in_map_range(cur_X+direction_X, cur_Y-1, map, map_width, map_height) == True) and \
                    (is_node_free_and_in_map_range(cur_X, cur_Y-1, map, map_width, map_height) == False)) \
                    or \
                    ((is_node_free_and_in_map_range(cur_X+direction_X, cur_Y+1, map, map_width, map_height) == True) and \
                     (is_node_free_and_in_map_range(cur_X, cur_Y+1, map, map_width, map_height) == False)):
                    cur_in_closed_set_array = get_node_idx_array_in_set(cur_X, cur_Y, closed_set)
                    if len(cur_in_closed_set_array) == 0:
                        if not child_search:
                            jump_point_node = create_node(cur_X, cur_Y, father_X, father_Y, father_g, goal_X, goal_Y)
                            open_set = np_append_in_numba(open_set, jump_point_node)
                        else:
                            father_point_node = create_node(father_X, father_Y, grand_X, grand_Y, grand_g, goal_X, goal_Y)
                            open_set = np_append_in_numba(open_set, father_point_node) 
                        return visited_node_set, open_set


    # 判断为垂直移动(Y方向)
    elif abs(direction_X) == 0 and abs(direction_Y) == 1:
        # 向上移动
        while True:
            # 移动一格:如果上侧free且unvisited，向上一格得到新的curXY， 否则停止搜索
            if (is_node_free_and_in_map_range(cur_X, cur_Y+direction_Y, map, map_width, map_height) == False):
                return visited_node_set, open_set
            if (not ignore_visited_set) and (visited_node_set[cur_X][cur_Y+direction_Y] == 1):
                return visited_node_set, open_set
            cur_Y += direction_Y
            visited_node_set[cur_X][cur_Y] = 1

            # 判断cur是否为终点
            if cur_X == goal_X and cur_Y == goal_Y:
                if not child_search:
                    jump_point_node = create_node(cur_X, cur_Y, father_X, father_Y, father_g, goal_X, goal_Y)
                    open_set = np_append_in_numba(open_set, jump_point_node)
                else:
                    father_point_node = create_node(father_X, father_Y, grand_X, grand_Y, grand_g, goal_X, goal_Y)
                    open_set = np_append_in_numba(open_set, father_point_node) 
                return visited_node_set, open_set

            # 判断cur是否为跳点: 上O 且 左上O，左X 或 右上O，右X 且cur不在closedset，将cur添加到openset，停止搜索，否则继续搜索
            if (is_node_free_and_in_map_range(cur_X, cur_Y+direction_Y, map, map_width, map_height) == True):
                if ((is_node_free_and_in_map_range(cur_X-1, cur_Y+direction_Y, map, map_width, map_height) == True) and \
                    (is_node_free_and_in_map_range(cur_X-1, cur_Y, map, map_width, map_height) == False)) \
                    or \
                    ((is_node_free_and_in_map_range(cur_X+1, cur_Y+direction_Y, map, map_width, map_height) == True) and \
                     (is_node_free_and_in_map_range(cur_X+1, cur_Y, map, map_width, map_height) == False)):
                    cur_in_closed_set_array = get_node_idx_array_in_set(cur_X, cur_Y, closed_set)
                    if len(cur_in_closed_set_array) == 0:
                        if not child_search:
                            jump_point_node = create_node(cur_X, cur_Y, father_X, father_Y, father_g, goal_X, goal_Y)
                            open_set = np_append_in_numba(open_set, jump_point_node)
                        else:
                            father_point_node = create_node(father_X, father_Y, grand_X, grand_Y, grand_g, goal_X, goal_Y)
                            open_set = np_append_in_numba(open_set, father_point_node) 
                        return visited_node_set, open_set
    
    # 判断为其他移动，警告
    else:
        print("线搜索或子线搜索的X,Y不符合其中只有一个绝对值为1另一个0的标准, 请检查...")
        return


@numba.njit
def create_node(X, Y, parent_X, parent_Y, parent_g, goal_X, goal_Y):
    return np.array([[X, Y, parent_g + hypot(parent_Y-Y, parent_X-X), cal_h(X, Y, goal_X, goal_Y), parent_X, parent_Y]])


def get_visited_XYs(visited_node_set):
    visited_XYs = []
    row, col = np.shape(visited_node_set)
    for i in range(row):
        for j in range(col):
            if visited_node_set[i][j] != 0:
                visited_XYs.append([i,j])
    return visited_XYs



### aStar function
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
    return expected_dist


# get the node with min cost from open_set
@numba.njit()
def get_min_cost_node_idx(open_set): # [X, Y, g, h, parent_X, parent_Y]
    return np.argmin(open_set[:,2] + open_set[:,3])


# get the idx of node in set that match with X, Y: if no math the len(idx) == 0, else the idx valid means there's matching in the set # 注意这里返回None, numba中 None == None 返回的是false,所以建议所有的比较用idx is None
@numba.njit()
def get_node_idx_array_in_set(X, Y, set): # Node的形式[X, Y, g, h, parent_X, parent_Y]
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