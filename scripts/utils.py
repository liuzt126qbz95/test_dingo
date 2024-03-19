import numba
import numpy as np
from math import sin, cos, atan2, hypot, floor



@numba.njit()
def XY_to_cmap_XY(X, Y, map_origin_X, map_origin_Y, map_resolution):
    '''
    从正常坐标XY转换为costmap内对应的XY(即行、列数)
    '''
    cmap_X = new_int((X- map_origin_X) / map_resolution)
    cmap_Y = new_int((Y- map_origin_Y) / map_resolution)
    return cmap_X, cmap_Y



@numba.njit()
def cmap_XY_to_XY(cmap_X, cmap_Y, map_origin_X, map_origin_Y, map_resolution):
    '''
    从costmap内的XY转换为正常世界参考系下的XY坐标值
    '''
    X = map_resolution * cmap_X + map_origin_X
    Y = map_resolution * cmap_Y + map_origin_Y
    return X, Y



@numba.njit()
def is_path_free_and_in_map_range(from_XY, to_XY, map, map_width, map_height):
    '''
    判断两点(cmap格式)间连线是否碰撞障碍物
    '''
    from_X, from_Y = from_XY[0], from_XY[1]
    to_X, to_Y = to_XY[0], to_XY[1]
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



@numba.njit()
def is_node_free_and_in_map_range(X, Y, map, map_width, map_height):
    '''
    判断当前点(cmap格式)是否碰撞障碍物
    '''
    int_X, int_Y = new_int(X), new_int(Y)
    if map[int_X][int_Y] > 0:
        # if (map[int_X+1][int_Y] != 0 and map[int_X-1][int_Y] != 0 and map[int_X][int_Y+1] != 0 and map[int_X][int_Y-1] != 0):
        #     return False
        return False
    
    # check if in map range
    if X < 0 or X > map_width - 1 or Y < 0 or Y > map_height:
        return False
    
    return True


@numba.njit()
def is_node_free_and_in_map_range_loose(X, Y, map, map_width, map_height):
    '''
    判断当前点(cmap格式)和四周的点全都碰撞到障碍物，则碰撞
    '''
    int_X, int_Y = new_int(X), new_int(Y)
    if map[int_X][int_Y] > 0:
        if (map[int_X+1][int_Y] != 0 and map[int_X-1][int_Y] != 0 and map[int_X][int_Y+1] != 0 and map[int_X][int_Y-1] != 0):
            return False
    
    # check if in map range
    if X < 0 or X > map_width - 1 or Y < 0 or Y > map_height:
        return False
    
    return True


@numba.njit()
def is_node_free_and_in_map_range_strict(X, Y, map, map_width, map_height):
    '''
    判断当前点(cmap格式)和四周的点只要有一个碰撞到障碍物，则碰撞
    '''
    # check if in map range
    if X < 0 or X > map_width - 1 or Y < 0 or Y > map_height:
        return False
    
    int_X, int_Y = new_int(X), new_int(Y)
    if map[int_X][int_Y] > 0:
        return False
    else:
        if (map[int_X+1][int_Y] == 0 and map[int_X-1][int_Y] == 0 and map[int_X][int_Y+1] == 0 and map[int_X][int_Y-1] == 0 and \
            map[int_X+1][int_Y+1] == 0 and map[int_X+1][int_Y-1] == 0 and map[int_X-1][int_Y+1] == 0 and map[int_X-1][int_Y-1] == 0):
            return True
    
    return False
    




@numba.njit()
def np_append_in_numba(arr, object):
    '''
    用在numba装饰的函数内,对array最后添加元素
    '''
    row, col = np.shape(arr)
    arr = np.append(arr, object)
    arr = np.reshape(arr, (row+1, col))
    return arr



@numba.njit()
def np_delete_in_numba(arr, idx):
    '''
    用在numba装饰的函数内,删除array内指定idx的元素
    '''
    row, col = np.shape(arr)
    del_idxs = list(range(idx*col, idx*col+col))
    arr = np.delete(arr, del_idxs)
    arr = np.reshape(arr, (row-1, col))
    return arr


@numba.njit()
def new_int(num):
    '''
    将num四舍五入得到int
    '''
    # XY turn to occuped format
    resolution = 1
    rem = num % resolution
    if rem >= resolution/2:
        return int(round(num - rem + resolution, 1))
    else:
        return int(round(num - rem, 1))
