"""
Omni model

state: X, Y, Theta, v_x, v_y, w_z
    X_t = X + vx*cos(Theta)*dt - vy*sin(Theta)*dt
    Y_t = Y + vx*sin(Theta)*dt + vy*cos(Theta)*dt
    Theta_t = Theta + wz*dt

    v_x_t = vx
    v_y_t = vy
    w_z_t = wz

input: v_x, v_y, w_z

"""



from math import cos, sin



# 根据状态量、控制量计算离散时间后的状态并返回
def get_next_state(state, control, dt):
    """
    机器人运动学模型
    Args:
        state(状态量): [X, Y, Theta, v_x, v_y, w_z]
        control(控制量): [v_x, v_y, w_z]
        dt(离散时间): dt

    Returns:
        state: 下一步的状态
    """
    X, Y, Theta = state[0], state[1], state[2]
    vx, vy, wz = control[0], control[1], control[2]

    X_t = X + vx*cos(Theta)*dt - vy*sin(Theta)*dt
    Y_t = Y + vx*sin(Theta)*dt + vy*cos(Theta)*dt
    Theta_t = Theta + wz*dt
    vx_t = vx
    vy_t = vy
    wz_t = wz

    return [X_t, Y_t, Theta_t, vx_t, vy_t, wz_t]

