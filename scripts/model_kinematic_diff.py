from math import cos, sin



# 根据状态量、控制量计算离散时间后的状态并返回
def get_next_state(state, control, dt):
    """
    机器人运动学模型
    Args:
        state(状态量): [X, Y, Theta, v, w]
        control(控制量): [v, w]
        dt(离散时间): dt

    Returns:
        state: 下一步的状态
    """
    X_t = state[0] + control[0] * cos(state[2]) * dt
    Y_t = state[1] + control[0] * sin(state[2]) * dt
    Theta_t = state[2] + control[1] * dt
    v_t = control[0]
    w_t = control[1]

    return [X_t, Y_t, Theta_t, v_t, w_t]

