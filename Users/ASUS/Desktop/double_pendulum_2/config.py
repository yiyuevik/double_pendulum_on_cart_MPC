"""
config.py

集中管理 MPC 相关设置（Q, R, P, Horizon, Ts 等）
以及初始状态采样与随机初始猜测生成函数等。
"""
import  numpy as np

Horizon = 250          # 预测步数
Ts = 0.01              # 采样时间
Num_State = 6         # 6个状态（包括两个摆杆的角度和角速度）
Num_Input = 1         # 控制量维数（推力）
Fmax = 6000

# 状态和控制量的权重矩阵
Q = np.diag([1,    # x
             1,    # xdot
             1,    # theta1
             1,    # omega1
             1,    # theta2
             1])   # omega2

R = 0.001             # 控制量的权重

P = np.diag([1,    # x
             1,    # xdot
             1,    # theta1
             1,    # omega1
             1,    # theta2
             1,
             ])   # omega2

def GenerateRandomInitialGuess(min_random=-6000.0, max_random=6000.0):
    """
    生成一个随机的 (u_ini_guess, x_ini_guess)
    其中 u_ini_guess 在 [min_random, max_random] 里均匀随机取,范围我不清楚，问！
    """
    u_ini_guess = np.random.uniform(min_random, max_random, 1)[0]
    if u_ini_guess >= 0:
        x_ini_guess =  np.zeros(8)
        x_ini_guess[2] = 2 * np.pi
        x_ini_guess[4] = 2 * np.pi
        # x_ini_guess[0] = 5
    else:
        x_ini_guess = np.zeros(8)
        # x_ini_guess[0] = -5
    return u_ini_guess, x_ini_guess
