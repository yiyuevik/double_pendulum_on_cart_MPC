"""
cartpole_closed_loop.py
我没有写main.py
此即为主入口脚本：读取/设置模型参数(在 config.py)，构造并求解 OCP，然后进行闭环仿真 + 可视化。
运行方式: python cartpole_closed_loop.py
"""


import config
from double_pendulum_ocp import create_ocp_solver, simulate_closed_loop  
from double_pendulum_utils import plot_cartpole_trajectories, animate_cartpole
import time
import matplotlib.pyplot as plt
import numpy as np

def main():

    # 2) 生成初始状态样本

    x0 = np.array([0, 0, 2*np.pi,0, 2*np.pi, 0])
    # 4) 闭环仿真
    N_sim = 120  # 模拟步数
    all_simX = np.zeros((N_sim+1,config.Num_State,10))
    all_simU = np.zeros((N_sim,1,10))
    all_time = np.zeros((10))
    i = 0
    ocp, ocp_solver, integrator = create_ocp_solver(x0)
    sim_round = 10
    for _ in range(sim_round):
        starttime = time.time()
        t, simX, simU = simulate_closed_loop(ocp, ocp_solver, integrator, x0, N_sim=N_sim)
        endtime = time.time()
        elapsed_time = endtime - starttime
        all_simX[:, :, i] = simX
        all_simU[:, :, i] = simU
        all_time[i] = elapsed_time
        print("elapsed_time: ", elapsed_time)
        i = i+1
        print(f"Simulation for initial state {x0} took {elapsed_time:.4f} seconds.")

        # 5) 绘制曲线
        # plot_cartpole_trajectories(t, simX, simU)
        # 6) 动画
        animate_cartpole(t, simX, interval=50)

    print("all_time: ", np.sum(all_time))
    print("time/turn: ", np.sum(all_time)/(N_sim*sim_round))

    theta_values = all_simX[:, 2, :]

    # 计算每个步骤的最大值、最小值和中位数
    theta_max = np.max(theta_values, axis=1)
    theta_min = np.min(theta_values, axis=1)
    theta_median = np.median(theta_values, axis=1)

    # 绘制 theta 的范围图
    plt.figure(figsize=(12, 6))

    # 填充最大值和最小值之间的区域，表示范围
    plt.fill_between(range(N_sim+1), theta_min, theta_max, color='lightblue', label='Range', alpha=0.5)

    # 绘制中位数线
    plt.plot(range(N_sim+1), theta_median, color='blue', label='Median', linewidth=2)

    # 设置图形标题和标签
    plt.title('Theta (State 3) Range and Median Over Time')
    plt.xlabel('Step')
    plt.ylabel('Theta Value')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()   
    # steps = 50
    # num_x0 = 10  # 10个x0

    # # 1. 将数据重新整理为适合绘制Boxplot的格式
    # data_for_boxplot = []

    # for step in range(steps):
    #     # 提取每个步长对应的10个控制输入值
    #     data_for_boxplot.append(all_simU[step, 0, :])

    # # 将数据转化为numpy数组，以便 matplotlib 进行绘制
    # data_for_boxplot = np.array(data_for_boxplot).T  # 转置，确保每一列为一个控制输入的样本

    # # 2. 使用matplotlib绘制Boxplot
    # plt.figure(figsize=(12, 6))

    # # 使用matplotlib的boxplot绘制
    # plt.boxplot(data_for_boxplot, widths=0.6)

    # # 3. 设置标签和标题
    # plt.title('Control Input in Dataset')
    # plt.xlabel('Step')
    # plt.ylabel('Control Input (ctrl)')

    # # 设置x轴标签
    # xticks = [i*10 + 5 for i in range(steps // 10)]  # 每10个步长在x轴上显示一个标签
    # plt.xticks(xticks, [str(i) for i in range(0, steps, 10)])

    # # 显示图形
    # plt.show()
        
if __name__ == "__main__":
    main()