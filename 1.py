import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_and_process_data(file_name, columns):
    """
    读取CSV文件并处理数据。
    """
    try:
        data = pd.read_csv(file_name, header=1)
        return data[columns].astype(float).values
    except Exception as e:
        print(f"Error reading or processing data: {e}")
        return None

def plot_aq_dot(x, y, color, group, ax=None, fig=None):
    """
    绘制数据点。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(x, y, 'o', color=color, label=group)
    plt.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Anscombe's quartet")
    return ax, fig

def plot_aq_line(x, w, ax, fig):
    """
    绘制拟合直线。
    """
    x_min = x[:, 0].min()
    x_max = x[:, 0].max()
    x_ = np.linspace(x_min, x_max, 20).reshape(-1, 1)
    x_ = np.concatenate([x_, np.ones((len(x_), 1))], axis=1)
    ax.plot(x_, x_ @ w, '-', color='#BEB8DC', linewidth=3)
    plt.xlim([x_min-0.5, x_max+0.5])
    plt.savefig('./1_picture/{}.png'.format(group))

def LSM(X, Y):
    """
    最小二乘法。
    """
    try:
        return np.linalg.inv(X.T @ X) @ X.T @ Y
    except np.linalg.LinAlgError as e:
        print(f"Error in linear algebra operation: {e}")
        return None

# 统计数据
def statistical(aq_):
    mean = aq_.mean()
    std = aq_.std()
    corr = np.corrcoef(aq_[:, 0], aq_[:, 1])
    print("mean: ", mean, "std: ", std, "corr: ", corr)
# 读取数据
aq1 = read_and_process_data('1-data.csv', ['x1', 'y1'])
statistical(aq1)
aq2 = read_and_process_data('1-data.csv', ['x2', 'y2'])
statistical(aq2)
aq3 = read_and_process_data('1-data.csv', ['x3', 'y3'])
statistical(aq3)
aq4 = read_and_process_data('1-data.csv', ['x4', 'y4'])
statistical(aq4)

# 绘制数据点和拟合线
colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2']
groups = ['1', '2', '3', '4']
flag = True
for i, (aq, color, group) in enumerate(zip([aq1, aq2, aq3, aq4], colors, groups)):
    ax, fig = plot_aq_dot(aq[:, 0], aq[:, 1], color, group)
    a = np.ones_like(aq[:, 0], dtype=float).reshape(-1, 1)
    x = np.concatenate([aq[:, 0].reshape(-1, 1), a], axis=1)
    y = aq[:, 1].reshape(-1, 1)
    w = LSM(x, y)
    print(w)
    if flag:
        if w is not None:
            plot_aq_line(x, w, ax, fig)
        else:
            print(f"Failed to plot line for group {group}")
    else:
        plt.savefig('./1_picture/{}.png'.format(group))

