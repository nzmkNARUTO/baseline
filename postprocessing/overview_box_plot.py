import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# 设置全局字体大小
plt.rcParams.update({"font.size": 18})
# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 或者使用其他支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 创建四组算法数据，每组包含一对算法
algo_data = {
    # 第一组算法对
    "组1": {
        "FedAvg": [
            93.21,
            96.02,
            84.75,
            87.00,
            98.65,
            99.32,
            89.23,
            99.35,
            83.33,
            84.94,
            88.29,
            90.46,
            77.93,
            88.79,
            76.96,
            84.09,
            87.33,
            92.10,
            78.65,
            92.73,
            51.09,
            72.83,
            48.08,
            65.99,
            45.06,
            66.63,
            52.02,
            71.37,
            37.32,
            39.41,
            37.77,
            43.61,
        ],
        "FedAvg-": [
            92.41,
            95.76,
            84.85,
            92.46,
            98.93,
            99.36,
            89.24,
            99.55,
            79.20,
            81.27,
            88.69,
            90.29,
            80.41,
            88.86,
            76.47,
            84.00,
            88.35,
            91.82,
            80.17,
            92.85,
            61.00,
            71.71,
            58.64,
            74.65,
            49.03,
            70.54,
            56.42,
            73.68,
            39.77,
            42.07,
            38.60,
            43.17,
        ],
    },
    # 第二组算法对
    "组2": {
        "FedNova": [
            93.50,
            95.99,
            84.63,
            87.15,
            98.91,
            99.21,
            89.07,
            99.36,
            82.90,
            84.74,
            87.90,
            90.31,
            80.37,
            88.79,
            77.28,
            84.10,
            87.51,
            91.87,
            78.68,
            92.62,
            50.11,
            73.86,
            48.55,
            65.20,
            47.36,
            65.82,
            50.20,
            71.66,
            36.74,
            38.45,
            38.31,
            41.99,
        ],
        "FedNova-": [
            92.68,
            95.84,
            84.38,
            86.94,
            98.72,
            99.21,
            88.97,
            99.37,
            79.44,
            84.71,
            88.26,
            90.15,
            81.11,
            88.72,
            76.96,
            83.92,
            86.34,
            91.93,
            79.77,
            92.52,
            56.40,
            70.57,
            49.35,
            71.26,
            48.91,
            71.20,
            56.48,
            73.23,
            39.23,
            38.93,
            39.16,
            43.28,
        ],
    },
    # 第三组算法对
    "组3": {
        "FedProx": [
            90.46,
            94.24,
            82.27,
            85.35,
            99.14,
            99.41,
            89.12,
            95.51,
            61.93,
            60.83,
            89.69,
            90.98,
            76.12,
            84.46,
            65.52,
            81.13,
            83.28,
            93.16,
            81.07,
            93.50,
            70.59,
            81.83,
            67.75,
            80.37,
            45.06,
            52.34,
            51.59,
            65.32,
            53.13,
            57.01,
            44.47,
            47.87,
        ],
        "FedProx-": [
            90.57,
            93.44,
            82.85,
            85.85,
            99.09,
            99.48,
            89.15,
            95.56,
            60.84,
            57.38,
            89.78,
            90.99,
            74.83,
            84.45,
            66.43,
            81.09,
            87.75,
            92.90,
            78.67,
            93.56,
            70.88,
            81.15,
            68.21,
            82.43,
            46.34,
            63.37,
            54.93,
            69.65,
            52.79,
            56.12,
            45.67,
            50.37,
        ],
    },
    # 第四组算法对
    "组4": {
        "SCAFFOLD": [
            76.01,
            84.55,
            68.38,
            93.54,
            97.71,
            98.96,
            88.86,
            98.88,
            85.52,
            88.71,
            41.18,
            60.38,
            42.81,
            64.35,
            84.57,
            89.25,
            74.16,
            89.94,
            54.96,
            67.50,
            55.06,
            66.13,
            53.03,
            70.32,
            52.02,
            70.89,
            29.89,
            32.16,
            34.32,
            40.08,
        ],
        "SCAFFOLD-": [
            76.02,
            95.23,
            85.66,
            95.43,
            98.78,
            99.26,
            89.25,
            99.31,
            87.79,
            90.22,
            68.51,
            47.66,
            58.39,
            77.55,
            87.26,
            91.38,
            78.41,
            91.92,
            68.56,
            75.56,
            63.96,
            77.20,
            54.38,
            70.43,
            53.04,
            73.53,
            42.30,
            44.35,
            36.86,
            40.93,
        ],
    },
}

# 设置颜色方案
colors = {
    "FedAvg": "#1f77b4",
    "FedAvg-": "#ff7f0e",  # 第一组颜色
    "FedNova": "#2ca02c",
    "FedNova-": "#d62728",  # 第二组颜色
    "FedProx": "#9467bd",
    "FedProx-": "#8c564b",  # 第三组颜色
    "SCAFFOLD": "#e377c2",
    "SCAFFOLD-": "#7f7f7f",  # 第四组颜色
}

# 设置图形大小
plt.figure(figsize=(14, 8))

# 计算箱线图的位置
group_width = 0.8  # 每组的宽度
box_width = 0.35  # 每个箱线图的宽度
group_gap = 0.2  # 组间距

# 箱线图位置计算
positions = []
group_positions = []
labels = []
color_list = []

for i, (group_name, group_data) in enumerate(algo_data.items()):
    # 计算组的位置
    group_start = i * (group_width + group_gap)
    group_center = group_start + group_width / 2
    group_positions.append(group_center)

    # 计算每个算法箱线图的位置
    algo_names = list(group_data.keys())

    pos1 = group_start + box_width / 2
    pos2 = group_start + group_width - box_width / 2

    positions.extend([pos1, pos2])
    labels.extend(algo_names)
    color_list.extend([colors[algo_names[0]], colors[algo_names[1]]])

# 准备箱线图数据
boxplot_data = []
for algo_name in labels:
    # 找到该算法所属的组
    for group_name, group_data in algo_data.items():
        if algo_name in group_data:
            boxplot_data.append(group_data[algo_name])
            break

# 绘制箱线图
boxplots = plt.boxplot(
    boxplot_data,
    positions=positions,
    widths=box_width,
    patch_artist=True,  # 填充箱体
    showfliers=True,  # 显示异常值
    medianprops={"color": "black", "linewidth": 2},  # 中位线样式
    flierprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markersize": 6,
    },  # 异常值样式
    whiskerprops={"linestyle": "-", "linewidth": 1.5},  # 须样式
)

# 设置箱体颜色
for box, color in zip(boxplots["boxes"], color_list):
    box.set(facecolor=color, alpha=0.7, edgecolor="black", linewidth=1.5)

# 添加组标签和分隔线
for i, group_pos in enumerate(group_positions):
    group_name = list(algo_data.keys())[i]
    plt.text(
        group_pos,
        plt.ylim()[0] - 3,
        group_name,
        ha="center",
        va="center",
        fontweight="bold",
    )

    # 添加组间分隔线（除了最后一组）
    if i < len(group_positions) - 1:
        separator_pos = (group_pos + group_positions[i + 1]) / 2
        plt.axvline(
            x=separator_pos, color="gray", linestyle="--", alpha=0.5, linewidth=1
        )

# 设置图表标题和标签
plt.title("四组算法对的性能对比", pad=20)
plt.ylabel("准确率 (%)", labelpad=10)

# 设置x轴刻度和标签
plt.xticks(positions, labels, rotation=0)

# 设置y轴范围
plt.ylim(20, 100)
plt.yticks(np.arange(30, 101, 5))

# 添加网格线
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 添加统计信息
for i, data in enumerate(boxplot_data):
    # 计算均值
    mean_val = np.mean(data)

    # 在箱线图上标记均值
    plt.plot(
        positions[i],
        mean_val,
        marker="*",
        markersize=10,
        color="red",
        markeredgecolor="black",
    )

# 添加图例说明均值标记
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor="red",
        markersize=10,
        markeredgecolor="black",
        label="均值",
    )
]
plt.legend(handles=legend_elements, loc="lower right")

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # 为组标签留出空间

# 显示图形
plt.show()
plt.savefig("figures/overview_box_plot.png")
