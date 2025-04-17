import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 或者使用其他支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 模拟数据：8个算法在5个数据集上的准确率
# 行: 算法, 列: 数据集
# 线性
accuracy_data = np.array(
    [
        [0.84, 0.76, 0.93, 0.80, 0.77],  # 算法A
        [0.86, 0.78, 0.94, 0.79, 0.79],  # 算法B
        [0.84, 0.77, 0.93, 0.82, 0.80],  # 算法C
        [0.85, 0.77, 0.94, 0.80, 0.81],  # 算法D
        [0.82, 0.65, 0.90, 0.61, 0.76],  # 算法E
        [0.83, 0.65, 0.91, 0.62, 0.75],  # 算法F
        [0.68, 0.42, 0.76, 0.14, 0.41],  # 算法G
        [0.86, 0.58, 0.77, 0.33, 0.54],  # 算法H
    ]
)

# CNN(缺失类别)
# accuracy_data = np.array(
#     [
#         [0.89, 0.78, 0.48],  # 算法A
#         [0.89, 0.82, 0.62],  # 算法B
#         [0.89, 0.78, 0.48],  # 算法C
#         [0.89, 0.81, 0.58],  # 算法D
#         [0.89, 0.81, 0.67],  # 算法E
#         [0.89, 0.80, 0.68],  # 算法F
#         [0.88, 0.74, 0.55],  # 算法G
#         [0.90, 0.77, 0.63],  # 算法H
#     ]
# )

# CNN(Dirichlet)
# accuracy_data = np.array(
#     [
#         [0.98, 0.88, 0.87, 0.51, 0.37],  # FedAvg
#         [0.99, 0.89, 0.89, 0.63, 0.48],  # FedAvg+
#         [0.99, 0.87, 0.87, 0.50, 0.36],  # FedNova
#         [0.99, 0.90, 0.87, 0.56, 0.48],  # FedNova+
#         [0.99, 0.89, 0.83, 0.70, 0.53],  # FedProx
#         [0.99, 0.90, 0.89, 0.70, 0.53],  # FedProx+
#         [0.97, 0.85, 0.84, 0.54, 0.29],  # SCAFFOLD
#         [0.98, 0.88, 0.86, 0.62, 0.44],  # SCAFFOLD+
#     ]
# )

# ResNet18
# accuracy_data = np.array(
#     [
#         [0.52, 0.46, 0.37],  # FedAvg
#         [0.55, 0.47, 0.44],  # FedAvg+
#         [0.50, 0.47, 0.38],  # FedNova
#         [0.55, 0.49, 0.43],  # FedNova+
#         [0.51, 0.45, 0.44],  # FedProx
#         [0.51, 0.44, 0.46],  # FedProx+
#         [0.52, 0.53, 0.34],  # SCAFFOLD
#         [0.54, 0.54, 0.39],  # SCAFFOLD+
#     ]
# )

# 算法和数据集名称
algorithms = [
    "FedAvg",
    "FedAvg+",
    "FedNova",
    "FedNova+",
    "FedProx",
    "FedProx+",
    "SCAFFOLD",
    "SCAFFOLD+",
]

# 确定哪些算法是带+号的，哪些不是
plus_algorithms = [algo for algo in algorithms if algo.endswith("+")]
base_algorithms = [algo for algo in algorithms if not algo.endswith("+")]

# 创建算法对应关系，将基础算法与其+版本对应起来
algorithm_pairs = {}
for base in base_algorithms:
    for plus in plus_algorithms:
        if plus.startswith(base):
            algorithm_pairs[base] = plus
            break
datasets = [
    "MNIST\n(缺失类别)",
    # "EMNIST\n(缺失类别)",
    "FashionMNIST\n(缺失类别)",
    # "CIFAR10\n(缺失类别)",
    "MNIST\n(Dirichlet)",
    "EMNIST\n(Dirichlet)",
    "FashionMNIST\n(Dirichlet)",
    # "CIFAR10\n(Dirichlet)",
    # "CIFAR100\n(Dirichlet)",
]

# 设置图形大小
plt.figure(figsize=(12, 8))

# 设置每个柱子的宽度
bar_width = 0.1

# 设置每组柱子的位置 - 修改为叠放式柱状图
dataset_positions = np.arange(len(datasets))

# 创建算法位置映射，使基础算法和对应的+算法共享相同的x轴位置
algorithm_positions = {}
position_map = {}

# 首先为基础算法分配位置
base_position_index = 0
for i, algo in enumerate(algorithms):
    if algo in base_algorithms:
        # 为基础算法分配位置
        algo_positions = [
            pos + base_position_index * bar_width for pos in dataset_positions
        ]
        algorithm_positions[algo] = algo_positions
        position_map[algo] = algo_positions
        base_position_index += 1

# 然后为+算法分配与对应基础算法相同的位置
for i, algo in enumerate(algorithms):
    if algo in plus_algorithms:
        # 找到对应的基础算法
        for base, plus in algorithm_pairs.items():
            if algo == plus:
                # 使用与基础算法相同的位置
                algorithm_positions[algo] = position_map[base]
                break

# 使用 tab10 色彩方案，但确保相关算法使用相同的基础颜色
base_colors = list(plt.cm.tab10(np.arange(len(base_algorithms))))
colors = []

# 为每个算法分配颜色，确保基础算法和+算法使用相同的基础颜色
color_map = {}
for i, algo in enumerate(algorithms):
    if algo in base_algorithms:
        base_index = base_algorithms.index(algo)
        color_map[algo] = base_colors[base_index]
    else:  # 带+的算法
        for base, plus in algorithm_pairs.items():
            if algo == plus:
                color_map[algo] = color_map[base]
                break

# 按原始顺序排列颜色
colors = [
    color_map[algo] if algo in color_map else plt.cm.tab10(i % 10)
    for i, algo in enumerate(algorithms)
]

# 绘制柱状图 - 先绘制基础算法，再绘制+算法，确保+算法在上层

# 然后绘制所有+算法
for i, algo in enumerate(algorithms):
    if algo.endswith("+"):
        pos = algorithm_positions[algo]
        color = colors[i]
        plt.bar(
            pos,
            accuracy_data[i],
            width=bar_width,
            label=algo,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.9,  # 较高的不透明度
        )
# 首先绘制所有基础算法
for i, algo in enumerate(algorithms):
    if not algo.endswith("+"):
        pos = algorithm_positions[algo]
        color = colors[i]
        plt.bar(
            pos,
            accuracy_data[i],
            width=bar_width,
            label=algo,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7,  # 半透明效果
            hatch="////",  # 斜线填充
        )

# 添加图例
plt.legend(loc="upper right", ncol=2, fontsize=12)

# 设置x轴刻度和标签 - 调整为新的柱状图位置
# 计算每组柱状图的中心位置
group_centers = [
    dataset_positions[i] + (len(base_algorithms) - 1) * bar_width / 2
    for i in range(len(datasets))
]
plt.xticks(group_centers, datasets, fontsize=12)

# 设置y轴范围和标签
plt.ylim(0.1, 1.0)
plt.ylabel("准确率", fontsize=14)

# 设置网格线
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 删除顶部标题
# plt.title("线性模型在三个数据集上的性能比较", fontsize=16, pad=20)

# 为每个柱子添加数值标签 - 调整标签位置以避免重叠
# 先为基础算法添加标签
for i, algo in enumerate(algorithms):
    if not algo.endswith("+"):
        for j, pos in enumerate(algorithm_positions[algo]):
            value = accuracy_data[i, j]
            plt.text(
                pos,
                value - 0.04,  # 将标签放在柱子内部
                f"{value:.2f}",
                ha="center",
                va="top",
                fontsize=12,  # 增大字体
                color="white",  # 白色文字更容易在柱子内部看清
                fontweight="bold",
                rotation=45,  # 文字倾斜45度
            )

# 再为+算法添加标签
for i, algo in enumerate(algorithms):
    if algo.endswith("+"):
        for j, pos in enumerate(algorithm_positions[algo]):
            value = accuracy_data[i, j]
            plt.text(
                pos,
                value + 0.01,  # 将标签放在柱子上方
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=12,  # 增大字体
                fontweight="bold",
                rotation=45,  # 文字倾斜45度
            )

# 调整布局，留出底部空间给标题
plt.tight_layout(rect=[0, 0.05, 1, 1])

# 在底部添加标题
plt.figtext(0.5, 0.01, "(a) 线性模型在三个数据集上的性能比较", ha="center", fontsize=16)

# 保存图像（可选）
# plt.savefig('algorithm_performance_overview.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
plt.savefig("overview_bar_chart.png")
