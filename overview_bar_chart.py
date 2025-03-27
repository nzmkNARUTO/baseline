import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 或者使用其他支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 模拟数据：8个算法在5个数据集上的准确率
# 行: 算法, 列: 数据集
accuracy_data = np.array(
    [
        [0.92, 0.88, 0.90, 0.92, 0.88, 0.90],  # 算法A
        [0.87, 0.85, 0.82, 0.87, 0.85, 0.82],  # 算法B
        [0.78, 0.75, 0.77, 0.78, 0.75, 0.77],  # 算法C
        [0.94, 0.91, 0.93, 0.94, 0.91, 0.93],  # 算法D
        [0.81, 0.79, 0.80, 0.81, 0.79, 0.80],  # 算法E
        [0.85, 0.82, 0.84, 0.85, 0.82, 0.84],  # 算法F
        [0.89, 0.87, 0.88, 0.89, 0.87, 0.88],  # 算法G
        [0.76, 0.74, 0.75, 0.76, 0.74, 0.75],  # 算法H
    ]
)

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
datasets = [
    "MNIST(缺失类别)",
    "EMNIST(缺失类别)",
    "FashionMNIST(缺失类别)",
    "MNIST(Dirichlet)",
    "EMNIST(Dirichlet)",
    "FashionMNIST(Dirichlet)",
]

# 设置图形大小
plt.figure(figsize=(12, 8))

# 设置每个柱子的宽度
bar_width = 0.09

# 设置每组柱子的位置
dataset_positions = np.arange(len(datasets))
algorithm_positions = [dataset_positions]

for i in range(1, len(algorithms)):
    algorithm_positions.append([x + bar_width for x in algorithm_positions[i - 1]])

# 使用 tab10 色彩方案
colors = list(plt.cm.tab10(np.arange(len(algorithms))))

# 绘制柱状图
for i, (algo, pos, color) in enumerate(zip(algorithms, algorithm_positions, colors)):
    plt.bar(
        pos,
        accuracy_data[i],
        width=bar_width,
        label=algo,
        color=color,
        edgecolor="black",
        linewidth=0.5,
    )

# 添加图例
plt.legend(loc="upper right", ncol=2, fontsize=12)

# 设置x轴刻度和标签
plt.xticks([p + bar_width * 3.5 for p in dataset_positions], datasets, fontsize=12)

# 设置y轴范围和标签
plt.ylim(0.6, 1.0)
plt.ylabel("准确率", fontsize=14)

# 设置网格线
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 添加标题
plt.title("线性模型在三个数据集上的性能比较", fontsize=16, pad=20)

# 为每个柱子添加数值标签
for i, algorithm_pos in enumerate(algorithm_positions):
    for j, pos in enumerate(algorithm_pos):
        value = accuracy_data[i, j]
        plt.text(
            pos,
            value + 0.01,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

# 调整布局
plt.tight_layout()

# 保存图像（可选）
# plt.savefig('algorithm_performance_overview.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
plt.savefig("overview_bar_chart.png")
