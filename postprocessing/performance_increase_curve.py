import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 或者使用其他支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["font.size"] = 18
# 数据集名称
datasets = ["MNIST", "FashionMNIST", "EMNIST", "CIFAR10", "CIFAR100"]

# 不同算法的准确率数据（以百分比表示）
accuracy_data = {
    "Linear": [2.74, 3.28, 5.47, np.nan, np.nan],
    "CNN": [0.38, 1.12, 2.31, 6.36, 9.55],
    "ResNet18": [np.nan, np.nan, np.nan, 3.14, 5.5],
}

# 创建图形和坐标轴
plt.figure(figsize=(12, 8))

# 设置线条样式和标记
line_styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "^", "D"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# 绘制每个算法的折线
for i, (algorithm, accuracies) in enumerate(accuracy_data.items()):
    plt.plot(
        datasets,
        accuracies,
        label=algorithm,
        linestyle=line_styles[i % len(line_styles)],
        marker=markers[i % len(markers)],
        markersize=10,
        linewidth=2.5,
        color=colors[i % len(colors)],
    )

# 添加数据标签
for i, (algorithm, accuracies) in enumerate(accuracy_data.items()):
    for j, acc in enumerate(accuracies):
        plt.text(
            j,
            acc + 0.5,
            f"{acc}%",
            ha="center",
            va="bottom",
            color=colors[i % len(colors)],
        )

# 设置图表标题和标签
plt.title("不同模型在各数据集上的平均准确率提升", pad=20)
plt.xlabel("数据集", labelpad=10)
plt.ylabel("准确率提升 (%)", labelpad=10)

# 设置刻度
plt.xticks(range(len(datasets)), datasets)
plt.yticks(np.arange(0, 10, 1))

# 设置y轴范围，留出足够空间显示数据标签
plt.ylim(0, 10)

# 添加网格线
plt.grid(True, linestyle="--", alpha=0.7)

# 添加图例
plt.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="gray")


# 调整布局
plt.tight_layout()

# 保存图形（可选）
# plt.savefig('dataset_accuracy_comparison.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
plt.savefig("figures/performance_increase_curve.png")
