from matplotlib import pyplot as plt

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 或者使用其他支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["font.size"] = 18
sparsity = [i for i in range(0, 101, 10)]
time = [431.33, 382.68, 346.97, 310.29, 277.91, 245.99, 211.84, 176.50, 140.52, 97.74]
x = [10, 30, 50, 70, 90]
acc = [92.24, 89.13, 91.56, 84.45, 71.58]
# Setting the figure size
plt.figure(figsize=(10, 6))

# Create the primary y-axis for time
ax1 = plt.gca()
ax1.plot(sparsity[1:], time, "b-", linewidth=2, label="时间")
ax1.set_xlabel("稀疏度 (%)")
ax1.set_ylabel("时间 (s)")
ax1.tick_params(axis="y")

# Create the secondary y-axis for accuracy
ax2 = ax1.twinx()
ax2.plot(x, acc, "r-", linewidth=2, label="准确率")
ax2.set_ylabel("准确率 (%)")
ax2.tick_params(axis="y")


# Create a single legend for both plots
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

# Add grid
ax1.grid(True, linestyle="--", alpha=0.7)

# Show the plot
plt.show()
plt.savefig("figures/sparsity_and_accuracy_vs_time", dpi=300, bbox_inches="tight")
