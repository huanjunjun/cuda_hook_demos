import matplotlib.pyplot as plt

# 数据准备
process_counts = [2, 4, 6, 8, 10]
switch_times = [102.46, 108.90, 115.36, 123.62, 131.34]  # 单位是秒/百万次
switch_times_per_million = [t for t in switch_times]  # 已经是每百万次耗时

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(process_counts, switch_times_per_million, marker='o', linestyle='-', color='b', linewidth=2)

# 添加标题和标签
plt.title('进程间GPU上下文切换开销', fontsize=15)
plt.xlabel('进程数量', fontsize=12)
plt.ylabel('每百万次切换耗时 (秒)', fontsize=12)

# 设置网格和刻度
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(process_counts)
plt.yticks(range(100, 140, 5))

# 在数据点上添加数值标签
for x, y in zip(process_counts, switch_times_per_million):
    plt.text(x, y, f'{y:.2f}s', ha='center', va='bottom', fontsize=10)

# 显示图表

plt.show()