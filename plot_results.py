import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# 读取CSV文件
df = pd.read_csv(sys.argv[1])

# 设置绘图风格
plt.style.use('seaborn-v0_8')  # 使用新版本的seaborn样式名称
sns.set_theme()  # 使用seaborn的默认主题

# 创建图表
plt.figure(figsize=(12, 8))

# 为每个kernel绘制一条线
for kernel in df['Kernel'].unique():
    kernel_data = df[df['Kernel'] == kernel]
    plt.plot(kernel_data['Size'], kernel_data['GFLOPs'], 
             marker='o', label=f'Kernel {kernel}')

# 设置图表属性
plt.title('Matrix Multiplication Performance Comparison')
plt.xlabel('Matrix Size')
plt.ylabel('GFLOPs')
plt.xscale('log', base=2)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(title='Kernels')

# 设置x轴刻度为实际数字
plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
plt.gca().xaxis.set_minor_formatter(plt.ScalarFormatter())

# 保存图表
plt.savefig('benchmark_results/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
