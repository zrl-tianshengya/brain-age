import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
csv_pathy = 'language.csv'  # 替换为你的CSV文件路径
csv_pathx = 'language.csv'
datax = pd.read_csv(csv_pathx)
datay = pd.read_csv(csv_pathy)
# 选择要分析的两列数据
column1 = 'prediction'  # 替换为第一列的列名
#2.CardSort_AgeAdj 4.
column2 = 'label'  # 替换为第二列的列名

x = datax[column1]
y = datay[column2]
#z = data[column3]
# 过滤含有 NaN 值的数据点
valid_indices = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
x_valid = x[valid_indices]
y_valid = y[valid_indices]

# 计算皮尔森相关系数
correlation_xy = np.corrcoef(x_valid, y_valid)[0, 1]
print("皮尔森相关系数(x-y)：", correlation_xy)
# 绘制 x-y 散点图
plt.scatter(x_valid, y_valid, s=10, alpha=0.6)
plt.plot(np.unique(x_valid), np.poly1d(np.polyfit(x_valid, y_valid, 1))(np.unique(x_valid)), color='b', linestyle='--', label='Regression Line')
plt.plot(y_valid, y_valid, color='r', linestyle='--', label='Diagonal Line')
plt.xlabel(column1)
plt.ylabel(column2)
plt.title('Language')
plt.text(np.nanmax(x_valid), np.nanmin(y_valid), f'Pearson Correlation: {correlation_xy:.2f}', ha='right', va='bottom')
plt.legend()
plt.grid(True)
plt.show()