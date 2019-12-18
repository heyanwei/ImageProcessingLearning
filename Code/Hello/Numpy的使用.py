import numpy as np

data1 = np.array([1,2,3,4,5])
print(data1)

data2 = np.array([[1,2],[3,4]])
print(data2)

# 维度打印
print(data1.shape, data2.shape)

print(np.zeros([2,3]), np.ones([2,3]))

# 修改
data2[1,0]=5
print(data2)
print(data2[1,1])

# 基本运算
data3 = np.ones([2,3])
print(data3*2)
print(data3/3)
print(data3+3)

# 矩阵加法乘法
data4 = np.array([[1,2,3],[4,5,6]])
print(data3*data4)