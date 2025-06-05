import numpy
import numpy as np

# 读取文件
data = np.load('stgcn_sequences/TEST-WCQvsOQLF_person_2.npy')

# 查看基本信息
print(f"数据形状: {data.shape}")  # 应为 [N,C,T,V]
print(f"数据类型: {data.dtype}")
print(f"数值范围: {data.min()} ~ {data.max()}")
