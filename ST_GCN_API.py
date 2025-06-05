import os
import numpy as np
import paddle
from Data_Processor_keypoint import SkeletonDataset, Normalize
from ST_GCN_Net import MyNet

DATAPATH = 'stgcn_sequences/person_crop.npy'


class InferenceDataset(SkeletonDataset):
    def __init__(self, data_path, window_size=50, stride=25):
        """
        滑动窗口推理数据集
        :param window_size: 窗口长度（与模型训练时一致）
        :param stride: 滑动步长
        """
        super().__init__(data_path=data_path, label_path=None)
        self.window_size = window_size
        self.stride = stride

        # 预计算窗口索引
        self.window_indices = []
        for idx in range(len(self.data)):
            seq = self.data[idx]  # 原始形状 [C, T, V]
            T = seq.shape[1]
            for start in range(0, T, stride):
                end = min(start + window_size, T)
                self.window_indices.append((idx, start, end))

    def __getitem__(self, index):
        data_idx, start, end = self.window_indices[index]
        skeleton = self.data[data_idx].astype(np.float32)

        # 提取窗口并补零
        window = skeleton[:2, start:end, :]  # [C=2, T=50, V=25]
        if window.shape[1] < self.window_size:
            window = np.pad(window, ((0, 0), (0, self.window_size - window.shape[1]), (0, 0)), 'constant')

        # 调整为 [N, C, T, V]
        return paddle.to_tensor(window[np.newaxis])  # 形状 [1, 2, 50, 25]

    def __len__(self):
        return len(self.window_indices)


# 初始化数据集
inference_dataset = InferenceDataset(
    data_path=DATAPATH,
    window_size=50,
    stride=25
)

# 验证数据形状
sample = inference_dataset[0]
print(f"单个窗口输入形状: {sample.shape}")  # 应为 [1, 2, 25, 50]

# 加载模型
model = MyNet()
model.set_state_dict(paddle.load('stgcn_model.pdparams'))
model.eval()

# 执行预测
all_outputs = []
with paddle.no_grad():
    for i in range(len(inference_dataset)):
        data = inference_dataset[i]
        output = model(data).numpy().squeeze()  # [num_classes]
        all_outputs.append(output)
print(f"总预测窗口数: {len(all_outputs)}")

# 结果聚合
import collections

results = collections.defaultdict(list)
for i, (data_idx, _, _) in enumerate(inference_dataset.window_indices):
    results[data_idx].append(all_outputs[i])

# 输出最终预测
for data_idx in sorted(results.keys()):
    probs = np.stack(results[data_idx]).mean(axis=0)
    print(f"样本{data_idx}的预测概率: {probs}")
    print(f"=> 最终类别: {np.argmax(probs)}")
