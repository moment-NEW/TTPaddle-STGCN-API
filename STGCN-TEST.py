class InferenceDataset(SkeletonDataset):
    def __init__(self, data_path, window_size=50, stride=25, transform=None):
        """
        添加滑动窗口功能的推理数据集
        :param window_size: 窗口长度（帧数）
        :param stride: 滑动步长
        """
        super().__init__(data_path=data_path, label_path=None, transform=transform)
        self.window_size = window_size
        self.stride = stride

        # 预计算所有窗口索引
        self.window_indices = []
        for idx in range(len(self.data)):
            seq = self.data[idx]
            T = seq.shape[1]  # 假设原始数据形状为 [C, T, V]
            for start in range(0, T, self.stride):
                end = start + self.window_size
                if end > T:
                    end = T
                self.window_indices.append((idx, start, end))

    def __getitem__(self, index):
        # 获取窗口对应的原始数据和位置
        data_idx, start, end = self.window_indices[index]
        skeleton = self.data[data_idx].astype(np.float32)  # 原始形状 [C, T, V]

        # 提取窗口数据
        window = skeleton[:, start:end, :]  # [C, window_frames, V]

        # 处理不足窗口长度的部分（末尾补零）
        if window.shape[1] < self.window_size:
            pad_width = ((0, 0), (0, self.window_size - window.shape[1]), (0, 0))
            window = np.pad(window, pad_width, mode='constant')

        # 数据预处理
        window = window[:2]  # 取XY坐标 [2, W, V]
        if self.transform:
            for t in self.transform:
                window = t(window)

        return paddle.to_tensor(window)

    def __len__(self):
        return len(self.window_indices)


# 初始化数据集（窗口参数根据需求调整）
inference_dataset = InferenceDataset(
    data_path=DATAPATH,
    window_size=50,  # 与模型训练时的时序长度一致
    stride=25,  # 50%重叠
    transform=[Normalize()]  # 移除HorizontalFlip()
)

# 模型预测与结果聚合
all_outputs = []
P_model.predict(inference_dataset, batch_size=32, verbose=1)  # 自动批处理

# 按原始样本分组结果
sample_results = {}
for i, (data_idx, start, end) in enumerate(inference_dataset.window_indices):
    output = all_outputs[i]
    if data_idx not in sample_results:
        sample_results[data_idx] = []
    sample_results[data_idx].append(output)

# 对每个样本的所有窗口结果取平均
final_preds = []
for data_idx in sorted(sample_results.keys()):
    outputs = np.array(sample_results[data_idx])
    avg_probs = outputs.mean(axis=0)
    final_pred = np.argmax(avg_probs)
    final_preds.append(final_pred)
    print(f"样本 {data_idx} 的预测结果: {final_pred}")
