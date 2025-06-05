# STGCN_API_Service.py
import os
import numpy as np
import paddle
import json
import collections
from flask import Flask, request, jsonify
from Data_Processor_keypoint import SkeletonDataset, Normalize
from ST_GCN_Net import MyNet
from NPYTEST import VideoToSTGCNConverter

app = Flask(__name__)

# 全局配置
CONFIG = {
    'output_dir': 'stgcn_sequences',
    'api_url': 'http://localhost:5003',
    'det_threshold': 0.3,
    'sequence_length': 50,
    'min_valid_ratio': 0.2,
    'model_path': 'stgcn_model.pdparams'
}

# 初始化模型
model = MyNet()
model.set_state_dict(paddle.load(CONFIG['model_path']))
model.eval()


class InferenceWrapper:
    def __init__(self, model):
        self.model = model

    def process_video(self, video_path):
        """处理视频生成npy"""
        converter = VideoToSTGCNConverter(CONFIG)
        npy_path = os.path.join(CONFIG['output_dir'],
                                os.path.basename(video_path).split('.')[0] + '.npy')
        converter.process_video(video_path)
        return npy_path if os.path.exists(npy_path) else None

    # 修复后的代码
    def predict(self, npy_path):
        """执行预测"""
        try:
            # 直接加载NPY文件
            raw_data = np.load(npy_path)
            print(f"原始数据维度: {raw_data.shape}")

            # 检查是否是4D数据 (序列数, 通道数, 时间步, 关节点)
            if len(raw_data.shape) == 4:
                num_sequences = raw_data.shape[0]
                results = []

                # 对每个序列分别进行预测
                for i in range(num_sequences):
                    # 提取单个序列并只保留前两个通道
                    sequence_data = raw_data[i][:2]  # 取该序列的前两个通道

                    # 添加批次维度 [1, 2, T, V]
                    input_data = paddle.to_tensor(sequence_data[np.newaxis, :, :, :], dtype='float32')
                    print(f"序列 {i + 1} 处理后维度: {input_data.shape}")

                    # 执行预测
                    with paddle.no_grad():
                        output = self.model(input_data).numpy().squeeze()

                    # 添加到结果
                    results.append({
                        "sample_id": i,
                        "probabilities": output.tolist(),
                        "predicted_class": int(np.argmax(output))
                    })

                return results
            else:
                # 处理3D数据的情况
                C, T, V = raw_data.shape

                # 只取前两个通道
                processed_data = raw_data[:2, :, :]

                # 创建适合模型的数据格式：[N=1, C=2, T, V]
                input_data = paddle.to_tensor(processed_data[np.newaxis, :, :, :], dtype='float32')

                # 执行预测
                with paddle.no_grad():
                    output = self.model(input_data).numpy().squeeze()

                return [{
                    "sample_id": 0,
                    "probabilities": output.tolist(),
                    "predicted_class": int(np.argmax(output))
                }]

        except Exception as e:
            print(f"预测过程中出错: {e}")
            raise Exception(f"数据预处理错误: {str(e)}")

inference_engine = InferenceWrapper(model)
# 修改后的InferenceDataset
class InferenceDataset(SkeletonDataset):
    def __init__(self, data_path, window_size=50, stride=25):
        super().__init__(data_path=data_path, label_path=None)

        # 维度验证（在SkeletonDataset中完成）
        self.window_size = window_size
        self.stride = stride
        self._precompute_indices()

    def _precompute_indices(self):
        """正确的时间窗口索引"""
        C, T, V = self.data.shape
        self.window_indices = []
        for start in range(0, T, self.stride):
            end = min(start + self.window_size, T)
            self.window_indices.append((0, start, end))  # data_idx固定为0

    # 修复后的代码
    def __getitem__(self, index):
        data_idx, start, end = self.window_indices[index]
        skeleton = self.data[:, start:end, :]  # [C, window_size, V]

        # 补零处理
        if skeleton.shape[1] < self.window_size:
            pad_width = ((0, 0), (0, self.window_size - skeleton.shape[1]), (0, 0))
            skeleton = np.pad(skeleton, pad_width, mode='constant')

        # 确保返回4D张量 [N=1, C, T, V]
        skeleton_4d = skeleton[np.newaxis, :, :, :]  # 显式添加批次维度
        return paddle.to_tensor(skeleton_4d)


@app.route('/infer', methods=['POST'])
def api_infer():
    try:
        # 参数解析
        req_data = request.json
        video_path = req_data.get('video_path')
        jump = req_data.get('jump', False)

        # 处理npy路径
        if jump:
            print("jump success")
            npy_path = 'stgcn_sequences/person_crop.npy'
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"预置文件不存在: {npy_path}")
        else:
            if not video_path or not os.path.exists(video_path):
                return jsonify({"status": "error", "message": "无效视频路径"}), 400
            npy_path = inference_engine.process_video(video_path)
            if not npy_path:
                return jsonify({"status": "error", "message": "视频处理失败"}), 500

        # 执行预测
        results = inference_engine.predict(npy_path)

        return jsonify({
            "status": "success",
            "results": results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    app.run(port=5006, debug=True)