# STGCN_API_Service.py
import os
import numpy as np
import paddle
import json
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
        # 加载动作标签（如果有）
        self.action_labels = self._load_action_labels()

    def _load_action_labels(self):
        """加载动作标签"""
        try:
            labels_path = 'action_labels.json'
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"加载动作标签失败: {e}")
            return None

    def process_video(self, video_path):
        """处理视频生成npy"""
        converter = VideoToSTGCNConverter(CONFIG)
        npy_path = os.path.join(CONFIG['output_dir'],
                                os.path.basename(video_path).split('.')[0] + '.npy')
        converter.process_video(video_path)
        return npy_path if os.path.exists(npy_path) else None

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

                print(f"检测到{num_sequences}个序列，开始分别预测...")

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

                    pred_class = int(np.argmax(output))
                    confidence = float(output[pred_class])

                    # 获取动作标签（如果有）
                    action_name = None
                    if self.action_labels and 0 <= pred_class < len(self.action_labels):
                        action_name = self.action_labels[pred_class]

                    # 添加到结果
                    result = {
                        "sequence_id": i,
                        "probabilities": output.tolist(),
                        "predicted_class": pred_class,
                        "confidence": confidence
                    }

                    if action_name:
                        result["action_name"] = action_name

                    results.append(result)
                    print(f"序列 {i + 1} 预测结果: 类别={pred_class}, 置信度={confidence:.4f}")

                print(f"所有{num_sequences}个序列预测完成")
                return results
            else:
                # 处理3D数据的情况
                print("检测到3D数据，作为单个序列处理")
                C, T, V = raw_data.shape

                # 只取前两个通道
                processed_data = raw_data[:2, :, :]

                # 创建适合模型的数据格式：[N=1, C=2, T, V]
                input_data = paddle.to_tensor(processed_data[np.newaxis, :, :, :], dtype='float32')

                # 执行预测
                with paddle.no_grad():
                    output = self.model(input_data).numpy().squeeze()

                pred_class = int(np.argmax(output))
                confidence = float(output[pred_class])

                # 获取动作标签（如果有）
                action_name = None
                if self.action_labels and 0 <= pred_class < len(self.action_labels):
                    action_name = self.action_labels[pred_class]

                result = {
                    "sequence_id": 0,
                    "probabilities": output.tolist(),
                    "predicted_class": pred_class,
                    "confidence": confidence
                }

                if action_name:
                    result["action_name"] = action_name

                print(f"单序列预测结果: 类别={pred_class}, 置信度={confidence:.4f}")
                return [result]

        except Exception as e:
            print(f"预测过程中出错: {e}")
            import traceback
            print(traceback.format_exc())
            raise Exception(f"数据预处理或预测错误: {str(e)}")


inference_engine = InferenceWrapper(model)


@app.route('/infer', methods=['POST'])
def api_infer():
    try:
        # 参数解析
        req_data = request.json
        video_path = req_data.get('video_path')
        jump = req_data.get('jump', False)

        # 处理npy路径
        if jump:
            print("使用预设NPY文件进行测试")
            npy_path = 'stgcn_sequences/TEST-WCQvsOQLF_person_2.npy'
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

        # 确保返回所有序列的预测结果
        return jsonify({
            "status": "success",
            "sequence_count": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    app.run(port=5006, debug=True)