import os
import sys
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from tqdm import tqdm


class STGCNPredictor:
    """用于调用ST-GCN API进行骨骼数据预测"""

    def __init__(self, config):
        """初始化预测器"""
        self.api_url = config['api_url']
        self.slide_step = config['slide_step']
        self.visualize = config.get('visualize', False)
        self.output_dir = config.get('output_dir', './results')

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化API连接
        self._initialize_api()

    def _initialize_api(self):
        """初始化ST-GCN API连接"""
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(f"{self.api_url}/initialize", headers=headers, json={})
            if response.status_code == 200:
                print("ST-GCN API连接成功")
            else:
                print(f"API初始化失败: {response.status_code} {response.text}")
        except Exception as e:
            print(f"连接ST-GCN API失败: {str(e)}")
            sys.exit(1)

    def predict_file(self, npy_path):
        """预测单个NPY文件"""
        if not os.path.exists(npy_path):
            print(f"错误: 找不到文件 {npy_path}")
            return

        try:
            # 调用API，直接发送文件路径
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                f"{self.api_url}/predict_file",
                headers=headers,
                json={"file_path": os.path.abspath(npy_path), "slide_step": self.slide_step}
            )

            if response.status_code != 200:
                print(f"API请求失败: {response.text}")
                return

            # 解析响应
            result = response.json()
            if result['status'] == 'success':
                results = result.get('predictions', [])
                timestamps = result.get('timestamps', [])

                # 输出结果
                if results:
                    print(f"\n检测到 {len(results)} ��动作:")
                    for i, (result, t) in enumerate(zip(results, timestamps)):
                        print(f"  [{t}-{t + 50}] 动作: {result['category']}, 置信度: {result['score']:.4f}")

                    # 保存结果
                    output_json = os.path.join(
                        self.output_dir,
                        f"prediction_{os.path.basename(npy_path).split('.')[0]}.json"
                    )
                    with open(output_json, 'w', encoding='utf-8') as f:
                        json.dump({
                            'file': npy_path,
                            'predictions': results,
                            'timestamps': timestamps
                        }, f, ensure_ascii=False, indent=2)

                    print(f"预测结果已保存至: {output_json}")

                    # 可视化
                    if self.visualize:
                        self.visualize_results(results, timestamps, npy_path)
                else:
                    print("没有检测到有效动作")
            else:
                print(f"预测失败: {result.get('message', '未知错误')}")

        except Exception as e:
            print(f"预测过程中出错: {str(e)}")

    def visualize_results(self, results, timestamps, npy_path):
        """可视化预测结果"""
        if not results:
            print("没有可视化的结果")
            return

        # 提取类别和置信度
        categories = [result['category'] for result in results]
        scores = [result['score'] for result in results]

        # 绘制结果
        plt.figure(figsize=(12, 6))

        # 绘制类别
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, categories, 'b-')
        plt.scatter(timestamps, categories, c='blue')
        plt.ylabel('动作类别')
        plt.title(f'滑动窗口预测结果: {os.path.basename(npy_path)}')
        plt.grid(True)

        # 绘制置信度
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, scores, 'r-')
        plt.scatter(timestamps, scores, c='red')
        plt.xlabel('帧索引')
        plt.ylabel('置信度')
        plt.grid(True)

        # 保存图像
        output_path = os.path.join(
            self.output_dir,
            f"prediction_{os.path.basename(npy_path).split('.')[0]}.png"
        )
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"可视化结果已保存至: {output_path}")

    def batch_predict(self, npy_dir):
        """批量预测目录中的所有NPY文件"""
        if not os.path.isdir(npy_dir):
            print(f"错误: {npy_dir} 不是有效目录")
            return

        npy_files = [
            os.path.join(npy_dir, f)
            for f in os.listdir(npy_dir)
            if f.endswith('.npy')
        ]

        if not npy_files:
            print(f"在 {npy_dir} 中没有找到NPY文件")
            return

        for npy_file in npy_files:
            print(f"\n处理文件: {npy_file}")
            self.predict_file(npy_file)


def main():
    # 配置
    config = {
        'api_url': 'http://localhost:5004',  # ST-GCN API地址
        'slide_step': 10,  # 滑动窗口步长
        'visualize': True,  # 是否可视化结果
        'output_dir': './stgcn_predictions'  # 输出目录
    }

    predictor = STGCNPredictor(config)

    # 处理单个文件
    # predictor.predict_file('./stgcn_sequences/person_crop.npy')

    # 批量处理目录
    predictor.batch_predict('./stgcn_sequences')


if __name__ == '__main__':
    main()