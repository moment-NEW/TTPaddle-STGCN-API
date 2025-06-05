import os
import numpy as np
import cv2
import traceback
import requests
import tempfile
from tqdm import tqdm


class Normalize:
    def __call__(self, skeleton):
        # 归一化处理
        skeleton = skeleton / np.max(np.abs(skeleton) + 1e-8, axis=(1, 2), keepdims=True)
        return skeleton


class VideoToSTGCNConverter:
    def __init__(self, config):
        """初始化视频转换器"""
        # 配置参数
        self.output_dir = os.path.normpath(config['output_dir'])
        self.sequence_length = config['sequence_length']
        self.det_threshold = config['det_threshold']
        self.min_valid_ratio = config['min_valid_ratio']
        self.api_url = config['api_url']

        # 创建临时目录存储帧图像
        self.temp_dir = tempfile.mkdtemp()
        print(f"临时文件目录: {self.temp_dir}")

        # 初始化API
        self._initialize_api()

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"输出目录: {self.output_dir}")

    def _initialize_api(self):
        """初始化骨骼检测API"""
        try:
            response = requests.post(f"{self.api_url}/initialize")
            if response.status_code == 200 and response.json()['status'] == 'success':
                print("骨骼检测API初始化成功")
            else:
                print(f"骨骼检测API初始化失败: {response.text}")
        except Exception as e:
            print(f"API连接错误: {str(e)}")

    def _map_coco_to_stgcn(self, coco_kpts):
        """将COCO格式17关键点映射到ST-GCN的25关键点"""
        stgcn_kpts = np.zeros((25, 3), dtype=np.float32)

        # 直接映射关系
        direct_mapping = {
            0: 3,  # nose -> head
            5: 4,  # left_shoulder
            6: 8,  # right_shoulder
            7: 5,  # left_elbow
            8: 9,  # right_elbow
            9: 6,  # left_wrist
            10: 10,  # right_wrist
            11: 12,  # left_hip
            12: 16,  # right_hip
            13: 13,  # left_knee
            14: 17,  # right_knee
            15: 14,  # left_ankle
            16: 18  # right_ankle
        }

        for coco_idx, stgcn_idx in direct_mapping.items():
            if coco_idx < len(coco_kpts):
                stgcn_kpts[stgcn_idx] = coco_kpts[coco_idx]

        # 脊柱基点 (左右臀中点)
        if stgcn_kpts[12, 2] > 0.1 and stgcn_kpts[16, 2] > 0.1:
            stgcn_kpts[0] = (stgcn_kpts[12] + stgcn_kpts[16]) / 2
            stgcn_kpts[0, 2] = min(stgcn_kpts[12, 2], stgcn_kpts[16, 2]) * 0.9

        # 中间脊柱 (脊柱基点和头部中点)
        if stgcn_kpts[0, 2] > 0.1 and stgcn_kpts[3, 2] > 0.1:
            stgcn_kpts[1] = (stgcn_kpts[0] + stgcn_kpts[3]) / 2
            stgcn_kpts[1, 2] = min(stgcn_kpts[0, 2], stgcn_kpts[3, 2]) * 0.9

        return stgcn_kpts

    def _process_frame(self, frame, frame_idx):
        """通过API处理单个视频帧"""
        try:
            # 保存帧到临时文件
            frame_path = os.path.join(self.temp_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)

            # 调用API获取关键点
            response = requests.post(
                f"{self.api_url}/predict",
                json={"image_path": frame_path}
            )

            # 检查响应
            if response.status_code != 200:
                print(f"API请求失败: {response.text}")
                return None

            data = response.json()
            if data['status'] != 'success' or not data['predictions']:
                # print(f"无有效预测结果: {data}")
                return None

            # 提取关键点
            keypoints = np.array(data['predictions'][0]['keypoints'])
            if keypoints is not None and len(keypoints) > 0:
                return self._map_coco_to_stgcn(keypoints)

            return None

        except Exception as e:
            print(f"帧处理失败: {str(e)}")
            print(f"错误详情:\n{traceback.format_exc()}")
            return None
        finally:
            # 尝试删除临时文件
            if os.path.exists(frame_path):
                try:
                    os.remove(frame_path)
                except:
                    pass

    def _adjust_sequence_length(self, sequence):
        """调整序列长度到目标尺寸"""
        seq = np.array(sequence, dtype=np.float32)
        current_length = len(sequence)

        # 填充或截断
        if current_length < self.sequence_length:
            pad = np.zeros((self.sequence_length - current_length, 25, 3), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[:self.sequence_length]

        # 转换为 [C, T, V] 格式
        return seq.transpose(2, 0, 1)

    def process_video(self, video_path):
        """处理单个视频文件"""
        # 读取视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频 {os.path.basename(video_path)} - FPS: {fps}, 总帧数: {total_frames}")

        # 存储所有关键点
        all_keypoints = []
        valid_frames = 0
        frame_idx = 0

        # 进度条
        pbar = tqdm(total=total_frames, desc=f"处理 {os.path.basename(video_path)}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 处理当前帧
            kpts = self._process_frame(frame, frame_idx)
            frame_idx += 1

            if kpts is not None:
                all_keypoints.append(kpts)
                valid_frames += 1
            else:
                all_keypoints.append(np.zeros((25, 3), dtype=np.float32))

            pbar.update(1)

        cap.release()
        pbar.close()

        # 输出检测统计
        print(f"视频总帧数: {len(all_keypoints)}, 有效帧数: {valid_frames}, "
              f"有效率: {valid_frames / len(all_keypoints):.2%}")

        # 分割为多个序列
        sequences = []
        num_frames = len(all_keypoints)

        sequence_stats = []
        for start_idx in range(0, num_frames, self.sequence_length):
            end_idx = min(start_idx + self.sequence_length, num_frames)
            sequence = all_keypoints[start_idx:end_idx]

            # 检查有效帧比例
            valid = sum(1 for kp in sequence if np.any(kp != 0))
            valid_ratio = valid / len(sequence) if len(sequence) > 0 else 0
            sequence_stats.append((valid, len(sequence), valid_ratio))

            if valid_ratio < self.min_valid_ratio:
                continue

            # 调整序列长度
            adjusted_seq = self._adjust_sequence_length(sequence)
            sequences.append(adjusted_seq)

        # 输出序列统计
        print(f"序列统计:")
        for i, (valid, total, ratio) in enumerate(sequence_stats):
            print(f"  序列 {i + 1}: {valid}/{total} 帧有效, 比例: {ratio:.2%} "
                  f"{'[通过]' if ratio >= self.min_valid_ratio else '[失败]'}")

        # 保存结果
        if sequences:
            # 应用归一化
            sequences_array = np.array(sequences)
            normalizer = Normalize()
            normalized_sequences = normalizer(sequences_array)

            output_name = os.path.basename(video_path).split('.')[0] + '.npy'
            output_path = os.path.join(self.output_dir, output_name)
            np.save(output_path, normalized_sequences)
            print(f"成功保存 {len(sequences)} 个归一化序列到 {output_path}")
        else:
            print(f"没有有效序列: {video_path}")

        # 清理临时目录
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass
        return output_path

    def convert_all(self, video_source):
        """处理所有视频"""
        if os.path.isdir(video_source):
            video_files = [os.path.join(video_source, f)
                           for f in os.listdir(video_source)
                           if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        elif os.path.isfile(video_source):
            video_files = [video_source]
        else:
            raise ValueError("无效的视频路径")

        for video_path in video_files:
            self.process_video(video_path)


# 主函数
def main():
    config = {
        'output_dir': 'stgcn_sequences',  # 输出目录
        'api_url': 'http://localhost:5003',  # API服务地址
        'det_threshold': 0.3,  # 检测阈值
        'sequence_length': 50,  # 序列长度
        'min_valid_ratio': 0.2  # 最小有效帧比例
    }

    converter = VideoToSTGCNConverter(config)
    converter.convert_all('person_crop.mp4')  # 替换为你的视频路径或目录


if __name__ == "__main__":
    main()