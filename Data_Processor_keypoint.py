import os
import sys
import numpy as np
import paddle
from paddle.io import Dataset


import numpy as np
### ================== 数据增强类 ==================
class Normalize:
    def __call__(self, skeleton):
        # 归一化处理
        skeleton = skeleton / np.max(np.abs(skeleton), axis=(1, 2), keepdims=True)
        return skeleton

class HorizontalFlip:
    def __call__(self, skeleton):
        # 对称翻转
        skeleton = skeleton[:, :, ::-1]
        return skeleton



# ================== 改进版骨骼数据集类（跨平台支持） ==================
class SkeletonDataset(Dataset):
    def __init__(self, data_path, label_path=None, transform=None):
        """
        改进版骨骼数据集类（跨平台支持）
        :param data_path: 骨骼数据文件路径（自动处理路径格式）
        :param label_path: 标签文件路径（可选）
        :param transform: 数据增强方法
        """
        super().__init__()

        # ================ 跨平台路径处理 ================
        def format_path(path):
            """统一处理路径格式"""
            return os.path.normpath(path.replace("/", os.sep))

        self.data_path = format_path(data_path)
        self.label_path = format_path(label_path) if label_path else None

        # ================ 增强型路径验证 ================
        self._validate_paths()

        # ================ 数据加载与处理 ================
        try:
            self.data = np.load(self.data_path)
            self.labels = np.load(self.label_path) if self.label_path else None
        except Exception as e:
            raise RuntimeError(f"数据加载失败: {str(e)}") from e

        # 只有用多余的数据集的情况下要用# 压缩单例维度M [N, 3, T, V]
        # self.data = np.squeeze(self.data, axis=-1)

        # ================ 动态维度验证 ================
        self._validate_dimensions()

        self.transform = transform

    def _validate_paths(self):
        """详细的路径验证"""
        error_messages = []

        if not os.path.exists(self.data_path):
            error_messages.append(
                f"数据文件不存在: {self.data_path}\n"
                f"可能的解决方案:\n"
                f"1. 检查文件路径是否正确\n"
                f"2. 确认文件已下载到指定位置\n"
                f"3. 检查文件权限（是否可读）"
            )

        if self.label_path and not os.path.exists(self.label_path):
            error_messages.append(
                f"标签文件不存在: {self.label_path}\n"
                f"可能的解决方案:\n"
                f"1. 如果不需要标签，请设置 label_path=None\n"
                f"2. 确认标签文件已正确生成"
            )

        if error_messages:
            raise FileNotFoundError("\n".join(error_messages))

    def _validate_dimensions(self):
        """动态维度验证"""
        expected_dims = 4
        if len(self.data.shape) != expected_dims:
            raise ValueError(
                f"数据维度错误，期望 {expected_dims}D[N,C,T,V]，实际维度 {self.data.shape}\n"
                f"可能的原因:\n"
                f"1. 数据预处理步骤未正确执行\n"
                f"2. 输入数据格式不符合要求"
            )

        channel_dim = 3
        if self.data.shape[1] != channel_dim:
            raise ValueError(
                f"通道维度应为 {channel_dim}（X/Y/Confidence），实际维度 {self.data.shape[1]}\n"
                f"可能的原因:\n"
                f"1. 数据预处理时未正确提取坐标信息\n"
                f"2. 数据文件版本不兼容"
            )

    def __getitem__(self, index):
        skeleton = self.data[index].astype(np.float32)
        skeleton = skeleton[:2]  # 取XY坐标 [2, T, V]

        if self.transform:
            # skeleton = self.transform(skeleton)
            for t in self.transform:
                skeleton = t(skeleton)

        label = self.labels[index] if self.labels is not None else -1
        #可能导致标签映射错误，总之先试试看再说
        if label >= 13:
            label = 12
        return (
            paddle.to_tensor(skeleton),
            paddle.to_tensor(label, dtype='int64') if label != -1 else None
        )

    def __len__(self):
        return len(self.labels) if self.labels is not None else len(self.data)


# ================== 跨平台路径配置 ==================
class PathConfig:
    """智能路径配置器"""

    def __init__(self):
        self._detect_environment()

    def _detect_environment(self):
        """自动检测运行环境"""
        self.is_aistudio = 'PADDLE_CLOUD' in os.environ
        self.is_windows = sys.platform.startswith('win')

    @property
    def data_root(self):
        """动态数据根目录"""
        if self.is_aistudio:
            return "/home/aistudio/data"
        elif self.is_windows:
            return r"D:\CODES\home\aistudio\data"  # Windows用户修改此处
        else:
            return "/data"

    def get_path(self, dataset_id, filename):
        """安全获取文件路径"""
        path = os.path.join(
            self.data_root,
            f"data{dataset_id}",
            filename
        )
        return os.path.normpath(path.replace("/", os.sep))


# ================== 使用示例 ==================

    # 初始化路径配置
path_config = PathConfig()

   # 动态创建数据集实例个样本
DATASET_ID = 324261
try:
    dataset = SkeletonDataset(
        data_path=path_config.get_path(DATASET_ID, "skeleton_data.npy"),
        label_path=path_config.get_path(DATASET_ID, "labels.npy"),
        transform=[Normalize(), HorizontalFlip()]
    )
except Exception as e:
    print(f"初始化失败: {str(e)}")
    sys.exit(1)

# 环境诊断信息
print("\n=== 环境诊断 ===")
print(f"操作系统: {'Windows' if path_config.is_windows else 'Linux/Mac'}")
print(f"运行环境: {'AI Studio' if path_config.is_aistudio else '本地'}")
print(f"数据根目录: {path_config.data_root}")
print(f"数据文件路径: {dataset.data_path}")

# 数据验证
sample, label = dataset[0]
print("\n=== 数据验证 ===")
print(f"样本形状: {sample.shape} (应为 [2, T, 25])")
print(f"标签示例: {label} (类型: {label.dtype})")
print(f"数据集大小: {len(dataset)}")
