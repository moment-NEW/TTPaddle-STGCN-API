import requests
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


# 测试API连接和单张图片处理
def test_skeleton_api(image_path):
    if not os.path.exists(image_path):
        print(f"错误: 图像路径不存在 - {image_path}")
        return

    api_url = "http://localhost:5003"

    # 1. 初始化模型
    print("初始化骨骼检测模型...")
    init_response = requests.post(f"{api_url}/initialize")
    if init_response.status_code != 200:
        print(f"API初始化失败: {init_response.text}")
        return
    print("模型初始化成功")

    # 2. 发送图像进行预测
    print(f"发送图像 {os.path.basename(image_path)} 进行骨骼检测...")
    predict_response = requests.post(
        f"{api_url}/predict",
        json={"image_path": os.path.abspath(image_path)}
    )

    # 3. 打印完整的响应内容用于调试
    print(f"状态码: {predict_response.status_code}")
    response_data = predict_response.json()
    print("响应内容:")
    print(json.dumps(response_data, indent=2, ensure_ascii=False))

    # 4. 如果成功，可视化结果
    if predict_response.status_code == 200 and response_data.get('status') == 'success':
        predictions = response_data.get('predictions', [])

        if predictions:
            # 加载原始图像
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 获取关键点
            keypoints = np.array(predictions[0]['keypoints'])

            # 绘制关键点
            plt.figure(figsize=(10, 10))
            plt.imshow(image)

            # COCO关键点连接线
            connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # 头和肩膀
                (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
                (5, 6), (5, 11), (6, 12),  # 躯干
                (11, 13), (13, 15), (12, 14), (14, 16)  # 腿
            ]

            # 画连接线
            for conn in connections:
                if conn[0] < len(keypoints) and conn[1] < len(keypoints):
                    if keypoints[conn[0]][2] > 0.3 and keypoints[conn[1]][2] > 0.3:
                        plt.plot(
                            [keypoints[conn[0]][0], keypoints[conn[1]][0]],
                            [keypoints[conn[0]][1], keypoints[conn[1]][1]],
                            'r-', linewidth=2
                        )

            # 画关键点
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.3:  # 只显示置信度高的点
                    plt.scatter(x, y, s=50, c='blue', marker='o')
                    plt.text(x + 10, y + 10, f"{i}", fontsize=8)

            plt.title(f"骨骼检测结果 - 置信度: {predictions[0]['confidence']:.2f}")
            plt.axis('off')

            # 保存结果图像
            output_dir = "./test_results"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"skeleton_{os.path.basename(image_path)}")
            plt.savefig(output_path)
            plt.close()
            print(f"结果已保存至: {output_path}")

            # 打印关键点信息
            print("\n关键点坐标 (x, y, confidence):")
            for i, (x, y, conf) in enumerate(keypoints):
                print(f"点 {i}: ({x:.1f}, {y:.1f}, {conf:.2f})")
        else:
            print("API返回成功，但未包含预测结果")
    else:
        print(f"API请求失败或返回错误")


if __name__ == "__main__":
    # 提示用户输入图片路径或使用默认路径
    default_image = "keypoint_detection_002.jpg"
    user_input = input(f"请输入测试图片路径 (直接回车使用默认图片 {default_image}): ").strip()
    image_path = user_input if user_input else default_image

    # 测试API
    test_skeleton_api(image_path)