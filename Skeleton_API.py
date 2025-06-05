import os
import uuid
import traceback
from flask import Flask, request, jsonify
from paddlex import create_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 启用跨域支持
model = None
OUTPUT_DIR = "./output/"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route('/initialize', methods=['POST'])
def initialize_model():
    global model
    try:
        model = create_model(model_name="PP-TinyPose_128x96")
        return jsonify({
            "status": "success",
            "message": "骨骼关键点检测模型初始化成功"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"模型初始化失败: {str(e)}"
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({
            "status": "error",
            "message": "请先初始化模型"
        }), 400

    data = request.json
    if not data or 'image_path' not in data:
        return jsonify({
            "status": "error",
            "message": "缺少必要参数: image_path"
        }), 400

    image_path = data['image_path']
    if not os.path.exists(image_path):
        return jsonify({
            "status": "error",
            "message": "指定的图片路径不存在"
        }), 400

    try:
        # 生成唯一文��名
        file_id = str(uuid.uuid4())
        output_img = os.path.join(OUTPUT_DIR, f"result_{file_id}.jpg")
        output_json = os.path.join(OUTPUT_DIR, f"result_{file_id}.json")

        # 执行预测
        result = model.predict(image_path, batch_size=1)

        # 处理结果 - 根据实际结构提取关键点
        predictions = []

        # 处理每个结果
        for res in result:
            # 保存可视化结果和JSON
            try:
                res.save_to_img(output_img)
                res.save_to_json(output_json)
            except Exception as e:
                print(f"保存结果图像失败: {e}")

            # 从res.json中提取关键点数据
            json_data = res.json

            # 提取关键点
            keypoints = []
            confidence = 0.0

            if 'res' in json_data and 'kpts' in json_data['res'] and json_data['res']['kpts']:
                kpt_data = json_data['res']['kpts'][0]
                if 'keypoints' in kpt_data:
                    # 确保转换为普通列表而不是numpy数组
                    if hasattr(kpt_data['keypoints'], 'tolist'):
                        keypoints = kpt_data['keypoints'].tolist()
                    else:
                        keypoints = kpt_data['keypoints']

                if 'kpt_score' in kpt_data:
                    confidence = float(kpt_data['kpt_score'])

            predictions.append({
                "input_path": image_path,
                "output_image": output_img,
                "output_json": output_json,
                "keypoints": keypoints,
                "confidence": confidence
            })

        return jsonify({
            "status": "success",
            "predictions": predictions
        })

    except Exception as e:
        error_details = traceback.format_exc()
        return jsonify({
            "status": "error",
            "message": f"预测失败: {str(e)}",
            "details": error_details
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)