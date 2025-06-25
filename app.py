import os
import shutil
import time
import torch
import numpy as np
import nibabel as nib
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from models import build_model
from util.data_utilities import reorient_to, resample_nib
import logging
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import uuid
from test import Tester, Args  # 导入 test.py 中的 Tester 和 Args

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 与 test.py 中的 test_dir 保持一致
TEST_DERIVATIVES_DIR = "data/train/derivatives"  # 注意路径调整
TRAIN_OUTPUT_DIR = "result_test/"

# 确保目录存在
os.makedirs(TEST_DERIVATIVES_DIR, exist_ok=True)
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)

# 静态文件路径
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# 模板文件路径
@app.route('/templates/<path:filename>')
def template_files(filename):
    return send_from_directory('templates', filename)

# 修改根路由，显示 welcome 页面
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 检查文件上传
        if not request.files.getlist('file'):
            logger.error("未收到上传的文件")
            return jsonify({'error': '请上传CT图像文件'}), 400

        file = request.files['file']
        filename = file.filename
        
        # 检查文件类型
        if not filename.endswith(".nii.gz"):
            logger.error("上传的文件格式不正确")
            return jsonify({'error': '请上传 .nii.gz 格式的CT图像文件'}), 400

        # 保存临时文件
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)

        case_id = filename.split("_")[0]
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        logger.info(f"临时文件保存成功: {file_path}")

        # 确保输入文件夹存在并创建子目录
        case_dir_derivatives = os.path.join(TEST_DERIVATIVES_DIR, case_id)
        os.makedirs(case_dir_derivatives, exist_ok=True)

        # 将上传的文件移动到测试输入目录的子目录
        shutil.copy(file_path, os.path.join(case_dir_derivatives, filename))

        # 调用测试功能
        cfg = Args()
        tester = Tester(cfg, "checkpoints/model_epoch_97_valloss_1.63.pth")
        tester.test()

        # 从 test.py 的输出文件夹获取结果
        output_path = os.path.join(TRAIN_OUTPUT_DIR, "keypoints_sagittal.png")
        if not os.path.exists(output_path):
            logger.error("未找到预测结果文件")
            return jsonify({'error': '预测结果文件未找到'}), 500

        # 读取结果文件并转换为 Base64
        with open(output_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

        # 将临时文件移动到指定位置（如果您需要保存上传的文件）
        destination = os.path.join(case_dir_derivatives, filename)
        shutil.move(file_path, destination)

        # 返回结果
        return jsonify({
            'success': True,
            'message': '预测成功',
            'image': image_base64,
            'prediction': '这里是预测结果信息'
        })

    except Exception as e:
        logger.error(f"预测失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500
    finally:
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"临时文件夹已删除: {temp_dir}")

# 添加其他页面的路由
@app.route('/diseases')
def diseases():
    return render_template('diseases.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/exercise')
def exercise():
    return render_template('exercise.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/welcome')
def welcome_page():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)