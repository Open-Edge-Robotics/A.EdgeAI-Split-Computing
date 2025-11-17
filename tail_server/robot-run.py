#!/usr/bin/env python3
"""
Minimal Distributed Inference Script:
Extracts head feature using head.onnx and sends it to the edge server.
"""
import requests
import numpy as np
import onnxruntime as ort
import json
import yaml
from PIL import Image

def get_head_feature(image):
    session = ort.InferenceSession("models/head.onnx")
    input_name = session.get_inputs()[0].name
    head_feature = session.run(None, {input_name: image})[0]
    return head_feature

def run_inference_and_send(image):
    # 1. 모델 실행해서 feature 추출
    head_feature = get_head_feature(image)
    
    # 2. 서버로 보낼 데이터 준비
    packed_feature = head_feature.tobytes()
    metadata = {
        "original_shape": list(head_feature.shape),
        "dtype": head_feature.dtype.name
    }
    
    # 3. 서버 설정 불러오기
    with open("config/experiment_config.yml", "r") as f:
        config = yaml.safe_load(f)
    
    url = config["server"]["edge_inference_url"] + "/infer_tail_none"
    files = {
        "file": ("data.bin", packed_feature, "application/octet-stream"),
        "metadata": (None, json.dumps(metadata), "application/json")
    }
    
    # 4. 서버로 데이터 전송하고, 서버의 응답 반환
    response = requests.post(url, files=files)
    return response.json()

if __name__ == "__main__":
    def load_test_image(path):
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224))
        img_data = np.array(img).astype("float32") / 255.0
        img_data = img_data.transpose(2, 0, 1)
        return np.expand_dims(img_data, axis=0)
        
    # 이미지 로드 후 서버로 추론 요청
    test_image = load_test_image("assets/cat.jpg")
    server_response = run_inference_and_send(test_image)
    
    print("Server response:", server_response)