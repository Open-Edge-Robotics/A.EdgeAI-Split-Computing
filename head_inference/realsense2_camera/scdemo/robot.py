import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import requests
import numpy as np
import json
from collections import OrderedDict

# ==============================================================================
# CONFIG
# ==============================================================================
HEAD_MODEL_PATH = "models_split/SP1-head.pt"
TARGET_IMAGE_PATH = "testimg/case1_5_Color.png"
SERVER_URL = "http://192.168.0.22:8000/infer_tail"
SPLIT_LAYER_NAME = 'layer1'


# ==============================================================================
# 1. Head 모델 아키텍처 정의
# ==============================================================================
class InternalHeadModel(nn.Module):
    def __init__(self, original_backbone, split_layer_name):
        super().__init__()
        layers = OrderedDict()
        for name, module in original_backbone.named_children():
            layers[name] = module
            if name == split_layer_name:
                break
        self.features = nn.Sequential(layers)

    def forward(self, x):
        return self.features(x)
# ==============================================================================


# ==============================================================================
# 2. 메인 실행 로직
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Head 모델 로드 ---
    original_backbone_for_head = models.resnet50(weights=None).to(device)
    head_model = InternalHeadModel(original_backbone_for_head, SPLIT_LAYER_NAME).to(device)
    head_model.load_state_dict(torch.load(HEAD_MODEL_PATH, map_location=device))
    head_model.eval()
    print(f"Head 모델 로드 완료. (Device: {device})")

    # --- 2. 이미지 로드 및 전처리 ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(TARGET_IMAGE_PATH).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # --- 3. Head 모델로 추론하여 중간 피처맵 추출 ---
    with torch.no_grad():
        feature_tensor = head_model(input_tensor)
    
    feature_np = feature_tensor.cpu().numpy()
    print(f"Head 모델 추론 완료. Feature Shape: {feature_np.shape}")

    # --- 4. 피처맵을 서버로 전송 ---
    packed_feature = feature_np.tobytes()
    metadata = {
        "original_shape": list(feature_np.shape),
        "dtype": feature_np.dtype.name
    }
    
    files = {
        "file": ("feature.bin", packed_feature, "application/octet-stream"),
        "metadata": (None, json.dumps(metadata), "application/json")
    }

    print("서버로 Feature 전송 중...")
    response = requests.post(SERVER_URL, files=files)

    # --- 5. 서버로부터 받은 최종 결과 출력 ---
    if response.status_code == 200:
        print("\n서버 응답 성공!")
        print(f"  - 최종 예측 결과: {response.json()}")
    else:
        print(f"\n서버 응답 실패: {response.status_code}")
        print(f"  - 에러 내용: {response.text}")

if __name__ == "__main__":
    main()