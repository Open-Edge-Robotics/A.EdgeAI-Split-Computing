import torch
from torch import nn
from torchvision import models
import os
from collections import OrderedDict

# ==============================================================================
# CONFIG
# ==============================================================================
# 1. 분할할 원본 모델 파일 경로
FULL_MODEL_PATH = "models/resnet50.pth"

# 2. 분할된 모델을 저장할 경로
OUTPUT_DIR = "models_split"
HEAD_MODEL_PATH = os.path.join(OUTPUT_DIR, "head.pt")
TAIL_MODEL_PATH = os.path.join(OUTPUT_DIR, "tail.pt")

# 3. 모델의 클래스 개수 (학습 코드와 동일하게 설정)
NUM_CASE_CLASSES = 4
NUM_LOC_CLASSES = 9

# 4. Backbone 분할 지점 설정 ['layer1', 'layer2', 'layer3']
SPLIT_LAYER_NAME = 'layer1'


# ==============================================================================
# 1. 모델 아키텍처 정의
# ==============================================================================

# 원본 전체 모델 (가중치 로드용)
class MultiHeadResNet50(nn.Module):
    def __init__(self, num_case=4, num_loc=9):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.case_head = nn.Linear(in_features, num_case)
        self.loc_head = nn.Linear(in_features, num_loc)

# 분할될 Head 모델 (Backbone의 일부)
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

# 분할될 Tail 모델 (Backbone의 나머지 + 최종 Heads)
class InternalTailModel(nn.Module):
    def __init__(self, original_backbone, split_layer_name, num_case, num_loc):
        super().__init__()
        layers = OrderedDict()
        found_split_point = False
        for name, module in original_backbone.named_children():
            if name == split_layer_name:
                found_split_point = True
                continue
            if found_split_point and name not in ['avgpool', 'fc']:
                layers[name] = module
        
        self.remaining_backbone = nn.Sequential(layers)
        self.avgpool = original_backbone.avgpool
        
        in_features = 2048
        self.case_head = nn.Linear(in_features, num_case)
        self.loc_head = nn.Linear(in_features, num_loc)
        
    def forward(self, x):
        feat = self.remaining_backbone(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        case_out = self.case_head(feat)
        loc_out = self.loc_head(feat)
        return case_out, loc_out


# ==============================================================================
# 2. 메인 실행 로직
# ==============================================================================
if __name__ == "__main__":
    print(f"모델 내부 분할을 시작합니다... (분할 기준: '{SPLIT_LAYER_NAME}')")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 원본 모델에 학습된 가중치 로드
    full_model = MultiHeadResNet50(NUM_CASE_CLASSES, NUM_LOC_CLASSES)
    full_model.load_state_dict(torch.load(FULL_MODEL_PATH, map_location=device))
    full_model.eval()
    print(f"'{FULL_MODEL_PATH}'에서 원본 모델 가중치 로드 완료.")

    # 2. 분할될 모델 인스턴스 생성
    # 원본 모델의 backbone 모듈을 전달하여 구조를 만듭니다.
    head_model = InternalHeadModel(full_model.backbone, SPLIT_LAYER_NAME)
    tail_model = InternalTailModel(full_model.backbone, SPLIT_LAYER_NAME, NUM_CASE_CLASSES, NUM_LOC_CLASSES)

    # 3. 원본 모델에서 최종 Head 가중치 복사
    tail_model.case_head.load_state_dict(full_model.case_head.state_dict())
    tail_model.loc_head.load_state_dict(full_model.loc_head.state_dict())
    print("Head 및 Tail 모델로 가중치 복사 완료.")

    # 4. 분할된 모델의 state_dict를 파일로 저장
    torch.save(head_model.state_dict(), HEAD_MODEL_PATH)
    torch.save(tail_model.state_dict(), TAIL_MODEL_PATH)

    print("\n모델 내부 분할이 성공적으로 완료되었습니다.")
    print(f"  - Head 모델 저장 경로: {HEAD_MODEL_PATH}")
    print(f"  - Tail 모델 저장 경로: {TAIL_MODEL_PATH}")