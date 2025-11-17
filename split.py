import torch
from torch import nn
from torchvision import models
import os
from collections import OrderedDict

FULL_MODEL_PATH = "models/ResNet50(9epoch).pth"

OUTPUT_DIR = "models_split"
SP = "SP1"
HEAD_MODEL_PATH = os.path.join(OUTPUT_DIR, SP+"-head.pt")
TAIL_MODEL_PATH = os.path.join(OUTPUT_DIR, SP+"-tail.pt")

NUM_CASE_CLASSES = 4
NUM_LOC_CLASSES = 9

SPLIT_LAYERS = {"SP1": "layer1",
                "SP2": "layer2",
                "SP3": "layer3",
                "SP4": "layer4",}

SPLIT_LAYER_NAME = SPLIT_LAYERS[SP]


class MultiHeadResNet50(nn.Module):
    def __init__(self, num_case=4, num_loc=9):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.case_head = nn.Linear(in_features, num_case)
        self.loc_head = nn.Linear(in_features, num_loc)

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

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_model = MultiHeadResNet50(NUM_CASE_CLASSES, NUM_LOC_CLASSES)
    full_model.load_state_dict(torch.load(FULL_MODEL_PATH, map_location=device))
    full_model.eval()
    print("원본 모델 로드 완료.")

    head_model = InternalHeadModel(full_model.backbone, SPLIT_LAYER_NAME)
    tail_model = InternalTailModel(full_model.backbone, SPLIT_LAYER_NAME, NUM_CASE_CLASSES, NUM_LOC_CLASSES)

    tail_model.case_head.load_state_dict(full_model.case_head.state_dict())
    tail_model.loc_head.load_state_dict(full_model.loc_head.state_dict())
    print("Head 및 Tail 모델로 가중치 복사 완료.")

    torch.save(head_model.state_dict(), HEAD_MODEL_PATH)
    torch.save(tail_model.state_dict(), TAIL_MODEL_PATH)

    print("모델 분할 완료")
    print(f"Head 모델 경로: {HEAD_MODEL_PATH}")
    print(f"Tail 모델 경로: {TAIL_MODEL_PATH}")