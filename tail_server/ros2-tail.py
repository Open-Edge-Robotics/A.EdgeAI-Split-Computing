import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import json
import jsonpickle # 객체를 JSON으로 직렬화/역직렬화할 때 유용
from torchvision import models
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import numpy as np
from collections import OrderedDict
import threading
import uvicorn

# ==============================================================================
# CONFIG: 설정
# ==============================================================================
TAIL_MODEL_PATH = "models_split/SP2-tail.pt"
NUM_CASE_CLASSES = 4
NUM_LOC_CLASSES = 9
SPLIT_LAYER_NAME = 'layer2' # 분할 시 사용한 이름과 동일해야 함
# ==============================================================================

with open("/robot_skills/skills.json", "r") as f:
    cases = json.load(f)

# ==============================================================================
# 1. Tail 모델 아키텍처 정의 (분할 코드와 동일)
# ==============================================================================
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


# ==============================================================================
# 2. ROS 2 노드 정의
# ==============================================================================
class RealTimeClassifier(Node):
    def __init__(self):
        super().__init__('real_time_classifier_tail')

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/sp_layer2',
            self.feature_callback,
            10
        )
        # 예측된 케이스 정보를 발행할 퍼블리셔
        self.publisher_ = self.create_publisher(String, '/predicted_case_info', 20)
        self.get_logger().info('ROS 2 Publisher Node has been started.')

        # --- 모델 로드 (한 번만 수행) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        original_backbone_for_tail = models.resnet50(weights=None).to(self.device)
        self.tail_model = InternalTailModel(
            original_backbone_for_tail, SPLIT_LAYER_NAME, NUM_CASE_CLASSES, NUM_LOC_CLASSES
        ).to(self.device)
        self.tail_model.load_state_dict(torch.load(TAIL_MODEL_PATH, map_location=self.device))
        self.tail_model.eval()
        print(f"Tail 모델 로드 완료. 추론 준비... (Device: {self.device})")

    def feature_callback(self, msg):

        feature_np = np.array(msg.data, dtype=np.float32)
        feature_tensor = torch.from_numpy(feature_np).view(1, 512, 28, 28).to(self.device)
        
        with torch.no_grad():
            case_out, loc_out = self.tail_model(feature_tensor)
            case_pred = torch.argmax(torch.softmax(case_out, dim=1)).item() + 1
            loc_pred = torch.argmax(torch.softmax(loc_out, dim=1)).item() + 1

        case_key = f"case{case_pred}"
        case_info = cases.get(case_key, None)

        # 발행할 메시지 생성
        msg_out = String()
       
        # 예측 결과를 딕셔너리 형태로 구성
        prediction_data = {
            "predicted_class": case_pred,
            "case_info": case_info
        }
       
        # jsonpickle을 사용해 딕셔너리를 JSON 문자열로 변환
        msg_out.data = jsonpickle.encode(prediction_data)
       
        # 토픽 발행
        self.publisher_.publish(msg_out)
        self.get_logger().info(f'Publishing: "{msg_out.data}"')
# ==============================================================================



# ==============================================================================
# 4. 메인 함수
# ==============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = RealTimeClassifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()