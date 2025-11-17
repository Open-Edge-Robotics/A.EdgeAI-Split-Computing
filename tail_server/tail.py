# ros_fastapi_server.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
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
class RosPublisherNode(Node):
    def __init__(self):
        super().__init__('tail_model_publisher')
        # 예측된 케이스 정보를 발행할 퍼블리셔
        self.publisher_ = self.create_publisher(String, '/predicted_case_info', 10)
        self.get_logger().info('ROS 2 Publisher Node has been started.')

    def publish_prediction(self, case_pred, case_info):
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
# 3. FastAPI 앱 및 모델 로드
# ==============================================================================
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 모델 로드 (한 번만 수행) ---
original_backbone_for_tail = models.resnet50(weights=None).to(device)
tail_model = InternalTailModel(
    original_backbone_for_tail, SPLIT_LAYER_NAME, NUM_CASE_CLASSES, NUM_LOC_CLASSES
).to(device)
tail_model.load_state_dict(torch.load(TAIL_MODEL_PATH, map_location=device))
tail_model.eval()
print(f"Tail 모델 로드 완료. 추론 준비... (Device: {device})")

# --- ROS 2 노드 초기화 ---
# FastAPI 엔드포인트에서 접근할 수 있도록 전역 변수로 노드를 생성합니다.
rclpy.init()
ros_node = RosPublisherNode()


@app.post("/infer_tail")
async def infer_tail(file: UploadFile = File(...), metadata: str = Form(...)):
    try:
        # 1. 데이터 수신 및 복원
        meta = json.loads(metadata)
        shape = tuple(meta["original_shape"])
        dtype = np.dtype(meta["dtype"])
        data_bytes = await file.read()
        feature_np = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
        feature_tensor = torch.from_numpy(feature_np).to(device)

        # 2. Tail 모델 추론
        with torch.no_grad():
            case_out, loc_out = tail_model(feature_tensor)
            case_pred = torch.argmax(torch.softmax(case_out, dim=1)).item() + 1
            loc_pred = torch.argmax(torch.softmax(loc_out, dim=1)).item() + 1
        
        case_key = f"case{case_pred}"
        case_info = cases.get(case_key, None)
       
        # 3. ROS 2 토픽 발행
        # 전역으로 생성된 ros_node의 publish 메소드 호출
        ros_node.publish_prediction(case_pred=case_pred, case_info=case_info)
           
        # 4. HTTP 응답 반환
        return {
            "case_prediction": case_pred,
            "location_prediction": loc_pred
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ==============================================================================


# ==============================================================================
# 4. 메인 실행 함수 (스레딩)
# ==============================================================================
def run_ros_spin():
    # 백그라운드에서 rclpy.spin을 실행하는 함수
    rclpy.spin(ros_node)

def main():
    # ROS 2 노드를 별도의 데몬 스레드에서 실행
    # 데몬 스레드는 메인 프로그램이 종료될 때 함께 종료됨
    ros_thread = threading.Thread(target=run_ros_spin, daemon=True)
    ros_thread.start()
   
    # 메인 스레드에서 FastAPI 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8020)

    # 프로그램 종료 시 ROS 2 노드 정리
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()