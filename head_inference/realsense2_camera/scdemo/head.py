import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from PIL import Image as PILImage
import torch
from torchvision import models, transforms
import torch.nn as nn
import json
# import jsonpickle
import requests
import numpy as np
import json
from collections import OrderedDict
import time

HEAD_MODEL_PATH = "models_split/SP1-head.pt"
TARGET_IMAGE_PATH = "testimg/case1_5_Color.png"
SERVER_URL = "http://192.168.0.179:8010/infer_tail"
SPLIT_LAYER_NAME = 'layer1'

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


class RealTimeClassifier(Node):
    def __init__(self):
        super().__init__('real_time_classifier')

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  
            self.image_callback,
            10
        )
        self.bridge = CvBridge()

        #self.publisher_ = self.create_publisher(String, '/predicted_case_info', 8)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        original_backbone_for_head = models.resnet50(weights=None).to(self.device)
        self.head_model = InternalHeadModel(original_backbone_for_head, SPLIT_LAYER_NAME).to(self.device)
        self.head_model.load_state_dict(torch.load(HEAD_MODEL_PATH, map_location=self.device))
        self.head_model.eval()
        print(f"Head 모델 로드 완료. (Device: {self.device})")
        

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        pil_image = PILImage.fromarray(cv_image)

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        print(next(self.head_model.parameters()).device)
        print(input_tensor.device)


        with torch.no_grad():
            start = time.time()
            feature_tensor = self.head_model(input_tensor)
            feature_np = feature_tensor.cpu().numpy()
            T_head = time.time() - start
            print("T_head: ", T_head)
            print(f"Head 모델 추론 완료. Feature Shape: {feature_np.shape}")

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


        if response.status_code == 200:
            print("\n서버 응답 성공!")
            print(f"  - 최종 예측 결과: {response.json()}")
        else:
            print(f"\n서버 응답 실패: {response.status_code}")
            print(f"  - 에러 내용: {response.text}")

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeClassifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
