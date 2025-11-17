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
    head_feature = get_head_feature(image)
    
    packed_feature = head_feature.tobytes()
    metadata = {
        "original_shape": list(head_feature.shape),
        "dtype": head_feature.dtype.name
    }
    
    with open("config/experiment_config.yml", "r") as f:
        config = yaml.safe_load(f)
    
    url = config["server"]["edge_inference_url"] + "/infer_tail_none"
    files = {
        "file": ("data.bin", packed_feature, "application/octet-stream"),
        "metadata": (None, json.dumps(metadata), "application/json")
    }
    
    response = requests.post(url, files=files)
    return response.json()

if __name__ == "__main__":
    def load_test_image(path):
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224))
        img_data = np.array(img).astype("float32") / 255.0
        img_data = img_data.transpose(2, 0, 1)
        return np.expand_dims(img_data, axis=0)
        
    test_image = load_test_image("assets/cat.jpg")
    server_response = run_inference_and_send(test_image)
    
    print("Server response:", server_response)