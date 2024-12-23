from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from common.inference import run_model
from common.quantization import clip_quantize

app = FastAPI()

HEAD_MODEL_PATH = "saved_models/head.onnx"
CLIP_MIN, CLIP_MAX, BITS = 0,32,4

class ImageData(BaseModel):
    image_data: list

@app.post("/infer_head")
def infer_head(data: ImageData):
    input_data = np.array(data.image_data, dtype=np.float32)
    if input_data.shape != (1,3,224,224):
        return {"error": "Invalid input shape"}
    head_output = run_model(HEAD_MODEL_PATH, input_data)
    original_feature = head_output.flatten().tolist()
    q = clip_quantize(head_output, CLIP_MIN, CLIP_MAX, BITS).astype(np.float32)
    return {
        "original_feature": original_feature,
        "quantized_feature": q.flatten().tolist(),
    }
