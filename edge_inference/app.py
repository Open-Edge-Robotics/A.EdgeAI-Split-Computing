from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from common.inference import run_model
from common.quantization import clip_dequantize
from postprocessing import postprocess

app = FastAPI()

TAIL_MODEL_PATH = "saved_models/tail.onnx"
CLIP_MIN, CLIP_MAX, BITS = 0,32,4

class FeatureData(BaseModel):
    quantized_feature: list

@app.post("/infer_tail")
def infer_tail(data: FeatureData):
    qf = np.array(data.quantized_feature, dtype=np.float32)
    if qf.shape != (1,64,56,56):
        return {"error": f"Invalid feature shape {qf.shape}"}
    dequant = clip_dequantize(qf, CLIP_MIN, CLIP_MAX, BITS, qf.shape).astype(np.float32)
    output = run_model(TAIL_MODEL_PATH, dequant)
    label = postprocess(output, 'imagenet_class_index.json')
    return {"result_label": label}
