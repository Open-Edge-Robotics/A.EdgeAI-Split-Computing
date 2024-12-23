from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.staticfiles import StaticFiles
import os
from uuid import uuid4
from model_ops import pt_to_onnx, get_split_points, split_model, infer_output_shapes
from preprocessing import preprocess
from common.quantization import clip_dequantize
from PIL import Image
import numpy as np
import requests

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SAVE_DIR_MODELS = "saved_models"
SAVE_DIR_IMAGES = "uploaded_images"
os.makedirs(SAVE_DIR_MODELS, exist_ok=True)
os.makedirs(SAVE_DIR_IMAGES, exist_ok=True)

ROBOT_INFERENCE_URL = "http://robot_inference:8001"
EDGE_INFERENCE_URL = "http://edge_inference:8002"

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "split_points": None, "message": None})

@app.post("/upload_pt", response_class=HTMLResponse)
async def upload_pt(request: Request, pt_file: UploadFile = File(...)):
    pt_path = os.path.join(SAVE_DIR_MODELS, f"{uuid4()}_{pt_file.filename}")
    with open(pt_path, "wb") as f:
        f.write(await pt_file.read())
    onnx_path = pt_path.replace(".pt", ".onnx")
    pt_to_onnx(pt_path, onnx_path)
    splits = get_split_points(onnx_path)

    shapes_info = infer_output_shapes(onnx_path, splits)
    detailed_splits = []
    for sp_name, shp in shapes_info:
        if len(shp) == 4:
            N,C,H,W = shp
            size = C*H*W
        else:
            C,H,W = 1,1,1
            size = 1
        detailed_splits.append({
            "name": sp_name,
            "C": C,
            "H": H,
            "W": W,
            "size": size
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "Model converted to ONNX",
        "pt_path": pt_path,
        "onnx_path": onnx_path,
        "split_points_detailed": detailed_splits
    })

@app.post("/split_model", response_class=HTMLResponse)
async def do_split(request: Request, onnx_path: str = Form(...), pt_path: str = Form(...), chosen_split: str = Form(...)):
    head_path = "saved_models/head.onnx"
    tail_path = "saved_models/tail.onnx"
    split_model(onnx_path, chosen_split, head_path, tail_path)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "Model split complete",
        "head_path": head_path,
        "tail_path": tail_path
    })

@app.post("/inference", response_class=HTMLResponse)
async def inference(request: Request, head_path: str = Form(...), tail_path: str = Form(...), img_file: UploadFile = File(...)):
    img_path = os.path.join(SAVE_DIR_IMAGES, f"{uuid4()}_{img_file.filename}")
    with open(img_path, "wb") as f:
        f.write(await img_file.read())

    img = Image.open(img_path)
    input_data = preprocess(img)

    # robot inference
    robot_resp = requests.post(f"{ROBOT_INFERENCE_URL}/infer_head", json={"image_data": input_data.tolist()})
    if robot_resp.status_code != 200:
        return templates.TemplateResponse("index.html", {"request": request, "message": "로봇 인퍼런스 실패."})
    robot_data = robot_resp.json()
    quantized_feature = np.array(robot_data["quantized_feature"], dtype=np.float32).reshape(1,64,56,56)

    # edge inference
    edge_resp = requests.post(f"{EDGE_INFERENCE_URL}/infer_tail", json={"quantized_feature": quantized_feature.tolist()})
    if edge_resp.status_code != 200:
        return templates.TemplateResponse("index.html", {"request": request, "message": "엣지 인퍼런스 실패."})
    edge_data = edge_resp.json()
    result_label = edge_data["result_label"]

    return templates.TemplateResponse("inference.html", {
        "message": "Inference complete",
        "request": request,
        "result_label": result_label,
    })
