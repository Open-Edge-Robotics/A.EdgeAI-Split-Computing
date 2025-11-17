import uvicorn
import torch
from torch import nn
from torchvision import models
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import numpy as np
import json
from collections import OrderedDict

TAIL_MODEL_PATH = "models_split/tail.pt"
NUM_CASE_CLASSES = 4
NUM_LOC_CLASSES = 9
SPLIT_LAYER_NAME = 'layer1'


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


app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

original_backbone_for_tail = models.resnet50(weights=None).to(device)
tail_model = InternalTailModel(
    original_backbone_for_tail, SPLIT_LAYER_NAME, NUM_CASE_CLASSES, NUM_LOC_CLASSES
).to(device)
tail_model.load_state_dict(torch.load(TAIL_MODEL_PATH, map_location=device))
tail_model.eval()
print("Tail 모델 로드 완료")


@app.post("/infer_tail")
async def infer_tail(file: UploadFile = File(...), metadata: str = Form(...)):
    try:
        meta = json.loads(metadata)
        shape = tuple(meta["original_shape"])
        dtype = np.dtype(meta["dtype"])
        data_bytes = await file.read()
        feature_np = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
        feature_tensor = torch.from_numpy(feature_np).to(device)

        with torch.no_grad():
            case_out, loc_out = tail_model(feature_tensor)
            case_pred = torch.argmax(torch.softmax(case_out, dim=1)).item()
            loc_pred = torch.argmax(torch.softmax(loc_out, dim=1)).item()
            
        return {
            "case_prediction": case_pred + 1,
            "location_prediction": loc_pred + 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)