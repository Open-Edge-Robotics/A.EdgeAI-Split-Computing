import json
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import uvicorn
import numpy as np
import onnxruntime as ort
from postprocessing import postprocess

app = FastAPI()

# 'tail' 모델만 로드
tail_session = ort.InferenceSession("models/tail.onnx")
tail_input_name = tail_session.get_inputs()[0].name

@app.post("/infer_tail_none")
async def infer_tail_none(file: UploadFile = File(...), metadata: str = Form(...)):
    try:
        # 1. 메타데이터에서 shape와 dtype 정보 파싱
        meta = json.loads(metadata)
        shape = tuple(meta["original_shape"])
        dtype = np.dtype(meta["dtype"])

        # 2. 전송받은 바이트(byte) 데이터를 다시 NumPy 배열로 복원
        data = await file.read()
        head_feature = np.frombuffer(data, dtype=dtype).reshape(shape)

        # 3. 복원된 데이터로 'tail' 모델 추론 실행
        result = tail_session.run(None, {tail_input_name: head_feature})
        
        # 4. 후처리 후 최종 결과 반환
        label = postprocess(result)
        return {"label": label}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)