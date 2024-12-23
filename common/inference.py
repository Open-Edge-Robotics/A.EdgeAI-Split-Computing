import onnxruntime as ort

def run_model(model_path, input_data):
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    return session.run([], {input_name: input_data})[0]
