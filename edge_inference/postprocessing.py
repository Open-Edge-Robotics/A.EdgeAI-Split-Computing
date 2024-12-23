import json
import numpy as np

def postprocess(model_output, class_index_path='imagenet_class_index.json'):
    with open(class_index_path) as f:
        annotations = json.load(f)
    class_id = int(np.argmax(model_output))
    return annotations[str(class_id)][1]
