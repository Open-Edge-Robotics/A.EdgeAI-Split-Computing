import numpy as np
from PIL import Image

def preprocess(img):
    img = img.resize((224, 224))
    img_data = np.array(img).transpose(2, 0, 1).astype('float32')
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    for i in range(img_data.shape[0]):
        img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return img_data.reshape(1, 3, 224, 224).astype('float32')
