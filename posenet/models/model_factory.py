import os
import tensorflow as tf

from posenet.models.mobilenet_v1 import MobileNetV1

MODEL_DIR = './_models'
DEBUG_OUTPUT = False


def load_model(model_id, output_stride=16, model_dir=MODEL_DIR):
    model_path = os.path.join(model_dir, 'ckpt')
    model = MobileNetV1(model_id, output_stride=output_stride)

    if not os.path.exists(model_path):
        return model
    else:
        model.load_weights(model_path)
        return model
