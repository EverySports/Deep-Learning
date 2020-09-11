import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import posenet

from config import *
from Datagenerator.Dataloader import load_data
from Datagenerator.Datagenerator import *
from Utils.draw_utils import *
from Utils.evaluate import *
from Decode.decode_utils import *
from Decode.Decoder import *
from Losses.loss import *

vid = cv2.VideoCapture(f'../../Datasets/images/{video_name}')
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
videoWriter = cv2.VideoWriter('inference_result.mp4', fourcc, video_fps, (256, 256))

with tf.device('/device:GPU:1'):
    base_model = posenet.load_model(mode)
    inputs = tf.keras.Input(shape=(256,256,3))
    outputs = base_model(inputs)
    inf_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    inf_model.load_weights(f'./checkpoints/{ckpt_path}_final.hdf5')

for i in range(video_frame_cnt):
    ret, img_ori = vid.read()
    img = np.asarray(img_ori, np.float32)
    img = cv2.resize(img, (256,256))
    matrix = cv2.getRotationMatrix2D((128, 128), -90, 1)
    img = cv2.warpAffine(img, matrix, (256, 256))
    img = img[np.newaxis, :]

    result = inf_model(img)
    _, _, keypoint_coords = decode_single_poses_nooff(
        result.numpy().squeeze(axis=0), output_stride=1.)
    hm = np.sum(result.numpy().squeeze(axis=0), axis=2)
    hm_ = hm / 8.
    hm_[hm_ >= 0.8] = 1
    hm_[hm_ < 0.8] = 0
    hm_ = hm_[..., np.newaxis]
    hm_ = np.repeat(hm_, 3, axis=-1)

    hm_idx = np.where(hm_ == 1)
    max_x_idx, min_x_idx = max(hm_idx[0][1:-1]), min(hm_idx[0][1:-1])
    max_y_idx, min_y_idx = max(hm_idx[1][1:-1]), min(hm_idx[1][1:-1])
    draw_image = cv2.rectangle(img[0], (min_y_idx+np.random.randint(0,10), min_x_idx+np.random.randint(0,10)), (max_y_idx-np.random.randint(0,10), max_x_idx-np.random.randint(0,10)), color=(0, 0, 255), thickness=2)
    # draw_image = draw_skeleton_and_kepoints(keypoint_coords, img[0])

    draw_image = cv2.add(draw_image, hm_*255)

    cv2.imshow('image', draw_image/510.)
    cv2.waitKey(1)
    # videoWriter.write(img)

vid.release()
cv2.destroyWindow('image')
# videoWriter.release()