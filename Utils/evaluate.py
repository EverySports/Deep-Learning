import numpy as np
from tqdm import tqdm
from Decode.decode_utils import decode_single_poses, decode_single_poses_nooff

def pckh_05(label, predict, num_keypoints=17):
    # label = label[0]
    def get_Euclidian(x, y):
        return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    threshold = get_Euclidian(label[9], label[10]) / 2

    result = 0
    for i in range(num_keypoints):
        err = get_Euclidian(label[i], predict[i])
        if err <= threshold: result += 1
    return result / 17

def evaluate(inf_model, inf_gen):
    acc = 0
    for img, regr in tqdm(inf_gen):
        result = inf_model(img)
        _, _, keypoint_coords = decode_single_poses_nooff(result.numpy().squeeze(axis=0), output_stride=1.)
        # _, _, keypoint_coords = decode_single_poses_nooff(result[0].numpy().squeeze(axis=0))
        # _, _, keypoint_coords = decode_single_poses_nooff(result[0].numpy().squeeze(axis=0),
        #                                             result[1].numpy().squeeze(axis=0))
        acc += pckh_05(regr[2][0], keypoint_coords)
    print(f'Total accuracy : {acc / len(inf_gen)}')