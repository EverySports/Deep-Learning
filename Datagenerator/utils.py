import numpy as np
from math import floor

def normalize_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std


def heatmap(keypoints, input_size, output_width=256, output_height=256, sigma=10):
    heatmap_result = np.zeros((output_width, output_height, 17))
    offset_result = np.zeros((output_width, output_height, 34))
    displacement_fwd_result = np.zeros((output_width, output_height, 32))
    displacement_bwd_result = np.zeros((output_width, output_height, 32))
    resize_rate_w = output_width / input_size[1]
    resize_rate_h = output_height / input_size[0]

    def get_coords(keypoints):
        keypoints = keypoints.reshape(17, 2)
        x_radius = (np.max(keypoints[:, 0]) - np.min(keypoints[:, 0])) / 8
        y_radius = (np.max(keypoints[:, 1]) - np.min(keypoints[:, 1])) / 8
        return keypoints, x_radius, y_radius

    def get_heatmap(p_x, p_y, sigma):
        X1 = np.linspace(1, output_width, output_height)
        Y1 = np.linspace(1, output_width, output_height)
        [X, Y] = np.meshgrid(X1, Y1)
        X = X - floor(p_x)
        Y = Y - floor(p_y)
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma ** 2
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        heatmap = heatmap[:, :, np.newaxis]
        return heatmap

    def get_offset(x, y, x_radius, y_radius):
        x_radius = np.max([2, floor(x_radius * resize_rate_h)])
        y_radius = np.max([2, floor(y_radius * resize_rate_w)])
        offset_x = np.zeros((output_width, output_height))
        offset_y = np.zeros((output_width, output_height))
        p_x = floor(x * resize_rate_w)
        p_y = floor(y * resize_rate_h)
        for idx in range(output_width):
            # offset_x[idx,:] = x - (p_x / resize_rate_w) + (1 / resize_rate_w) * (idx - p_x)
            # offset_y[:,idx] = y - (p_y / resize_rate_h) + (1 / resize_rate_h) * (idx - p_y)
            if p_y - y_radius <= idx <= p_y + y_radius:
                offset_x[idx, p_x - x_radius:p_x + x_radius] = y - idx / resize_rate_h
            if p_x - x_radius <= idx <= p_x + x_radius:
                offset_y[p_y - y_radius:p_y + y_radius, idx] = x - idx / resize_rate_w
        return offset_x, offset_y

    keypoints, x_radius, y_radius = get_coords(keypoints)

    for idx, keypoint in enumerate(keypoints):
        if -1 in keypoint: continue
        heatmap = get_heatmap(keypoint[0] * resize_rate_w,
                              keypoint[1] * resize_rate_h,
                              sigma)

        heatmap_result[:, :, idx] = np.maximum(heatmap_result[:, :, idx], heatmap[:, :, 0])

        # offset_x, offset_y = get_offset(keypoint[0], keypoint[1], x_radius, y_radius)
        # offset_result[:, :, idx] = offset_x
        # offset_result[:, :, 17 + idx] = offset_y

    return heatmap_result, offset_result, displacement_fwd_result, displacement_bwd_result