import matplotlib.pyplot as plt
import numpy as np
import cv2

def draw_heatmap(batch, heatmap_result, offset_result, displacement_fwd_result=None, displacement_bwd_result=None):
    plt.figure(figsize=(10, 30))

    ### Heatmap
    for idx in range(17):
        plt.subplot(12, 5, idx + 1)
        plt.imshow(heatmap_result[batch][:, :, idx])
        plt.colorbar()
        plt.title(idx)

    ### Offset
    for idx in range(17):
        plt.subplot(12, 5, idx + 1 + 20)
        plt.imshow(offset_result[batch][:, :, idx])
        plt.colorbar()
        plt.title(idx)
        plt.subplot(12, 5, idx + 18 + 20)
        plt.imshow(offset_result[batch][:, :, 17 + idx])
        plt.colorbar()
        plt.title(17 + idx)
        # plt.show()


def get_adjacent(keypoint_coords):
    adjacent_list = []
    list_dic = [(0, 1), (1, 2), (2, 3), (3, 2), (2, 1), (1, 0),
                (0, 4), (4, 5), (5, 6), (6, 5), (5, 4), (4, 0),
                (0, 7), (7, 8), (8, 9), (9, 10), (10, 9), (9, 8),
                (8, 11), (11, 12), (12, 13), (13, 12), (12, 11), (11, 8),
                (8, 14), (14, 15), (15, 16), (16, 15)]
    for start, end in list_dic:
        start_node = keypoint_coords[start]
        end_node = keypoint_coords[end]
        if sum(start_node) == 0 or sum(end_node) == 0: continue
        adjacent_list.append(start_node)
        adjacent_list.append(end_node)
    adjacent_list = np.array(adjacent_list, dtype=np.int32)
    adjacent_list[:, 0], adjacent_list[:, 1] = adjacent_list[:, 1], adjacent_list[:, 0].copy()
    return adjacent_list


def get_keypoints(keypoint_coords):
    result = []
    for p in keypoint_coords:
        if sum(p) == 0: continue
        result.append(p)
    result = np.array(result, dtype=np.int32)
    result[:, 0], result[:, 1] = result[:, 1], result[:, 0].copy()
    return result


def draw_keypoint(keypoints, img):
    for pt in keypoints:
        img = cv2.circle(img, tuple(pt), radius=5, color=(255, 255, 0), thickness=-1)
    return img


def draw_skeleton_and_kepoints(keypoint_coords, img):
    adjacent_list = get_adjacent(keypoint_coords)
    draw_image = cv2.polylines(img, [adjacent_list], isClosed=False, color=(255,0,0), thickness=3)
    cv_keypoints = get_keypoints(keypoint_coords)
    draw_image = draw_keypoint(cv_keypoints, draw_image)
    return draw_image


def draw_result(img, idx, keypoint_coords, keypoint_coords_nooff, result, inf_gen):
    plt.figure(figsize=(10, 14))

    plt.subplot(321)
    plt.imshow(img[idx])
    plt.scatter(keypoint_coords[:, 1], keypoint_coords[:, 0], c='red')
    plt.title('with offset')

    plt.subplot(322)
    plt.imshow(img[idx])
    plt.scatter(keypoint_coords_nooff[:, 1], keypoint_coords_nooff[:, 0], c='blue')
    plt.title('No offset')

    hm = np.sum(result[0].numpy().squeeze(axis=0), axis=2)

    plt.subplot(323)
    plt.imshow(hm)
    plt.title('Full heatmap')

    plt.subplot(324)
    plt.imshow(img[0])
    plt.imshow(cv2.resize(hm, (256, 256)), alpha=0.7, cmap=plt.cm.gray)
    plt.title('Heatmap with image')

    plt.subplot(325)
    draw_image = draw_skeleton_and_kepoints(keypoint_coords, img[idx])
    plt.imshow(draw_image)
    plt.title('Predict')

    plt.subplot(326)
    img, regr = inf_gen.__getitem__(11)
    gt_keypoints = regr[2][0]
    gt_keypoints[:, [0, 1]] = gt_keypoints[:, [1, 0]]
    draw_image = draw_skeleton_and_kepoints(gt_keypoints, img[idx])
    plt.imshow(draw_image)
    plt.title('Ground Truth')

    plt.show()

def draw_train_result(history):
    fig, loss_ax = plt.subplots(3, 1, figsize=(7, 21))

    loss_ax[0].plot(history.history['loss'], 'y', label='train loss')
    loss_ax[0].plot(history.history['val_loss'], 'r', label='val loss')

    best = min(history.history['val_loss'])
    loss_ax[0].set_title(f'Best Loss : {best}')

    loss_ax[0].set_xlabel('epoch')
    loss_ax[0].set_ylabel('loss')

    loss_ax[0].legend(loc='lower left')

    loss_ax[1].plot(history.history['mobile_net_v1_loss'], 'y', label='train loss')
    loss_ax[1].plot(history.history['val_mobile_net_v1_loss'], 'r', label='val loss')

    best = min(history.history['val_mobile_net_v1_loss'])
    loss_ax[1].set_title(f'Best Loss : {best}')

    loss_ax[1].set_xlabel('epoch')
    loss_ax[1].set_ylabel('loss')

    loss_ax[1].legend(loc='lower left')

    loss_ax[2].plot(history.history['mobile_net_v1_1_loss'], 'y', label='train loss')
    loss_ax[2].plot(history.history['val_mobile_net_v1_1_loss'], 'r', label='val loss')

    best = min(history.history['val_mobile_net_v1_1_loss'])
    loss_ax[2].set_title(f'Best Loss : {best}')

    loss_ax[2].set_xlabel('epoch')
    loss_ax[2].set_ylabel('loss')

    loss_ax[2].legend(loc='lower left')

    plt.show()