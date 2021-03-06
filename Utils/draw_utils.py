import matplotlib.pyplot as plt
import numpy as np
import cv2

def draw_heatmap(batch, heatmap_result, offset_result=None, displacement_fwd_result=None, displacement_bwd_result=None):
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
    plt.show()


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
        if sum(start_node) <= 0 or sum(end_node) <= 0: continue
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
        if sum(pt) <= 0: continue
        img = cv2.circle(img, tuple(pt), radius=5, color=(255, 255, 0), thickness=-1)
    return img

def draw_skeleton(keypoint_coords, img):
    list_dic = [(0, 1), (1, 2), (2, 3),
                (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9), (9, 10),
                (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16)]
    for adj in list_dic:
        pt1 = keypoint_coords[adj[0]]
        pt2 = keypoint_coords[adj[1]]
        if sum(pt1) <= 0 or sum(pt2) <= 0: continue
        img = cv2.line(img, (int(pt1[1]), int(pt1[0])),
                       (int(pt2[1]), int(pt2[0])), color=(255,0,0), thickness=3)
    return img


def draw_skeleton_and_kepoints(keypoint_coords, img):
    adjacent_list = get_adjacent(keypoint_coords)
    draw_image = cv2.polylines(img, [adjacent_list], isClosed=False, color=(255,0,0), thickness=3)
    # draw_image = draw_skeleton(keypoint_coords, img)
    cv_keypoints = get_keypoints(keypoint_coords)
    draw_image = draw_keypoint(cv_keypoints, draw_image)
    return draw_image


def draw_result(img, idx, keypoint_coords, keypoint_coords_nooff, result, inf_gen, seed):
    plt.figure(figsize=(10, 18))

    plt.subplot(521)
    plt.imshow(img[idx])
    plt.scatter(keypoint_coords[:, 1], keypoint_coords[:, 0], c='red')
    plt.title('with offset')

    plt.subplot(522)
    plt.imshow(img[idx])
    plt.scatter(keypoint_coords_nooff[:, 1], keypoint_coords_nooff[:, 0], c='blue')
    plt.title('No offset')

    plt.subplot(523)
    draw_image = draw_skeleton_and_kepoints(keypoint_coords, img[idx])
    plt.imshow(draw_image)
    plt.title('Predict')

    plt.subplot(524)
    img, regr = inf_gen.__getitem__(seed)
    img /= 255.
    gt_keypoints = regr[2][0]
    gt_keypoints[:, [0, 1]] = gt_keypoints[:, [1, 0]]
    draw_image = draw_skeleton_and_kepoints(gt_keypoints, img[idx])
    plt.imshow(draw_image)
    plt.title('Ground Truth')
    
    # hm = np.sum(result[0].numpy().squeeze(axis=0), axis=2)
    hm = np.sum(result.numpy().squeeze(axis=0), axis=2)

    plt.subplot(525)
    plt.imshow(hm)
    plt.title('Full heatmap')

    plt.subplot(526)
    img, regr = inf_gen.__getitem__(seed)
    img /= 255.
    plt.imshow(img[0])
    plt.imshow(cv2.resize(hm, (256, 256)), alpha=0.7, cmap=plt.cm.gray)
    plt.title('Heatmap with image')

    plt.subplot(527)
    hm_ = hm/8.
    hm_[hm_ >= 0.4] = 1
    hm_[hm_ < 0.4] = 0
    plt.imshow(hm_)
    plt.title('Image Segmentation')

    plt.subplot(528)
    img, regr = inf_gen.__getitem__(seed)
    img /= 255.
    plt.imshow(img[0])
    plt.imshow(cv2.resize(hm_, (256, 256)), alpha=0.5)
    plt.title('Segmetation with image')

    hm_idx = np.where(hm_ == 1)
    max_x_idx, min_x_idx = max(hm_idx[0][1:-1]), min(hm_idx[0][1:-1])
    max_y_idx, min_y_idx = max(hm_idx[1][1:-1]), min(hm_idx[1][1:-1])
    plt.subplot(529)
    img, regr = inf_gen.__getitem__(seed)
    img /= 255.
    cv2.rectangle(img[0], (min_y_idx, min_x_idx), (max_y_idx, max_x_idx), color=(255,0,0), thickness=2)
    plt.imshow(img[0])
    plt.title('BBox with image')

    plt.show()

def draw_train_result(history):
    fig, loss_ax = plt.subplots(1, 1, figsize=(7, 21))

    loss_ax[0].plot(history.history['loss'], 'y', label='train loss')
    loss_ax[0].plot(history.history['val_loss'], 'r', label='val loss')

    best = min(history.history['val_loss'])
    loss_ax[0].set_title(f'Best Loss : {best}')

    loss_ax[0].set_xlabel('epoch')
    loss_ax[0].set_ylabel('loss')

    loss_ax[0].legend(loc='lower left')

    # loss_ax[1].plot(history.history['mobile_net_v1_loss'], 'y', label='train loss')
    # loss_ax[1].plot(history.history['val_mobile_net_v1_loss'], 'r', label='val loss')
    #
    # best = min(history.history['val_mobile_net_v1_loss'])
    # loss_ax[1].set_title(f'Best Loss : {best}')
    #
    # loss_ax[1].set_xlabel('epoch')
    # loss_ax[1].set_ylabel('loss')
    #
    # loss_ax[1].legend(loc='lower left')

    # loss_ax[2].plot(history.history['mobile_net_v1_1_loss'], 'y', label='train loss')
    # loss_ax[2].plot(history.history['val_mobile_net_v1_1_loss'], 'r', label='val loss')
    #
    # best = min(history.history['val_mobile_net_v1_1_loss'])
    # loss_ax[2].set_title(f'Best Loss : {best}')
    #
    # loss_ax[2].set_xlabel('epoch')
    # loss_ax[2].set_ylabel('loss')
    #
    # loss_ax[2].legend(loc='lower left')

    plt.show()