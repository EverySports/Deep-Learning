import numpy as np
import scipy.ndimage as ndi

def within_nms_radius_fast(pose_coords, squared_nms_radius, point):
    if not pose_coords.shape[0]:
        return False
    return np.any(np.sum((pose_coords - point) ** 2, axis=1) <= squared_nms_radius)


def build_part_with_score(score_threshold, local_max_radius, scores):
    parts = []
    num_keypoints = scores.shape[2]
    lmd = 2 * local_max_radius + 1

    for keypoint_id in range(num_keypoints):
        kp_scores = scores[:, :, keypoint_id].copy()
        kp_scores[kp_scores < score_threshold] = 0.
        max_vals = ndi.maximum_filter(kp_scores, size=lmd, mode='constant')
        max_loc = np.logical_and(kp_scores == max_vals, kp_scores > 0)
        max_loc_idx = max_loc.nonzero()

        for y, x in zip(*max_loc_idx):
            parts.append((
                scores[y, x, keypoint_id],
                keypoint_id,
                np.array((y, x))
            ))

    return parts


def get_instance_score_fast(
        exist_pose_coords,
        squared_nms_radius,
        keypoint_scores, keypoint_coords):
    if exist_pose_coords.shape[0]:
        s = np.sum((exist_pose_coords - keypoint_coords) ** 2, axis=2) > squared_nms_radius
        not_overlapped_scores = np.sum(keypoint_scores[np.all(s, axis=0)])
    else:
        not_overlapped_scores = np.sum(keypoint_scores)
    return not_overlapped_scores / len(keypoint_scores)


def decode_single_poses(scores, offsets=None, NUM_KEYPOINTS=17, nms_radius=20,
                        score_threshold=0.5, LOCAL_MAXIMUM_RADIUS=1, output_stride=256 / 32):
    pose_scores = np.zeros(1)
    pose_keypoint_scores = np.zeros((1, NUM_KEYPOINTS))
    pose_keypoint_coords = np.zeros((1, NUM_KEYPOINTS, 2))

    squared_nms_radius = nms_radius ** 2

    scored_parts = build_part_with_score(score_threshold, LOCAL_MAXIMUM_RADIUS, scores)
    scored_parts = sorted(scored_parts, key=lambda x: x[0], reverse=True)

    keypoint_scores = np.zeros((NUM_KEYPOINTS))
    keypoint_coords = np.zeros((NUM_KEYPOINTS, 2))

    for root_score, root_id, root_coord in scored_parts:
        # print(offsets[root_coord[0], root_coord[1], root_id])
        root_image_coords = root_coord * output_stride + offsets[
            root_coord[0], root_coord[1], root_id]

        if within_nms_radius_fast(
                pose_keypoint_coords[:, root_id, :], squared_nms_radius, root_image_coords):
            continue

        keypoint_scores[root_id] = root_score
        keypoint_coords[root_id, :] = root_image_coords

    pose_scores = get_instance_score_fast(
        pose_keypoint_coords, squared_nms_radius, keypoint_scores, keypoint_coords)

    pose_keypoint_scores = keypoint_scores
    pose_keypoint_coords = keypoint_coords

    return [pose_scores], pose_keypoint_scores, pose_keypoint_coords


def decode_single_poses_nooff(scores, offsets=None, NUM_KEYPOINTS=17, nms_radius=20,
                              score_threshold=0.5, LOCAL_MAXIMUM_RADIUS=1, output_stride=256 / 32):
    pose_scores = np.zeros(1)
    pose_keypoint_scores = np.zeros((1, NUM_KEYPOINTS))
    pose_keypoint_coords = np.zeros((1, NUM_KEYPOINTS, 2))

    squared_nms_radius = nms_radius ** 2

    scored_parts = build_part_with_score(score_threshold, LOCAL_MAXIMUM_RADIUS, scores)
    scored_parts = sorted(scored_parts, key=lambda x: x[0], reverse=True)

    keypoint_scores = np.zeros((NUM_KEYPOINTS))
    keypoint_coords = np.zeros((NUM_KEYPOINTS, 2))

    for root_score, root_id, root_coord in scored_parts:
        root_image_coords = root_coord * output_stride

        if within_nms_radius_fast(
                pose_keypoint_coords[:, root_id, :], squared_nms_radius, root_image_coords):
            continue

        keypoint_scores[root_id] = root_score
        keypoint_coords[root_id, :] = root_image_coords

    pose_scores = get_instance_score_fast(
        pose_keypoint_coords, squared_nms_radius, keypoint_scores, keypoint_coords)

    pose_keypoint_scores = keypoint_scores
    pose_keypoint_coords = keypoint_coords

    return [pose_scores], pose_keypoint_scores, pose_keypoint_coords