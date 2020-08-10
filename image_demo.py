import cv2
import time
import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import posenet
import matplotlib.pyplot as plt

def showInfo(heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result):
    ###
    plt.figure(figsize=(20, 15))
    plt.subplot(421)
    hmdata = heatmaps_result[0][:,:,-2]
    plt.imshow(hmdata)

    plt.subplot(423)
    offdata_x = offsets_result[0][:,:,0]
    plt.imshow(offdata_x)

    plt.subplot(424)
    offdata_y = offsets_result[0][:,:,1]
    plt.imshow(offdata_y)

    plt.subplot(425)
    disbwd_s = displacement_bwd_result[0][:,:,0]
    plt.imshow(disbwd_s)

    plt.subplot(426)
    disbwd_e = displacement_bwd_result[0][:,:,1]
    plt.imshow(disbwd_e)

    plt.subplot(427)
    disfwd_s = displacement_fwd_result[0][:,:,0]
    plt.imshow(disfwd_s)

    plt.subplot(428)
    disfwd_e = displacement_fwd_result[0][:,:,1]
    plt.imshow(disfwd_e)
    plt.show()
    ###

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mobilenet_v1')
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

def main():
    model = posenet.load_model(args.model)
    output_stride = model.output_stride

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    filenames = [
        f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    start = time.time()
    for f in filenames:
        input_image, draw_image, output_scale = posenet.read_imgfile(
            f, scale_factor=args.scale_factor, output_stride=output_stride)

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

        ###
        # showInfo(heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result)
        ###

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.numpy().squeeze(axis=0),
            offsets_result.numpy().squeeze(axis=0),
            displacement_fwd_result.numpy().squeeze(axis=0),
            displacement_bwd_result.numpy().squeeze(axis=0),
            output_stride=output_stride,
            max_pose_detections=10,
            min_pose_score=0.25)
        # 10 person
        # print(pose_scores[0])
        # print(keypoint_scores[0])
        # print(keypoint_coords[0])

        keypoint_coords *= output_scale

        if args.output_dir:
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)

            cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

        if not args.notxt:
            print()
            print("Results for image: %s" % f)
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

    print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()