from .utils import *

import tensorflow as tf
import numpy as np
import cv2

class EvalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, df, batch_size=4, shuffle=True,
                 random_state=42, image_paths=None, mode='fit', ):
        self.list_IDs = list_IDs
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.image_paths = image_paths
        self.mode = mode

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_batch = [self.list_IDs[i] for i in indexes]

        X, img_hs, img_ws = self.__generate_X(list_IDs_batch)

        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch, img_hs, img_ws)
            return X, y

        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        X = []
        img_ws = []
        img_hs = []
        for i, ID in enumerate(list_IDs_batch):
            img_path = self.image_paths[ID]
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32)
            # img /= 255.
            img_hs.append(img.shape[0])
            img_ws.append(img.shape[1])

            ### Show Img ###
            # plt.imshow(img)
            # plt.title(f'ID: {ID}, Shape: {img.shape}')
            ################
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            X.append(img)
        X = np.array(X)
        return X, img_hs, img_ws

    def __generate_y(self, list_IDs_batch, img_hs, img_ws):
        heatmap_result = []
        offset_result = []
        displacement_fwd_result = []
        displacement_bwd_result = []
        keypoints_result = []

        for i, ID in enumerate(list_IDs_batch):
            keypoints = self.df[self.df.index == ID].iloc[:, 1:].to_numpy()[0]
            hm, offset, _, _ = heatmap(keypoints, input_size=(img_hs[i], img_ws[i]))
            heatmap_result.append(hm)
            offset_result.append(offset)

            keypoints = keypoints.reshape(17, 2)
            keypoints[:, 0] *= (256 / img_ws[i])
            keypoints[:, 1] *= (256 / img_hs[i])
            keypoints_result.append(keypoints)
            # displacement_fwd_result.append(displacement_fwd)
            # displacement_bwd_result.append(displacement_bwd)

        heatmap_result = np.array(heatmap_result)
        offset_result = np.array(offset_result)
        keypoints_result = np.array(keypoints_result)
        # displacement_fwd_result = np.array(displacement_fwd_result)
        # displacement_bwd_result = np.array(displacement_bwd_result)

        #         return [heatmap_result, offset_result, displacement_fwd_result, displacement_bwd_result]
        return [heatmap_result, offset_result, keypoints_result]


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, df, batch_size=4, shuffle=True,
                 random_state=42, image_paths=None, mode='fit', ):
        self.list_IDs = list_IDs
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.image_paths = image_paths
        self.mode = mode

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_batch = [self.list_IDs[i] for i in indexes]

        X, img_hs, img_ws = self.__generate_X(list_IDs_batch)

        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch, img_hs, img_ws)
            return X, y

        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        X = []
        img_ws = []
        img_hs = []
        for i, ID in enumerate(list_IDs_batch):
            img_path = self.image_paths[ID]
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32)
            # img /= 255.
            img_hs.append(img.shape[0])
            img_ws.append(img.shape[1])

            ### Show Img ###
            # plt.imshow(img)
            # plt.title(f'ID: {ID}, Shape: {img.shape}')
            ################
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            X.append(img)
        X = np.array(X)
        return X, img_hs, img_ws

    def __generate_y(self, list_IDs_batch, img_hs, img_ws):
        heatmap_result = []
        offset_result = []
        displacement_fwd_result = []
        displacement_bwd_result = []
        for i, ID in enumerate(list_IDs_batch):
            keypoints = self.df[self.df.index == ID].iloc[:, 1:].to_numpy()[0]
            hm, offset, _, _ = heatmap(keypoints, input_size=(img_hs[i], img_ws[i]))
            heatmap_result.append(hm)
            offset_result.append(offset)
            # displacement_fwd_result.append(displacement_fwd)
            # displacement_bwd_result.append(displacement_bwd)

        heatmap_result = np.array(heatmap_result)
        offset_result = np.array(offset_result)
        # displacement_fwd_result = np.array(displacement_fwd_result)
        # displacement_bwd_result = np.array(displacement_bwd_result)

        #         return [heatmap_result, offset_result, displacement_fwd_result, displacement_bwd_result]
        return [heatmap_result, offset_result]