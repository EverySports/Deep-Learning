import pandas as pd
import numpy as np

def load_data():
    TRAIN_ROOT_PATH = f'../../Datasets/mpii_human_pose_v1'
    pd.set_option('display.max_columns', None)
    df_ = pd.read_csv(TRAIN_ROOT_PATH + f'/mpii_human_pose_v1_u12_2/mpii_dataset.csv')
    df_ = df_.iloc[:, 1:-3]

    df_ = df_.iloc[:, [0, 13, 14, 5, 6, 3, 4, 1, 2, 7, 8, 9, 10, 11, 12,
                      15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 32, 25, 26, 23, 24, 21, 22]]

    spine = (np.array(df_.iloc[:, [15, 16]]) +  np.array(df_.iloc[:, [1, 2]])) // 2

    df_ = pd.concat([df_, pd.DataFrame(spine)], axis=1)
    df = df_.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                     33, 34, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]
    df.rename(columns={0:'spine_X', 1:'spine_Y'}, inplace=True)

    df = df.drop(df[df.iloc[:, 1] == -1].index)
    df = df.drop(df[df.iloc[:, 17] == -1].index)

    df.iloc[:, 0] = TRAIN_ROOT_PATH + f'/images/' + df.iloc[:, 0]
    return df