import pandas as pd
import numpy as np
from rigid_transform_3D import rigid_transform_3D
from utils.error import analyse_err

def naive_align(df):
    err = []
    file_num = []

    for i in range(len(df.index)):
        test = df.iloc[i]
        A_test = np.array([test["cam_center_x"], test["cam_center_y"], test["cam_center_z"]]).reshape(3, 1)
        B_test = np.array([test["utm_x"], test["utm_y"], test["utm_z"]]).reshape(3, 1)

        train = df.drop([i])
        A_train = np.array([train["cam_center_x"], train["cam_center_y"], train["cam_center_z"]]).reshape(3, -1)
        B_train = np.array([train["utm_x"], train["utm_y"], train["utm_z"]]).reshape(3, -1)

        ret_R, ret_t = rigid_transform_3D(A_train, B_train)

        # Compare the recovered R and t with the original
        B2 = (ret_R @ A_test).reshape(3, 1) + ret_t

        dist = B2 - B_test
        dist = np.sqrt(np.sum(dist * dist))
        err.append(dist)
        file_num.append(test["file_number"])

    analyse_err(err, file_num)

def bottom_only_align(df, max_file_num, n_samp):
    err = []
    file_num = []

    for i in range(len(df.index)):
        test = df.iloc[i]
        A_test = np.array([test["cam_center_x"], test["cam_center_y"], test["cam_center_z"]]).reshape(3, 1)
        B_test = np.array([test["utm_x"], test["utm_y"], test["utm_z"]]).reshape(3, 1)

        train = df.drop([i])
        train = train[train.file_number <= max_file_num]
        train = train.sample(n=n_samp)
        A_train = np.array([train["cam_center_x"], train["cam_center_y"], train["cam_center_z"]]).reshape(3, -1)
        B_train = np.array([train["utm_x"], train["utm_y"], train["utm_z"]]).reshape(3, -1)

        ret_R, ret_t = rigid_transform_3D(A_train, B_train)

        # Compare the recovered R and t with the original
        B2 = (ret_R @ A_test).reshape(3, 1) + ret_t

        dist = B2 - B_test
        dist = np.sqrt(np.sum(dist * dist))
        err.append(dist)
        file_num.append(test["file_number"])

    analyse_err(err, file_num)


if __name__ == '__main__':
    df = pd.read_csv('data/GPS_aligned_0529.csv')
    df = df.sort_values(by=['file_number'])
    df = df.reset_index(drop=True)

    naive_align(df)
    #bottom_only_align(df, 120, 20)
