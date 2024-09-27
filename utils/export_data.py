import pandas as pd
from parse_exif import ImagesMeta
import numpy as np

if __name__ == '__main__':
    images_dir = "/home/cviss/PycharmProjects/OpenSfM/data/Ford_Tower_0529/images"
    images_txt = ("/home/cviss/PycharmProjects/OpenSfM/0529/triangulated/images.txt")
    save_path = "../data/GPS_aligned_0529.csv"

    img_meta = ImagesMeta(images_dir, images_txt)
    cam_centers = np.array(img_meta.cam_centers)
    utm = np.array(img_meta.utm)
    ecef = np.array(img_meta.ecef)
    lla = np.array(img_meta.lla)
    rtk_std = np.array(img_meta.rtk_std)

    weights = (rtk_std - rtk_std.min()) / (rtk_std.max() - rtk_std.min())
    weights = 1 - weights

    df = pd.DataFrame(
        {'file_number': img_meta.file_num,
         'cam_center_x': cam_centers[:, 0], 'cam_center_y': cam_centers[:, 1], 'cam_center_z': cam_centers[:, 2],
         'utm_x': utm[:, 0], 'utm_y': utm[:, 1], 'utm_z': utm[:, 2],
         'ecef_x': ecef[:, 0], 'ecef_y': ecef[:, 1], 'ecef_z': ecef[:, 2],
         'lla_x': lla[:, 0], 'lla_y': lla[:, 1], 'lla_z': lla[:, 2],
         'weights': weights,
         'rtk_std': rtk_std,
         'filename': img_meta.files,
        })

    df.to_csv(save_path, index=False)