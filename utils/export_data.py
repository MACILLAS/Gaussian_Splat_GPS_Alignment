import pandas as pd
from parse_exif import ImagesMeta

if __name__ == '__main__':
    images_dir = "/home/cviss/PycharmProjects/OpenSfM/data/Ford_Tower_0529_0607/images"
    images_txt = "/home/cviss/PycharmProjects/OpenSfM/0529/triangulated/images.txt"
    save_path = "/home/cviss/PycharmProjects/Gaussian_Splat_GPS_Alignment/out.csv"

    img_meta = ImagesMeta(images_dir, images_txt)

    df = pd.DataFrame(
        {'file_number': img_meta.file_num,
         'cam_center': img_meta.cam_centers,
         'ecef': img_meta.ecef,
         'lla': img_meta.lla
        })

    df.to_csv(save_path, index=False)