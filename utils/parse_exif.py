from os.path import join
from pyproj import Proj
import numpy as np
from camera_pose import compose_44, decompose_44, qvec2rotmat, rotmat2qvec

ZoneNo = "17"
# Peterborough to Windsor (Ontario) if you decide to use UTM coordinates please change.
myProj = Proj("+proj=utm +zone="+ZoneNo+" +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs") # Northern Hemisphere

a = 6378137.0  # Semi-major axis in meters
f = 1 / 298.257223563  # Flattening
e_sq = f * (2 - f)  # Square of eccentricity

def latlonalt_to_ecef_matrix(lat, lon, alt):
    """
    This function was authored by Anas and validated by Max

    :param lat:
    :param lon:
    :param alt:
    :return: transformation matrix, ECEF coordinates
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = a / np.sqrt(1 - e_sq * np.sin(lat_rad) ** 2)
    X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = ((1 - e_sq) * N + alt) * np.sin(lat_rad)
    transformation_matrix = np.array([
        [np.cos(lat_rad) * np.cos(lon_rad), -np.sin(lon_rad), 0, X],
        [np.cos(lat_rad) * np.sin(lon_rad), np.cos(lon_rad), 0, Y],
        [np.sin(lat_rad), 0, 1, Z],
        [0, 0, 0, 1]])

    return transformation_matrix, np.array([X, Y, Z])

def get_dji_meta(filepath: str) -> dict:
    """
     Authored by Max for GS_Stream
     Returns a dict with DJI-specific metadata stored in the XMB portion of the image
     @param filepath: filepath to referenced image with DJI formatted meta-data
     @returns dictionary with metadata tags
     """

    # list of metadata tags
    djimeta = ["AbsoluteAltitude", "RelativeAltitude", "GimbalRollDegree", "GimbalYawDegree", \
               "GimbalPitchDegree", "FlightRollDegree", "FlightYawDegree", "FlightPitchDegree", "GpsLatitude", "GpsLongitude"]

    # read file in binary format and look for XMP metadata portion
    fd = open(filepath, 'rb')
    d = fd.read()
    xmp_start = d.find(b'<x:xmpmeta')
    xmp_end = d.find(b'</x:xmpmeta')

    # convert bytes to string
    xmp_b = d[xmp_start:xmp_end + 12]
    xmp_str = xmp_b.decode()

    fd.close()

    # parse the XMP string to grab the values
    xmp_dict = {}
    for m in djimeta:
        istart = xmp_str.find(m)
        ss = xmp_str[istart:istart + len(m) + 20]
        val = float(ss.split('"')[1])
        xmp_dict.update({m: val})

    return xmp_dict

class ImagesMeta:
    # Custom ImageMeta class for testing
    def __init__(self, data_path, images_txt_file):
        self.img_id = [] # image id from colmap
        self.files = [] # filenames
        self.t_vec = [] # W2C t_vec
        self.q_vec = [] # W2C q_vec
        self.cam_centers = [] # C2W t_vec
        self.lla = []
        self.utm = []
        self.ecef = []
        self.file_num =[] # the image number from DJI filename

        with open(images_txt_file, 'r') as f:
            for count, line in enumerate(f, start=0):
                if count < 4:
                    pass
                else:
                    if count % 2 == 0:
                        str_parsed = line.split()
                        self.img_id.append(str_parsed[0])
                        self.files.append(str_parsed[9])
                        self.file_num.append(int(str_parsed[9].split('_')[2]))
                        q_raw = np.array(str_parsed[1:5], dtype=np.float64)
                        R_raw = qvec2rotmat(q_raw)
                        t_raw = np.array(str_parsed[5:8], dtype=np.float64)
                        cam_center = (-R_raw.T @ t_raw)
                        self.q_vec.append(q_raw)
                        self.t_vec.append(t_raw)
                        self.cam_centers.append(cam_center)
                        meta = get_dji_meta(join(data_path, str_parsed[9]))
                        self.lla.append(np.array([meta['GpsLatitude'], meta['GpsLongitude'], meta['AbsoluteAltitude']]))
                        UTMx, UTMy = myProj(meta['GpsLongitude'], meta['GpsLatitude'])
                        #Lon2, Lat2 = myProj(UTMx, UTMy, inverse=True)
                        self.utm.append([UTMx, UTMy, meta['AbsoluteAltitude']])
                        _, ecef = latlonalt_to_ecef_matrix(meta['GpsLatitude'], meta['GpsLongitude'],
                                                           meta['AbsoluteAltitude'])
                        self.ecef.append(ecef)

        self.q_vec = np.array(self.q_vec, dtype=np.float32)
        self.t_vec = np.array(self.t_vec, dtype=np.float32)

    def get_closest_n(self, pose, n=4):
        """
        :param pose: 4x4 extrinsic matrix
        :param n: number of closest images to provide
        :return: list of image filenames
        """
        R, t = decompose_44(pose)

        t = np.matmul(-R.T, t)

        #R = Rotation.from_matrix(R)
        #R = R.as_quat()[[3, 0, 1, 2]]  # Change from x,y,z,w to w,x,y,z

        # First filter cameras by translation
        t_dist = np.linalg.norm(self.cam_centers - t, axis=1)

        lowest_t_idx = np.argsort(t_dist)[0:n]  # Find the closest 2*n idxs
        filtered_files = [self.files[i] for i in lowest_t_idx.tolist()]

        # Then rank by camera rotation (Depreciated June 18 2024, fixed closes images bug.)
        #filtered_q_vec = self.q_vec[lowest_t_idx]
        #q_dist = np.linalg.norm(filtered_q_vec - R, axis=1)
        #lowest_q_idx = np.argsort(q_dist)[0:n]
        #q_filtered_files = [filtered_files[i] for i in lowest_q_idx.tolist()]
        #return q_filtered_files

        return filtered_files, np.sort(t_dist)[0:n]

    def get_pose_by_filename(self, filename, colmap=False):
        """
        :param filename: filename of the image
        :param colmap: boolean flag to indicate whether to return in original colmap convention (i.e., T_w_c)
        :return: 4x4 Transformation matrix
        """
        idx = self.files.index(filename)
        R = qvec2rotmat(self.q_vec[idx])
        #Rotation.from_quat(self.q_vec[idx][[1, 2, 3, 0]]).as_matrix()
        if colmap:
            t = self.t_vec[idx]
        else:
            R = np.linalg.inv(R)
            t = self.cam_centers[idx]
        return compose_44(R, t)

    def get_cam_center_by_filename(self, filename):
        idx = self.files.index(filename)
        return self.cam_centers[idx]

    def get_ecef_by_filename(self, filename):
        idx = self.files.index(filename)
        return self.ecef[idx]

    def get_utm_by_filename(self, filename):
        idx = self.files.index(filename)
        return self.utm[idx]