import cv2
import numpy as np
import pickle


def open_depth(meta_data, data_num):
    return cv2.imread(meta_data[data_num][1], -1)  # use -1 to read as 16 bit not 8 bit


def open_rgb(meta_data, data_num):
    return cv2.imread(meta_data[data_num][0], -1)


def load_SUNRGBD_meta():
    with open('SUN-RGBD_convert_matlab.pickle','rb') as pickle_in:
        meta_data = pickle.load(pickle_in)
    # meta_data in format:
    # [rgb_file_path, depth_file_path, Rtilt, K, label, BB_2D, basis_3D, coeff_3D, centroid_3D]
    return meta_data


def make_point_cloud(depth, meta_data, data_num):   # meta_data needed as holds camera position and tilt
    # convert int16 values to depth (inversely proportional)
    depth_input = ((depth>>3)|(depth<<13))/1000  # taken from sunrgbd toolbox --- fuck knows how this works
    # threshold max depth
    depth_input[np.where(depth_input > 8)] = 8   # max depth measurement at 8 meters
    # extract 'K' from meta_data 
    [[fx, _, cx], [_, fy, cy], [_, _, _]] = meta_data[data_num][3] # K is [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # fx, fy are the focal focal length of x and y
    # cx, cy are the optical centers    of x and y 

    x, y = np.meshgrid(range(depth_input.shape[1]), range(depth_input.shape[0]))
    # translate from optical center and scale inversely proportional to focal lengths
    x3 = (x-cx) * (depth_input*(1/fx))
    y3 = (y-cy) * (depth_input*(1/fy))
    z3 = depth_input
    # convert to coordinate style list
    points3d = [x3.reshape(-1), z3.reshape(-1), -y3.reshape(-1)]
    # tilt points so that the floor is flat using camera tilt
    Rtilt = meta_data[data_num][2]
    return np.transpose(np.matmul(Rtilt, points3d))
