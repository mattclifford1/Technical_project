import cv2
import numpy as np
import pickle

# bunch of functions to help extract/ manipulate the dataset in a readable format.


def open_depth(meta_data, data_num):
    return cv2.imread(meta_data[data_num][1], -1)  # use -1 to read as 16 bit not 8 bit (need 16 bit for relative depth)


def open_rgb(meta_data, data_num):
    return cv2.imread(meta_data[data_num][0], -1)


def get_2D_bb(meta_data, data_num):
    return meta_data[data_num][5]


def get_3D_bb(meta_data, data_num):
    return (meta_data[data_num][6], meta_data[data_num][7], meta_data[data_num][8])


def get_camera_pos(meta_data, data_num):
    # extract 'K' from meta_data 
    # fx, fy are the focal focal length of x and y
    # cx, cy are the optical centers    of x and y
    [[fx, _, cx], [_, fy, cy], [_, _, _]] = meta_data[data_num][3] # K is [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    Rtilt = meta_data[data_num][2]
    return (fx, fy, cx, cy, Rtilt)


def get_label(meta_data, data_num):
    return meta_data[data_num][4]


def load_SUNRGBD_meta(path_to_dataset = 'SUN-RGBD_convert_matlab.pickle'):
    with open(path_to_dataset,'rb') as pickle_in:
        meta_data = pickle.load(pickle_in)
    # meta_data in format:
    # [rgb_file_path, depth_file_path, Rtilt, K, label, BB_2D, basis_3D, coeff_3D, centroid_3D]
    return meta_data


def make_point_cloud(depth, meta_data, data_num):   # meta_data needed as holds camera position and tilt
    # convert int16 values to depth (inversely proportional)
    depth_input = ((depth>>3)|(depth<<13))/1000  # taken from sunrgbd matlab toolbox --- fuck knows how this works
    # threshold max depth
    depth_input[np.where(depth_input > 8)] = 8   # max depth measurement at 8 meters
    # get camera parameters to scale and tilt camera
    (fx, fy, cx, cy, Rtilt) = get_camera_pos(meta_data, data_num)

    x, y = np.meshgrid(range(depth_input.shape[1]), range(depth_input.shape[0]))
    # translate from optical center and scale inversely proportional to focal lengths
    x3 = (x-cx) * (depth_input*(1/fx))
    y3 = (y-cy) * (depth_input*(1/fy))
    z3 = depth_input
    # convert to coordinate style list
    points3d = [x3.reshape(-1), z3.reshape(-1), -y3.reshape(-1)]
    # tilt points so that the floor is flat using camera tilt
    return np.transpose(np.matmul(Rtilt, points3d))
