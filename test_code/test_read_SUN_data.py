# examples/Python/Tutorial/Basic/rgbd_sun.py

from open3d import *
import matplotlib.pyplot as plt

color_raw = read_image("/Users/mattclifford/Documents/Technical_project/datasets/SUN-RGBD/SUNRGBD/kv2/kinect2data/000024_2014-05-26_14-44-24_260595134347_rgbf000055-resize/image/0000055.jpg")
depth_raw = read_image("/Users/mattclifford/Documents/Technical_project/datasets/SUN-RGBD/SUNRGBD/kv2/kinect2data/000024_2014-05-26_14-44-24_260595134347_rgbf000055-resize/depth/0000055.png")
rgbd_image = create_rgbd_image_from_sun_format(color_raw, depth_raw);
# print(rgbd_image)
# plt.subplot(1, 2, 1)
# plt.title('SUN grayscale image')
# plt.imshow(rgbd_image.color)
# plt.subplot(1, 2, 2)
# plt.title('SUN depth image')
# plt.imshow(rgbd_image.depth)
# plt.show()
pcd = create_point_cloud_from_rgbd_image(rgbd_image, PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
draw_geometries([pcd])