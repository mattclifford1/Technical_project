# examples/Python/Tutorial/Basic/rgbd_sun.py

from open3d import *
import matplotlib.pyplot as plt


print("Read SUN dataset")
color_raw = read_image("../datasets/SUN-RGBD/SUNRGBD/xtion/sun3ddata/brown_bm_1/brown_bm_1/0000001-000000020431/fullres/0000001-000000020431.jpg")
depth_raw = read_image("../datasets/SUN-RGBD/SUNRGBD/xtion/sun3ddata/brown_bm_1/brown_bm_1/0000001-000000020431/fullres/0000002-000000033369.png")
rgbd_image = create_rgbd_image_from_sun_format(color_raw, depth_raw);
print(rgbd_image)
plt.subplot(1, 2, 1)
plt.title('SUN grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('SUN depth image')
plt.imshow(rgbd_image.depth)
plt.show()
pcd = create_point_cloud_from_rgbd_image(rgbd_image, PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
draw_geometries([pcd])