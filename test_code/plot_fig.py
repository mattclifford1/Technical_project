import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import open3d
import pptk

import utils


def bb_2D(im, bb):
	# write code to show image with bounding box drawn 
	# funcitonality:
	# 			might need to give different colours for ground truth, correct, incorrect
	#			do this by do variable number of arguments *kargs?
	return 0


def bb_3D(points, bb_sizes, bb_centers, bb_rotations):
	# downsample the point cloud (for mpl)
	pcd = open3d.PointCloud()
	pcd.points = open3d.Vector3dVector(points)
	# open3d.draw_geometries([pcd])
	downpcd = open3d.voxel_down_sample(pcd, voxel_size = 0.05)

	# extract downsampled points to plot in matplotlib
	downpoints = np.asarray(downpcd.points)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(downpoints[:,0], downpoints[:,1], downpoints[:,2], s=0.1)

	# plot bounding boxes
	for label_num in range(len(bb_sizes)):
		# get current label's bounding box parameters
		bb_rotation = bb_rotations[label_num]
		bb_size = bb_sizes[label_num]
		bb_center = bb_centers[label_num]
		plot_cuboid(ax, bb_size, bb_center, bb_rotation, colour='red')

	plt.axis('equal')
	plt.show()


def plot_cuboid(ax, bb_size, bb_center, bb_rotation, colour='green'):
	# set up corners around the origin
	corners = np.array([[-bb_size[0], bb_size[0], -bb_size[0], bb_size[0], -bb_size[0], bb_size[0], -bb_size[0], bb_size[0]],
	                    [-bb_size[1],-bb_size[1], -bb_size[1],-bb_size[1],  bb_size[1], bb_size[1],  bb_size[1], bb_size[1]],
	                    [-bb_size[2],-bb_size[2],  bb_size[2], bb_size[2], -bb_size[2],-bb_size[2],  bb_size[2], bb_size[2]]])
	# rotate points
	corners = np.dot(corners.T, bb_rotation)
	# translate points
	for i in range(3):
	    corners[:, i] += bb_center[i]
	# plot edges
	edge_list = [[0,1], [1,3], [3,2], [2,0], [4,5], [5,7], [7,6], [6,4], [0,4], [2,6], [3,7], [1,5]]
	for i in edge_list:
	    ax.plot3D([corners[i[0], 0], corners[i[1], 0]], [corners[i[0], 1], corners[i[1], 1]], [corners[i[0], 2], corners[i[1], 2]], colour) 


if __name__ == '__main__':
	data_num = 0
	meta_data = utils.load_SUNRGBD_meta()
	depth = utils.open_depth(meta_data, data_num)
	points = utils.make_point_cloud(depth, meta_data, data_num)

	bb_rotations = meta_data[data_num][6]
	bb_sizes = meta_data[data_num][7]
	bb_centers = meta_data[data_num][8]
	bb_3D(points, bb_sizes, bb_centers, bb_rotations)





# pptk has the nicest viewer
# v = pptk.viewer(points3d)

# v =pptk.viewer(points)
# # v.load(points)
# v.load(points3d)

# points = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],
#           [0,0,1],[1,0,1],[0,1,1],[1,1,1]]
# lines = [[0,1],[0,2],[1,3],[2,3],
#          [4,5],[4,6],[5,7],[6,7],
#          [0,4],[1,5],[2,6],[3,7]]
# colors = [[1, 0, 0] for i in range(len(lines))]
# line_set = open3d.LineSet()
# line_set.points = open3d.Vector3dVector(points)
# line_set.lines = open3d.Vector2iVector(lines)
# line_set.colors = open3d.Vector3dVector(colors)
# open3d.draw_geometries([line_set])
