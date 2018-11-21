## how to extract file names and files from particular paths


from pathlib import Path
import glob
import json
import pickle
import os  
import time

from matplotlib.image import imread
import matplotlib.pyplot as plt


def crop_im(im, x, y):
	print(im.shape)
	print(min(x))
	print(max(x)-min(x))
	print(min(y))
	print(max(y)-min(y))
	fig,ax = plt.subplots(1)
	ax.imshow(im)
	time.sleep(1)
	# rect = patches.Rectangle((min(y),min(x)),max(y)-min(y),max(x)-min(x),linewidth=1,edgecolor='r',facecolor='none')
	# ax.add_patch(rect)
	plt.show(block=False)
	plt.pause(3)
	plt.close()
	time.sleep(20)
	# return im[min(x): min(y), max(x), max(y), :]



# check if list of file names has been made already or not
if not os.path.isfile('SUN-RGBD_file_list.pickle'):
	# get path to dataset from cwd
	path_to_data = Path.cwd().parents[0].joinpath('datasets', 'SUN-RGBD', 'SUNRGBD')
	path_list = path_to_data.glob('**/*/*/image/*.jpg')
	path_list = path_to_data.glob('**/*/*')
	# create list of all the paths
	image_list = [0]*14686    # number of images in the dataset (use list.append if not known) 
	file_counter = 0		  # to index image_list - performance based
	for path in path_list:
		image_paths = path.glob('**/image/*.jpg')   # format to get images from dataset
		# annotations_path = path.glob('**/annotation2Dfinal/*.json')
		# file_num = len(list(image_paths))           # for use with iterating over dataset
		image_paths = path.glob('**/image/*.jpg')   # redo as gets comsumed in list()
		for image_file_path in image_paths:
			parent_path = image_file_path.parents[1]    # go two .. up to image dir
			annotation_path = parent_path.joinpath('annotation2Dfinal', 'index.json')
			image_list[file_counter] = [image_file_path, annotation_path]
			file_counter += 1
	# save list so we don't have to re-run everytime
	with open('SUN-RGBD_file_list.pickle','wb') as pickle_out:
		pickle.dump(image_list, pickle_out)
else:      # load pickled file if it's already been made
	with open('SUN-RGBD_file_list.pickle','rb') as pickle_in:
		image_list = pickle.load(pickle_in)


# read all the json files and extract objects and bounding boxes
if not os.path.isfile('SUN-RGBD_subset.pickle'):
	labels = []    # don't know size due to some corrupt data
	images = []
	class_we_care = ['chair', 'door']
	for i in range(len(image_list)):
		try:  # some of the json files are corrupted with \'s -- no point in fixing them all
			with open(image_list[i][1]) as json_file:
				data = json.load(json_file)
			object_counter = 0
			for objects in data['frames']:
				for obj in objects['polygon']:
					if data['objects'][object_counter]['name'] in class_we_care:  # only take subset of objects
						labels.append([obj['x'], obj['y'], data['objects'][object_counter]['name']])
						## crop the images before saving.
						print('0')
						images.append(imread(image_list[i][0]))
						im = imread(image_list[i][0])
						print(data['objects'][object_counter]['name'])
						crop_im(im, obj['x'], obj['y'])
						print('2')
						# plt.imshow(im)
						plt.show(block=False)
						plt.pause(3)
						plt.close()
						time.sleep(5)
						object_counter += 1
		except:
			print('Error found in json file \'{}\''.format(image_list[i][1]))

	with open('SUN-RGBD_subset.pickle','wb') as pickle_out:
		pickle.dump([images, labels], pickle_out)
else:
	with open('SUN-RGBD_subset.pickle','rb') as pickle_in:
		images, labels = pickle.load(pickle_in)

## how to plot image with bounding box annotation
# fig,ax = plt.subplots(1)
# ax.imshow(im)
# # show where it thinks the box should be 
# anno = np.round(anno)
# rect = patches.Rectangle((anno[0],anno[1]),anno[2],anno[3],linewidth=1,edgecolor='r',facecolor='none')
# ax.add_patch(rect)		
# plt.show(block=False)
# plt.pause(5)
# plt.close()


## once got file names try out the code from youtube.py
