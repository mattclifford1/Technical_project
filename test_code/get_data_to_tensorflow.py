## read dataset from meta data - pre process into tensorflow format

import pickle
import numpy as np
from matplotlib.image import imread
from matplotlib import patches
import matplotlib.pyplot as plt
import sys

def crop_image(im, anno):
	return im[int(anno[1]): int(round(anno[1]+anno[3])), int(anno[0]):int(round(anno[0]+anno[2]))]


# load dataset's metadata
try:
	with open('SUN-RGBD_convert_matlab.pickle','rb') as pickle_in:
		dataset = pickle.load(pickle_in)
		# dataset is in format:
		# rows: each entry
		# columns:  0: filepath to rgb image .jpg
		#		    1: filepath to depth image .png
		#			2: object bounding boxes 
		#			3: object labels
except:
	print('Need to make the python data list from \'get_data_from_matlab.py\'')

images_not_found = 0                 # counter for lost images
class_we_care = ['chair', 'door']    # taking subset of objects
for entry in range(len(dataset)):    # iterate over all images
	for obj in range(len(dataset[entry][3])):    # over all objects in an image
		if dataset[entry][3][obj] in class_we_care:   # select only objects we interested in
			anno = dataset[entry][2][obj]   # current object annotation box					
			im = imread(dataset[entry][0])	
			try:
				cropped_im = crop_image(im, anno)
				# ax.imshow(cropped_im)
			except:
				print('{object} cannot be cropped with annotation: {ann}'.format(
																	    object = dataset[entry][3][obj],
																        ann = anno))
				print('image size is {size} \n'.format(size = im.shape))
				print(dataset[entry][0])
				sys.stdout.flush()
				# fig,ax = plt.subplots(1)
				# ax.imshow(im)
				# # show where it thinks the box should be 
				# anno = np.round(anno)
				# rect = patches.Rectangle((anno[0],anno[1]),anno[2],anno[3],linewidth=1,edgecolor='r',facecolor='none')
				# ax.add_patch(rect)		
				# plt.show(block=False)
				# plt.pause(5)
				# plt.close()
	if entry % 100 == 0:
		print('{}% '.format(int((entry/len(dataset)*100))))

