## read dataset from meta data - pre process into tensorflow format

import tensorflow as tf
import pickle
import numpy as np
from matplotlib.image import imread
from matplotlib import patches
import matplotlib.pyplot as plt
import sys
from random import shuffle
import cv2


def crop_image(im, anno):
	return im[int(anno[1]): int(round(anno[1]+anno[3])), int(anno[0]):int(round(anno[0]+anno[2]))]


# tensorflow feature functions
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_dataset():
	# load dataset's metadata
	try:
		with open('SUN-RGBD_convert_matlab.pickle','rb') as pickle_in:
			return pickle.load(pickle_in)
			# dataset is in format:
			# rows: each entry
			# columns:  0: filepath to rgb image .jpg
			#		    1: filepath to depth image .png
			#			2: object bounding boxes 
			#			3: object labels
	except:
		print('Need to make the python data list from \'get_data_from_matlab.py\'')


def create_data_record(out_filename, dataset):
	# open the TFRecords file
	writer = tf.python_io.TFRecordWriter(out_filename)

	class_we_care = ['chair', 'door']    # taking subset of objects
	for entry in range(len(dataset)):    # iterate over all images
		im = imread(dataset[entry][0])   # evalute now to save re compution	
		for obj in range(len(dataset[entry][3])):    # over all objects in an image
			if dataset[entry][3][obj] in class_we_care:   # select only objects we interested in
				anno = dataset[entry][2][obj]   # current object annotation box	
				label_str = dataset[entry][3][obj]
				label = class_we_care.index(label_str)  # use class_we_care as the values of labels
				try:
					cropped_im = crop_image(im, anno)
					# ax.imshow(cropped_im)
				except:
					print('{object} cannot be cropped with annotation: {ann}'.format(
																			object = dataset[entry][3][obj],
																			ann = anno))
					print('image size is {size} \n'.format(size = im.shape))
					print(dataset[entry][0])
				
				# resize cropped image
				im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_CUBIC)   #use 256 as standard preprocessing for ImageNet

				# Create a feature
				feature = {
					'image_raw': _bytes_feature(im.tostring()),
					'label': _int64_feature(label)
				}
				# Create an example protocol buffer
				example = tf.train.Example(features=tf.train.Features(feature=feature))
				
				# Serialize to string and write on the file
				writer.write(example.SerializeToString())

		if entry % 100 == 0:
			print('{}% '.format(int((entry/len(dataset)*100))))

	# make sure to close file writer
	writer.close()
	sys.stdout.flush()

if __name__ == '__main__':
	dataset = load_dataset()
	# Divide the data into train and test
	shuffle(dataset)   # first shuffle dataset
	train = dataset[0:int(0.8*len(dataset))]
	test = dataset[int(0.8*len(dataset)):]

	# create tensorflow records
	create_data_record('train.tfrecords', train)
	create_data_record('test.tfrecords', test)
