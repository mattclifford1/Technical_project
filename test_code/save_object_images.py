## read dataset from meta data - pre process into images of objects

import os
import pickle
from matplotlib.image import imread
import matplotlib.pyplot as plt


def crop_image(im, anno):
    return im[int(anno[1]): int(anno[1]+anno[3]), int(anno[0]): int(anno[0]+anno[2])]


def load_dataset():
    # load dataset's metadata
    try:
        with open('SUN-RGBD_convert_matlab.pickle','rb') as pickle_in:
            return pickle.load(pickle_in)
            # dataset is in format:
            # rows: each entry
            # columns:  0: filepath to rgb image .jpg
            #           1: filepath to depth image .png
            #           2: object bounding boxes 
            #           3: object labels
    except:
        print('Need to make the python data list from \'get_data_from_matlab.py\'')


def create_data_record(dataset):
    class_we_care = ['chair', 'door', 'table']    # taking subset of objects
    name_counter = 0
    chair_counter = 0
    door_counter = 0
    table_counter = 0
    failed = 0
    # create image directories
    for dir_name in class_we_care:
        dir_path = 'SUNRGBD_objects/' + dir_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print("Directory " , dir_name ,  " Created ")
        else:    
            print("Directory " , dir_name ,  " already exists") 
    #save objects as images       
    for entry in range(len(dataset)):    # iterate over all images
        im = imread(dataset[entry][0])   # evalute now to save re compution 
        for obj in range(len(dataset[entry][3])):    # over all objects in an image
            if dataset[entry][3][obj] in class_we_care:   # select only objects we interested in
                anno = dataset[entry][2][obj]   # current object annotation box 
                label_str = dataset[entry][3][obj]
                label = class_we_care.index(label_str)  # use class_we_care as the values of labels
                try:
                    cropped_im = crop_image(im, anno)
                    if cropped_im.shape[0] > 0 and cropped_im.shape[1] > 0:  # need to find out why this is happening
                        file_name = 'SUNRGBD_objects/' + label_str + '/' + str(name_counter) +'.jpg'
                        plt.imsave(file_name, cropped_im)
                except:
                    failed += 1
                if label_str == 'chair':
                    chair_counter += 1
                if label_str == 'door':
                    door_counter += 1
                if label_str == 'table':
                    table_counter += 1
                name_counter += 1
                break   # take new scece so objects not overly similar

            if chair_counter >499 or door_counter>499 or table_counter>499:
                if chair_counter >199:
                    chair_counter = 0
                if door_counter >199:
                    door_counter = 0 
                if table_counter >199:
                    table_counter = 0
                class_we_care.pop(class_we_care.index(label_str))  # can do this as should have gone over threshold this it
        if entry % 100 == 0:
            print('{}% '.format(int((entry/len(dataset)*100))))

    print('{}% of objects couldn\'t be cropped'.format(int((failed/name_counter*100)))) 
    print('{} number of objects found'.format(int(name_counter))) 

if __name__ == '__main__':
    dataset = load_dataset()
    # create image records
    create_data_record(dataset)

