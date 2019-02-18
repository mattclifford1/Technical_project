import pickle
import numpy as np
from random import shuffle

import utils

'''
get dataset object to use in batch training
usage: 
    import get_data
    data = get_data.SUNRGBD(batch_size, data_type)    # change for number of batch size
    (rgb, depth, points, bb_2D, bb_3D, labels) = data.train_batch()
'''


def one_hot_vector(class_id, num_classes):   # turn class labels into one-hot-vector format
    hot = np.zeros(num_classes, dtype=np.int)
    hot[class_id] = 1
    return hot


def crop_image_2D(im, anno):
    im = 2*(im/255.0) -1   # normalise
    # re = re.reshape(224,224,3)
    return im[int(anno[1]): int(anno[1]+anno[3]), int(anno[0]): int(anno[0]+anno[2])]


class SUNRGBD:

    def __init__(self, batch_size):   #TODO work data_type
        self.batch_size = batch_size
        self.train_index = 0
        self.test_index = 0
        self.num_of_test = 0
        self.train_epoch = 0
        self.train_size = 1000#7750   # 75%
        self.test_size = 20#2605    # 25%
        self.label_white_list = ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']
        self.num_classes = len(self.label_white_list)

        self.load_dataset()


    # define methods
    def load_dataset(self):
        # open the dataset file
        try:
            meta_data = utils.load_SUNRGBD_meta()
        except IOError:
            print("Need to have 'SUN-RGBD_convert_matlab.pickle' file in current directory")

        ## set up TRAINING data properties -------------------------------------------------
        # we don't know the size of these:
        self.train_2D_im = []     
        self.train_labels = []
        self.train_labels_image = []
        for entry in range(self.train_size):
            current_rgb = utils.open_rgb(meta_data, entry)
            bbs_2D = utils.get_2D_bb(meta_data, entry)
            labels = utils.get_label(meta_data, entry)
            for objekt in range(len(bbs_2D)):
                if not labels[objekt] in self.label_white_list:  # only take the objects we care about
                    continue
                bb_2D = bbs_2D[objekt]
                cropped_im = crop_image_2D(current_rgb, bb_2D)
                if cropped_im.shape[0] == 0 or cropped_im.shape[1] == 0:
                    continue     # cropping didn't work 
                self.train_2D_im.append(utils.cv2.resize(cropped_im, (224,224)))
                self.train_labels_image.append(entry)   # need to keep track of what labels matches each image
                self.train_labels.append(one_hot_vector(self.label_white_list.index(labels[objekt]), self.num_classes))


        ## set up TESTING data properties --------------------------------------------------
        # we don't know the size of these:
        self.test_2D_im = []
        self.test_labels = []
        self.test_labels_image = []
        test_counter = 0
        for entry in range(self.train_size, self.train_size+self.test_size):
            current_rgb = utils.open_rgb(meta_data, entry)
            bbs_2D = utils.get_2D_bb(meta_data, entry)
            labels = utils.get_label(meta_data, entry)
            for objekt in range(len(bbs_2D)):
                if not labels[objekt] in self.label_white_list:  # only take the objects we care about
                    continue
                bb_2D = bbs_2D[objekt]
                self.test_2D_im.append(crop_image_2D(current_rgb, bb_2D))
                self.test_labels_image.append(entry)   # need to keep track of what labels matches each image
                self.test_labels.append(one_hot_vector(self.label_white_list.index(labels[objekt]), self.num_classes))
            test_counter += 1
        # unsure if i will need this??
        # self.train_labels = tuple(self.train_labels)


        # shuffle datasets 
        self.shuffle_train()
        # self.shuffle_test()

        # assign validation set as we want to validate on bigger that batch size
        # val_size = self.batch_size * 4
        # self.val_id, self.test_id = self.test_id[:val_size], self.test_id[val_size:]
        # self.val_labels, self.test_labels = self.test_labels[:val_size], self.test_labels[val_size:]
        # self.val_data, self.test_data = self.test_data[:val_size], self.test_data[val_size:]


        self.num_train = len(self.train_labels)
        self.num_test = len(self.test_labels)




###### TODO: change these function to support new data properties
    def train_batch(self):      # use to get a training batch
        rgb = [0]*self.batch_size
        labels = [0]*self.batch_size

        # see if we are at the last batch (could allow for end batch smaller?)      
        if self.train_index + self.batch_size > self.num_train:
            self.train_index = 0      # reset the index
            self.shuffle_train()      # shuffle so we dont get the same order
            self.train_epoch += 1     # we have seen dataset once
        
        for i in range(self.batch_size):
            current_data = i + self.train_index
            rgb[i] = self.train_2D_im[current_data]
            labels[i] = self.train_labels[current_data]

        self.train_index += self.batch_size
        return (rgb, labels)
    
    def val_set(self):
        return (self.val_data, self.val_labels)

    def test_batch(self):      # use to get a test batch
        rgb = [0]*self.batch_size
        depth = [0]*self.batch_size
        points = [0]*self.batch_size
        bb_2D = [0]*self.batch_size
        bb_3D = [0]*self.batch_size
        labels = [0]*self.batch_size

        # see if we are at the last batch (could allow for end batch smaller?)      
        if self.test_index + self.batch_size > self.num_test:
            self.test_index = 0      # reset the index
            self.shuffle_train()      # shuffle so we dont get the same order
        
        for i in range(self.batch_size):
            current_data = i + self.train_index
            bb_2D[i] = self.test_2D_bb[current_data]
            bb_3D[i] = self.test_3D_bb[current_data]
            labels[i] = self.test_labels[current_data]
            # images have different indexing as multiple labels map to one image
            current_image = self.test_labels_image[current_data]
            rgb[i] = self.test_rgb[current_image]
            depth[i] = self.test_depth[current_image]
            points[i] = self.test_points[current_image]

        self.test_index += self.batch_size
        return (rgb, depth, points, bb_2D, bb_3D, labels)


    def shuffle_train(self):
        # shuffle train sets as segments of songs are in order together
        # zip together to preserve entries
        zipped_train = list(zip(self.train_2D_im, self.train_labels, self.train_labels_image))
        shuffle(zipped_train)
        # unzip shuffled sets
        self.train_2D_im, self.train_labels, self.train_labels_image = zip(*zipped_train)
        

    def shuffle_test(self):
        # shuffle test sets as segments of songs are in order together
        # zip together to preserve entries
        zipped_test = list(zip(self.test_rgb, self.test_depth, self.test_points, self.test_2D_bb, self.test_3D_bb, self.test_labels, self.test_labels_image))
        shuffle(zipped_test)
        # unzip shuffled sets
        self.test_rgb, self.test_depth, self.test_points, self.test_2D_bb, self.test_3D_bb, self.test_labels, self.test_labels_image = zip(*zipped_test)


    def reset(self):      # reset the indexing of the batches
        self.test_index = 0
        self.train_index = 0


    def get_one_class(self, label):   # return a single class for test evalution
        int_label = self.label_white_list.index(label)
        indicies = np.where(np.array(self.test_labels)[:, int_label] == 1)[0]
        class_size = len(indicies)
        rgb = [0]*class_size
        depth = [0]*class_size
        points = [0]*class_size
        bb_2D = [0]*class_size
        bb_3D = [0]*class_size
        labels = [0]*class_size
        for i in range(class_size):
            bb_2D[i] = self.test_2D_bb[indicies[i]]
            bb_3D[i] = self.test_3D_bb[indicies[i]]
            labels[i] = self.test_labels[indicies[i]]
            # images have different indexing as multiple labels map to one image
            current_image = self.test_labels_image[indicies[i]]
            rgb[i] = self.test_rgb[current_image]
            depth[i] = self.test_depth[current_image]
            points[i] = self.test_points[current_image]

        return (rgb, depth, points, bb_2D, bb_3D, labels)



