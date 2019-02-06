import pickle
import numpy as np
from random import shuffle

'''
get dataset object to use in batch training
usage: 
    import get_data
    data = get_data.GTZAN(batch_size, data_type)    # change for number of batch size
    (train_data, train_labels) = data.train_batch()
'''

def one_hot_vector(class_id, num_classes):   # turn class labels into one-hot-vector format
    hot = np.zeros(num_classes, dtype=np.int)
    hot[class_id] = 1
    return hot


class SUNRGBD:
    train_data = np.array([])
    train_labels = np.array([])
    train_id = np.array([])
    test_data = np.array([])
    test_labels = np.array([])
    test_id = np.array([])

    num_train = 0
    num_test = 0

    def __init__(self, batch_size, data_type):
        self.batch_size = batch_size
        self.num_classes = 10
        self.train_index = 0
        self.test_index = 0
        self.num_of_test_songs = 0
        self.train_epoch = 0
        self.load_dataset(data_type)


    # define methods
    def load_dataset(self, data_type):
        # open the dataset file
        try:
            with open('music_genres_dataset.pkl', 'rb') as f:
                train_set = pickle.load(f)     # contains 11250 entries 75%
                test_set = pickle.load(f)    # contains  3750 entries   25%
        except IOError:
            print("Need to have 'music_genres_dataset.pkl' file in current directory")

        # see if we want to augment data
        if data_type == 'aug':
            # this will take about 40 mins
            train_set = augment_data(train_set)

        # set up training data properties
        self.train_id = train_set['track_id']
        self.train_labels = [0]*len(self.train_id)
        self.train_data = [0]*len(self.train_id)
        for entry in range(len(self.train_id)):  # need to convert raw data 
            self.train_labels[entry] = one_hot_vector(train_set['labels'][entry], self.num_classes)
            self.train_data[entry] = melspectrogram(train_set['data'][entry])
        self.train_labels = tuple(self.train_labels)

        # set up testing data properties
        self.test_id = test_set['track_id']
        self.test_labels = [0]*len(self.test_id)
        self.test_data = [0]*len(self.test_id)
        for entry in range(len(self.test_id)):   # need to convert raw data 
            self.test_labels[entry] = one_hot_vector(test_set['labels'][entry], self.num_classes)
            self.test_data[entry] = melspectrogram(test_set['data'][entry])
        
        self.train_labels = tuple(self.train_labels)
        self.num_of_test_songs = max(self.test_id) + 1

        # shuffle datasets
        self.shuffle_train()
        self.shuffle_test()

        #assign validation set as we want to validate on bigger that batch size
        val_size = self.batch_size * 4
        self.val_id, self.test_id = self.test_id[:val_size], self.test_id[val_size:]
        self.val_labels, self.test_labels = self.test_labels[:val_size], self.test_labels[val_size:]
        self.val_data, self.test_data = self.test_data[:val_size], self.test_data[val_size:]


        self.num_train = len(self.train_labels)
        self.num_test = len(self.test_labels)

    def train_batch(self):      # use to get a training batch
        data = [0]*self.batch_size
        labels = [0]*self.batch_size
        # see if we are at the last batch (could allow for end batch smaller?)
        
        if self.train_index + self.batch_size > self.num_train:
            self.train_index = 0      # reset the index
            self.shuffle_train()      # shuffle so we dont get the same order
            self.train_epoch += 1     # we have seen dataset once
        
        for i in range(self.batch_size):
            data[i] = self.train_data[i + self.train_index]
            labels[i] = self.train_labels[i + self.train_index]
        self.train_index += self.batch_size
        return (data, labels)
    
    def val_set(self):
        return (self.val_data, self.val_labels)

    def test_batch(self):      # use to get a test batch
        data = [0]*self.batch_size
        labels = [0]*self.batch_size
        # see if we are at the last batch (could allow for end batch smaller?)
        if self.test_index + self.batch_size > self.num_test:
            self.test_index = 0      # reset the index
            self.shuffle_test()      # shuffle so we dont get the same order
        for i in range(self.batch_size):
            data[i] = self.test_data[i + self.test_index]
            labels[i] = self.test_labels[i + self.test_index]
        self.test_index += self.batch_size
        return (data, labels)


    def shuffle_train(self):
        # shuffle train sets as segments of songs are in order together
        # zip together to preserve entries
        zipped_train = list(zip(self.train_data, self.train_labels, self.train_id))
        shuffle(zipped_train)
        # unzip shuffled sets
        self.train_data, self.train_labels, self.train_id = zip(*zipped_train)
        

    def shuffle_test(self):
        # shuffle test sets as segments of songs are in order together
        # zip together to preserve entries
        zipped_test = list(zip(self.test_data, self.test_labels, self.test_id))
        shuffle(zipped_test)
        # unzip shuffled sets
        self.test_data, self.test_labels, self.test_id = zip(*zipped_test)


    def reset(self):      # reset the indexing of the batches
        self.test_index = 0
        self.train_index = 0


    def get_one_test_track(self, song_num):   # return a single track from track id's for test evalution
        indicies = np.where(np.array(self.test_id) == song_num)[0]
        data = [0]*len(indicies)
        labels = [0]*len(indicies)
        for i in range(len(indicies)):
            data[i] = self.test_data[indicies[i]]
            labels[i] =self.test_labels[indicies[i]]
        return (data, labels)


