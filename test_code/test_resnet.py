'''
scipt to re train resnet100v2 pre trained on imagenet
arguments:
    train_type: 'all' or 'last'
    max_steps: number of steps (int)
    train_per_class: number of examples of each training class (int)
EXAMPLE USAGE:
$ python test_resnet.py last 100 200
'''

import tensorflow as tf
import os
import sys

from resnet_v2 import resnet_v2_101, resnet_arg_scope
import utils
import get_data_2D_objects as gd

# set up model variables -- from command line input
try:
    train_type = sys.argv[1]
    max_steps = int(sys.argv[2])
    train_per_class = int(sys.argv[3])
    test_per_class = int(train_per_class*0.25)
except:
    raise Exception('Need to enter max_steps to train and number of examples of each class to train with at command line')

print('loading data')
# train on new data ---------------------------------------------------
data = gd.SUNRGBD(16, train_per_class, test_per_class)
print('data loaded')


print('Loading resnet')
slim = tf.contrib.slim

# define input to network (takes 244 square rgb batches)
X = tf.placeholder(tf.float32, shape=([None,224,224,3]))

# load network
with slim.arg_scope(resnet_arg_scope()):
    net, end_points = resnet_v2_101(X, 1001, is_training=False)

# load pretrained weights onto the network
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'resnet_v2_101.ckpt')

# truncate final classifying layer
reduced_mean = tf.get_default_graph().get_operation_by_name('resnet_v2_101/pool5')
# postnorm = tf.get_default_graph().get_operation_by_name('resnet_v2_101/postnorm/Relu')

# freeze the lower layers of the network that have been pre trained
reduced_mean_stop = tf.stop_gradient(reduced_mean.outputs) # reduced_mean.outputs [0]

# get current set of variable (need to know what new variables are added)
old_vars = set(tf.global_variables())

# add a new classifying layer
if train_type == 'last':
    net = slim.conv2d(reduced_mean_stop, data.num_classes, (1,1,1), activation_fn=None,normalizer_fn=None, scope='logits_new')
else:
    net = slim.conv2d(reduced_mean, data.num_classes, (1,1,1), activation_fn=None,normalizer_fn=None, scope='logits_new')
net = tf.squeeze(net, [0, 2, 3], name='SpatialSqueeze_new')    # get rid of reduntant dimentions
net = tf.nn.softmax(net)

# define labels
y_ = tf.placeholder(tf.float32, [None, data.num_classes]) # in logit form

# error function for backprop to optimise against
cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=net))

# calculate the prediction and the accuracy
correct_pred = tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(y_, 1)), 'float32')
accuracy = tf.reduce_mean(correct_pred)

# find new variables added to the graph
with_last_layer_vars = set(tf.global_variables())
last_layer_vars = with_last_layer_vars - old_vars

# initalise the new wieghts from new layer only (we dont want to re initalise pre trained)
init_last_layer_op = tf.initialize_variables(last_layer_vars)

# define optimiser
global_step = tf.Variable(0, trainable=False)
if train_type == 'last':
    optim_op = tf.train.AdamOptimizer(learning_rate=0.001, name='optimiser').minimize(cross_entropy, var_list=list(last_layer_vars))
    # optim_op = tf.train.AdamOptimizer(learning_rate=0.001, name='optimiser').minimize(tf.losses.get_total_loss(), var_list=list(last_layer_vars))
else:
    optim_op = tf.train.AdamOptimizer(learning_rate=0.001, name='optimiser').minimize(cross_entropy, global_step=global_step)
# get adam vars and initiliastion
with_optim_vars = set(tf.global_variables())
optim_vars = with_optim_vars - with_last_layer_vars
init_optim_op = tf.initialize_variables(optim_vars)

# initialise last layer and optimiser
sess.run(init_last_layer_op)
sess.run(init_optim_op)


# define and set up summaries for tensorboard ----------------------------------
validation_summary = tf.summary.merge([tf.summary.scalar('Accuracy', accuracy)])
training_summary = tf.summary.merge([tf.summary.scalar('Loss', cross_entropy)])
# set directory to save logs for tensorboard
run_name = train_type+'_BS'+str(data.batch_size)+'_MS'+str(max_steps)+'_E'+str(train_per_class)
run_dir = os.path.join('{cwd}/logs/'.format(cwd=os.getcwd()), run_name)
# set up summary writers
train_writer = tf.summary.FileWriter(run_dir + "_train", sess.graph)
val_writer = tf.summary.FileWriter(run_dir + "_validate", sess.graph)

print('training')
epoch = 0
for i in range(max_steps):
    # keep track of when we reach a new epoch
    if data.train_epoch > epoch:
        epoch = data.train_epoch

    # train a batch 
    (train_data, train_labels) = data.train_batch()
    [_, train_summary_str] = sess.run([optim_op, training_summary], feed_dict = {X:train_data, y_:train_labels})

    if i%2 == 0: # check against training data (should set some data asside for validation really?)
        (test_data, test_labels) = data.test_batch(64)
        [acc, val_summary_str] = sess.run([accuracy, validation_summary], feed_dict = {X:test_data, y_:test_labels})
        print(acc)
        train_writer.add_summary(train_summary_str, i)
        val_writer.add_summary(val_summary_str, i)

print('testing')
data.reset()
data.test_epoch = 0
it_counter = 0
test_accuracy_sum = 0
while data.test_epoch < 1:
    (test_data, test_labels) = data.test_batch()
    test_accuracy_sum += sess.run([accuracy], feed_dict = {X:test_data, y_:test_labels})[0]   
    it_counter += 1

test_accuracy_mean = test_accuracy_sum / it_counter
print('Mean accuracy of test set is: '+str(test_accuracy_mean))


# save the model 
saver = tf.train.Saver()
saver.save(sess, 'res_net_101_v2_transfered')


