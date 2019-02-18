import tensorflow as tf

from resnet_v2 import resnet_v2_101, resnet_arg_scope
import utils
import get_data_2D_objects as gd


print('loading data')
# train on new data ---------------------------------------------------
data = gd.SUNRGBD(16)
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
net = slim.conv2d(reduced_mean_stop, data.num_classes, (1,1,1), activation_fn=None,normalizer_fn=None, scope='logits_new')
net = tf.squeeze(net, [0, 2, 3], name='SpatialSqueeze_new')    # get rid of reduntant dimentions

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
# optim_last_layer = tf.train.AdamOptimizer(learning_rate=0.001, name='optimiser').minimize(cross_entropy, global_step=global_step)
optim_last_layer = tf.train.AdamOptimizer(learning_rate=0.001, name='optimiser').minimize(tf.losses.get_total_loss(), var_list=list(last_layer_vars))
# get adam vars and initiliastion
with_optim_vars = set(tf.global_variables())
optim_vars = with_optim_vars - with_last_layer_vars
init_optim_op = tf.initialize_variables(optim_vars)

# initialise last layer and optimiser
sess.run(init_last_layer_op)
sess.run(init_optim_op)


print('training')
for i in range(100):
    (train_data, train_labels) = data.train_batch()
    sess.run([optim_last_layer], feed_dict = {X:train_data, y_:train_labels})
    if i%2 == 0:
        acc = sess.run([accuracy], feed_dict = {X:train_data, y_:train_labels})
        print(acc)










