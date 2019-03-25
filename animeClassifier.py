
## This is largely a tutorial tuned to fit the problem of classifying images as anime or cartoons

import tensorflow as tf
import sys
import os
from random import shuffle
import time
import dataset

session = tf.Session()

summary_writer = tf.summary.FileWriter('tmp/logs', session.graph)
summary_writer_train = tf.summary.FileWriter('tmp/logs/train', session.graph)
summary_writer_test  = tf.summary.FileWriter('tmp/logs/test', session.graph)

#initial parameters
imgSize = 128
nChannels = 3
batch_size = 32
max_steps = 1000000
classes = ['anime','cartoons']
num_classes = len(classes)
validation_size = 0.2


path = '/Users/mazinkhader/Downloads'
data = dataset.read_train_sets(path, imgSize, classes, validation_size=validation_size)   

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))


# Use truncated normal distributions to initialize.
def createWeights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
 
def createBiases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


# Convolutional layer has 3 steps: convolution -> max-pooling -> activation
def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters,name):  
    
    ## Initial weights
    weights = createWeights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## Initial biases
    biases = createBiases(num_filters)
 
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
 
    layer += biases
 

    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
   
    layer = tf.nn.relu(layer)
    
    filter_summary = []
    with tf.variable_scope(name) as scope_conv:
        nf = 0
        for sw in tf.split(weights,num_filters,3): 
          filter_summary.append(tf.summary.image(name+'_'+str(nf),sw))
          nf+=1
 
    return layer,filter_summary

def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
 
    return layer

def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    

    weights = createWeights(shape=[num_inputs, num_outputs])
    biases = createBiases(num_outputs)
 
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
 
    return layer


filter_size_conv1  = 10
num_filters_conv1  = 10
filter_size_conv2  = 20
num_filters_conv2  = 230
filter_size_conv3  = 3
num_filters_conv3  = 64

fc_layer_size=128

x = tf.placeholder(tf.float32, shape=[None, imgSize,imgSize,nChannels], name='x')

layer_conv1,filter_sum1 = create_convolutional_layer(input=x,
               num_input_channels=nChannels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1,
               name="conv1")
 
layer_conv2,filter_sum2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2,
               name="conv2")
 
layer_conv3,filter_sum3 = create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3,
               name="conv3")
          
layer_flat = create_flatten_layer(layer_conv3)
 
layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)
 
layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=2,
                     use_relu=False)




y_pred = tf.nn.softmax(layer_fc2,name="y_pred")
y_pred_cls = tf.argmax(y_pred, dimension=1)


## labels
y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

session.run(tf.global_variables_initializer()) 

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
tf.add_to_collection('losses', cost)
loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

acc_ten  = tf.summary.scalar('training accuracy', accuracy)
vacc_ten = tf.summary.scalar('test accuracy', accuracy)

merged = tf.summary.merge_all()


session.run(tf.global_variables_initializer()) 


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)

    acc_ten  = tf.summary.scalar('training accuracy', acc)
    vacc_ten = tf.summary.scalar('test accuracy', val_acc)

    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))
    summary_writer_train.add_summary(session.run(acc_ten), epoch+1)
    summary_writer_test.add_summary(session.run(vacc_ten), epoch+1)

total_iterations = 0

saver = tf.train.Saver()


def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):
        #lb = tf.reshape(label_batch, [batch_size])
        #x_batch, y_true_batch = session.run([image_batch,label_batch])
        #lbv = tf.reshape(label_batch_v, [batch_size])
        #x_valid_batch, y_valid_batch = session.run([image_batch_v,label_batch_v])

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
        
        #print x_batch
        #print y_true_batch
        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        

    
        if i % int(data.train.num_examples/batch_size) == 0: 
            sumLoss,vals = session.run([merged,cost], feed_dict=feed_dict_tr)
            sumVLoss,val_loss = session.run([merged,cost], feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            sumAcc,acc = session.run([merged,accuracy], feed_dict=feed_dict_tr)
            sumVacc,val_acc = session.run([merged,accuracy], feed_dict=feed_dict_val)
            #show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
            print(msg.format(epoch + 1, acc, val_acc, val_loss))
            summary_writer_train.add_summary(sumAcc, epoch+1)
            summary_writer_test.add_summary(sumVacc, epoch+1)
            summary_writer_train.add_summary(sumLoss, epoch+1)
            summary_writer_test.add_summary(sumVLoss, epoch+1)

            for lay in filter_sum1:
                summary_writer.add_summary(session.run(lay), epoch+1)
            for lay in filter_sum2:
                summary_writer.add_summary(session.run(lay), epoch+1)
            for lay in filter_sum3:
                summary_writer.add_summary(session.run(lay), epoch+1)
            #summary_writer.add_summary(session.run(filter_sum2), epoch+1)
            #summary_writer.add_summary(session.run(filter_sum3), epoch+1)
        saver.save(session, './anime-model') 


    total_iterations += num_iteration



train(num_iteration=1700)

