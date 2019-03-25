import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse


# First, pass the path of the image
#dir_path = os.path.dirname(os.path.realpath(__file__))
#image_path=sys.argv[1] 
filename = sys.argv[1]#dir_path +'/' +image_path
image_size=128
num_channels=3
images = []
# Reading the image using OpenCV
image = cv2.imread(filename)
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('anime-model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

#Mazin - adding filter visualization?

with tf.variable_scope('conv1') as scope:
  for V in tf.global_variables():
    #print var.name, 
    sess.run(V)
    print V.get_shape()
    V  = tf.slice(V,(0,0,0,0),(1,-1,-1,-1)) #V[0,...]
    iy = V.get_shape().as_list()[1]
    ix = V.get_shape().as_list()[2]
    nFilters = V.get_shape().as_list()[3]
    
    V = tf.reshape(V,(ix,iy,nFilters))
  
    ix += 4
    iy += 4
    V = tf.image.resize_image_with_crop_or_pad(V, iy, ix)
    cy = 4
    cx = 8
    
    if(nFilters > 32):
    	cy = 8
    	cx = 16

    V = tf.reshape(V,(iy,ix,cy,cx))
    V = tf.transpose(V,(2,0,3,1)) #cy,iy,cx,ix
    # image_summary needs 4d input
    V = tf.reshape(V,(1,cy*iy,cx*ix,1))
    filter_summary = tf.summary.image('conv1/filters', V)#max_images=3)
    summary_writer = tf.summary.FileWriter('tmp/logs', graph)
    summary_writer.add_summary(sess.run(filter_summary))
    break

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 2)) 


### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probabiliy_of_rose probability_of_sunflower]
print(result)
