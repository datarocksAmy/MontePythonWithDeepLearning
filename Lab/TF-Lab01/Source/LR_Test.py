import tensorflow as tf
import data_helpers
import numpy as np

# Set threshold for what messages will be logged
tf.logging.set_verbosity(tf.logging.DEBUG)

# Start the tensorflow session
sess = tf.Session()

# Import CIFAR-10 data using data_helpers.py to unpack
CIFAR_data = data_helpers.load_data()

# Restore the saved model ( Import the data from the Training session )
new_saver = tf.train.import_meta_graph('tfboard/model/00000001/export.meta')
new_saver.restore(sess, 'tfboard/model/00000001/export')

# Observe the restored variables
for v in tf.get_collection('variables'):
    print(v.name)
print(sess.run(tf.global_variables()))

# Initialize weight and bias
# Obtain saved weights ( using existing weights and bias from the trained model )
W = tf.get_collection('variables')[0]
b = tf.get_collection('variables')[1]

# Create Placeholders for testing images and labels
labels_placeholder = tf.placeholder(tf.int64, shape=[None])
images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])

# Prediction model using softmax
y = tf.nn.softmax(tf.matmul(images_placeholder, W) + b, name='y')

# Compare predicted label and actual label
correct_prediction = tf.equal(tf.argmax(y, 1), labels_placeholder)

# Calculate accuracy and create the accuracy op for tf session
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accu = sess.run(accuracy, feed_dict={images_placeholder: CIFAR_data['images_test'],
                                     labels_placeholder: CIFAR_data['labels_test']})

# Save summaries for visualization ( Histograms + Scalar )
with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels_placeholder, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)


# Merge all summaries into one op
merged = tf.summary.merge_all()

# FileWriter for tensorboard graph log set-up
testwriter = tf.summary.FileWriter('tfboard/model'+'/logs/test', sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

# Evaluate accuracy for the testing model
print("\n ---> Accuracy : " + str(accu))
