# TensorFlow (CNN)

import tensorflow as tf
import glob
import numpy as np
import datetime
from tensorflow.contrib.session_bundle import exporter

# Print out the timestamp when the CNN starts
startTime = datetime.datetime.now()

# Connection between tensorflow & python
sess = tf.InteractiveSession()
tf.logging.set_verbosity(tf.logging.INFO)

# Create Index for our data
labelGlob = glob.glob("datasets/Train/*")
labelMaster = np.ndarray(shape=(5), dtype=np.dtype('S20'))

loop = 0
for folder in labelGlob:
    labelKey = folder.split("/")[2]
    labelMaster.itemset(loop, labelKey)
    loop = loop + 1

print("Number of Classes: ", len(labelMaster))

# Load in Test and Train datasets
files = glob.glob("datasets/Train/*/*.jpg")
testFiles = glob.glob("datasets/Test/*/*.jpg")

print("Number of Training Images: ", len(files))
print("Number of Test Images:     ", len(testFiles))
print("")

# -------------------------------------------------- Train --------------------------------------------------
# Shape (total of Images = 20, Size of Image = 100*100 pixels)
allImages = np.ndarray(shape=(100, 10000))
allLabels = np.ndarray(shape=(100, 5))  # 5 Classes

testImages = np.ndarray(shape=(50, 10000))
testLabels = np.ndarray(shape=(50, 5))

# For all images, read them into the allImages tensor, then construct a one-hot
# Matrix for the label (with a 1 only in the element matching the label)
# Because our images are natively 400 x 400, compress to 200 x 200.
for loop in range(0, len(files)):
    curFile = files[loop]
    curLabel = files[loop].split("/")[2]

    # Build our label matrix
    indexOf = np.where(labelMaster == str.encode(curLabel))
    print("class: ", curLabel, "  index: ", indexOf[0], "  fileName: ", curFile)

    curLabels = np.zeros(shape=(5))
    curLabels[indexOf[0]] = 1
    allLabels[loop] = curLabels

    # Build the pixel map
    file_contents = tf.read_file(files[loop])
    readImage = tf.image.decode_jpeg(file_contents, channels=1)
    image = tf.image.resize_images(readImage, [100, 100])

    with sess.as_default():
        flatImage = image.eval().ravel()
    flatImage = np.multiply(flatImage, 1.0 / 255.0)
    allImages[loop] = flatImage

print ("Length of allImages: ", len(allImages))
print ("Length of allLabels: ", len(allLabels))

# -------------------------------------------------- TEST --------------------------------------------------
for loop in range(0, len(testFiles)):
    curFile = testFiles[loop]
    curLabel = testFiles[loop].split("/")[2]

    #Build our label matrix
    indexOf = np.where(labelMaster == str.encode(curLabel))
    print("class: ", curLabel, "  index: ", indexOf[0], "  fileName: ", curFile)

    curLabels = np.zeros(shape=(5))
    curLabels[indexOf[0]] = 1
    testLabels[loop] = curLabels

    #Build the pixel map
    file_contents = tf.read_file(testFiles[loop])
    readImage = tf.image.decode_jpeg(file_contents, channels=1)
    image = tf.image.resize_images(readImage, [100, 100])

    with sess.as_default():
        flatImage = image.eval().ravel()
    flatImage = np.multiply(flatImage, 1.0 / 255.0)
    testImages[loop] = flatImage

# -------------------------------------------------- Split pixels --------------------------------------------------
x = tf.placeholder(tf.float32, [None, 10000],name='x')  # pixel * pixel we set
y_ = tf.placeholder(tf.float32, [None, 5],name='y_')  # Labels for number of classes

init = tf.global_variables_initializer()
sess.run(init)

# Log Session data for Tensorboard
def variable_summaries(var, var2):
  # Attach summaries for TensorBoard visualization
  with tf.name_scope(var2):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# CNN Layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 100, 100, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# CNN Layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully Connected Layer 1 ( Weights + Bias )
W_fc1 = weight_variable([25 * 25 * 64, 1024])
b_fc1 = bias_variable([1024])

# Flatten Pool 2 (Resize image) + Set up ReLU
h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 25 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully Connected Layer 2 ( Weights + Bias )
W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Cross Entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# Optimization + Prediction + Calculate Accuracy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TensorBoard graph summaries collection
variable_summaries(W_conv1, 'Convolution_Weights_1')
variable_summaries(h_conv1, 'Convolution_1')
variable_summaries(W_conv2, 'onvolution_Weights_2')
variable_summaries(h_conv2, 'Convolution_2')
variable_summaries(keep_prob, 'Keep_Probabilities')
variable_summaries(h_fc1_drop, 'Drop_ReLU')
variable_summaries(cross_entropy, 'Cross_Entropy')
variable_summaries(accuracy, 'Accuracy')

# Merge all summaries into one op
merged = tf.summary.merge_all()
trainwriter = tf.summary.FileWriter('data/logs', sess.graph) # FileWriter for tensorboard graph log set-up
sess.run(tf.global_variables_initializer())

# Train 500 Steps
for i in range(500):
    print("", datetime.datetime.now(), "Running Iteration ", i)
    # Create samples for data ( 10% of all data  every time )
    mask = np.random.choice([True, False], len(allImages), p=[0.10, 0.90])
    trainImages = allImages[mask]
    trainLabels = allLabels[mask]
    # print(datetime.datetime.now(), "  Generated data mask.  About to run the training step...")
    summary, _ = sess.run([merged, train_step], feed_dict={x: trainImages, y_: trainLabels, keep_prob: 0.5})
    # print(datetime.datetime.now(), "  Training Step complete.  Log summary...")
    trainwriter.add_summary(summary, i)
    if i % 25 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: trainImages, y_: trainLabels, keep_prob: 1.0})
        print("Evalutating iteration %d, training accuracy %g" % (i, train_accuracy))

print("Final Test Accuracy: %g" % accuracy.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0}))


endModelTime = datetime.datetime.now()

# model export path
export_path = 'data/model'
print('Exporting trained model to', export_path)

saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
model_exporter.init(
    sess.graph.as_graph_def(),
    named_graph_signatures={
        'inputs': exporter.generic_signature({'images': x}),
        'outputs': exporter.generic_signature({'scores': y_})})

model_exporter.export(export_path, tf.constant(1), sess)

print("Program Complete!")
print("CNN Start Time:     ", startTime)
print("Model Built: ", endModelTime)
print("Finish Time:       ", datetime.datetime.now())