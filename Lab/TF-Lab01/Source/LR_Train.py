import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
import data_helpers
import numpy as np

# Set up batch size
batch_size = 100

# Import the CIFAR-10 data using the data_helpers.py to unpack
CIFAR_data = data_helpers.load_data()

# Initialize a tensorflow session
sess = tf.Session()

# Input Placeholders : Labels + Images
labels_placeholder = tf.placeholder(tf.int64, shape=[None])
images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])

# Initialize Weights + Bias as 0
W = tf.Variable(tf.zeros([3072, 10]), name="Weight")
b = tf.Variable(tf.zeros([10]), name="Bias")

# Prediction model using softmax
y = tf.nn.softmax(tf.matmul(images_placeholder, W) + b, name="predictY")

# Create different collections ( layers ) and add them together
tf.add_to_collection('variables', W)
tf.add_to_collection('variables', b)

# Use Cross Entropy to calculate the cost
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=labels_placeholder))

# Training using Gradient Descent Algorithm
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# Save summaries for visualization ( Histograms + Scalar )
tf.summary.histogram('weights', W)
tf.summary.histogram('max_weight', tf.reduce_max(W))
tf.summary.histogram('bias', b)
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.histogram('cross_hist', cross_entropy)

# Merge all summaries into one op
merged = tf.summary.merge_all()

# FileWriter to set up for tensorboard
trainwriter = tf.summary.FileWriter('tfboard/model'+'/logs/train', sess.graph)

# Initialize global variables
init = tf.global_variables_initializer()

# Start the session
sess.run(init)

# Train the model for 10000 epochs
for i in range(10000):
    indices = np.random.choice(CIFAR_data['images_train'].shape[0], batch_size)
    images_batch = CIFAR_data['images_train'][indices]
    labels_batch = CIFAR_data['labels_train'][indices]
    summary, _ = sess.run([merged, train_step], feed_dict={images_placeholder: images_batch, labels_placeholder: labels_batch})
    trainwriter.add_summary(summary, i)

# Export the model to local path
export_path = 'tfboard/model'
print('Exporting Trained Model ---> ', export_path)

# Save and export all the information
saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
model_exporter.init(
    sess.graph.as_graph_def(),
    named_graph_signatures={
        'inputs': exporter.generic_signature({'images': images_placeholder}),
        'outputs': exporter.generic_signature({'scores': y})})

# Save the snapshot of the trained model -- > Load for testing model
model_exporter.export(export_path, tf.constant(1), sess)

