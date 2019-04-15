from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf

from utils import *
from metrics import *
from models import GCN_Align

# Set random seed
seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 2, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gamma', 1.0, 'Hyper-parameter for margin based loss.')
flags.DEFINE_integer('k', 5, 'Number of negative samples for each positive seed.')

flags.DEFINE_integer('se_dim', 100, 'Dimension for SE.')
flags.DEFINE_integer('seed', 5, 'Proportion of seeds, 3 means 30%')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight for L2 loss on embedding matrix.')

# Load data
adj, train, test, KG, e = load_sn_data()

# Some preprocessing
support = [preprocess_adj(adj)]
num_supports = 1
model_func = GCN_Align
k = FLAGS.k

ph_se = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder_with_default(0, shape=())
}

# Create model
model_se = model_func(ph_se, input_dim=e, output_dim=FLAGS.se_dim, ILL=train, sparse_inputs=False, featureless=True, decay=False, logging=True)
# Initialize session
sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

t = len(train)
L = np.ones((t, k)) * (train[:, 0].reshape((t, 1)))
neg_left = L.reshape((t * k,))
L = np.ones((t, k)) * (train[:, 1].reshape((t, 1)))
neg2_right = L.reshape((t * k,))

# Train model
for epoch in range(FLAGS.epochs):
    if epoch % 10 == 0:
        neg2_left = np.random.choice(e, t * k)
        neg_right = np.random.choice(e, t * k)
    # Construct feed dictionary
    feed_dict_se = construct_feed_dict(1.0, support, ph_se)
    feed_dict_se.update({ph_se['dropout']: FLAGS.dropout})
    feed_dict_se.update({'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left, 'neg2_right:0': neg2_right})
    # Training step
    outs_se = sess.run([model_se.opt_op, model_se.loss], feed_dict=feed_dict_se)
    cost_val.append(outs_se[1])

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "SE_train_loss=", "{:.5f}".format(outs_se[1]))

print("Optimization Finished!")

# Testing
print('testing...', len(train), len(test))
feed_dict_se = construct_feed_dict(1.0, support, ph_se)
vec_se = sess.run(model_se.outputs, feed_dict=feed_dict_se)
print("SE")
get_hits(vec_se, test, top_k=[30])
