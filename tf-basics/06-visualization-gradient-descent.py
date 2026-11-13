import os
import time
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

tf.compat.v1.disable_eager_execution()

now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
logdir = f"tf_logs/run-{now}/"
os.makedirs("data", exist_ok=True)          # fix: folder for checkpoints

# Data
X_synth, y_synth = make_regression(n_samples=20640, n_features=8, noise=0.1, random_state=42)
m, n = X_synth.shape
scaled = StandardScaler().fit_transform(X_synth)
X_data = np.c_[np.ones((m, 1)), scaled].astype(np.float32)
y_data = y_synth.reshape(-1, 1).astype(np.float32)

n_epochs = 100                               # reduced for speed
learning_rate = 0.01
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

# Graph
X = tf.compat.v1.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.compat.v1.Variable(tf.compat.v1.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
mse = tf.reduce_mean(tf.square(y_pred - y), name="mse")
training_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(mse)

mse_summary = tf.compat.v1.summary.scalar("MSE", mse)   # fix: v1 API
init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()

start = time.perf_counter()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    file_writer = tf.compat.v1.summary.FileWriter(logdir, sess.graph)

    for epoch in range(n_epochs):
        perm = np.random.permutation(m)
        Xs, ys = X_data[perm], y_data[perm]

        for b in range(n_batches):
            s = b * batch_size
            feed = {X: Xs[s:s + batch_size], y: ys[s:s + batch_size]}

            if b % 10 == 0:
                summary_str = sess.run(mse_summary, feed_dict=feed)
                file_writer.add_summary(summary_str, epoch * n_batches + b)

            sess.run(training_op, feed_dict=feed)

        if epoch % 10 == 0:
            saver.save(sess, "./data/05-grad-des-model.ckpt")
            cur = sess.run(mse, feed_dict={X: X_data, y: y_data})
            print(f"Epoch {epoch} MSE = {cur}")

    best_theta = sess.run(theta)
    saver.save(sess, "./data/05-grad-des-model-final.ckpt")
    file_writer.close()

print(f"\nTraining completed in {time.perf_counter() - start:.2f} s")
print("Best Theta:", best_theta.ravel())