import numpy as np
import tensorflow as tf
import time

# Disable eager execution for TF 1.x graph style
tf.compat.v1.disable_eager_execution()

from sklearn.datasets import make_regression
X_synth, y_synth = make_regression(n_samples=20640, n_features=8, noise=0.1, random_state=42)
class HousingMock: pass
housing = HousingMock()
housing.data = X_synth
housing.target = y_synth

m, n = housing.data.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 1000
learning_rate = 0.01

X = tf.compat.v1.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="y")

batch_size = 100
n_batches = int(np.ceil(m / batch_size))

theta = tf.Variable(tf.compat.v1.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.compat.v1.global_variables_initializer()
target_reshaped = housing.target.reshape(-1, 1)

# model saver
saver = tf.compat.v1.train.Saver()

start_time = time.perf_counter()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        permutation = np.random.permutation(m)
        shuffled_X = scaled_housing_data_plus_bias[permutation]
        shuffled_y = target_reshaped[permutation]
        
        for batch_index in range(n_batches):
            start_idx = batch_index * batch_size
            X_batch = shuffled_X[start_idx:start_idx + batch_size]
            y_batch = shuffled_y[start_idx:start_idx + batch_size]
            
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        
        # Print MSE every 100 epochs using the full dataset
        if epoch % 100 == 0:
            save_path = saver.save(sess, "./data/05-grad-des-model.ckpt")
            current_mse = sess.run(mse, feed_dict={X: scaled_housing_data_plus_bias, y: target_reshaped})
            print(f"Epoch {epoch} MSE = {current_mse}")

    best_theta = sess.run(theta)
    save_path = saver.save(sess, "./data/05-grad-des-model-final.ckpt")

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"\nTraining completed in {elapsed_time:.4f} seconds")
print("Best Theta:", best_theta)