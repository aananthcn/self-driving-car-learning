import numpy as np
import tensorflow as tf
import time

# 1. Disable eager execution for TF 1.x graph style
tf.compat.v1.disable_eager_execution()

from sklearn.datasets import make_regression
X_synth, y_synth = make_regression(n_samples=20640, n_features=8, noise=0.1, random_state=42)
class HousingMock: pass
housing = HousingMock()
housing.data = X_synth
housing.target = y_synth

m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

from sklearn.preprocessing import StandardScaler
# Scale the features
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)

# Add the bias term to the scaled data
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")

# 2. Updated to tf.compat.v1 functions
theta = tf.Variable(tf.compat.v1.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# without optimizer
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.compat.v1.assign(theta, theta - learning_rate * gradients)

# with optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


init = tf.compat.v1.global_variables_initializer()

start_time = time.perf_counter()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            # 3. Use sess.run() instead of .eval()
            print("Epoch", epoch, "MSE =", sess.run(mse))
        sess.run(training_op)

    best_theta = sess.run(theta)

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"\nTraining completed in {elapsed_time:.4f} seconds")
print("Best Theta:", best_theta)