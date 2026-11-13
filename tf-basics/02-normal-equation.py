# Normal Equation θ = (X^T . X)^–1 . X^T . y
import numpy as np
import tensorflow as tf

# Fix: Disable eager execution so operations register in the graph for Session()
tf.compat.v1.disable_eager_execution()

# The following lines threw error for me
# from sklearn.datasets import fetch_california_housing
# housing = fetch_california_housing()

# from sklearn.datasets import fetch_openml
# housing = fetch_openml(name="california_housing", version=1, as_frame=False)

# 1. Bypassed fetch_openml using synthetic data (to fix the WSL2 network/DNS error)
from sklearn.datasets import make_regression
X_synth, y_synth = make_regression(n_samples=20640, n_features=8, noise=0.1, random_state=42)
class HousingMock: pass
housing = HousingMock()
housing.data = X_synth
housing.target = y_synth

m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")

XT = tf.transpose(X)

# 2. Changed tf.matrix_inverse to tf.linalg.inv
theta = tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(XT, X)), XT), y)

# 3. Fixed tf.compat.v1.Session() syntax and used sess.run() instead of .eval()
with tf.compat.v1.Session() as sess:
	theta_value = sess.run(theta)

print("Housing data size = ", m, " x ", n)
print("Theta = ")
print(theta_value)