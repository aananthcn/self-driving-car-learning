import tensorflow as tf

tf.compat.v1.disable_eager_execution()

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

sess = tf.compat.v1.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print("Evaluating `f = x*x*y + y + 2` where x = 3 and y = 4 and f =")
print(result)


x1 = tf.Variable(1)
x1.graph is tf.compat.v1.get_default_graph()

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.compat.v1.Session() as sess:
    print("y = x + 5, where x = w + 2 and w = 3; y = ")
    print(y.eval())
    print("z = x * 3; z = ")
    print(z.eval())
