import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

# enable eager mode

tfe.enable_eager_execution()

a=tf.constant(2)
b=tf.constant(3)
print('a+b=%i\n' % (a+b))

# Define constant tensors
c = tf.constant([[2., 1.],
                 [1., 0.]], dtype=tf.float32)
print("Tensor:\n c = %s" % a)
d = np.array([[3., 0.],
              [5., 1.]], dtype=np.float32)
print("NumpyArray:\n d = %s" % b)

# Run the operation without the need for tf.Session
print("Running operations, without tf.Session")

e = a + b
print("c + d = %s" % e)

g = tf.matmul(c, d)
print("c * d = %s" % g)