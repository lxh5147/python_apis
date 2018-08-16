
import tensorflow as tf

print('tensorflow hello world example\n')

hello=tf.constant('hello world')
sess=tf.Session()
print(sess.run(hello))