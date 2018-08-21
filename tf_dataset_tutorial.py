import tensorflow as tf
import numpy as np


def simple_dataset_with_error():
    x = np.arange(0, 10)
    # create dataset object from the numpy array
    dx = tf.data.Dataset.from_tensor_slices(x)
    # create a one-shot iterator
    iterator = dx.make_one_shot_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(11):
            val = sess.run(next_element)
            print(val)

def simple_dataset_initializer():
    x = np.arange(0, 10)
    dx = tf.data.Dataset.from_tensor_slices(x)
    # create an initializable iterator
    iterator = dx.make_initializable_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(15):
            val = sess.run(next_element)
            print(val)
            if i % 9 == 0 and i > 0:
                sess.run(iterator.initializer)

def simple_dataset_batch():
    x = np.arange(0, 10)
    dx = tf.data.Dataset.from_tensor_slices(x).batch(3)
    # create a one-shot iterator
    iterator = dx.make_initializable_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(15):
            val = sess.run(next_element)
            print(val)
            if (i + 1) % (10 // 3) == 0 and i > 0:
                sess.run(iterator.initializer)

def simple_zip_example():
    x = np.arange(0, 10)
    y = np.arange(1, 11)
    # create dataset objects from the arrays
    dx = tf.data.Dataset.from_tensor_slices(x)
    dy = tf.data.Dataset.from_tensor_slices(y)
    # zip the two datasets together
    dcomb = tf.data.Dataset.zip((dx, dy)).batch(3)
    iterator = dcomb.make_initializable_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(15):
            val = sess.run(next_element)
            print(val)
            if (i + 1) % (10 // 3) == 0 and i > 0:
                sess.run(iterator.initializer)

if __name__ == "__main__":
    # simple_dataset_with_error()
    # simple_dataset_initializer()
    # simple_dataset_batch()
    simple_zip_example()


# more examples here: https://github.com/adventuresinML/adventures-in-ml-code/blob/master/r_learning_tensorflow.py