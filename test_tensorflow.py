# from tensorflow.python.client import device_lib
#
# print(device_lib.list_local_devices())
#

# import tensorflow as tf
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#
# import tensorflow as tf
# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)
#
# # with tf.Session() as sess:
# #     print (sess.run(c))
#
# with tf.Session() as sess:
#   devices = sess.list_devices()


# import tensorflow as tf
# sess = tf.InteractiveSession()
#
#
# from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())


# import keras
# # Using TensorFlow backend.
# print(keras.__version__)


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())