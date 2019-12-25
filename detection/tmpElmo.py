import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import numpy

# to jest ważne, bez tego nie działa z tf v2.0
tf.disable_eager_execution()

# elmo_tmp = hub.Module("https://tfhub.dev/google/elmo/3")
elmo_tmp = hub.Module('/home/filip/Documents/projects/new_mgr/ironyDetectionPy/detection/elmoDir/3')


def elmo_vectors22(elmo, x):
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(embeddings)


x = ['aa','ddd','oooo']
vector = elmo_vectors22(elmo_tmp, x)
print(vector)
print(vector.shape)
