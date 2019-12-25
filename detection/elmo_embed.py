import tensorflow.compat.v1 as tf
import tensorflow_hub as hub



def provideElmo():
    # tf.disable_eager_execution()
    print('get elmo')
    # elmo = hub.Module("https://tfhub.dev/google/elmo/3")
    elmo = hub.Module('/home/filip/Documents/projects/new_mgr/ironyDetectionPy/detection/elmoDir/3')
    # elmo = hub.load('/home/filip/Documents/projects/new_mgr/ironyDetectionPy/detection/elmoDir/3')
    print('return elmo')
    return elmo


def elmo_vectors(elmo, x):
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(embeddings)
