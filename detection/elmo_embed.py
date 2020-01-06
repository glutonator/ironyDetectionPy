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

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(embeddings)


# Yield successive n-sized
# chunks from l.
def divide_chunks(my_list, nuber_of_elem_in_batch):
    # looping till length l
    for i in range(0, len(my_list), nuber_of_elem_in_batch):
        yield my_list[i:i + nuber_of_elem_in_batch]

    # How many elements each


# list should have
