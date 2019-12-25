import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

# to jest ważne, bez tego nie działa z tf v2.0
tf.disable_eager_execution()

elmo = hub.Module("https://tfhub.dev/google/elmo/3")


def elmo_vectors(x):
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(embeddings)


x = [':)']
vector = elmo_vectors(x)
print(vector)
