import tensorflow_hub as hub
import tensorflow as tf


elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
