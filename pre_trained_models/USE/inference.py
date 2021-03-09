import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def load_model():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    model = hub.Module(module_url)

    config = tf.ConfigProto()
    config.graph_options.rewrite_options.shape_optimization = 2
    session = tf.Session(config=config)
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    messages = tf.placeholder(dtype=tf.string, shape=[None])
    input_plc = model(messages)

    return (model, session, messages, input_plc)


def generate_vecs(models, document):
    model, session, messages, input_plc = models
    embeddings = session.run(input_plc, feed_dict={messages:document})

    if len(np.shape(embeddings)) > 2:
        embeddings = np.reshape(embeddings, [np.shape(embeddings)[0], np.shape(embeddings)[2]])

    return embeddings
