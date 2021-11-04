# -*- coding: utf-8 -*-
"""
Use pre-trained model to train for center.
"""
import pickle

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
from image_vector import load_pair_vector, load_vector


def cosine(a, b):
    t = np.float(np.dot(a.T, b))
    k = np.linalg.norm(a) * np.linalg.norm(b)
    cos = t / k

    return (1 - cos)


def run(threshold=0.43):
    ckpt = tf.train.latest_checkpoint('model-pretrained')
    saver = tf.train.import_meta_graph(ckpt + '.meta')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        file_writer = tf.summary.FileWriter('./logs', sess.graph)
        x = tf.get_default_graph().get_tensor_by_name("input/x:0")
        vector = tf.get_default_graph().get_tensor_by_name("Conv_layer_2/max-pooling:0")
        # 256 dims
        # vector = tf.get_default_graph().get_tensor_by_name("DeepID/Relu:0")  # 256 dims

        amount = 20000  # how many pairs to extract for testing? Larger is better.
        real_x_1, real_x_2,real_y = load_pair_vector("image/test_vector_dataset.pkl", amount)

        pre_vector_1 = sess.run(vector, {x: real_x_1})
        pre_vector_2 = sess.run(vector, {x: real_x_2})

        output = open('test1.pkl', 'wb')
        pickle.dump(pre_vector_1, output)
        output = open('test2.pkl', 'wb')
        pickle.dump(pre_vector_2, output)




if __name__ == "__main__":
    run()
