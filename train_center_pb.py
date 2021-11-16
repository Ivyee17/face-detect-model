# -*- coding: utf-8 -*-
# @Time    : 2021-11
# @Author  : Wenhan Wu
# @FileName: train_center_pb.py
# @Project :
# @GitHub  : Ivyee17
"""
We use the pre-trained pb file to train in the center.
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
    return 1 - cos


def run(threshold=0.43):


    with tf.Session() as sess:
        output_graph_def = tf.GraphDef()
        with open("model-center/model.pb", "rb") as f:
            output_graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            _ = tf.import_graph_def(output_graph_def, name="")

        sess.run(tf.global_variables_initializer())

        x = sess.graph.get_tensor_by_name("input/x:0")
        vector = sess.graph.get_tensor_by_name("Conv_layer_2/max-pooling:0")

        amount = 20000  # how many pairs to extract for testing? Larger is better.
        real_x_1, real_y = load_vector("image/train_vector_dataset.pkl", amount)

        features = sess.run(vector, {x: real_x_1})
        output = open('datapb.pkl', 'wb')
        pickle.dump(features, output)


if __name__ == "__main__":
    run()
