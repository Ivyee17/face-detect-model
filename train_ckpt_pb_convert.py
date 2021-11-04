# -*- coding: utf-8 -*-
"""From the pretrained-model, extract the model of the first two level as the center code. It can be trained using
train_center.py by using all images in the dataset, but we give an extraction which can be faster to get the data
from the center. Remember that if you use the model trained by yourself, you'll also need to run this script in order
to transform from checkpoint file to pb file for train_institution.py to use.
By using this file, you can use train_center_pb.py in public. """

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
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
        # file_writer = tf.summary.FileWriter('./logs', sess.graph)

        x = tf.get_default_graph().get_tensor_by_name("input/x:0")
        vector = tf.get_default_graph().get_tensor_by_name("Conv_layer_2/max-pooling:0")
        # vector.name = "Conv_layer_2/max-pooling:0"
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Conv_layer_2/max-pooling'])
        with tf.gfile.FastGFile('model-center/model2.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())



if __name__ == "__main__":
    run()
