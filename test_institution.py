# -*- coding: utf-8 -*-
"""
Testing on pre-trained model.
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
    
    ckpt = tf.train.latest_checkpoint('model')
    saver = tf.train.import_meta_graph(ckpt + '.meta')
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        file_writer = tf.summary.FileWriter('./logs', sess.graph)
        x = tf.get_default_graph().get_tensor_by_name("input/x:0")
        vector = tf.get_default_graph().get_tensor_by_name("DeepID/Relu:0")
        # 256 dims
        # vector = tf.get_default_graph().get_tensor_by_name("DeepID/Relu:0")  # 256 dims
        amount=20000
        _, _, real_y = load_pair_vector("image/test_vector_dataset.pkl", amount)

        amount=20000 # how many pairs to extract for testing? Larger is better.
        output1 = open("test1.pkl", 'rb')
        output2 = open("test2.pkl", 'rb')
        real_x_1 = pickle.load(output1)
        real_x_2 = pickle.load(output2)
        
        pre_vector_1 = sess.run(vector, {x: real_x_1})
        pre_vector_2 = sess.run(vector, {x: real_x_2})

        pre_y = np.array([cosine(x, y) for x, y in zip(pre_vector_1, pre_vector_2)])
        thre_y = []
        print(pre_y)
        for i in range(len(pre_y)):

            if pre_y[i] < threshold:
                thre_y.append(1)
            else:
                thre_y.append(0)

        print("------------------------------")
        print(u"predict: ", np.array(thre_y))
        print(u"label: ", np.array(real_y))
        print(u"acc=", 1-np.sum(np.abs(thre_y-real_y))/amount)
        print("------------------------------")

if __name__ == "__main__":
    run()
    