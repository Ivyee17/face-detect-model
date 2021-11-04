# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 21:31:56 2018

@author: shen1994
"""

import os
import pickle

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from image_vector import load_vector
from image_vector import load_pair_vector
from image_vector import load_vector_from_index
from image_vector import safe_file_close
from image_split import load_train_test_number
from model_finetune import deepid_1

def run():
    
    epocs = 100001
    batch_size = 256
    
    log_dir = "log"
    if tf.io.gfile.exists(log_dir):
        tf.io.gfile.rmtree(log_dir)
    tf.io.gfile.makedirs(log_dir)
    
    model_dir = "model"
    if tf.io.gfile.exists(model_dir):
        tf.io.gfile.rmtree(model_dir)
    tf.io.gfile.makedirs(model_dir)
    
    train_samples_number, _ = load_train_test_number("image/train_test_number.pkl")
    class_num = train_samples_number + 1

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 12,10,40], name='x')
        y = tf.placeholder(tf.float32, [None, class_num], name='y')

    merged, loss, accuracy, optimizer = deepid_1(x, y, class_num=class_num)
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
        
        t_big_number, t_size, t_index, t_pkl_file = 0, 0, 0, None
        v_big_number, v_size, v_index, v_pkl_file = 0, 0, 0, None
        step = 0

        while step < epocs:
            pkl_file = open("data.pkl", "rb")
            aaaa=pickle.load(pkl_file)
            _, train_y, t_index, t_big_number, t_size, t_pkl_file = load_vector_from_index(
                "image/train_vector_dataset.pkl",
                batch_size,
                t_index,
                t_big_number,
                t_size,
                t_pkl_file)
            times_per_round = (int)(t_big_number / batch_size)
            train_onehot_y = (np.arange(class_num) == train_y[:, None]).astype(np.float32)

            if train_onehot_y.shape[0]!=256:
                pass
                # _ = sess.run(optimizer, {x: aaaa[step * 256:step*256+train_onehot_y.shape[0]], y: train_onehot_y})
            else:
                _ = sess.run(optimizer, {x: aaaa[(step%times_per_round)*256:((step%times_per_round)+1)*256], y: train_onehot_y})

            if train_onehot_y.shape[0] != 256:
                pass
            else:
                if step % 1000 == 0:

                    summary = sess.run(merged, {x: aaaa[(step%times_per_round)*256:((step%times_per_round)+1)*256], y: train_onehot_y})
                    train_writer.add_summary(summary, step)
                

                
                    t_cost, t_acc = sess.run([loss, accuracy], {x: aaaa[(step%times_per_round)*256:((step%times_per_round)+1)*256], y: train_onehot_y})
                
                    print(str(step) + ": train --->" + "cost:" + str(t_cost) + ", accuracy:" + str(t_acc))
                    print("----------------------------------------")
                
                if step % 1000 == 0 and step != 0:
                    saver.save(sess, 'model/deepid%d.ckpt' % step)
                
            step += 1
            
        safe_file_close(t_pkl_file)
        safe_file_close(v_pkl_file)

if __name__ == "__main__":
    run()