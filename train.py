import os
import tensorflow as tf
from copy import deepcopy
import numpy as np
from parameters import *
from model import *
from data_processor import *

tf.reset_default_graph()

pretrain_times = 100
batch_size = 30

def tqdm(a):
    try:
        from tqdm import tqdm
        return tqdm(a)
    except:
        return a

def create_model(session):
    if (os.path.exists("model/save1")):
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint("model/save1"))
        print("load the model")
    else:
        session.run(tf.global_variables_initializer(), feed_dict={})
        print("create a new model")

def save_model(session, number):
    saver = tf.train.Saver()
    saver.save(session, "model/save" + str(number) + "/model.cpkt")

#def save_model_time(session, number)


def g_pretrain(sess, model, batch_data):

    _, pre, a = sess.run([model.optim, model.correct_prediction, model.cross_entropy],
                         feed_dic = {
                             model.original_tokens: batch_data.original_tokens,
                             model.original_chars:  batch_data.original_chars,
                             model.original_masks:  batch_data.original_masks,
                             model.diff_tokens:     batch_data.diff_tokens,
                             model.diff_chars:      batch_data.diff_chars,
                             model.diff_masks:      batch_data.diff_masks,
                             model.diff_matrixs:    batch_data.diff_matrixs,
                             model.target_rules:    batch_data.target_rules,
                             model.target_nodes:    batch_data.target_nodes,
                             model.target_parents:  batch_data.target_parents,
                             model.target_sons:     batch_data.target_sons,
                             model.target_masks:    batch_data.target_masks,
                             model.target_matrixs:  batch_data.target_matrixs,
                             model.loss_masks:      batch_data.target_masks,

                             model.keep_prob: 0.85,
                         })

    return pre

def g_eval(sess, model, batch_data):

    acc, pre, pre_rules = sess.run([model.accuracy, model.correct_prediction, model.max_res],
                            feed_dict={
                                model.original_tokens: batch_data.original_tokens,
                                model.original_chars: batch_data.original_chars,
                                model.original_masks: batch_data.original_masks,
                                model.diff_tokens: batch_data.diff_tokens,
                                model.diff_chars: batch_data.diff_chars,
                                model.diff_masks: batch_data.diff_masks,
                                model.diff_matrixs: batch_data.diff_matrixs,
                                model.target_rules: batch_data.target_rules,
                                model.target_nodes: batch_data.target_nodes,
                                model.target_parents: batch_data.target_parents,
                                model.target_sons: batch_data.target_sons,
                                model.target_masks: batch_data.target_masks,
                                model.target_matrixs: batch_data.target_matrixs,
                                model.loss_mask: batch_data.target_masks,

                                model.keep_prob: 1.0
                            })

    p = []
    max_res = []
    for i in range(len(batch_data.target_rules)):
        for t in range(parameters.TARGET_LEN):
            if batch_data.target_rules != 0: # 这里有问题的，0
                p.append(pre[i][t])
                max_res.append(pre_rules[i][t])
            else:
                p.pop()
                max_res.pop()
    return acc, p, max_res

def run():

    data_processor = DataProcessor(rule_path="Files/RuleCodes",
                                   token_path="Files/TokenCodes")

    global trainset

    trainset["train"] = data_processor.resolve_data("Files/data_train")
    trainset["dev"]   = data_processor.resolve_data("Files/data_dev")
    trainset["test"]  = data_processor.resolve_data("Files/data_test")


    transform_model = Model(rule_size=len(data_processor.rules),
                            class_size=len(data_processor.rules) + parameters.TARGET_LEN,
                            token_size=len(data_processor.tokens),
                            char_size=len(data_processor.chars))

    valid_batch = batch_data(batch_size, "dev")


    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    output = open("out.txt", "w")

    with tf.Session(config=config) as sess:
        create_model(sess)

        for i in tqdm(range(pretrain_times)):
            batch = batch_data(batch_size, "train")

            for j in tqdm(range(len(batch))):
                if i % 3 == 0 and j % 2000 == 0:
                    ac = 0
                    res = []
                    for k in range(len(valid_batch)):
                        ac1, loss1, _ = g_eval(sess, transform_model, valid_batch[k])
                        res.extend(loss1)
                        ac += ac1
                    ac /= len(valid_batch)

                if i % 50 == 0 and j == 0:
                    ac = 0
                    res = []
                    for k in range(len(valid_batch)):
                        ac1, loss1, _ = g_eval(sess, transform_model, valid_batch[k])

                        res.extend(loss1)
                        ac += ac1
                    print(len(res))

                    ac /= len(valid_batch)

                    print("current accuracy " +
                          str(ac))
                    save_model(sess, i)

                    print("current accuracy " + str(ac))

                g_pretrain(sess, transform_model, batch[j])
                tf.train.global_step(sess, transform_model)

        save_model(sess, 0)
    output.close()
    return


def main():
    run()

with tf.device('/cpu:0'):
    main()










