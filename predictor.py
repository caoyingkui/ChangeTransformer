import os
import tensorflow as tf
from model import *
from data_processor import *
from generator import *
import queue

class predictor():
    def __init__(self):
        self.predict()

    def create_model(self, session):
        path = "model/save0"
        if (os.path.exists(path)):
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(path))
            print("load model: " + path)
        else:
            raise Exception("model does not exist!")

    def predict(self):
        processor = DataProcessor("Files/RuleCodes", "Files/TokenCodes")
        batch_data(1, "test")
        code_gen_model = Model(processor.rule_size,
                               processor.class_size,
                               len(processor.tokens),
                               len(processor.chars))

        config = tf.ConfigProto(device_count={"GPU": 0})

        with tf.Session(config=config) as sess:
            self.create_model(sess)
            for batch in batch_data(1, "test"):
                data = batch[0]

                generator = Generator(data.original_tokens, data.diff_tokens, data.diff_matrix)
                self.beam_search(sess, code_gen_model, generator)


    def beam_search(self, sess, model, generator):

        beam_size = 5

        Beam = [model]

        step = 0
        while True:
            step += 1
            result = queue.PriorityQueue()

            for ge in Beam:
                if ge.finished:
                    result.put(ge)

                action = self.get_action(sess, model, generator)
                action_list = [[action[i], i] for i in range(len(action))]
                action_list = sorted(action_list, reverse=True)

                extend_num = 0
                for i in range(len(action_list)):
                    class_num = action_list[i][1] # 操作编号

                    if not ge.is_valid(class_num): # 扩展的节点类型不合法
                        continue

                    extend_num += 1
                    ge_temp = deepcopy(ge)
                    ge_temp.append(class_num)

                    ge_temp.priority = self.get_problility(
                        ge_temp.probility, action_list[i][0], len(ge_temp.rules)
                    )

                    result.put(ge_temp)

                    if extend_num >= beam_size:
                        break

            Beam = []
            num = 0
            while (not result.empty()) and len(Beam) < beam_size:
                ge = result.get()
                Beam.append(ge)
                if ge.finished:
                    num += 1

            if num >= beam_size or step >= parameters.TARGET_LEN:
                for ge in Beam:
                    if ge.finished:
                        ge.output()

                break


    def get_action(self, sess, model, generator):
        y = sess.run(model.y_result, feed_dict= {
            model.original_tokens: [generator.original_token_seq],
            model.original_chars : [generator.original_chars],
            model.original_mask: [generator.original_mask],
            model.diff_tokens: [generator.diff_tokens],
            model.diff_chars: [generator.diff_chars],
            model.diff_mask: [generator.diff_mask],
            model.target_rules: [generator.rule_seq],
            model.target_nodes: [generator.rule_node],
            model.target_parents: [generator.parents],
            model.target_sons: [generator.rule_son],
            model.target_mask: [generator.rule_mask],
            model.target_matrix: [generator.rule_matrix],
            model.loss_mask: [generator.rule_mask]
        })

        index = len(generator.rule_seq) - 1

        return y[0][index]

    def get_problility(self, ge_problility, extended_probility, rule_len):
        apa = 0.6
        return ( (ge_problility * math.pow(rule_len, apa) + math.log(max(1e-10, extended_probility))) /
                 math.pow(rule_len + 1, apa) )

pre = predictor()

