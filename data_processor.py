import numpy as np
from parameters import *
import math
from copy import deepcopy

trainset  = {}


def batch_data(batch_size, data_set):
    global trainset
    if trainset == {}:
        p = DataProcessor("Files/RuleCodes", "Files/TokenCodes")
        trainset["train"] = p.resolve_data("Files/data_train")
        trainset["dev"] = p.resolve_data("Files/data_dev")
        trainset["test"] = p.resolve_data("Files/data_test")

    all_data = trainset[data_set]

    batches = []
    size = all_data.size() // batch_size
    for i in range(size):
        batch = Data()
        batch.init(all_data, i * batch_size, (i + 1) * ( batch_size))
        batches.append(batch)

    return batches


class Data:
    def __init__(self):
        self.original_tokens = []
        self.original_chars  = []
        self.original_masks  = []

        self.diff_tokens  = []
        self.diff_chars   = []
        self.diff_matrixs = []
        self.diff_masks   = []

        self.target_rules   = []
        self.target_parents = []
        self.target_nodes   = []
        self.target_sons    = []
        self.target_masks   = []
        self.target_matrixs = []

    def init(self, data, start, end):
        end = min(data.size(), end)

        if end < start:
            raise Exception("index error")
            return

        self.original_tokens = self.__copy__(data.original_tokens, start, end)
        self.original_chars  = self.__copy__(data.original_chars, start, end)
        self.original_masks = self.__copy__(data.original_masks, start, end)

        self.diff_tokens = self.__copy__(data.diff_tokens, start, end)
        self.diff_chars  = self.__copy__(data.diff_chars, start, end)
        self.diff_matrixs = self.__copy__(data.diff_matrixs, start, end)
        self.diff_masks   = self.__copy__(data.diff_masks, start, end)

        self.target_rules   = self.__copy__(data.target_rules, start, end)
        self.target_parents = self.__copy__(data.target_parents, start, end)
        self.target_nodes   = self.__copy__(data.target_nodes, start, end)
        self.target_sons    = self.__copy__(data.target_sons, start, end)
        self.target_masks    = self.__copy__(data.target_masks, start, end)
        self.target_matrixs  = self.__copy__(data.target_matrixs, start, end)

    def size(self):
        return len(self.original_tokens)

    def __copy__(self, data, start, end):
        res = []
        for i in range(start, end):
            res.append(deepcopy(data[i]))

        return res


class DataProcessor:

    rules = []
    tokens = []
    token_map = {}
    chars = []
    char_map = {}

    UNK = "<UNK>"
    EMPTY = "$Empty"
    START = "$Start"
    METHOD = "$MethodDeclaration"
    COPY   = "$Copy"
    END    = "<END>"

    def __init__(self, rule_path, token_path):
        self.rule_path = rule_path
        self.token_path = token_path

        self.get_token_list()
        self.get_char_list()
        self.get_rule_list()

    def get_rule_list(self):
        self.rules = []
        file = open(self.rule_path)
        self.heads = set()


        for line in file.readlines():
            line = line.strip()
            rule = []
            tokens = line.split(" ")

            head = self.token2num(tokens[0])
            rule.append(head)
            self.heads.add(head)

            for i in range(len(tokens))[2:]: # rule的格式： head -> a b c, tokens[1]是 ->
                rule.append(self.token2num(tokens[i]))
            self.rules.append(rule)

        self.START_RULE = len(self.rules)
        self.rules.append([self.token2num(self.START), self.token2num(self.METHOD)])
        self.heads.add(self.token2num(self.START))
        self.heads.add(self.token2num(self.COPY))

        self.EMPTY_RULE = len(self.rules)
        self.rules.append([])


        self.rule_size = len(self.rules)
        self.class_size = self.rule_size + parameters.TARGET_LEN
        class_num = self.class_size

    def get_token_list(self):
        self.tokens = []
        self.token_map = {}
        file = open(self.token_path)

        for index, token in enumerate(file.readlines()):
            token = token.strip()
            self.tokens.append(token)
            self.token_map[token] = index

        self.token_map[self.EMPTY] = len(self.tokens)
        self.tokens.append(self.EMPTY)

        self.token_map[self.UNK] = len(self.tokens)
        self.tokens.append(self.UNK)

        self.token_map[self.START] = len(self.tokens)
        self.tokens.append(self.START)

        self.token_map[self.END] = len(self.tokens)
        self.tokens.append(self.END)

    def get_char_list(self):
        self.chars = []
        self.char_map = {}

        char_set = set()
        for token in self.tokens:
            for c in token:
                char_set.add(c)

        index = 0
        for c in char_set:
            self.chars.append(c)
            self.char_map[c] = index

            index += 1

        self.char_map[self.EMPTY] = len(self.chars)
        self.chars.append(self.EMPTY)

        self.char_map[self.UNK] = len(self.chars)
        self.chars.append(self.UNK)

    def char2num(self, c):
        if c in self.char_map:
            return self.char_map[c]
        else:
            print("error from char2num")
            return self.char_map[self.UNK]

    def token2num(self, token):
        if token in self.token_map:
            return self.token_map[token]
        else:
            return self.token_map[self.UNK]

    def rule2num(self, rule):
        num = int(rule)
        if num < self.rule_size:
            return num
        elif num >= 100000:
            return self.rule_size + (num - 100000)
        else:
            print("error")

        return 0

    def line2rules(self, line):
        rule_seq = []

        nums = line.split(" ")
        for num in nums:
            rule_seq.append(self.rule2num(num))
        return rule_seq

    # rule 是编号
    # parent 节点
    # rule_seq是一条数据的rule list
    def rule2ast(self, rule, parent, rule_seq):
        empty = self.token_map[self.EMPTY]
        vec = np.zeros([1], dtype="int32")
        vecson = np.array([empty] * parameters.RULE_SON_SIZE)

        num = int(rule)

        if num < self.rule_size:
            rule = self.rules[num] # rule[0] 是父亲token，rule[1:]是孩子token
            vec[0] = rule[0]
            for i in range(min(parameters.RULE_SON_SIZE, len(rule) - 1)):
                vecson[i] = rule[i+1]
        elif num < self.class_size:
            index = rule_seq[num - self.rule_size]
            vec[0] = parent
            vecson[0] = self.rules[index][1]

        return vec, vecson

    def line2ast(self, line, parents):
        rule_seq = self.line2rules(line)

        empty = self.token_map[self.EMPTY]
        seq = np.array([self.EMPTY_RULE] * parameters.TARGET_LEN)
        vec_node = np.array([empty] * parameters.TARGET_LEN)
        vec_son = np.array([[empty] * parameters.RULE_SON_SIZE] * parameters.TARGET_LEN)

        for index, rule in enumerate(rule_seq):
            if (index >= parameters.TARGET_LEN):
                break
            parent = parents[index]
            head, son = self.rule2ast(rule, parent, rule_seq)

            seq[index] = rule_seq[index]
            vec_node[index] = head[0]
            vec_son[index] = son

        return seq, vec_node, vec_son

    def line2vec(self, line, length):
        empty = self.token_map[self.EMPTY]
        vec = np.array([empty] * length)

        for index, token in enumerate(line.split(" ")):
            if index >= length:
                break
            vec[index] = int(token)
        return vec

    def token2chars(self, token):
        empty = self.char_map[self.EMPTY]
        res = np.array([empty] * parameters.CHAR_PER_TOKEN)

        for index, c in enumerate(token):
            if index >= parameters.CHAR_PER_TOKEN:
                break
            res[index] = self.char_map[c]

        return res

    def line2chars(self, line, length):
        empty = self.char_map[self.EMPTY]

        chars = np.array([[empty] * parameters.CHAR_PER_TOKEN] * length)

        for index, token_num in enumerate(line.split(" ")):
            if index >= length:
                break
            token = self.tokens[int(token_num)]
            chars[index] = self.token2chars(token)
        return chars

    def get_mask(self, data_length, mask_length):
        mask = np.zeros([mask_length])

        for i in range(min(data_length, mask_length)):
            mask[i] = 1

        return mask

    def line2matrix(self, line, length):
        matrix = np.zeros([length, length])

        cordinates = line.split()

        for i in range(len(cordinates))[0::2]:
            x = int(cordinates[i])
            y = int(cordinates[i+1])
            if x < length and y < length:
                matrix[x][y] = 1

        return matrix

    def resolve_data(self, path):
        if len(self.tokens) is 0:
            self.get_token_list()

        if len(self.chars) is 0:
            self.get_char_list()

        if len(self.rules) is 0:
            self.get_rule_list()

        file = open(path)
        lines = file.readlines()

        data = Data()

        for i in range(0, 300, 7):
            original_str = lines[i]
            diff_str     = lines[i + 1]
            matrix       = lines[i + 2]
            targetSeq    = lines[i + 3]
            parent       = lines[i + 4]
            target_matrix = lines[i + 5]
            #lines[i+6] 为空行

            original_token_seq = self.line2vec(original_str, parameters.INPUT_LEN)
            original_char_seq  = self.line2chars(original_str, parameters.INPUT_LEN)
            original_mask      = self.get_mask(len(original_token_seq), parameters.INPUT_LEN)
            data.original_tokens.append(original_token_seq)
            data.original_chars. append(original_char_seq)
            data.original_masks. append(original_mask)

            diff_token_seq = self.line2vec(diff_str, parameters.DIFF_LEN)
            diff_char_seq  = self.line2chars(diff_str, parameters.DIFF_LEN)
            diff_mask      = self.get_mask(len(diff_token_seq), parameters.DIFF_LEN)
            diff_matrix    = self.line2matrix(matrix, parameters.DIFF_LEN)
            data.diff_tokens. append(diff_token_seq)
            data.diff_chars.  append(diff_char_seq)
            data.diff_masks.  append(diff_mask)
            data.diff_matrixs.append(diff_matrix)


            rule_parent = self.line2vec(parent, parameters.TARGET_LEN)
            rule_seq, rule_node, rule_son = self.line2ast(targetSeq, rule_parent)
            rule_mask = self.get_mask(len(rule_seq), parameters.TARGET_LEN)
            rule_matrix = self.line2matrix(target_matrix, parameters.TARGET_LEN)

            data.target_rules.  append(rule_seq)
            data.target_parents.append(rule_parent)
            data.target_nodes.  append(rule_node)
            data.target_sons.   append(rule_son)
            data.target_masks.  append(rule_mask)
            data.target_matrixs.append(rule_matrix)

        self.data_check(data)

        return data

    def recover(self, data, type):
        if type == "tokens":

            return [self.tokens[token] for token in data]
        elif type == "chars":
            return [[self.chars[c] for c in char] for char in data ]
        elif type == "mask":
            for i in range(len(data)):
                if (data[i] == 0):
                    return i
            return len(data)
        elif type == "rule":
            res = []
            for i in range(len(data)):
                if (data[i] == self.EMPTY_RULE):
                    break
                if data[i] >= self.rule_size:
                    res.append(["$Copy", str(data[i] - self.rule_size)])
                else:
                    res.append(self.recover(self.rules[data[i]], "tokens"))
            return res
        elif type == "node":
            return [self.tokens[n] for n in data]
        elif type == "son":
            return [[self.tokens[s] for s in son] for son in data]

        return None

    def data_check(self, data, shape=None, length=None, error=None):
        if shape and length and error:
            if len(data) != length:
                print(error)
                return
            for d in data:
                if d.shape != shape:
                    print(error)
                    return
        else:
            length = len(data.original_tokens)

            self.data_check(data.original_tokens, (parameters.INPUT_LEN,), length, "error1")
            self.data_check(data.original_chars, (parameters.INPUT_LEN, parameters.CHAR_PER_TOKEN), length, "error2")
            self.data_check(data.original_masks, (parameters.INPUT_LEN,), length, "error3")
            self.data_check(data.target_rules, (parameters.TARGET_LEN,), length, "error7")
            self.data_check(data.target_nodes, (parameters.TARGET_LEN,), length, "error8")
            self.data_check(data.target_parents, (parameters.TARGET_LEN,), length, "error9")
            self.data_check(data.target_sons, (parameters.TARGET_LEN, parameters.RULE_SON_SIZE), length, "error10")
            self.data_check(data.target_masks, (parameters.TARGET_LEN,), length, "error11")


processor = DataProcessor("Files/RuleCodes", "Files/TokenCodes")
#data = batch_data(1, "test")

