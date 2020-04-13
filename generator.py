from data_processor import *

rule_path = "Files/RuleCodes"
token_path = "Files/TokenCodes"


class Generator:
    # input 修改前的代码， 是token sequence
    # diff  修改信息      是token sequence
    def __init__(self, original, diff, diff_matrix):
        self.probility = 1.0
        self.finished = False

        self.processor = DataProcessor(rule_path, token_path)

        self.original_tokens = original
        self.original_chars  = self.seq2chars(original, parameters.INPUT_LEN)
        self.original_mask   = self.processor.get_mask(len(self.original_tokens), parameters.INPUT_LEN)

        self.diff_tokens     = diff
        self.diff_chars      = self.seq2chars(diff, parameters.DIFF_LEN)
        self.diff_mask       = self.processor.get_mask(len(self.diff_tokens), parameters.DIFF_LEN)
        self.diff_matrix     = diff_matrix

        self.rule_parent, self.rule_seq, self.rule_node, self.rule_son, self.rule_mask, self.rule_matrix = self.initialize(self.processor)

        self.extend = list()
        start_rule = self.processor.rules[self.processor.START_RULE]
        for i in range(len(start_rule)-1, 0, -1):
            self.extend.append(start_rule[i])

        #self.rule_result = [self.processor.START_RULE] # rule 序列
        self.rule_result = [] #不用加start_rule, 否则后面的copy的信息将产生移动
        self.path        = [self.processor.token_map[self.processor.START]] # 从顶节点到扩展节点的路径 每个为node对应的token
        self.path_index  = [0] # path上每个节点对应的index
        self.result      = list() # 产生的token序列
        self.used_tokens = set()  # 在扩展过程中产生已经被用过的token

    def seq2chars(self, seq, length):
        empty_char = self.processor.char_map[self.processor.EMPTY]
        empty = np.array([empty_char] * parameters.CHAR_PER_TOKEN)

        res = []
        for i in range(min(len(seq), length)):
            token = self.processor.tokens[seq[i]]
            res.append(self.processor.token2chars(token))

        while len(res) < length:
            res.append(empty)

        return res

    def update(self):
        parent_token = self.path[-2] # -1是当前新增的节点
        self.rule_parent.append(parent_token)

        rule_index = self.rule_result[-1]
        added_rule = self.processor.rules[rule_index]
        self.rule_seq.append(rule_index)

        self.rule_node = added_rule[0]

        empty = self.processor.token_map[self.processor.EMPTY]
        son_seq = np.array([empty] * parameters.RULE_SON_SIZE)
        son_size = len(added_rule) - 1
        for i in range(min(son_size, parameters.RULE_SON_SIZE)):
            son_seq[i] = added_rule[i+1]
        self.rule_son.append(son_seq)

        length = len(self.rule_result)
        self.rule_mask[length-1] = 1

        parent = self.path_index[-2]
        cur = self.path_index[-1]
        if parent < parameters.TARGET_LEN and cur < parameters.TARGET_LEN:
            self.rule_matrix[cur][parent] = 1

    def initialize(self, processor):
        rule_parent = processor.line2vec(str(self.processor.token_map[self.processor.UNK]), parameters.TARGET_LEN)
        rule_seq, rule_node, rule_son = processor.line2ast(str(processor.START_RULE), rule_parent)
        rule_mask = processor.get_mask(1, parameters.TARGET_LEN)

        rule_matrix = processor.line2matrix("", parameters.TARGET_LEN)

        return rule_parent, rule_seq, rule_node, rule_son, rule_mask, rule_matrix

    def next(self):
        end = self.processor.token_map[self.processor.END]
        while len(self.extend) > 0:
            top = self.extend[-1]

            if (top == end):
                self.extend.pop()
                self.path_op("pop", None)
                continue
            elif self.processor.heads.issuperset([top]): # top 是非叶子节点
                return top
            else:
                self.result.append(top)
                self.extend.pop()
        return None

    # class_num 应该是rule的编号
    def append(self, class_num):

        if not self.is_valid(class_num): #扩展的指令不合法
            return False

        # 增加一条记录
        self.rule_result.append(class_num)

        top = self.next()
        self.extend.pop()
        self.path_op("append", top)

        end = self.processor.token_map[self.processor.END]
        self.extend.append(end)

        # 当为复制指令的时候，top和class_num的信息不一致
        if class_num >= self.processor.rule_size: #复制指令
            index = class_num - self.processor.rule_size
            self.extend.append(self.processor.rules[self.rule_result[index]][1])
        else:
            if class_num >= len(self.processor.rules):
                a = 2
                pass

            rule = self.processor.rules[class_num]
            for i in range(len(rule)-1, 0, -1):

                self.extend.append(rule[i])

        if self.next() == None:
            self.finished = True
        return True

    #class_num是即将产生的代码的编号
    #判断是否是当前需要的可扩展节点
    def is_valid(self, class_num):
        top = self.next()

        if top is None:
            return False

        #代待扩展的是复制语句
        if top == self.processor.token_map[self.processor.COPY]:
            if class_num < self.processor.rule_size: #添加的不是复制命令
                return False
            index = class_num - self.processor.rule_size # 复制第几条指令

            # 拷贝指令越界
            if (index >= len(self.rule_result)):
                return False

            copy_rule = self.rule_result[index] # 指令的编码

            copy_head = self.processor.rules[copy_rule][0] # 由什么节点产生的变量

            if not self.is_extend_tokens(self.processor.tokens[copy_head]):
                return False

            parent = self.path[-1] if len(self.path) >= 1 else None #有疑问，到底是-1 还是-2

            if parent != copy_head:
                return False

            return True

        else: # 待扩展的是普通指令
            if class_num >= self.processor.rule_size:
                return False

            #扩展的指令 与预期不一致
            extend_head = self.processor.rules[class_num][0]
            if extend_head != top:
                return False


            if self.is_extend_tokens(extend_head): #扩展的指令为产生name的类型，那么产生的name不能出现在之前已经产生过的
                literal = self.processor.rules[class_num][1]
                if self.used_tokens.issuperset([literal]): #该literal已经被用过
                    return False
                else:
                    return True
            else: #扩展的指令为其他指令

                return True

    def is_extend_tokens(self, token):
        return (token == "$SimpleName" or
                token == "$CharacterLiteral" or
                token == "$NumberLiteral" or
                token == "$StringLiteral" or
                token == "$QualifiedName" or
                token == "$QualifiedType")

    def path_op(self, op, node=None):
        if op == "append":
            if node == None:
                raise Exception("path operator error: node is empty")
            else:
                index = len(self.path)
                self.path_index.append(index)
                self.path.append(node)
        elif op == "pop":
            if len(self.path) == 0:
                raise Exception("path operator error: path in empty")
            else:
                self.path_index.pop()
                self.path.pop()
        else:
            raise Exception("path operator error: operator is valid")

    def output(self):
        res = ""
        for token in self.result:
            res += self.processor.tokens[token] + " "
        print(res)

def main():

    batches = batch_data(1, "test")
    for batch in batches:
        ge = Generator(batch.original_tokens[0], batch.diff_tokens[0], batch.diff_matrixs[0])

        for class_num in batch.target_rules[0]:
            ge.append(class_num)

        ge.output()

#main()