import tensorflow as tf
from parameters import *
import math
import numpy as np
from tensorflow.contrib import *
from copy import deepcopy

tf.reset_default_graph()

def noise_from_step_num():
  """Quantization noise equal to (phi * (step_num + 1)) mod 1.0.
  Not using random_uniform here due to a problem on TPU in that random seeds
  are not respected, which may cause the parameters on different replicas
  to go out-of-sync.
  Returns:
    a float32 scalar
  """
  step = tf.to_int32(tf.train.get_or_create_global_step()) + 1
  phi = ((5 ** 0.5) - 1) / 2
  # Naive computation tf.mod(phi * step, 1.0) in float32 would be disastrous
  # due to loss of precision when the step number gets large.
  # Computation in doubles does not work on TPU, so we use this complicated
  # alternative computation which does not suffer from these roundoff errors.
  ret = 0.0
  for i in range(30):
    ret += (((phi * (2 ** i)) % 1.0)  # double-precision computation in python
            * tf.to_float(tf.mod(step // (2 ** i), 2)))
  return tf.mod(ret, 1.0)

class AdafactorOptimizer(tf.train.Optimizer):
  """Optimizer that implements the Adafactor algorithm.
  Adafactor is described in https://arxiv.org/abs/1804.04235.
  Adafactor is most similar to Adam (Kingma and Ba), the major differences are:
  1. For a two-dimensional AxB weight matrix, Adafactor uses only A+B auxiliary
     parameters to maintain the second-moment estimator, instead of AB.
     This is advantageous on memory-limited systems.  In addition, beta1
     (momentum) is set to zero by default, saving an additional auxiliary
     parameter per weight.  Variables with >=3 dimensions are treated as
     collections of two-dimensional matrices - factorization is over the final
     two dimensions.
  2. Adafactor incorporates "update-clipping" - a scale-invariant analog of
     gradient clipping.  This adds stability
  3. Adafactor does not require an external "learning rate".  By default, it
     incorporates a relative-update-scale schedule, corresponding to
     inverse-square-root learning-rate-decay in ADAM.  We hope this works well
     for most applications.
  ALGORITHM:
  parameter -= absolute_update_scale * clip(grad / grad_scale)
  where:
    absolute_update_scale := relative_update_scale * parameter_scale
    relative_update_scale := min((step_num + 1)**-0.5, 1e-2)
    parameter_scale := max(rms(var)), epsilon2)
    clip(x) := x / max(1.0, rms(x))
    grad_scale := tf.sqrt(v)   (v is the second-moment estimator)
  The second-moment estimator v is maintained in a manner similar to Adam:
  We initialize
  ```
  if var is 2-dimensional:
    v_r <- zeros([num_rows])
    v_c <- zeros([num_cols])
  if var is 0-dimensional or 1-dimensional:
    v <- zeros(shape(var))
  ```
  The update rule is as follows:
  ```
  decay_rate = 1 - (step_num + 1) ^ -0.8
  grad_squared = tf.square(grad) + epsilon1
  if var is 2-dimensional:
    v_r <- decay_rate * v_r + (1 - decay_rate) * reduce_mean(grad_squared, 1)
    v_c <- decay_rate * v_c + (1 - decay_rate) * reduce_mean(grad_squared, 0)
    v = outer_prod(v_r, v_c) / reduce_mean(v_r)
  if var is 0-dimensional or 1-dimensional:
    v <- decay_rate * v + (1 - decay_rate) * grad_squared
  ```
  For variables with >=3 dimensions, we factorize the second-moment accumulator
  over the final 2 dimensions.  See the code for details.
  Several parts of this algorithm are configurable from the initializer.
    multiply_by_parameter_scale:  If True, then compute absolute_update_scale
      as described above.  If False, let absolute_update_scale be the externally
      supplied learning_rate.
    learning_rate: represents relative_update_scale if
      multiply_by_parameter_scale==True, or absolute_update_scale if
      multiply_by_parameter_scale==False.
    decay_rate: Decay rate of the second moment estimator (varies by step_num).
      This should be set to a function such that:
      1-1/(step_num + 1) <= decay_rate(step_num) < 1.0
    beta1: enables momentum, as in Adam.  Uses extra memory if nonzero.
    clipping_threshold: should be >=1.0 or None for no update clipping
    factored: whether to factor the second-moment estimator.  True means
      less memory usage.
  """

  def __init__(self,
               multiply_by_parameter_scale=True,
               learning_rate=None,
               decay_rate=None,
               beta1=0.0,
               clipping_threshold=1.0,
               factored=True,
               simulated_quantize_bits=None,
               parameter_encoding=None,
               use_locking=False,
               name="Adafactor",
               epsilon1=1e-30,
               epsilon2=1e-3):
    """Construct a new Adafactor optimizer.
    See class comment.
    Args:
      multiply_by_parameter_scale: a boolean
      learning_rate: an optional Scalar.
      decay_rate: an optional Scalar.
      beta1: a float value between 0 and 1
      clipping_threshold: an optional float >= 1
      factored: a boolean - whether to use factored second-moment estimator
        for 2d variables
      simulated_quantize_bits: train with simulated quantized parameters
        (experimental)
      parameter_encoding: a ParameterEncoding object to use in the case of
        bfloat16 variables.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "AdafactorOptimizer".
      epsilon1: Regularization constant for squared gradient.
      epsilon2: Regularization constant for parameter scale.
    Raises:
      ValueError: if absolute_update_scale and relative_update_scale_fn are both
        present or both absent.
    """
    super(AdafactorOptimizer, self).__init__(use_locking, name)
    self._multiply_by_parameter_scale = multiply_by_parameter_scale
    if learning_rate is None:
      learning_rate = self._learning_rate_default(multiply_by_parameter_scale)
    self._learning_rate = learning_rate
    if decay_rate is None:
      decay_rate = self._decay_rate_default()
    self._decay_rate = decay_rate
    self._beta1 = beta1
    self._clipping_threshold = clipping_threshold
    self._factored = factored
    self._simulated_quantize_bits = simulated_quantize_bits
    self._parameter_encoding = parameter_encoding
    self._quantization_noise = noise_from_step_num()
    self._epsilon1 = epsilon1
    self._epsilon2 = epsilon2

  def _should_use_factored_second_moment_estimate(self, shape):
    """Should we use a factored second moment estimator.
    Based on the shape of the variable.
    Args:
      shape: a list of integers
    Returns:
      a boolean
    """
    return self._factored and len(shape) >= 2

  def _create_slots(self, var_list):
    for var in var_list:
      shape = var.get_shape().as_list()
      if self._beta1:
        self._zeros_slot(var, "m", self._name)
      if self._should_use_factored_second_moment_estimate(shape):
        r_val = tf.zeros(shape[:-1], dtype=tf.float32)
        c_val = tf.zeros(shape[:-2] + shape[-1:], dtype=tf.float32)
        self._get_or_make_slot(var, r_val, "vr", self._name)
        self._get_or_make_slot(var, c_val, "vc", self._name)
      else:
        v_val = tf.zeros(shape, dtype=tf.float32)
        self._get_or_make_slot(var, v_val, "v", self._name)

  def _apply_dense(self, grad, var):
    return self._resource_apply_dense(grad, var)

  def _apply_sparse(self, grad, var):
    return self._apply_dense(tf.convert_to_tensor(grad), var)

  def _resource_apply_sparse(self, grad, handle, indices):
    return self._resource_apply_dense(
        tf.convert_to_tensor(tf.IndexedSlices(grad, indices, tf.shape(handle))),
        handle)

  def _parameter_scale(self, var):
    """Estimate the scale of the parameters from the current values.
    We include a minimum value of 0.001 to give it a chance to escape 0
    if it was zero-initialized.
    Instead of using the value, we could impute the scale from the shape,
    as initializers do.
    Args:
      var: a variable or Tensor.
    Returns:
      a Scalar
    """
    return tf.maximum(reduce_rms(var), self._epsilon2)

  def _resource_apply_dense(self, grad, handle):
    var = handle
    grad = tf.to_float(grad)
    grad_squared = tf.square(grad) + self._epsilon1
    grad_squared_mean = tf.reduce_mean(grad_squared)
    decay_rate = self._decay_rate
    update_scale = self._learning_rate
    old_val = var
    if var.dtype.base_dtype == tf.bfloat16:
      old_val = tf.to_float(self._parameter_encoding.decode(old_val))
    if self._multiply_by_parameter_scale:
      update_scale *= tf.to_float(self._parameter_scale(old_val))
    # HACK: Make things dependent on grad.
    # This confounds the XLA rewriter and keeps it from fusing computations
    # across different variables.  This fusion is a bad for HBM usage, since
    # it causes the gradients to persist in memory.
    decay_rate += grad_squared_mean * 1e-30
    update_scale += grad_squared_mean * 1e-30
    # END HACK
    mixing_rate = 1.0 - decay_rate
    shape = var.get_shape().as_list()
    updates = []
    if self._should_use_factored_second_moment_estimate(shape):
      grad_squared_row_mean = tf.reduce_mean(grad_squared, -1)
      grad_squared_col_mean = tf.reduce_mean(grad_squared, -2)
      vr = self.get_slot(var, "vr")
      new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
      vc = self.get_slot(var, "vc")
      new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
      vr_update = tf.assign(vr, new_vr, use_locking=self._use_locking)
      vc_update = tf.assign(vc, new_vc, use_locking=self._use_locking)
      updates = [vr_update, vc_update]
      long_term_mean = tf.reduce_mean(new_vr, -1, keepdims=True)
      r_factor = tf.rsqrt(new_vr / long_term_mean)
      c_factor = tf.rsqrt(new_vc)
      x = grad * tf.expand_dims(r_factor, -1) * tf.expand_dims(c_factor, -2)
    else:
      v = self.get_slot(var, "v")
      new_v = decay_rate * v + mixing_rate * grad_squared
      v_update = tf.assign(v, new_v, use_locking=self._use_locking)
      updates = [v_update]
      x = grad * tf.rsqrt(new_v)
    if self._clipping_threshold is not None:
      clipping_denom = tf.maximum(1.0, reduce_rms(x) / self._clipping_threshold)
      x /= clipping_denom
    subtrahend = update_scale * x
    if self._beta1:
      m = self.get_slot(var, "m")
      new_m = self._beta1 * tf.to_float(m) + (1.0 - self._beta1) * subtrahend
      subtrahend = new_m
      new_m = common_layers.cast_like(new_m, var)
      updates.append(tf.assign(m, new_m, use_locking=self._use_locking))
    new_val = tf.to_float(old_val) - subtrahend
    if var.dtype.base_dtype == tf.bfloat16:
      new_val = self._parameter_encoding.encode(
          new_val, self._quantization_noise)
    if self._simulated_quantize_bits:
      new_val = quantization.simulated_quantize(
          var - subtrahend, self._simulated_quantize_bits,
          self._quantization_noise)
    var_update = tf.assign(var, new_val, use_locking=self._use_locking)
    updates = [var_update] + updates
    return tf.group(*updates)

  def _decay_rate_default(self):
    return adafactor_decay_rate_pow(0.8)

  def _learning_rate_default(self, multiply_by_parameter_scale):
    learning_rate = tf.minimum(tf.rsqrt(step_num() + 1.0), 0.01)
    if not multiply_by_parameter_scale:
      learning_rate *= 0.05
    return learning_rate


def adafactor_decay_rate_adam(beta2):
  """Second-moment decay rate like Adam, subsuming the correction factor.
  Args:
    beta2: a float between 0 and 1
  Returns:
    a scalar
  """
  t = tf.to_float(tf.train.get_or_create_global_step()) + 1.0
  decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / (1.0 - tf.pow(beta2, t))
  # decay = tf.cond(tf.equal(t, 1.0), lambda: beta2, lambda: decay)
  return decay


def adafactor_decay_rate_pow(exponent):
  """Second moment decay rate where memory-length grows as step_num^exponent.
  Args:
    exponent: a float between 0 and 1
  Returns:
    a scalar
  """
  return 1.0 - tf.pow((step_num() + 1.0), -exponent)


def step_num():
  return tf.to_float(tf.train.get_or_create_global_step())


def adafactor_optimizer_from_hparams(hparams, lr):
  """Create an Adafactor optimizer based on model hparams.
  Args:
    hparams: model hyperparameters
    lr: learning rate scalar.
  Returns:
    an AdafactorOptimizer
  Raises:
    ValueError: on illegal values
  """
  if hparams.optimizer_adafactor_decay_type == "adam":
    decay_rate = adafactor_decay_rate_adam(
        hparams.optimizer_adafactor_beta2)
  elif hparams.optimizer_adafactor_decay_type == "pow":
    decay_rate = adafactor_decay_rate_pow(
        hparams.optimizer_adafactor_memory_exponent)
  else:
    raise ValueError("unknown optimizer_adafactor_decay_type")
  if hparams.weight_dtype == "bfloat16":
    parameter_encoding = quantization.EighthPowerEncoding()
  else:
    parameter_encoding = None
  return AdafactorOptimizer(
      multiply_by_parameter_scale=(
          hparams.optimizer_adafactor_multiply_by_parameter_scale),
      learning_rate=lr,
      decay_rate=decay_rate,
      beta1=hparams.optimizer_adafactor_beta1,
      clipping_threshold=hparams.optimizer_adafactor_clipping_threshold,
      factored=hparams.optimizer_adafactor_factored,
      simulated_quantize_bits=getattr(
          hparams, "simulated_parameter_quantize_bits", 0),
      parameter_encoding=parameter_encoding,
      use_locking=False,
      name="Adafactor")


def reduce_rms(x):
  return tf.sqrt(tf.reduce_mean(tf.square(x)))

class Model:
    def gelu(self, x):
        # return tf.nn.tanh(x)
        cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf

    def drop(self, input, drop_rate=0.4):
        return tf.nn.dropout(input, self.keep_prob) #这里的drop_rate无用，并且keep_prob是一个占位符

    def layer_norm(self, vec, name=None, axis=2):
        return tf.contrib.layers.layer_norm(vec, scope=name, begin_norm_axis=axis, reuse=None)

    def sepconv(self, state, size, mask):
        state = self.drop(
            tf.layers.separable_conv1d(
                tf.expand_dims(mask, -1) * self.drop(
                    tf.layers.separable_conv1d(state, size, 3, activation=self.gelu, padding="SAME", name="conv")), ##存疑
                size,
                3,
                padding="SAME",
                name="dense_2") + state)
        return state

    def weights_nonzero(self, labels):
        return tf.to_float(tf.not_equal(labels, 0))

    def weights_zero(self, labels):
        return tf.to_float(tf.equal(labels, 0))

    def tree_conv(self, input, matrix, kernel):
        l = [input]
        cur = input #这为啥没转置
        for i in range(kernel - 1):
            cur = tf.transpose(tf.matmul(cur, matrix, transpose_a=True), [0, 2, 1]) # 这个transpose是否需要
            l.append(cur)

        data_flow = tf.stack(l, 2)

        data_flow = self.drop(
            tf.layers.separable_conv2d(data_flow, parameters.EMBEDDING_SIZE, [1, kernel], name="separ_1")
        )
        data_flow = self.gelu(data_flow)
        data_flow = tf.reduce_max(data_flow, 2)

        l = [data_flow]
        cur = data_flow
        for i in range(kernel - 1):
            cur = tf.transpose(tf.matmul(cur, matrix, transpose_a=True), [0, 2, 1])
            l.append(cur)

        data_flow = tf.stack(l, 2)

        data_flow = self.drop(
            tf.layers.separable_conv2d(data_flow, parameters.EMBEDDING_SIZE, [1, kernel], name="separ_2")
        )
        data_flow = tf.reduce_max(data_flow, 2)
        return data_flow


    def position(self, block_index, item_size, embedding_len, min_timescale=1.0, max_timescalse=1.0e4):
        pos = tf.to_float(tf.range(item_size) + block_index)

        num_timescales = embedding_len // 2
        log_timescale_increment = (math.log(float(max_timescalse) / float(min_timescale)) /
                                   tf.maximum(tf.to_float(num_timescales)-1, 1))

        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(pos, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(embedding_len, 2)]])
        signal = tf.reshape(signal, [1, item_size, embedding_len])
        return signal

    # flag 用于表示是否需要antimask
    def head_attention(self, Q, K, V, mask, antimask, flag):
        factor = math.sqrt(int(Q.shape[-1]))

        matrix = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / factor

        mask_temp = antimask if flag else tf.expand_dims(mask, -2)

        matrix = matrix * mask_temp #将不相关的项置零
        matrix = matrix + self.weights_zero(mask_temp) * -1e-18 #将零项置为无穷小

        matrix = tf.nn.softmax(matrix)

        matrix = matrix * mask_temp

        return tf.matmul(matrix, V)

    def multi_head_attention(self, Q, K, V, mask, antimask, flag):
        d = int(Q.shape[-1])

        list_concat = []

        heads = parameters.STATEMENT_TRAN_HEADS

        with tf.variable_scope("multi_head_attention") as scope:
            for i in range(heads):
                head_Q = tf.layers.dense(Q, d // heads, name="head_Q_" + str(i), use_bias=False)
                head_K = tf.layers.dense(K, d // heads, name="head_K_" + str(i), use_bias=False)
                head_V = tf.layers.dense(V, d // heads, name="head_V_" + str(i), use_bias=False)

                list_concat.append(self.head_attention(head_Q, head_K, head_V, mask, antimask, flag))

        result = tf.concat(list_concat, -1)

        result = tf.layers.dense(result, d, name="head", use_bias=False)
        return result

    def gating(self, query, values1, values2):
        d = int(query.shape[-1])
        list_concat = []
        heads = 8
        qd = math.sqrt(float(d//heads))
        for i in range(8):
            convert_query = tf.layers.dense(query, d//heads, name="qkv2headq"+str(i), use_bias=False)

            factor_temp1 = tf.layers.dense(query, d//heads, name="factor1_"+str(i), use_bias=False)
            factor_temp2 = tf.layers.dense(query, d//heads, name="factor2_"+str(i), use_bias=False)

            factor1 = tf.reduce_sum(convert_query * factor_temp1, -1, keep_dims=True)
            factor2 = tf.reduce_sum(convert_query * factor_temp2, -1, keep_dims=True)

            factor_maximum = tf.maximum(factor1, factor2)

            factor1 = tf.exp(factor1 - factor_maximum)
            factor2 = tf.exp(factor2 - factor_maximum)

            contribute_factor1 = factor1 / (factor1 + factor2)
            contribute_factor2 = factor2 / (factor1 + factor2)

            contribution1 = tf.layers.dense(values1, d//heads, name="contribution1_"+str(i), use_bias=False)
            contribution2 = tf.layers.dense(values2, d//heads, name="contribution2_"+str(i), use_bias=False)

            list_concat.append(contribution1 * contribute_factor1 + contribution2 * contribute_factor2)

        concat_head = tf.concat(list_concat, -1)

        result = tf.layers.dense(concat_head, d, name="gating_result", use_bias=False)
        return result

    def combine_input(self, token_embedding, char_embedding):
        conver_embedding = tf.layers.dense(char_embedding, parameters.CHAR_EMBEDDING_SIZE, use_bias=False)
        conver_embedding = tf.reduce_max(conver_embedding, reduction_indices=[-2], keep_dims=False)

        return tf.concat([token_embedding, conver_embedding], -1)

    # block_input = [batch, input_len, embedding_size]
    # em_char = [batch, input_len, char_size, embedding_size]
    # mask = [batch, input_len]
    def input_reader(self, block_index, block_input, em_char, mask):
        item_size = int(block_input.shape[-2])
        embedding_length = int(block_input.shape[-1])

        input_start = block_input + self.position(block_index, item_size, embedding_length) # 这里有点不一样

        # self attention
        state = self.layer_norm(
            self.drop(self.multi_head_attention(input_start, input_start, input_start, mask, None, False) + input_start), "norm1"
        )

        # gating
        with tf.variable_scope("Char_Att", reuse=None):
            state = self.layer_norm(
                self.drop(self.gating(state, state, em_char) + state), name="norm2"
            )

        state *= tf.expand_dims(mask, -1)


        with tf.variable_scope("Dense", reuse=None):
            state = self.sepconv(state, parameters.EMBEDDING_SIZE, mask)

        state = self.layer_norm(state, "norm3")
        return state

    def input_transf(self, input_token, input_char, mask):
        state = input_token

        # input_char[batch, input_len, token_len, embedding]
        char_conv = self.drop(tf.layers.conv2d(input_char, parameters.EMBEDDING_SIZE, [1, parameters.CHAR_PER_TOKEN]))
        char_conv = tf.reduce_max(char_conv, reduction_indices=[-2])

        for i in range(self.max_steps):
            with tf.variable_scope("input_transf" + str(i), reuse=None):
                state = self.input_reader(i, state, char_conv, mask)

        return state

    def ast_reader(self, block_index, block_input, em_input, em_diff, rule_combination, matrix, input_mask, diff_mask, ast_mask):
        rule_size = int(block_input.shape[-2]) #有多少条rule
        embedding_size = int(block_input.shape[-1])

        data_flow = block_input

        data_flow += self.position(block_index, rule_size, embedding_size)

        with tf.variable_scope("ast_reader_self_attention", reuse=None):
            data_flow = self.layer_norm(
                self.drop(
                    self.multi_head_attention(data_flow, data_flow, data_flow, ast_mask, None, False) + data_flow
                ), "norm1"
            )

        with tf.variable_scope("ast_reader_ast_gating", reuse=None):
            data_flow=self.layer_norm(
                self.drop(
                    self.gating(data_flow, data_flow, rule_combination) + data_flow
                ), "norm2"
            )



        with tf.variable_scope("ast_reader_ast_input_attention", reuse=None):
            data_flow = self.layer_norm(
                self.drop(
                    self.multi_head_attention(data_flow, em_input, em_input, input_mask, None, False) + data_flow
                ), "norm3"
            )

        with tf.variable_scope("ast_reader_ast_diff_attention", reuse=None):
            data_flow = self.layer_norm(
                self.drop(self.multi_head_attention(data_flow, em_diff, em_diff, diff_mask, None, False) + data_flow
                ), "norm4"
            )

        data_flow = self.drop(self.tree_conv(data_flow, matrix, 3))
        data_flow = self.layer_norm(data_flow, "norm5")

        return data_flow

    def ast_transf(self, em_input, em_diff, target_rules, target_parents, target_nodes, target_sons, matrix, input_mask, diff_mask, ast_mask):
        em_son_conv = self.drop(tf.layers.conv2d(target_sons, parameters.EMBEDDING_SIZE, [1, parameters.RULE_SON_SIZE]))
        em_son_conv = tf.reduce_max(em_son_conv, reduction_indices=[-2])
        em_son_conv = self.layer_norm(em_son_conv) #为啥这里没有drop

        em_combination = tf.layers.conv2d(tf.stack([target_nodes, target_rules, em_son_conv], -2), parameters.EMBEDDING_SIZE, [1, 3])
        em_combination = tf.reduce_max(em_combination, reduction_indices=[-2])
        em_combination = self.layer_norm(self.drop(em_combination))

        data_flow = em_combination
        for i in range(int(self.max_steps - 1)):
            with tf.variable_scope("ast_transf" + str(i), reuse=None):
                data_flow = self.ast_reader(i, data_flow, em_input, em_diff, em_combination, matrix, input_mask, diff_mask, ast_mask)
        return data_flow

    def query_decoder(self, em_state, em_input, em_ast, em_diff, input_mask, diff_mask, ast_mask):
        data_flow = em_state

        with tf.variable_scope("self_attention", reuse=None):
            data_flow = self.layer_norm(
                self.drop(
                    self.multi_head_attention(data_flow, em_ast, em_ast, ast_mask, self.antimask, True) + data_flow
                ), "norm1"
            )

        with tf.variable_scope("input_attention", reuse=None):
            data_flow = self.layer_norm(
                self.drop(
                    self.multi_head_attention(data_flow, em_input, em_input, input_mask, None, False) + data_flow
                ), "norm2"
            )

        with tf.variable_scope("diff_attention", reuse=None):
            data_flow = self.layer_norm(
                self.drop(
                    self.multi_head_attention(data_flow, em_diff, em_diff, diff_mask, None, False) + data_flow
                )
            )

        data_flow = self.drop(
            tf.layers.dense(
                self.drop(
                    self.gelu(tf.layers.dense(data_flow, parameters.EMBEDDING_SIZE * 4)
                )
            ), parameters.EMBEDDING_SIZE) + data_flow
        )

        data_flow = self.layer_norm(data_flow, "norm3")

        return data_flow

    def query_trans(self, state, em_input, em_ast, em_diff, input_mask, diff_mask, ast_mask):

        data_flow = state
        for i in range(self.max_steps):
        #for i in range(parameters.TARGET_LEN - 1):
            print("decoder " +  str(i))
            with tf.variable_scope("query_trans" + str(i), reuse=None):
                data_flow = self.query_decoder(data_flow, em_input, em_ast, em_diff, input_mask, diff_mask, ast_mask)

        return data_flow

    def copy_multihead_attention(self, query, key, value, anti_mask):
        d = value.shape[2]

        w_q = tf.layers.dense(query, d, name="w_q", use_bias=False)
        w_k = tf.layers.dense(key, d, name="w_l", use_bias=False)

        w_q = tf.expand_dims(w_q, 2)
        w_k = tf.expand_dims(w_k, 1)

        return tf.reduce_sum(tf.layers.dense(tf.tanh(w_q + w_k), 1), -1)

    def generate_anti_mask(self, length):
        mask = np.zeros([length, length])
        for i in range(length):
            for j in range(i + 1):
                mask[i][j] = 1

        return tf.cast(mask, dtype=tf.float32)

    def __init__(self, rule_size, class_size, token_size, char_size):

        self.global_step=tf.Variable(1, trainable=False, name="global_step")

        self.keep_prob = tf.placeholder(tf.float32)
        self.max_steps = 5
        self.antimask = self.generate_anti_mask(parameters.TARGET_LEN)

        self.original_tokens    = tf.placeholder(tf.int32, shape=[None, parameters.INPUT_LEN])
        self.original_chars     = tf.placeholder(tf.int32, shape=[None, parameters.INPUT_LEN, parameters.CHAR_PER_TOKEN])
        self.original_masks     = tf.placeholder(tf.float32, shape=[None, parameters.INPUT_LEN])

        self.diff_tokens        = tf.placeholder(tf.int32, shape=[None, parameters.DIFF_LEN])
        self.diff_chars         = tf.placeholder(tf.int32, shape=[None, parameters.DIFF_LEN, parameters.CHAR_PER_TOKEN])
        self.diff_matrixs       = tf.placeholder(tf.float32, shape=[None, parameters.DIFF_LEN, parameters.DIFF_LEN])
        self.diff_masks         = tf.placeholder(tf.float32, shape=[None, parameters.DIFF_LEN])



        self.target_rules       = tf.placeholder(tf.int32, shape=[None, parameters.TARGET_LEN])
        self.target_parents     = tf.placeholder(tf.int32, shape=[None, parameters.TARGET_LEN])
        self.target_nodes       = tf.placeholder(tf.int32, shape=[None, parameters.TARGET_LEN])
        self.target_sons        = tf.placeholder(tf.int32, shape=[None, parameters.TARGET_LEN, parameters.RULE_SON_SIZE])
        self.target_masks       = tf.placeholder(tf.float32, shape=[None, parameters.TARGET_LEN])
        self.target_matrixs     = tf.placeholder(tf.float32, shape=[None, parameters.TARGET_LEN, parameters.TARGET_LEN])
        self.input_y            = tf.one_hot(self.target_rules, class_size)

        self.rules_embedding = tf.get_variable("rule_embedding", shape=[class_size, parameters.RULE_EMBEDDING_SIZE], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
        self.token_embedding = tf.get_variable("token_embedding", shape=[token_size, parameters.TOKEN_EMBEDDING_SIZE], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
        self.char_embedding  = tf.get_variable("char_embeding", shape=[char_size, parameters.CHAR_EMBEDDING_SIZE], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))

        self.loss_masks = tf.placeholder(tf.float32, shape=[None, parameters.TARGET_LEN])
        loss_masks = self.loss_masks


        #embedding for code
        with tf.variable_scope("input_transform", reuse=None):
            em_input_original_tokens = tf.nn.embedding_lookup(self.token_embedding, self.original_tokens)
            em_input_original_chars = tf.nn.embedding_lookup(self.char_embedding, self.original_chars)
            input_trans_original = self.input_transf(em_input_original_tokens, em_input_original_chars, self.original_masks)

        with tf.variable_scope("diff_transform", reuse=None):
            em_input_diff_tokens = tf.nn.embedding_lookup(self.token_embedding, self.diff_tokens)
            em_input_diff_chars = tf.nn.embedding_lookup(self.char_embedding, self.diff_chars)
            em_input_diff        = self.input_transf(em_input_diff_tokens, em_input_diff_chars, self.diff_masks)
            em_input_diff        = self.tree_conv(em_input_diff, self.diff_matrixs, 3)



        em_target_rules = tf.nn.embedding_lookup(self.rules_embedding, self.target_rules)
        em_target_parents = tf.nn.embedding_lookup(self.token_embedding, self.target_parents)
        em_target_nodes = tf.nn.embedding_lookup(self.token_embedding, self.target_nodes)
        em_target_sons = tf.nn.embedding_lookup(self.token_embedding, self.target_sons)

        with tf.variable_scope("ast_transform", reuse=None):
            em_ast = self.ast_transf(input_trans_original, em_input_diff, em_target_rules, em_target_parents, em_target_nodes, em_target_sons,
                                     self.target_matrixs, self.original_masks, self.diff_masks, self.target_masks)

        with tf.variable_scope("target_transform", reuse=None):
            em_query_decoder = self.query_trans(em_target_nodes, input_trans_original, em_ast, em_input_diff, self.original_masks, self.diff_masks, self.target_masks)

        em_query_decoder = tf.layers.dense(em_query_decoder, rule_size)

        self.y_result = tf.nn.softmax(em_query_decoder)

        copy_result = self.copy_multihead_attention(em_query_decoder, em_target_rules, em_target_rules, self.antimask)
        copy_result *= tf.expand_dims(self.target_masks, 1)
        copy_result = tf.exp(copy_result - tf.reduce_max(copy_result, -1))
        copy_result *= tf.expand_dims(self.target_masks, 1)
        copy_result = copy_result / tf.reduce_sum(copy_result, reduction_indices=[-1], keep_dims=True)

        P_gen = tf.layers.dense(em_query_decoder, 1, activation=tf.nn.sigmoid)

        self.y_result *= P_gen
        copy_result *= (1 - P_gen)

        self.y_result = tf.concat([self.y_result, copy_result], -1)

        self.max_res = tf.argmax(self.y_result, 2)

        self.correct_prediction = tf.cast(tf.equal(tf.arg_max(self.y_result, 2), tf.arg_max(self.input_y, 2)), tf.float32) * loss_masks
        self.accuracy = tf.reduce_mean(self.correct_prediction * tf.expand_dims( parameters.TARGET_LEN / tf.reduce_sum(loss_masks, reduction_indices=[1]), -1))
        self.cross_entropy = tf.reduce_sum(
            tf.reduce_sum(
                loss_masks * -tf.reduce_sum(
                    self.input_y * tf.log(tf.clip_by_value(self.y_result, 1e-10, 1.0)), reduction_indices=[2]
                ),
                reduction_indices=[1]
            )
        ) / tf.reduce_sum(loss_masks, reduction_indices=[0, 1])

        tf.add_to_collection("losses", self.cross_entropy)

        self.loss = self.cross_entropy
        self.params = [param for param in tf.trainable_variables()]
        global_step = tf.cast(self.global_step, dtype=tf.float32)
        self.optim = AdafactorOptimizer().minimize(self.loss , global_step=self.global_step)