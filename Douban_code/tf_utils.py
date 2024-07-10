from __future__ import division

import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import random
import h5py

dtype = tf.float32
minval = -0.01
maxval = 0.01
mean = 0
stddev = 0.001

def matrix_coefficient(mat_a, mat_b):
    mean1 = tf.reduce_mean(mat_a)
    mean2 = tf.reduce_mean(mat_b)
    mat_a = mat_a - mean1
    mat_b = mat_b - mean2
    numerator = tf.reduce_sum(mat_a * mat_b)
    square1 = tf.reduce_sum(tf.square(mat_a))
    square2 = tf.reduce_sum(tf.square(mat_b))
    denominator = tf.clip_by_value(tf.sqrt(square1 * square2), 1e-6, tf.reduce_max(tf.sqrt(square1 * square2)))
    return numerator / denominator

def matrix_cosine_similarity(mat_a, mat_b):
    inners = tf.reduce_sum(mat_a * mat_b, axis=-2)
    lens1 = tf.sqrt(tf.reduce_sum(tf.square(mat_a), axis=-2))
    lens2 = tf.sqrt(tf.reduce_sum(tf.square(mat_b), axis=-2))
    lens = tf.clip_by_value(lens1 * lens2, 1e-4, tf.reduce_max(lens1 * lens2))
    return inners / lens        # mat_a: [M, N] -> [N,]

def cosine(emb1, emb2):
    # emb1, emb2: [B, embed_size]
    inner = tf.reduce_sum(emb1 * emb2, axis=-1)
    l1 = tf.sqrt(tf.reduce_sum(emb1 * emb1, axis=-1))
    l1 = tf.clip_by_value(l1, 1e-8, tf.reduce_max(l1))
    l2 = tf.sqrt(tf.reduce_sum(emb2 * emb2, axis=-1))
    l2 = tf.clip_by_value(l2, 1e-8, tf.reduce_max(l2))
    cosines = inner / l1 * l2
    return cosines

def get_variable(init_type='xavier', shape=None, name=None, minval=minval, maxval=maxval, mean=mean,
                 stddev=stddev, dtype=dtype, ):
    if type(init_type) is str:
        init_type = init_type.lower()
    if init_type == 'tnormal':
        return tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype), name=name)
    elif init_type == 'uniform':
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'normal':
        return tf.Variable(tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype), name=name)
    elif init_type == 'xavier':
        maxval = np.sqrt(6. / np.sum(shape))
        minval = -maxval
        print(name, 'initialized from:', minval, maxval, " shape:", shape)
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'xavier_out':
        maxval = np.sqrt(3. / shape[1])
        minval = -maxval
        print(name, 'initialized from:', minval, maxval)
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'xavier_in':
        maxval = np.sqrt(3. / shape[0])
        minval = -maxval
        print(name, 'initialized from:', minval, maxval)
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'zero':
        return tf.Variable(tf.zeros(shape=shape, dtype=dtype), name=name)
    elif init_type == 'one':
        return tf.Variable(tf.ones(shape=shape, dtype=dtype), name=name)
    elif init_type == 'identity' and len(shape) == 2 and shape[0] == shape[1]:
        return tf.Variable(tf.diag(tf.ones(shape=shape[0], dtype=dtype)), name=name)
    elif 'int' in init_type.__class__.__name__ or 'float' in init_type.__class__.__name__:
        return tf.Variable(tf.ones(shape=shape, dtype=dtype) * init_type, name=name)


def selu(x):
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def activate(weights, act_type):
    if type(act_type) is str:
        act_type = act_type.lower()
    if act_type == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif act_type == 'softmax':
        return tf.nn.softmax(weights)
    elif act_type == 'relu':
        return tf.nn.relu(weights)
    elif act_type == 'tanh':
        return tf.nn.tanh(weights)
    elif act_type == 'elu':
        return tf.nn.elu(weights)
    elif act_type == 'selu':
        return selu(weights)
    elif act_type == 'none':
        return weights
    else:
        return weights


def get_optimizer(opt_algo):
    opt_algo = opt_algo.lower()
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer
    elif opt_algo == 'moment':
        return tf.train.MomentumOptimizer
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer
    elif opt_algo == 'gd' or opt_algo == 'sgd':
        return tf.train.GradientDescentOptimizer
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer
    else:
        return tf.train.GradientDescentOptimizer


def get_loss(loss_func):
    loss_func = loss_func.lower()
    if loss_func == 'weight' or loss_func == 'weighted':
        return tf.nn.weighted_cross_entropy_with_logits
    elif loss_func == 'sigmoid':
        return tf.nn.sigmoid_cross_entropy_with_logits
    elif loss_func == 'softmax':
        return tf.nn.softmax_cross_entropy_with_logits


def check(x):
    try:
        return x is not None and x is not False and float(x) > 0
    except TypeError:
        return True


def get_l1_loss(params, variables):
    _loss = None
    with tf.name_scope('l1_loss'):
        for p, v in zip(params, variables):
            print('add l1', p, v)
            if not type(p) is list:
                if check(p):
                    if type(v) is list:
                        for _v in v:
                            if _loss is None:
                                _loss = tf.contrib.layers.l1_regularizer(p)(_v)# tf.nn.l1_loss(_v)
                            else:
                                _loss += tf.contrib.layers.l1_regularizer(p)(_v)
                    else:
                        if _loss is None:
                            _loss = tf.contrib.layers.l1_regularizer(p)(v)
                        else:
                            _loss += tf.contrib.layers.l1_regularizer(p)(v)
            else:
                for _lp, _lv in zip(p, v):
                    if _loss is None:
                        _loss = tf.contrib.layers.l1_regularizer(_lp)(_lv)
                    else:
                        _loss += tf.contrib.layers.l1_regularizer(_lp)(_lv)
    return _loss


def get_l2_loss(params, variables):
    _loss = None
    with tf.name_scope('l2_loss'):
        for p, v in zip(params, variables):
            print('add l2', p, v)
            if not type(p) is list:
                if check(p):
                    if type(v) is list:
                        for _v in v:
                            if _loss is None:
                                _loss = p * tf.nn.l2_loss(_v)
                            else:
                                _loss += p * tf.nn.l2_loss(_v)
                    else:
                        if _loss is None:
                            _loss = p * tf.nn.l2_loss(v)
                        else:
                            _loss += p * tf.nn.l2_loss(v)
            else:
                for _lp, _lv in zip(p, v):
                    if _loss is None:
                        _loss = _lp * tf.nn.l2_loss(_lv)
                    else:
                        _loss += _lp * tf.nn.l2_loss(_lv)
    return _loss


def normalize(norm, x, scale):
    if norm:
        return x * scale
    else:
        return x


def mul_noise(noisy, x, training=None):
    if check(noisy) and training is not None:
        with tf.name_scope('mul_noise'):
            noise = tf.truncated_normal(
                shape=tf.shape(x),
                mean=1.0, stddev=noisy)
            return tf.where(
                training,
                tf.multiply(x, noise),
                x)
    else:
        return x


def add_noise(noisy, x, training):
    if check(noisy):
        with tf.name_scope('add_noise'):
            noise = tf.truncated_normal(
                shape=tf.shape(x),
                mean=0, stddev=noisy)
            return tf.where(
                training,
                x + noise,
                x)
    else:
        return x


def drop_out(training, keep_probs, ):
    with tf.name_scope('drop_out'):
        keep_probs = tf.where(training,
                              keep_probs,
                              np.ones_like(keep_probs),
                              name='keep_prob')
    return keep_probs


def linear(xw):
    with tf.name_scope('linear'):
        l = tf.squeeze(tf.reduce_sum(xw, 1))
    return l


def output(x):
    with tf.name_scope('output'):
        if type(x) is list:
            logits = sum(x)
        else:
            logits = x
        outputs = tf.nn.sigmoid(logits)
    return logits, outputs


def layer_normalization(x, reduce_dim=1, out_dim=None, scale=None, bias=None):
    if type(reduce_dim) is int:
        reduce_dim = [reduce_dim]
    if type(out_dim) is int:
        out_dim = [out_dim]
    with tf.name_scope('layer_norm'):
        layer_mean, layer_var = tf.nn.moments(x, reduce_dim, keep_dims=True)
        x = (x - layer_mean) / tf.sqrt(layer_var)
        if scale is not False:
            scale = scale if scale is not None else tf.Variable(tf.ones(out_dim), dtype=dtype, name='g')
        if bias is not False:
            bias = bias if bias is not None else tf.Variable(tf.zeros(out_dim), dtype=dtype, name='b')
        if scale is not False and bias is not False:
            return x * scale + bias
        elif scale is not False:
            return x * scale
        elif bias is not False:
            return x + bias
        else:
            return x

def row_col_fetch(xv_embed, num_inputs):
    """
    for field-aware embedding
    :param xv_embed: batch * num * (num - 1) * k
    :param num_inputs: num
    :return:
    """
    rows = []
    cols = []
    for i in range(num_inputs - 1):
        for j in range(i + 1, num_inputs):
            rows.append([i, j - 1])
            cols.append([j, i])
    with tf.name_scope('lookup'):
        # batch * pair * k
        xv_p = tf.transpose(
            # pair * batch * k
            tf.gather_nd(
                # num * (num - 1) * batch * k
                tf.transpose(xv_embed, [1, 2, 0, 3]),
                rows),
            [1, 0, 2])
        xv_q = tf.transpose(
            tf.gather_nd(
                tf.transpose(xv_embed, [1, 2, 0, 3]),
                cols),
            [1, 0, 2])
    return xv_p, xv_q


def row_col_expand(xv_embed, num_inputs):
    """
    for universal embedding and field-aware param
    :param xv_embed: batch * num * k
    :param num_inputs:
    :return:
    """
    rows = []
    cols = []
    for i in range(num_inputs - 1):
        for j in range(i + 1, num_inputs):
            rows.append(i)
            cols.append(j)
    with tf.name_scope('lookup'):
        # batch * pair * k
        xv_p = tf.transpose(
            # pair * batch * k
            tf.gather(
                # num * batch * k
                tf.transpose(
                    xv_embed, [1, 0, 2]),
                rows),
            [1, 0, 2])
        # batch * pair * k
        xv_q = tf.transpose(
            tf.gather(
                tf.transpose(
                    xv_embed, [1, 0, 2]),
                cols),
            [1, 0, 2])
    return xv_p, xv_q


def bin_mlp(layer_sizes, layer_acts, layer_keeps, h, training=True, reuse=False, name=None):
    layer_kernels = []
    layer_biases = []
    nn_h = []
    with tf.name_scope(str(name)):
        for i in range(len(layer_sizes)):
            h = tf.layers.dense(h, layer_sizes[i], activation=None, name='mlp_%d' % i + str(name), reuse=reuse)
            if i < len(layer_sizes) - 1:
                h = tf.layers.batch_normalization(h, training=training, name='mlp_bn_%d' % i + str(name), reuse=reuse)
            h = tf.nn.dropout(
                activate(
                    h, layer_acts[i]),
                layer_keeps[i])
            nn_h.append(h)
    return h, layer_kernels, layer_biases, nn_h

def bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h, input_dim=None, init='xavier', training=True,
              reuse=False, name=None):
    layer_kernels = []
    layer_biases = []
    nn_h = []
    for i in range(len(layer_sizes)):
        with tf.name_scope('hidden_%d' % i + str(name)):
            if i == 0:
                in_dim = input_dim
                out_dim = layer_sizes[0]
            else:
                in_dim = layer_sizes[i-1]
                out_dim = layer_sizes[i]

            Wi = tf.Variable(tf.truncated_normal([in_dim[1], out_dim], stddev=0.1), name='mlp_%d' % i + str(name))
            bi = tf.Variable(tf.constant(0.1, shape=[out_dim]), name='mlp_%d' % i + str(name))

            # Wi = get_variable(init, name='mlp_%d' % i + str(name), shape=[in_dim, out_dim])
            # bi = get_variable('zero', name='mlp_b%d' % i + str(name), shape=[1, out_dim])
            h = tf.matmul(h, Wi)
            h = tf.nn.bias_add(h, bi)
            if i < len(layer_sizes) - 1:
                h = tf.layers.batch_normalization(h, training=training, name='mlp_bn_%d' % i + str(name), reuse=reuse)
            h = tf.nn.dropout(
                activate(
                    h, layer_acts[i]),
                layer_keeps[i])
            layer_kernels.append(Wi)
            layer_biases.append(bi)
            nn_h.append(h)
    return h, layer_kernels, layer_biases, nn_h


def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    #
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    try:
        group_auc = float(total_auc) / impression_total
        # group_auc = round(group_auc, 4)
        return group_auc
    except:
        return None


def init_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    # tf.random.set_seed(random_seed)


def attention(queries, keys, keys_length, name, training, attention_size=[160, 80], reuse=False):
    '''
    queries:     [Batch, K]
    keys:        [Batch, history length, K]
    keys_length: [Batch]
    '''
    print("attention_size:", attention_size)
    queries_hidden_units = queries.get_shape().as_list()[-1]  # K
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  # B, K * H
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  # B, H, K
    din_all = tf.concat([queries, keys, queries-keys, queries*keys, queries+keys], axis=-1)  # B, H, 5K
    d_layer_1_all = tf.layers.dense(din_all, attention_size[0], activation=None, name='f1_att' + str(name), reuse=reuse)
    d_layer_1_all = tf.layers.batch_normalization(d_layer_1_all, training=training, name='bn_attention_f1'+str(name), reuse=reuse)
    d_layer_1_all = tf.nn.relu(d_layer_1_all)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, attention_size[1], activation=None, name='f2_att' + str(name), reuse=reuse)
    d_layer_2_all = tf.layers.batch_normalization(d_layer_2_all, training=training, name='bn_attention_f2' + str(name), reuse=reuse)
    d_layer_2_all = tf.nn.relu(d_layer_2_all)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att'+str(name), reuse=reuse)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # B, 1, H
    outputs = d_layer_3_all
    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, H]  set True according to key_length
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, H]
    # paddings = tf.ones_like(outputs) * (-float("inf"))#  (-2 ** 32 + 1)
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, H]
    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, H]
    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, K]
    return outputs


def attention_without_para(queries, keys, keys_length, temperature):  # for multi-head attention
    # keys: batch, 3, his, k or batch1, batch2, 3, his, k
    kq_inter = keys * queries
    atten = tf.reduce_sum(kq_inter, axis=-1)  # batch, 3, his
    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[-2])  # batch, 3, 30
    paddings = tf.ones_like(atten) * (-2 ** 32 + 1)  # (-2 ** 32 + 1)
    outputs = tf.where(key_masks, atten, paddings)  # batch, 3, 30
    # outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
    outputs = outputs / temperature
    outputs = tf.nn.softmax(outputs)
    outputs = tf.expand_dims(outputs, -2)  # batch, 3, 1, 30
    outputs = tf.matmul(outputs, keys)
    outputs = tf.squeeze(outputs, -2)
    # print(outputs.shape)
    return outputs


def attention1_weight(queries, keys, keys2, keys_length, name, training, temperature, is_softmax=True, reuse=False, attention_size=[160, 80]):
    print("get attention1 weight")
    queries_hidden_units = queries.get_shape().as_list()[-1]  # K
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  # B, K * H
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  # B, H, K
    if keys2 is not None:
        din_all = tf.concat([queries, keys, keys2, queries*keys, queries*keys2, keys*keys2], axis=-1)  # B, H, 4K
    else:
        din_all = tf.concat([queries, keys, queries * keys], axis=-1)  # B, H, 4K
    d_layer_1_all = tf.layers.dense(din_all, attention_size[0], activation=None, name='f1_att' + str(name), reuse=reuse)
    with tf.variable_scope('f1_att' + str(name), reuse=True):
        w111 = tf.get_variable('kernel')
    d_layer_1_all = tf.layers.batch_normalization(d_layer_1_all, training=training, name='bn_attention_f1'+str(name), reuse=reuse)
    d_layer_1_all = tf.nn.relu(d_layer_1_all)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, attention_size[1], activation=None, name='f2_att' + str(name), reuse=reuse)
    d_layer_2_all = tf.layers.batch_normalization(d_layer_2_all, training=training, name='bn_attention_f2' + str(name), reuse=reuse)
    d_layer_2_all = tf.nn.relu(d_layer_2_all)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att'+str(name), reuse=reuse)
    weight = tf.squeeze(d_layer_3_all, 2)
    # key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, H]  set True according to key_length
    if is_softmax:
        # paddings = tf.ones_like(weight) * (-2 ** 15 + 1)  # (-2 ** 32 + 1)
        # weight = tf.where(key_masks, weight, paddings)  # batch, 3, 30
        weight = weight / temperature
        weight = tf.nn.softmax(weight)
        return weight, w111
    else:
        # paddings = tf.zeros_like(weight)
        # paddings = tf.ones_like(weight) * (-2 ** 15 + 1)  # (-2 ** 32 + 1)
        # weight = tf.where(key_masks, weight, paddings)
        weight = tf.nn.sigmoid(weight)
        return weight, w111


def attention_most_similar_weight(queries, keys, keys_length, temperature, is_softmax=True):  # for multi-head attention
    # queries: [batch, k]  keys: [batch, len, k]  keys_length: [batch]
    kq_inter = keys * tf.expand_dims(queries, axis=1)
    atten = tf.reduce_sum(kq_inter, axis=-1)  # batch, len
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[-2])  # batch, len
    if is_softmax:
        paddings = tf.ones_like(atten) * (-2 ** 15 + 1)  # (-2 ** 32 + 1)
        weight = tf.where(key_masks, atten, paddings)  # batch, 3, 30
        weight = weight / temperature
        weight = tf.nn.softmax(weight)
        return weight
    else:
        paddings = tf.zeros_like(atten)
        weight = tf.where(key_masks, atten, paddings)
        return weight


def attention_most_unsimilar_weight(queries, keys, keys_length, temperature, is_softmax=True):  # for multi-head attention
    # queries: [batch, k]  keys: [batch, len, k]  keys_length: [batch]
    kq_inter = keys * tf.expand_dims(queries, axis=1)
    atten = -tf.reduce_sum(kq_inter, axis=-1)  # batch, len
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[-2])  # batch, len
    if is_softmax:
        paddings = tf.ones_like(atten) * (-2 ** 15 + 1)  # (-2 ** 32 + 1)
        weight = tf.where(key_masks, atten, paddings)  # batch, 3, 30
        weight = weight / temperature
        weight = tf.nn.softmax(weight)
        return weight
    else:
        paddings = tf.zeros_like(atten)
        weight = tf.where(key_masks, atten, paddings)
        return weight


def self_attention(x, x_length, name, reuse, attention_size):
    print("self attention")
    # keys batch, 3, his, k or batch1, batch2, 3, his, k
    Q = tf.layers.dense(x, attention_size, name=name + 'att_Q', reuse=reuse)
    K = tf.layers.dense(x, attention_size, name=name + 'att_K', reuse=reuse)
    V = tf.layers.dense(x, attention_size, name=name + 'att_V', reuse=reuse)
    score = tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2]))
    # Mask
    key_masks = tf.sequence_mask(x_length, tf.shape(x)[-2])  # batch, 3, 30
    key_masks = tf.expand_dims(key_masks, -1)  #
    key_masks = tf.tile(key_masks, [1, 1, 1, tf.shape(x)[-2]])
    paddings = tf.ones_like(score) * (-2 ** 32 + 1)  # (-2 ** 32 + 1)
    score = tf.where(key_masks, score, paddings)  # [B, 1, H]
    score = score / (K.get_shape().as_list()[-1] ** 0.5)
    score = tf.nn.softmax(score, axis=-1)
    outputs = tf.matmul(score, V)
    return outputs

def from_hdf(path, key, start=0, stop='last'):
    with h5py.File(str(path), 'r') as hf:
        if stop == 'last':
            data = hf[key][start:]
        else:
            data = hf[key][start:stop]
    return data


def load_hdf_datasets(path, keys=None, start=0, stop='last'):
    if keys is None:
        x = from_hdf(str(path), 'x', start=start, stop=stop)
        y = from_hdf(str(path), 'y', start=start, stop=stop)
        userid = from_hdf(str(path), 'userid', start=start, stop=stop)
        visual_property = from_hdf(str(path), 'visual_property', start=start, stop=stop)
        index = from_hdf(str(path), 'index', start=start, stop=stop)
        return x, visual_property, y, userid, index
    else:
        res = []
        for key in keys:
            res.append(from_hdf(str(path), str(key), start=start, stop=stop))
        return res


def KL_divergence(p, q, norm=False):
    if norm:
        p = tf.nn.l2_normalize(p, axis=-1)
        q = tf.nn.l2_normalize(q, axis=-1)
    p = tf.clip_by_value(p, +1e-8, 1 - 1e-8)
    q = tf.clip_by_value(q, +1e-8, 1 - 1e-8)
    log_p = tf.log(p)
    log_q = tf.log(q)
    neg_ent = tf.reduce_sum(p * log_p, axis=-1)
    neg_cross_ent = tf.reduce_sum(p * log_q, axis=-1)
    kl = neg_ent - neg_cross_ent
    return kl


def JS_divergence(p, q):
    return 0.5 * KL_divergence(p, 0.5*(p+q)) + 0.5 * KL_divergence(q, 0.5*(p+q))


