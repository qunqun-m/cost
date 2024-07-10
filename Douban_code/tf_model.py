# coding: utf-8
from __future__ import print_function

import math
from abc import abstractmethod
import tensorflow as tf

from tf_utils import drop_out, output, get_l1_loss, \
    bin_mlp, get_variable, get_l2_loss, attention, activate, matrix_coefficient, matrix_cosine_similarity, \
    bin_mlp_2
import numpy as np

dtype = tf.float32
epsilon = 1e-4


class Model:
    inputs = None
    outputs = None
    logits = None
    labels = None
    learning_rate = None
    loss = None
    l1_loss = None
    l2_loss = None
    optimizer = None
    grad = None

    @abstractmethod
    def compile(self, **kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__


class SingleTower(Model):
    def __init__(self, init='xavier', user_max_id=None, src_item_max_id=None, trg_item_max_id=None,
                 embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, batch_norm=False, layer_norm=False,
                 l1_w=None, l1_v=None, layer_l1=None, user_his_len=None, hist_type=None, cotrain=False):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.layer_l2 = layer_l2
        self.l1_w = l1_w
        self.layer_l1 = layer_l1
        self.l1_v = l1_v
        self.hist_type = hist_type
        self.embed_size = embed_size
        self.user_his_len = user_his_len
        self.cotrain = cotrain

        with tf.name_scope('input'):
            self.user_id = tf.placeholder(tf.int32, [None], name='user_id')
            self.item_id = tf.placeholder(tf.int32, [None], name='item_id')
            self.history1 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items1')
            self.history_len1 = tf.placeholder(tf.int32, [None], name='history_items_len1')
            self.history2 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items2')
            self.history_len2 = tf.placeholder(tf.int32, [None], name='history_items_len2')
            self.labels = tf.placeholder(tf.float32, [None], name='label')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.is_src = tf.placeholder(dtype=tf.bool, name='whether_is_source')

        layer_keeps = drop_out(self.training, layer_keeps)
        v_user = get_variable(init, name='v_user', shape=[user_max_id, embed_size])
        v_item = get_variable(init, name='v_item', shape=[src_item_max_id + trg_item_max_id, embed_size])
        b = get_variable('zero', name='b', shape=[1])

        xv1 = tf.gather(v_user, self.user_id)
        xv2 = tf.gather(v_item, self.item_id)

        hist1 = tf.gather(v_item, self.history1)
        hist2 = tf.gather(v_item, self.history2)

        self.a = xv1

        if hist_type == 1:
            # source
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history1 = user_history1 / tf.expand_dims(tf.cast(self.history_len1, tf.float32), 1)

            # target
            user_history2 = tf.reduce_sum(hist2, axis=-2)
            user_history2 = user_history2 / tf.expand_dims(tf.cast(self.history_len2, tf.float32), 1)
        elif hist_type == 2:
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history2 = tf.reduce_sum(hist2, axis=-2)
        elif hist_type == 3:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)
        elif hist_type == 4:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention1', self.training), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention2', self.training), 1)

        if hist_type > 0:
            user_feat1 = tf.concat([xv1, user_history1], axis=-1)
            user_feat2 = tf.concat([xv1, user_history2], axis=-1)
        else:
            user_feat1 = xv1
            user_feat2 = xv1

        h1 = tf.concat([user_feat1, xv2], axis=-1)
        h2 = tf.concat([user_feat2, xv2], axis=-1)
        if cotrain:
            h1, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                   training=self.training, name='mlp',
                                                                   reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                   training=self.training, name='mlp',
                                                                   reuse=tf.AUTO_REUSE)
        else:
            h1, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                   training=self.training, name='mlp1',
                                                                   reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                   training=self.training, name='mlp2',
                                                                   reuse=tf.AUTO_REUSE)
        h1 = tf.squeeze(h1)
        h2 = tf.squeeze(h2)
        self.logits1, self.outputs1 = output([h1, b])
        self.logits2, self.outputs2 = output([h2, b])

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.entropy1 = loss(logits=self.logits1, targets=self.labels, pos_weight=pos_weight)
                self.loss1 = tf.reduce_mean(self.entropy1)

                self.entropy2 = loss(logits=self.logits2, targets=self.labels, pos_weight=pos_weight)
                self.loss2 = tf.reduce_mean(self.entropy2)

                _loss_ = self.loss
                self.optimizer1 = optimizer.minimize(loss=self.loss1,
                                                     global_step=global_step)
                self.optimizer2 = optimizer.minimize(loss=self.loss2,
                                                     global_step=global_step)


class Model_NegSampling_positive(Model):
    def __init__(self, init='xavier', user_max_id=None, src_item_max_id=None, trg_item_max_id=None,
                 embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, batch_norm=False, layer_norm=False,
                 l1_w=None, l1_v=None, layer_l1=None, user_his_len=None, hist_type=None, cotrain=False):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.layer_l2 = layer_l2
        self.l1_w = l1_w
        self.layer_l1 = layer_l1
        self.l1_v = l1_v
        self.hist_type = hist_type
        self.embed_size = embed_size
        self.user_his_len = user_his_len
        self.cotrain = cotrain

        with tf.name_scope('input'):
            self.user_id = tf.placeholder(tf.int32, [None], name='user_id')
            self.item_id = tf.placeholder(tf.int32, [None], name='item_id')
            self.history1 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items1')
            self.history_len1 = tf.placeholder(tf.int32, [None], name='history_items_len1')
            self.history2 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items2')
            self.history_len2 = tf.placeholder(tf.int32, [None], name='history_items_len2')
            self.labels = tf.placeholder(tf.float32, [None], name='label')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.is_src = tf.placeholder(dtype=tf.bool, name='whether_is_source')

        layer_keeps = drop_out(self.training, layer_keeps)
        v_user = get_variable(init, name='v_user', shape=[user_max_id, embed_size])
        v_item = get_variable(init, name='v_item', shape=[src_item_max_id + trg_item_max_id, embed_size])
        b = get_variable('zero', name='b', shape=[1])

        xv1 = tf.gather(v_user, self.user_id)
        xv2 = tf.gather(v_item, self.item_id)

        hist1 = tf.gather(v_item, self.history1)
        hist2 = tf.gather(v_item, self.history2)

        self.a = xv1

        if hist_type == 1:
            # source
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history1 = user_history1 / tf.expand_dims(tf.cast(self.history_len1, tf.float32), 1)

            # target
            user_history2 = tf.reduce_sum(hist2, axis=-2)
            user_history2 = user_history2 / tf.expand_dims(tf.cast(self.history_len2, tf.float32), 1)
        elif hist_type == 2:
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history2 = tf.reduce_sum(hist2, axis=-2)
        elif hist_type == 3:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)
        elif hist_type == 4:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention1', self.training), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention2', self.training), 1)

        if hist_type > 0:
            user_feat1 = tf.concat([xv1, user_history1], axis=-1)
            user_feat2 = tf.concat([xv1, user_history2], axis=-1)
        else:
            user_feat1 = xv1
            user_feat2 = xv1

        h1 = tf.concat([user_feat1, xv2], axis=-1)
        h2 = tf.concat([user_feat2, xv2], axis=-1)
        if cotrain:
            h1, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                   training=self.training, name='mlp',
                                                                   reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                   training=self.training, name='mlp',
                                                                   reuse=tf.AUTO_REUSE)
        else:
            h1, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                   training=self.training, name='mlp1',
                                                                   reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                   training=self.training, name='mlp2',
                                                                   reuse=tf.AUTO_REUSE)
        h1 = tf.squeeze(h1)
        h2 = tf.squeeze(h2)
        self.logits1, self.outputs1 = output([h1, b])
        self.logits2, self.outputs2 = output([h2, b])

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.entropy1 = loss(logits=self.logits1, targets=self.labels, pos_weight=pos_weight)
                self.loss1 = tf.reduce_mean(self.entropy1)

                self.entropy2 = loss(logits=self.logits2, targets=self.labels, pos_weight=pos_weight)
                self.loss2 = tf.reduce_mean(self.entropy2)

                _loss_ = self.loss
                self.optimizer1 = optimizer.minimize(loss=self.loss1,
                                                     global_step=global_step)
                self.optimizer2 = optimizer.minimize(loss=self.loss2,
                                                     global_step=global_step)


class CDTrnas(Model):
    def __init__(self, init='xavier', user_max_id=None, src_item_max_id=None, trg_item_max_id=None,
                 embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, batch_norm=False, layer_norm=False,
                 l1_w=None, l1_v=None, layer_l1=None, user_his_len=None, hist_type=None, cotrain=False):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.layer_l2 = layer_l2
        self.l1_w = l1_w
        self.layer_l1 = layer_l1
        self.l1_v = l1_v
        self.hist_type = hist_type
        self.embed_size = embed_size
        self.user_his_len = user_his_len
        self.cotrain = cotrain

        with tf.name_scope('input'):
            self.user_id = tf.placeholder(tf.int32, [None], name='user_id')
            self.item_id = tf.placeholder(tf.int32, [None], name='item_id')
            self.history1 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items1')
            self.history_len1 = tf.placeholder(tf.int32, [None], name='history_items_len1')
            self.history2 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items2')
            self.history_len2 = tf.placeholder(tf.int32, [None], name='history_items_len2')
            self.labels = tf.placeholder(tf.float32, [None], name='label')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.is_src = tf.placeholder(dtype=tf.bool, name='whether_is_source')

        layer_keeps = drop_out(self.training, layer_keeps)
        v_user = get_variable(init, name='v_user', shape=[user_max_id, embed_size])
        v_item = get_variable(init, name='v_item', shape=[src_item_max_id + trg_item_max_id, embed_size])
        b = get_variable('zero', name='b', shape=[1])

        xv1 = tf.gather(v_user, self.user_id)
        xv2 = tf.gather(v_item, self.item_id)

        hist1 = tf.gather(v_item, self.history1)
        hist2 = tf.gather(v_item, self.history2)

        self.a = xv1

        if hist_type == 1:
            # source
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history1 = user_history1 / tf.expand_dims(tf.cast(self.history_len1, tf.float32), 1)

            # target
            user_history2 = tf.reduce_sum(hist2, axis=-2)
            user_history2 = user_history2 / tf.expand_dims(tf.cast(self.history_len2, tf.float32), 1)
        elif hist_type == 2:
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history2 = tf.reduce_sum(hist2, axis=-2)
        elif hist_type == 3:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)
        elif hist_type == 4:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention1', self.training), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention2', self.training), 1)

        if hist_type > 0:
            user_feat1 = tf.concat([xv1, user_history1], axis=-1)
            user_feat2 = tf.concat([xv1, user_history2], axis=-1)
        else:
            user_feat1 = xv1
            user_feat2 = xv1

        h1 = tf.concat([user_feat1, xv2], axis=-1)
        h2 = tf.concat([user_feat2, xv2], axis=-1)
        if cotrain:
            h1, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                   training=self.training, name='mlp',
                                                                   reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                   training=self.training, name='mlp',
                                                                   reuse=tf.AUTO_REUSE)
        else:
            h1, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                   training=self.training, name='mlp1',
                                                                   reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                   training=self.training, name='mlp2',
                                                                   reuse=tf.AUTO_REUSE)
        h1 = tf.squeeze(h1)
        h2 = tf.squeeze(h2)
        self.logits1, self.outputs1 = output([h1, b])
        self.logits2, self.outputs2 = output([h2, b])

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.entropy1 = loss(logits=self.logits1, targets=self.labels, pos_weight=pos_weight)
                self.loss1 = tf.reduce_mean(self.entropy1)

                self.entropy2 = loss(logits=self.logits2, targets=self.labels, pos_weight=pos_weight)
                self.loss2 = tf.reduce_mean(self.entropy2)

                _loss_ = self.loss
                self.optimizer1 = optimizer.minimize(loss=self.loss1,
                                                     global_step=global_step)
                self.optimizer2 = optimizer.minimize(loss=self.loss2,
                                                     global_step=global_step)


class Model_Anchors_old(Model):
    def __init__(self, init='xavier', user_max_id=None, src_item_max_id=None, trg_item_max_id=None,
                 embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, batch_norm=False, layer_norm=False,
                 l1_w=None, l1_v=None, layer_l1=None, user_his_len=None, hist_type=None, anchor_num=5, anchor_dim=None,
                 fuse=1, meta_layer_sizes=None, meta_layer_acts=None, meta_layer_keeps=None, cotrain=False):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.layer_l2 = layer_l2
        self.l1_w = l1_w
        self.layer_l1 = layer_l1
        self.l1_v = l1_v
        self.hist_type = hist_type
        self.embed_size = embed_size
        self.user_his_len = user_his_len
        self.anchor_num = anchor_num
        self.anchor_dim = anchor_dim
        self.layer_sizes = layer_sizes
        self.layer_acts = layer_acts
        self.layer_keeps = layer_keeps
        self.cotrain = cotrain

        with tf.name_scope('input'):
            self.user_id = tf.placeholder(tf.int32, [None], name='user_id')
            self.item_id = tf.placeholder(tf.int32, [None], name='item_id')
            self.history1 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items1')
            self.history_len1 = tf.placeholder(tf.int32, [None], name='history_items_len1')
            self.history2 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items2')
            self.history_len2 = tf.placeholder(tf.int32, [None], name='history_items_len2')
            self.labels = tf.placeholder(tf.float32, [None], name='label')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.is_src = tf.placeholder(dtype=tf.bool, name='whether_is_source')

        v_user = get_variable(init, name='v_user', shape=[user_max_id, embed_size])
        v_item = get_variable(init, name='v_item', shape=[src_item_max_id + trg_item_max_id, embed_size])
        b = get_variable('zero', name='b', shape=[1])

        xv1 = tf.gather(v_user, self.user_id)
        xv2 = tf.gather(v_item, self.item_id)

        hist1 = tf.gather(v_item, self.history1)
        hist2 = tf.gather(v_item, self.history2)

        self.a = xv1

        if hist_type == 1:
            # source
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history1 = user_history1 / tf.expand_dims(tf.cast(self.history_len1, tf.float32), 1)

            # target
            user_history2 = tf.reduce_sum(hist2, axis=-2)
            user_history2 = user_history2 / tf.expand_dims(tf.cast(self.history_len2, tf.float32), 1)
        elif hist_type == 2:
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history2 = tf.reduce_sum(hist2, axis=-2)
        elif hist_type == 3:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)
        elif hist_type == 4:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention1', self.training), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention2', self.training), 1)

        if hist_type > 0:
            user_feat1 = tf.concat([xv1, user_history1], axis=-1)
            user_feat2 = tf.concat([xv1, user_history2], axis=-1)
            self.concat_shape = embed_size * 2
        else:
            user_feat1 = xv1
            user_feat2 = xv1
            self.concat_shape = embed_size * 1

        h1 = tf.concat([user_feat1, xv2], axis=-1)
        h2 = tf.concat([user_feat2, xv2], axis=-1)
        self.concat_shape += embed_size

        anchors = get_variable(init, name='anchors', shape=[anchor_num, anchor_dim])
        # self.meta_bias = get_variable(init, name='meta_bias', shape=[len(layer_sizes)])

        if fuse == 1:
            domain_network_inputs = []
            for domain in range(2):
                anchor_weights = get_variable(init, name='anchors_weights', shape=[2, anchor_num])
                domain_weights = anchor_weights[domain]  # [anchor_num]
                domain_weights = tf.expand_dims(tf.nn.softmax(domain_weights), axis=0)  # [1, anchor_num]
                network_inputs = tf.matmul(domain_weights, anchors)  # [1, anchor_dim]
                domain_network_inputs.append(network_inputs)
        elif fuse == 2:
            domain_network_inputs = []
            for domain in range(2):
                anchor_weights = get_variable(init, name='anchors_weights', shape=[2, 1, anchor_dim])
                domain_weights = anchor_weights[domain]  # [1, anchor_dim]
                domain_weights = tf.reduce_sum(anchors * domain_weights, axis=-1)  # [anchor_num]
                domain_weights = tf.expand_dims(tf.nn.softmax(domain_weights), axis=0)  # [1, anchor_num]
                network_inputs = tf.matmul(domain_weights, anchors)  # [1, anchor_dim]
                domain_network_inputs.append(network_inputs)

        meta_networks = []
        for domain in range(2):
            net, _, _, _ = bin_mlp(meta_layer_sizes, meta_layer_acts, meta_layer_keeps, domain_network_inputs[domain],
                                   training=self.training, name='meta_learning', reuse=tf.AUTO_REUSE)
            meta_networks.append(net * 2)
        self.meta_networks = meta_networks
        self.a = meta_networks[0]
        self.b = meta_networks[1]

        h1 = self.meta_mlp(h1, 0, 'meta1')
        h2 = self.meta_mlp(h2, 1, 'meta2')

        h1 = tf.squeeze(h1)
        h2 = tf.squeeze(h2)
        self.logits1, self.outputs1 = output([h1, b])
        self.logits2, self.outputs2 = output([h2, b])

    def meta_mlp(self, h, domain, name=None):
        pre = ''
        if not self.cotrain:
            pre = str(domain)
        out = h
        meta_network = self.meta_networks[domain]
        start = 0
        for i in range(len(self.layer_sizes)):
            with tf.name_scope('hidden_%d' % i + str(name)):
                size_i = self.layer_sizes[i]
                out = tf.layers.dense(out, size_i, activation=None, name='mlp%s_%d' % (pre, i) + str(name), reuse=False)
                matrix = meta_network[:, start: start + size_i]
                # print('meta_network', self.layer_sizes, size_i, matrix.shape, meta_network.shape)
                start += size_i
                out = out * matrix
                if i < len(self.layer_sizes) - 1:
                    out = tf.layers.batch_normalization(out, training=self.training,
                                                        name='mlp%s_bn_%d' % (pre, i) + str(name),
                                                        reuse=False)
                out = tf.nn.dropout(
                    activate(
                        out, self.layer_acts[i]),
                    self.layer_keeps[i])
        return out

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.entropy1 = loss(logits=self.logits1, targets=self.labels, pos_weight=pos_weight)
                self.loss1 = tf.reduce_mean(self.entropy1)

                self.entropy2 = loss(logits=self.logits2, targets=self.labels, pos_weight=pos_weight)
                self.loss2 = tf.reduce_mean(self.entropy2)

                _loss_ = self.loss
                self.optimizer1 = optimizer.minimize(loss=self.loss1,
                                                     global_step=global_step)
                self.optimizer2 = optimizer.minimize(loss=self.loss2,
                                                     global_step=global_step)


class Model_Anchors(Model):
    def __init__(self, init='xavier', user_max_id=None, src_item_max_id=None, trg_item_max_id=None,
                 embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, batch_norm=False, layer_norm=False,
                 l1_w=None, l1_v=None, layer_l1=None, user_his_len=None, hist_type=None, anchor_num=5, anchor_dim=None,
                 fuse=1, meta_layer_sizes=None, meta_layer_acts=None, meta_layer_keeps=None, cotrain=False, orth=None,
                 orth_alpha=1.0, head_num=None, meta_share=False, use_meta_base=False, seperate_loss=False):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.layer_l2 = layer_l2
        self.l1_w = l1_w
        self.layer_l1 = layer_l1
        self.l1_v = l1_v
        self.hist_type = hist_type
        self.embed_size = embed_size
        self.user_his_len = user_his_len
        self.anchor_num = anchor_num
        self.anchor_dim = anchor_dim
        self.layer_sizes = layer_sizes
        self.layer_acts = layer_acts
        self.layer_keeps = layer_keeps
        self.cotrain = cotrain
        self.orth = orth
        self.orth_alpha = orth_alpha
        self.use_meta_base = use_meta_base
        self.seperate_loss = seperate_loss

        with tf.name_scope('input'):
            self.user_id_src = tf.placeholder(tf.int32, [None], name='user_id_src')
            self.item_id_src = tf.placeholder(tf.int32, [None], name='item_id_src')
            self.user_id_trg = tf.placeholder(tf.int32, [None], name='user_id_trg')
            self.item_id_trg = tf.placeholder(tf.int32, [None], name='item_id_trg')
            self.history1 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items1')
            self.history_len1 = tf.placeholder(tf.int32, [None], name='history_items_len1')
            self.history2 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items2')
            self.history_len2 = tf.placeholder(tf.int32, [None], name='history_items_len2')
            self.labels_src = tf.placeholder(tf.float32, [None], name='label_src')
            self.labels_trg = tf.placeholder(tf.float32, [None], name='label_trg')
            self.training = tf.placeholder(dtype=tf.bool, name='training')

        v_user = get_variable(init, name='v_user', shape=[user_max_id, embed_size])
        v_item = get_variable(init, name='v_item', shape=[src_item_max_id + trg_item_max_id, embed_size])
        b = get_variable('zero', name='b', shape=[1])

        xv1_src = tf.gather(v_user, self.user_id_src)
        xv2_src = tf.gather(v_item, self.item_id_src)
        xv1_trg = tf.gather(v_user, self.user_id_trg)
        xv2_trg = tf.gather(v_item, self.item_id_trg)

        hist1 = tf.gather(v_item, self.history1)
        hist2 = tf.gather(v_item, self.history2)

        if hist_type == 1:
            # source
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history1 = user_history1 / tf.expand_dims(tf.cast(self.history_len1, tf.float32), 1)

            # target
            user_history2 = tf.reduce_sum(hist2, axis=-2)
            user_history2 = user_history2 / tf.expand_dims(tf.cast(self.history_len2, tf.float32), 1)
        elif hist_type == 2:
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history2 = tf.reduce_sum(hist2, axis=-2)
        elif hist_type == 3:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)
        elif hist_type == 4:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention1', self.training), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention2', self.training), 1)

        if hist_type > 0:
            user_feat1 = tf.concat([xv1_src, user_history1], axis=-1)
            user_feat2 = tf.concat([xv1_trg, user_history2], axis=-1)
            self.concat_shape = embed_size * 2
        else:
            user_feat1 = xv1_src
            user_feat2 = xv1_trg
            self.concat_shape = embed_size * 1

        h1 = tf.concat([user_feat1, xv2_src], axis=-1)
        h2 = tf.concat([user_feat2, xv2_trg], axis=-1)
        self.concat_shape += embed_size

        anchors = get_variable(init, name='anchors', shape=[anchor_num, anchor_dim])
        self.anchors = anchors
        # self.meta_bias = get_variable(init, name='meta_bias', shape=[len(layer_sizes)])

        if fuse == 1:
            domain_network_inputs = []
            anchor_weights = get_variable(init, name='anchors_weights', shape=[2, anchor_num])
            for domain in range(2):
                domain_weights = anchor_weights[domain]  # [anchor_num]
                domain_weights = tf.expand_dims(tf.nn.softmax(domain_weights), axis=0)  # [1, anchor_num]
                network_inputs = tf.matmul(domain_weights, anchors)  # [1, anchor_dim]
                domain_network_inputs.append(network_inputs)
        elif fuse == 2:
            domain_network_inputs = []
            anchor_weights = get_variable(init, name='anchors_weights', shape=[2, 1, anchor_dim])
            for domain in range(2):
                domain_weights = anchor_weights[domain]  # [1, anchor_dim]
                domain_weights = tf.reduce_sum(anchors * domain_weights, axis=-1)  # [anchor_num]
                domain_weights = tf.expand_dims(tf.nn.softmax(domain_weights), axis=0)  # [1, anchor_num]
                network_inputs = tf.matmul(domain_weights, anchors)  # [1, anchor_dim]
                domain_network_inputs.append(network_inputs)
        elif fuse == 3:
            domain_network_inputs = []
            anchor_weights = get_variable(init, name='anchors_weights', shape=[2, head_num, 1, anchor_dim])
            for domain in range(2):
                domain_weights = anchor_weights[domain]  # [head_num, 1, anchor_dim]
                domain_weights = tf.reduce_sum(tf.expand_dims(anchors, axis=0) * domain_weights,
                                               axis=-1)  # [head_num, anchor_num]
                domain_weights = tf.nn.softmax(domain_weights, dim=-1)  # [head_num, anchor_num]
                network_inputs = tf.matmul(domain_weights, anchors)  # [head_num, anchor_dim]
                network_inputs = tf.reshape(network_inputs, [1, -1])  # [1, head_num * anchor_dim]
                domain_network_inputs.append(network_inputs)
        self.anchor_weights = anchor_weights

        if use_meta_base:
            domain_weights = get_variable(init, name='anchors_weights_base', shape=[head_num, 1, anchor_dim])
            domain_weights = tf.reduce_sum(tf.expand_dims(anchors, axis=0) * domain_weights,
                                           axis=-1)  # [head_num, anchor_num]
            domain_weights = tf.nn.softmax(domain_weights, dim=-1)  # [head_num, anchor_num]
            network_inputs = tf.matmul(domain_weights, anchors)  # [head_num, anchor_dim]
            base_network_inputs = tf.reshape(network_inputs, [1, -1])  # [1, head_num * anchor_dim]

        meta_networks = []
        for domain in range(2):
            suffix = ''
            if not meta_share:
                suffix += str(domain)
            net, _, _, _ = bin_mlp(meta_layer_sizes, meta_layer_acts, meta_layer_keeps, domain_network_inputs[domain],
                                   training=self.training, name='meta_learning' + suffix, reuse=tf.AUTO_REUSE)
            net = net * 2
            pads = tf.where(tf.less(tf.abs(net), epsilon), tf.ones_like(net), tf.zeros_like(net))
            pads = tf.stop_gradient(pads)
            net = net + pads
            self.a = net
            meta_networks.append(net)

        self.meta_networks = meta_networks
        # self.a = meta_networks[0]
        self.b = meta_networks[1]

        if use_meta_base:
            h_base = self.meta_base_mlp(tf.concat([h1, h2], axis=0), name='metabase')
            h_base = tf.squeeze(h_base)
            self.logits_base, self.outputs_base = output([h_base, b])

        h1 = self.meta_mlp(h1, domain=0, name='meta1')
        h2 = self.meta_mlp(h2, domain=1, name='meta2')

        h1 = tf.squeeze(h1)
        h2 = tf.squeeze(h2)

        self.logits1, self.outputs1 = output([h1, b])
        self.logits2, self.outputs2 = output([h2, b])

    def meta_mlp(self, h, domain, name=None):
        suffix = ''
        if not self.cotrain:
            suffix = str(domain)
        name = name + suffix
        out = h
        meta_network = self.meta_networks[domain]
        start = 0
        for i in range(len(self.layer_sizes)):
            with tf.name_scope('hidden_%d' % i + str(name)):
                size_i = self.layer_sizes[i]
                out = tf.layers.dense(out, size_i, activation=None, name='mlp_%d' % i + str(name), reuse=False)
                matrix = meta_network[:, start: start + size_i]
                # print('meta_network', self.layer_sizes, size_i, matrix.shape, meta_network.shape)
                start += size_i
                out = out * matrix
                if i < len(self.layer_sizes) - 1:
                    out = tf.layers.batch_normalization(out, training=self.training, name='mlp_bn_%d' % i + str(name),
                                                        reuse=False)
                out = tf.nn.dropout(
                    activate(
                        out, self.layer_acts[i]),
                    self.layer_keeps[i])
        return out

    def meta_base_mlp(self, h, name=None):
        out = h
        meta_network = self.base_network
        start = 0
        for i in range(len(self.layer_sizes)):
            with tf.name_scope('hidden_%d' % i + str(name)):
                size_i = self.layer_sizes[i]
                out = tf.layers.dense(out, size_i, activation=None, name='mlp_%d' % i + str(name), reuse=False)
                matrix = meta_network[:, start: start + size_i]
                # print('meta_network', self.layer_sizes, size_i, matrix.shape, meta_network.shape)
                start += size_i
                out = out * matrix
                if i < len(self.layer_sizes) - 1:
                    out = tf.layers.batch_normalization(out, training=self.training, name='mlp_bn_%d' % i + str(name),
                                                        reuse=False)
                out = tf.nn.dropout(
                    activate(
                        out, self.layer_acts[i]),
                    self.layer_keeps[i])
        return out

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.entropy1 = loss(logits=self.logits1, targets=self.labels_src, pos_weight=pos_weight)
                self.loss1 = tf.reduce_mean(self.entropy1)

                self.entropy2 = loss(logits=self.logits2, targets=self.labels_trg, pos_weight=pos_weight)
                self.loss2 = tf.reduce_mean(self.entropy2)

                if self.orth == 1:
                    inner = tf.reduce_mean(tf.squeeze(tf.reduce_prod(self.anchor_weights, axis=0)))
                    norm_prod = tf.reduce_prod(
                        tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(self.anchor_weights), axis=-1))), axis=0)
                    self.orth_loss = inner / tf.clip_by_value(norm_prod, 1e-8, norm_prod)
                    self.orth_loss = self.orth_loss * self.orth_alpha
                    self.loss1 += self.orth_loss * 0.5
                    self.loss2 += self.orth_loss * 0.5
                elif self.orth == 2:
                    grad_vec1 = tf.gradients(self.loss1, self.anchors)[0]  # [anchor_num, anchor_dim]
                    grad_vec2 = tf.gradients(self.loss2, self.anchors)[0]  # [anchor_num, anchor_dim]
                    inners = tf.reduce_sum(grad_vec1 * grad_vec2, axis=-1)  # [anchor_num]
                    norms1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec1), axis=-1)))  # [anchor_num]
                    norms2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec2), axis=-1)))  # [anchor_num]
                    self.orth_loss = inners / tf.clip_by_value(norms1 * norms2, 1e-8, tf.reduce_max(norms1 * norms2))
                    self.orth_loss = tf.reduce_sum(self.orth_loss)
                    self.orth_loss = self.orth_loss * self.orth_alpha
                    self.loss1 += self.orth_loss * 0.5
                    self.loss2 += self.orth_loss * 0.5
                elif self.orth == 3:
                    grad_vec1 = tf.gradients(self.loss1, self.anchors)[0]  # [anchor_num, anchor_dim]
                    grad_vec2 = tf.gradients(self.loss2, self.anchors)[0]  # [anchor_num, anchor_dim]
                    inners = tf.reduce_sum(grad_vec1 * grad_vec2, axis=-1)  # [anchor_num]
                    norms1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec1), axis=-1)))  # [anchor_num]
                    norms2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec2), axis=-1)))  # [anchor_num]
                    self.orth_loss = inners / tf.clip_by_value(norms1 * norms2, 1e-8, tf.reduce_max(norms1 * norms2))
                    self.orth_loss = tf.reduce_sum(tf.abs(self.orth_loss))
                    self.orth_loss = self.orth_loss * self.orth_alpha
                    self.loss1 += self.orth_loss * 0.5
                    self.loss2 += self.orth_loss * 0.5
                elif self.orth == 4:
                    grad_vec1 = tf.gradients(self.loss1, self.anchors)[0]  # [anchor_num, anchor_dim]
                    grad_vec2 = tf.gradients(self.loss2, self.anchors)[0]  # [anchor_num, anchor_dim]
                    inners = tf.reduce_sum(grad_vec1 * grad_vec2, axis=-1)  # [anchor_num]
                    norms1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec1), axis=-1)))  # [anchor_num]
                    norms2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec2), axis=-1)))  # [anchor_num]
                    self.orth_loss = inners / tf.clip_by_value(norms1 * norms2, 1e-8, tf.reduce_max(norms1 * norms2))
                    self.orth_loss = tf.where(tf.less(self.orth_loss, 0), self.orth_loss, tf.zeros_like(self.orth_loss))
                    self.orth_loss = tf.reduce_sum(-self.orth_loss)
                    self.orth_loss = self.orth_loss * self.orth_alpha
                    self.loss1 += self.orth_loss * 0.5
                    self.loss2 += self.orth_loss * 0.5

                if self.use_meta_base:
                    labels = tf.concat([self.labels_src, self.labels_trg], axis=-1)
                    self.entropy_base = loss(logits=self.logits_base, targets=labels, pos_weight=pos_weight)
                    loss_base = tf.reduce_mean(self.entropy_base)
                    self.loss1 += loss_base * 0.5
                    self.loss2 += loss_base * 0.5

                if self.seperate_loss:
                    loss_ = -tf.sqrt(
                        tf.nn.relu(tf.reduce_mean(tf.square(self.meta_networks[0] - self.meta_networks[1]), axis=-1)))
                    loss_ = tf.reduce_mean(loss_)
                    self.c = loss_
                    self.loss1 += loss_ * 0.5
                    self.loss2 += loss_ * 0.5

                _loss_ = self.loss
                self.optimizer1 = optimizer.minimize(loss=self.loss1,
                                                     global_step=global_step)
                self.optimizer2 = optimizer.minimize(loss=self.loss2,
                                                     global_step=global_step)


class Model_Voting(Model):
    def __init__(self, init='xavier', user_max_id=None, src_item_max_id=None, trg_item_max_id=None,
                 embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, batch_norm=False, layer_norm=False,
                 l1_w=None, l1_v=None, layer_l1=None, user_his_len=None, hist_type=None, cotrain=None,
                 subnet_N=5, voting=None, orth=None, orth_alpha=None, matrix_reg=None, anchor_user=None,
                 anchor_user_num=None, anchor_user_orth=None, anchor_user_orth_alpha=None):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.layer_l2 = layer_l2
        self.l1_w = l1_w
        self.layer_l1 = layer_l1
        self.l1_v = l1_v
        self.hist_type = hist_type
        self.embed_size = embed_size
        self.user_his_len = user_his_len
        self.subnet_N = subnet_N
        self.layer_sizes = layer_sizes
        self.layer_acts = layer_acts
        self.layer_keeps = layer_keeps
        self.cotrain = cotrain
        self.voting = voting
        self.orth = orth
        self.orth_alpha = orth_alpha
        self.matrix_reg = matrix_reg
        self.anchor_user = anchor_user
        self.anchor_user_num = anchor_user_num
        self.anchor_user_orth = anchor_user_orth
        self.anchor_user_orth_alpha = anchor_user_orth_alpha

        with tf.name_scope('input'):
            self.user_id_src = tf.placeholder(tf.int32, [None], name='user_id_src')
            self.item_id_src = tf.placeholder(tf.int32, [None], name='item_id_src')
            self.user_id_trg = tf.placeholder(tf.int32, [None], name='user_id_trg')
            self.item_id_trg = tf.placeholder(tf.int32, [None], name='item_id_trg')
            self.history1 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items1')
            self.history_len1 = tf.placeholder(tf.int32, [None], name='history_items_len1')
            self.history2 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items2')
            self.history_len2 = tf.placeholder(tf.int32, [None], name='history_items_len2')
            self.labels_src = tf.placeholder(tf.float32, [None], name='label_src')
            self.labels_trg = tf.placeholder(tf.float32, [None], name='label_trg')
            self.training = tf.placeholder(dtype=tf.bool, name='training')

        v_user = get_variable(init, name='v_user', shape=[user_max_id, embed_size])
        v_item = get_variable(init, name='v_item', shape=[src_item_max_id + trg_item_max_id, embed_size])
        b = get_variable('zero', name='b', shape=[1])

        xv1_src = tf.gather(v_user, self.user_id_src)
        xv2_src = tf.gather(v_item, self.item_id_src)
        xv1_trg = tf.gather(v_user, self.user_id_trg)
        xv2_trg = tf.gather(v_item, self.item_id_trg)

        if anchor_user == 1:
            anchors = get_variable(init, name='anchors', shape=[anchor_user_num, embed_size])
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src * anchor_src
            xv1_trg = xv1_trg * anchor_trg
        elif anchor_user == 2:
            anchors = get_variable(init, name='anchors', shape=[anchor_user_num, embed_size])
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src * domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg * domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src * anchor_src
            xv1_trg = xv1_trg * anchor_trg
        elif anchor_user == 3:
            anchors = get_variable(init, name='anchors', shape=[anchor_user_num, embed_size])
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = (xv1_src + domain_emb[:1]) * 0.5  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = (xv1_trg + domain_emb[1:]) * 0.5
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src * anchor_src
            xv1_trg = xv1_trg * anchor_trg
        elif anchor_user == 5:
            anchors = get_variable(init, name='anchors', shape=[anchor_user_num, embed_size])
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # # 222
            # score_src = tf.nn.softmax(score_src, dim=-1)
            # score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src * anchor_src
            xv1_trg = xv1_trg * anchor_trg
        elif anchor_user == 6:
            anchors = get_variable(init, name='anchors', shape=[anchor_user_num, embed_size])
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src + anchor_src
            xv1_trg = xv1_trg + anchor_trg
        elif anchor_user == 7:
            anchors = get_variable(init, name='anchors', shape=[anchor_user_num, embed_size])
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src + anchor_src + xv1_src * anchor_src
            xv1_trg = xv1_trg + anchor_trg + xv1_trg * anchor_trg
        elif anchor_user == 8:
            anchors = get_variable(init, name='anchors', shape=[anchor_user_num, embed_size])
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            # score_src = tf.nn.softmax(score_src, dim=-1)
            # score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src + anchor_src + xv1_src * anchor_src
            xv1_trg = xv1_trg + anchor_trg + xv1_trg * anchor_trg
        elif anchor_user == 9:
            anchors = get_variable(init, name='anchors', shape=[anchor_user_num, embed_size])
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            # score_src = tf.nn.softmax(score_src, dim=-1)
            # score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src + xv1_src * anchor_src
            xv1_trg = xv1_trg + xv1_trg * anchor_trg
        elif anchor_user == 10:
            anchors = get_variable(init, name='anchors', shape=[anchor_user_num, embed_size])
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            query_len_src = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(query_src), axis=-1)), [-1, 1])
            query_len_trg = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(query_trg), axis=-1)), [-1, 1])
            anchors_len = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(anchors), axis=-1)), [1, -1])
            denominator_src = tf.clip_by_value(query_len_src * anchors_len, 1e-6,
                                               tf.reduce_max(query_len_src * anchors_len))
            denominator_trg = tf.clip_by_value(query_len_trg * anchors_len, 1e-6,
                                               tf.reduce_max(query_len_trg * anchors_len))
            score_src = score_src / denominator_src
            score_trg = score_trg / denominator_trg

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src * anchor_src
            xv1_trg = xv1_trg * anchor_trg
        elif anchor_user == 11:
            # 444
            anchors = get_variable('one', name='anchors', shape=[anchor_user_num, embed_size])
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            query_len_src = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(query_src), axis=-1)), [-1, 1])
            query_len_trg = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(query_trg), axis=-1)), [-1, 1])
            anchors_len = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(anchors), axis=-1)), [1, -1])
            denominator_src = tf.clip_by_value(query_len_src * anchors_len, 1e-6,
                                               tf.reduce_max(query_len_src * anchors_len))
            denominator_trg = tf.clip_by_value(query_len_trg * anchors_len, 1e-6,
                                               tf.reduce_max(query_len_trg * anchors_len))
            score_src = score_src / denominator_src
            score_trg = score_trg / denominator_trg

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src * anchor_src
            xv1_trg = xv1_trg * anchor_trg
        elif anchor_user == 12:
            # 444
            anchors = get_variable('uniform', name='anchors', shape=[anchor_user_num, embed_size])
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            query_len_src = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(query_src), axis=-1)), [-1, 1])
            query_len_trg = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(query_trg), axis=-1)), [-1, 1])
            anchors_len = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(anchors), axis=-1)), [1, -1])
            denominator_src = tf.clip_by_value(query_len_src * anchors_len, 1e-6,
                                               tf.reduce_max(query_len_src * anchors_len))
            denominator_trg = tf.clip_by_value(query_len_trg * anchors_len, 1e-6,
                                               tf.reduce_max(query_len_trg * anchors_len))
            score_src = score_src / denominator_src
            score_trg = score_trg / denominator_trg

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src * anchor_src
            xv1_trg = xv1_trg * anchor_trg

        hist1 = tf.gather(v_item, self.history1)
        hist2 = tf.gather(v_item, self.history2)

        if hist_type == 1:
            # source
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history1 = user_history1 / tf.expand_dims(tf.cast(self.history_len1, tf.float32), 1)

            # target
            user_history2 = tf.reduce_sum(hist2, axis=-2)
            user_history2 = user_history2 / tf.expand_dims(tf.cast(self.history_len2, tf.float32), 1)
        elif hist_type == 2:
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history2 = tf.reduce_sum(hist2, axis=-2)
        elif hist_type == 3:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)
        elif hist_type == 4:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention1', self.training), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention2', self.training), 1)

        if hist_type > 0:
            user_feat1 = tf.concat([xv1_src, user_history1], axis=-1)
            user_feat2 = tf.concat([xv1_trg, user_history2], axis=-1)
            self.concat_shape = embed_size * 2
        else:
            user_feat1 = xv1_src
            user_feat2 = xv1_trg
            self.concat_shape = embed_size * 1

        h1 = tf.concat([user_feat1, xv2_src], axis=-1)
        h2 = tf.concat([user_feat2, xv2_trg], axis=-1)
        self.concat_shape += embed_size

        subnet_ws = []
        subnet_bs = []
        for i in range(len(layer_sizes)):
            if i == 0:
                in_shape = self.concat_shape
                out_shape = layer_sizes[i]
            else:
                in_shape = layer_sizes[i - 1]
                out_shape = layer_sizes[i]
            ws = []
            bs = []
            for n in range(subnet_N):
                subw = get_variable(init, name='w_layer%d_num%d' % (i, n), shape=[in_shape, out_shape])
                subb = get_variable(init, name='b_layer%d_num%d' % (i, n), shape=[1])
                ws.append(subw)
                bs.append(subb)
            ws = tf.stack(ws)  # [N, in_shape, out_shape]
            bs = tf.stack(bs)  # [N, 1]
            subnet_ws.append(ws)
            subnet_bs.append(bs)

        self.subnet_ws = subnet_ws
        self.subnet_bs = subnet_bs

        if voting in [1, 2]:
            domain_weights = get_variable(init, name='domain_weights', shape=[2, subnet_N])

        if voting > 0:
            self.domain_weights = domain_weights

            h1 = self.sub_mlp(h1, domain=0, name='domain1')
            h2 = self.sub_mlp(h2, domain=1, name='domain2')

            h1 = tf.squeeze(h1)
            h2 = tf.squeeze(h2)

        if cotrain > 0:
            # h1_spec, self.layer_kernels, self.layer_biases, _ = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h1,
            #                     input_dim=self.concat_shape, training=self.training, name='mlp1', reuse=tf.AUTO_REUSE)
            #
            # h2_spec, self.layer_kernels, self.layer_biases, _ = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h2,
            #                     input_dim=self.concat_shape, training=self.training, name='mlp2', reuse=tf.AUTO_REUSE)

            h1_spec, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                        training=self.training, name='mlp1',
                                                                        reuse=tf.AUTO_REUSE)

            h2_spec, self.layer_kernels, self.layer_biases, _ = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                        training=self.training, name='mlp2',
                                                                        reuse=tf.AUTO_REUSE)

            h1_spec = tf.reshape(h1_spec, [-1])
            h2_spec = tf.reshape(h2_spec, [-1])

        if cotrain == 0:
            out1 = [h1, b]
            out2 = [h2, b]
        elif cotrain == 1:
            out1 = [h1 + h1_spec, b]
            out2 = [h2 + h2_spec, b]
        elif cotrain == 2:
            w1 = get_variable(init, name='w1', shape=[1])
            w2 = get_variable(init, name='w2', shape=[1])
            w3 = get_variable(init, name='w3', shape=[1])
            w4 = get_variable(init, name='w4', shape=[1])
            out1 = [w1 * h1, w2 * h1_spec, b]
            out2 = [w3 * h2, w4 * h2_spec, b]
        elif cotrain == 3:
            out1 = [h1_spec, b]
            out2 = [h2_spec, b]

        self.logits1, self.outputs1 = output(out1)
        self.logits2, self.outputs2 = output(out2)

    def sub_mlp(self, h, domain, name=None):
        suffix = ''
        if not self.cotrain:
            suffix = str(domain)
        name = name + suffix
        out = h
        weights = self.domain_weights[domain]  # [N,]
        if self.voting in [2]:
            weights = tf.nn.softmax(weights)

        for i in range(len(self.layer_sizes)):
            with tf.name_scope('hidden_%d' % i + str(name)):
                domain_w = self.subnet_ws[i]  # [N, in_shape, out_shape]
                domain_b = self.subnet_bs[i]  # [N, 1]
                domain_w = tf.reduce_sum(tf.reshape(weights, [-1, 1, 1]) * domain_w, axis=0)  # [in_shape, out_shape]
                domain_b = tf.reduce_sum(tf.reshape(weights, [-1, 1]) * domain_b, axis=0)  # [1]
                out = tf.matmul(out, domain_w) + domain_b
                if i < len(self.layer_sizes) - 1:
                    out = tf.layers.batch_normalization(out, training=self.training, name='mlp_bn_%d' % i + str(name),
                                                        reuse=False)
                out = tf.nn.dropout(
                    activate(
                        out, self.layer_acts[i]),
                    self.layer_keeps[i])
        return out

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.entropy1 = loss(logits=self.logits1, targets=self.labels_src, pos_weight=pos_weight)
                self.origin_loss1 = tf.reduce_mean(self.entropy1)
                self.loss1 = self.origin_loss1

                self.entropy2 = loss(logits=self.logits2, targets=self.labels_trg, pos_weight=pos_weight)
                self.origin_loss2 = tf.reduce_mean(self.entropy2)
                self.loss2 = self.origin_loss2

                if self.orth == 1:
                    self.orth_loss = 0
                    for i in range(len(self.layer_sizes)):
                        W = self.subnet_ws[i]
                        grad1 = tf.gradients(self.origin_loss1, W)[0]  # [N, in_shape, out_shape]
                        grad2 = tf.gradients(self.origin_loss2, W)[0]  # [N, in_shape, out_shape]
                        cosine_vec = matrix_cosine_similarity(grad1, grad2)  # [N, out_shape]
                        orth_loss = tf.reduce_mean(tf.nn.relu(-cosine_vec))
                        self.orth_loss += orth_loss * self.orth_alpha

                    self.loss1 += self.orth_loss * 0.5
                    self.loss2 += self.orth_loss * 0.5
                elif self.orth == 2:
                    self.orth_loss = 0
                    for i in range(len(self.layer_sizes)):
                        W = self.subnet_ws[i]
                        grad1 = tf.gradients(self.origin_loss1, W)[0]  # [N, in_shape, out_shape]
                        grad2 = tf.gradients(self.origin_loss2, W)[0]  # [N, in_shape, out_shape]
                        cosine_vec = matrix_cosine_similarity(grad1, grad2)  # [N, out_shape]
                        orth_loss = tf.reduce_mean(tf.nn.relu(cosine_vec))
                        self.orth_loss += orth_loss * self.orth_alpha

                    self.loss1 += self.orth_loss * 0.5
                    self.loss2 += self.orth_loss * 0.5
                elif self.orth == 3:
                    self.orth_loss = 0
                    for i in range(len(self.layer_sizes)):
                        W = self.subnet_ws[i]
                        grad1 = tf.gradients(self.origin_loss1, W)[0]  # [N, in_shape, out_shape]
                        grad2 = tf.gradients(self.origin_loss2, W)[0]  # [N, in_shape, out_shape]
                        cosine_vec = matrix_cosine_similarity(grad1, grad2)  # [N, out_shape]
                        orth_loss = tf.reduce_mean(tf.abs(cosine_vec))
                        self.orth_loss += orth_loss * self.orth_alpha

                    self.loss1 += self.orth_loss * 0.5
                    self.loss2 += self.orth_loss * 0.5

                if self.matrix_reg == 1:
                    reg = 0
                    for l in range(len(self.layer_sizes)):
                        mat = self.subnet_ws[l]
                        for i in range(self.subnet_N - 1):
                            for j in range(self.subnet_N):
                                m1 = mat[i]
                                m2 = mat[j]
                                reg += tf.abs(matrix_coefficient(m1, m2))
                    self.loss1 += reg * 0.5
                    self.loss2 += reg * 0.5
                elif self.matrix_reg == 2:
                    reg = 0
                    for l in range(len(self.layer_sizes)):
                        mat = self.subnet_ws[l]
                        regl = 0
                        cnt = 0
                        for i in range(self.subnet_N - 1):
                            for j in range(self.subnet_N):
                                m1 = mat[i]
                                m2 = mat[j]
                                regl += tf.abs(matrix_coefficient(m1, m2))
                                cnt += 1
                        reg += regl / cnt
                    self.loss1 += reg * 0.5
                    self.loss2 += reg * 0.5
                elif self.matrix_reg == 3:
                    reg = 0
                    for l in range(len(self.layer_sizes)):
                        mat = self.subnet_ws[l]
                        regl = 0
                        cnt = 0
                        for i in range(self.subnet_N - 1):
                            for j in range(self.subnet_N):
                                m1 = mat[i]
                                m2 = mat[j]
                                regl += tf.reduce_mean(tf.abs(matrix_cosine_similarity(m1, m2)))
                                cnt += 1
                        reg += regl / cnt
                    self.loss1 += reg * 0.5
                    self.loss2 += reg * 0.5
                elif self.matrix_reg == 4:
                    reg = 0
                    for l in range(len(self.layer_sizes)):
                        mat = self.subnet_ws[l]
                        cnt = 0
                        for i in range(self.subnet_N - 1):
                            for j in range(self.subnet_N):
                                m1 = mat[i]
                                m2 = mat[j]
                                reg += tf.reduce_mean(tf.abs(matrix_cosine_similarity(m1, m2)))
                                cnt += 1
                    reg = reg / cnt
                    self.loss1 += reg * 0.5
                    self.loss2 += reg * 0.5
                elif self.matrix_reg == 5:
                    reg = 0
                    for l in range(len(self.layer_sizes)):
                        mat = self.subnet_ws[l]
                        regl = 0
                        cnt = 0
                        for i in range(self.subnet_N - 1):
                            for j in range(self.subnet_N):
                                m1 = mat[i]
                                m2 = mat[j]
                                inners = tf.reduce_sum(m1 * m2, axis=0)
                                regl += tf.reduce_mean(tf.abs(inners))
                                cnt += 1
                        reg += regl / cnt
                    self.loss1 += reg * 0.5
                    self.loss2 += reg * 0.5

                if self.anchor_user_orth == 1:
                    grad_vec1 = tf.gradients(self.origin_loss1, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    grad_vec2 = tf.gradients(self.origin_loss2, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    inners = tf.reduce_sum(grad_vec1 * grad_vec2, axis=-1)  # [anchor_user_num]
                    norms1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec1), axis=-1)))  # [anchor_user_num]
                    norms2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec2), axis=-1)))  # [anchor_user_num]
                    self.user_orth_loss = inners / tf.clip_by_value(norms1 * norms2, 1e-8,
                                                                    tf.reduce_max(norms1 * norms2))
                    self.user_orth_loss = tf.reduce_sum(tf.nn.relu(-self.user_orth_loss))
                    self.user_orth_loss = self.user_orth_loss * self.anchor_user_orth_alpha
                    self.loss1 += self.user_orth_loss * 0.5
                    self.loss2 += self.user_orth_loss * 0.5
                elif self.anchor_user_orth == 2:
                    grad_vec1 = tf.gradients(self.origin_loss1, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    grad_vec2 = tf.gradients(self.origin_loss2, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    inners = tf.reduce_sum(grad_vec1 * grad_vec2, axis=-1)  # [anchor_user_num]
                    norms1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec1), axis=-1)))  # [anchor_user_num]
                    norms2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec2), axis=-1)))  # [anchor_user_num]
                    self.user_orth_loss = inners / tf.clip_by_value(norms1 * norms2, 1e-8,
                                                                    tf.reduce_max(norms1 * norms2))
                    self.user_orth_loss = tf.reduce_mean(tf.nn.relu(-self.user_orth_loss))
                    self.user_orth_loss = self.user_orth_loss * self.anchor_user_orth_alpha
                    self.loss1 += self.user_orth_loss * 0.5
                    self.loss2 += self.user_orth_loss * 0.5
                elif self.anchor_user_orth == 3:
                    grad_vec1 = tf.gradients(self.origin_loss1, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    grad_vec2 = tf.gradients(self.origin_loss2, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    inners = tf.reduce_sum(grad_vec1 * grad_vec2, axis=-1)  # [anchor_user_num]
                    norms1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec1), axis=-1)))  # [anchor_user_num]
                    norms2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec2), axis=-1)))  # [anchor_user_num]
                    self.user_orth_loss = inners / tf.clip_by_value(norms1 * norms2, 1e-8,
                                                                    tf.reduce_max(norms1 * norms2))
                    self.user_orth_loss = tf.reduce_mean(tf.abs(self.user_orth_loss))
                    self.user_orth_loss = self.user_orth_loss * self.anchor_user_orth_alpha
                    self.loss1 += self.user_orth_loss * 0.5
                    self.loss2 += self.user_orth_loss * 0.5

                _loss_ = self.loss
                self.optimizer1 = optimizer.minimize(loss=self.loss1,
                                                     global_step=global_step)
                self.optimizer2 = optimizer.minimize(loss=self.loss2,
                                                     global_step=global_step)


class Model_User_Anchors(Model):
    def __init__(self, init='xavier', user_max_id=None, src_item_max_id=None, trg_item_max_id=None,
                 embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, batch_norm=False, layer_norm=False,
                 l1_w=None, l1_v=None, layer_l1=None, user_his_len=None, hist_type=None, anchor_num=5, anchor_user=1,
                 cotrain=None, cluster_embeddings=None, user_embeddings=None, l2=None, anchor_user_orth=None,
                 anchor_user_orth_alpha=None, anchor_score_reg=None, head_num=None, k_ratio=None, k_ratio2=None,
                 tau=None):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.layer_l2 = layer_l2
        self.l1_w = l1_w
        self.layer_l1 = layer_l1
        self.l1_v = l1_v
        self.hist_type = hist_type
        self.embed_size = embed_size
        self.user_his_len = user_his_len
        self.layer_sizes = layer_sizes
        self.layer_acts = layer_acts
        self.layer_keeps = layer_keeps
        self.cotrain = cotrain
        self.anchor_num = anchor_num
        self.l2 = l2
        self.anchor_user_orth = anchor_user_orth
        self.anchor_user_orth_alpha = anchor_user_orth_alpha
        self.concat_shape = 0
        self.anchor_score_reg = anchor_score_reg
        self.head_num = head_num
        k = math.ceil(k_ratio * anchor_num)
        k2 = math.ceil(k_ratio2 * anchor_num)
        self.tau = tau

        with tf.name_scope('input'):
            self.user_id_src = tf.placeholder(tf.int32, [None], name='user_id_src')
            self.item_id_src = tf.placeholder(tf.int32, [None], name='item_id_src')
            self.user_id_trg = tf.placeholder(tf.int32, [None], name='user_id_trg')
            self.item_id_trg = tf.placeholder(tf.int32, [None], name='item_id_trg')
            self.history1 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items1')
            self.history_len1 = tf.placeholder(tf.int32, [None], name='history_items_len1')
            self.history2 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items2')
            self.history_len2 = tf.placeholder(tf.int32, [None], name='history_items_len2')
            self.labels_src = tf.placeholder(tf.float32, [None], name='label_src')
            self.labels_trg = tf.placeholder(tf.float32, [None], name='label_trg')
            self.training = tf.placeholder(dtype=tf.bool, name='training')

        if user_embeddings is not None:
            v_user = tf.Variable(user_embeddings)
        else:
            v_user = get_variable(init, name='v_user', shape=[user_max_id, embed_size])
        v_item = get_variable(init, name='v_item', shape=[src_item_max_id + trg_item_max_id, embed_size])
        b = get_variable('zero', name='b', shape=[1])
        self.v_user = v_user
        self.v_item = v_item

        xv1_src = tf.gather(v_user, self.user_id_src)
        xv2_src = tf.gather(v_item, self.item_id_src)
        xv1_trg = tf.gather(v_user, self.user_id_trg)
        xv2_trg = tf.gather(v_item, self.item_id_trg)

        if anchor_user > 0:
            if anchor_user <= 28:
                if cluster_embeddings is not None:
                    anchors = tf.Variable(cluster_embeddings)
                else:
                    anchors = get_variable(init, name='anchors', shape=[anchor_num, embed_size])
            else:
                anchors1 = get_variable(init, name='anchors1', shape=[anchor_num, embed_size])
                anchors2 = get_variable(init, name='anchors2', shape=[anchor_num, embed_size])

        if anchor_user == 1:
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src * anchor_src
            xv1_trg = xv1_trg * anchor_trg
        elif anchor_user == 9:

            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            # score_src = tf.nn.softmax(score_src, dim=-1)
            # score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src + xv1_src * anchor_src
            xv1_trg = xv1_trg + xv1_trg * anchor_trg
        elif anchor_user == 10:
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            query_len_src = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(query_src), axis=-1)), [-1, 1])
            query_len_trg = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(query_trg), axis=-1)), [-1, 1])
            anchors_len = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(anchors), axis=-1)), [1, -1])
            denominator_src = tf.clip_by_value(query_len_src * anchors_len, 1e-6,
                                               tf.reduce_max(query_len_src * anchors_len))
            denominator_trg = tf.clip_by_value(query_len_trg * anchors_len, 1e-6,
                                               tf.reduce_max(query_len_trg * anchors_len))
            score_src = score_src / denominator_src
            score_trg = score_trg / denominator_trg

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src * anchor_src
            xv1_trg = xv1_trg * anchor_trg
        elif anchor_user == 14:

            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = anchor_src
            xv1_trg = anchor_trg
        elif anchor_user == 15:

            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = domain_emb[:1]  # [1, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [1, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [1, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [1, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src * anchor_src
            xv1_trg = xv1_trg * anchor_trg
        elif anchor_user == 16:
            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [B, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [B, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [B, anchor_user_num]

            # 222
            query_len_src = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(query_src), axis=-1)), [-1, 1])
            query_len_trg = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(query_trg), axis=-1)), [-1, 1])
            anchors_len = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(anchors), axis=-1)), [1, -1])
            denominator_src = tf.clip_by_value(query_len_src * anchors_len, 1e-6,
                                               tf.reduce_max(query_len_src * anchors_len))
            denominator_trg = tf.clip_by_value(query_len_trg * anchors_len, 1e-6,
                                               tf.reduce_max(query_len_trg * anchors_len))
            score_src = score_src / denominator_src
            score_trg = score_trg / denominator_trg

            anchor_src = tf.matmul(score_src, anchors)  # [B, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = anchor_src
            xv1_trg = anchor_trg
        elif anchor_user == 17:

            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = domain_emb[:1]  # [1, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [1, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [1, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [1, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = (xv1_src + anchor_src) / 2
            xv1_trg = (xv1_trg + anchor_trg) / 2
        elif anchor_user == 18:

            domain_emb = get_variable(init, name='domain_emb', shape=[2, embed_size])
            self.anchors = anchors

            # 111
            query_src = xv1_src + domain_emb[:1]  # [1, embed_size]       --> 1/2 ? ; '*' ?
            query_trg = xv1_trg + domain_emb[1:]
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [1, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [1, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [1, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = (xv1_src + anchor_src) / 2
            xv1_trg = (xv1_trg + anchor_trg) / 2
        elif anchor_user == 19:
            self.anchors = anchors
            # 111
            query_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_src,
                                         training=self.training, name='query_src',
                                         reuse=tf.AUTO_REUSE)
            query_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_trg,
                                         training=self.training, name='query_trg',
                                         reuse=tf.AUTO_REUSE)
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [1, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [1, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [1, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = (xv1_src + anchor_src) / 2
            xv1_trg = (xv1_trg + anchor_trg) / 2
        elif anchor_user == 20:
            self.anchors = anchors
            # 111
            query_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_src,
                                         training=self.training, name='query_src',
                                         reuse=tf.AUTO_REUSE)
            query_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_trg,
                                         training=self.training, name='query_trg',
                                         reuse=tf.AUTO_REUSE)
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [1, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [1, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [1, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = xv1_src + anchor_src
            xv1_trg = xv1_trg + anchor_trg
        elif anchor_user == 21:
            self.anchors = anchors
            # 111
            query_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_src,
                                         training=self.training, name='query_src',
                                         reuse=tf.AUTO_REUSE)
            query_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_trg,
                                         training=self.training, name='query_trg',
                                         reuse=tf.AUTO_REUSE)
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [1, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [1, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [1, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
        elif anchor_user == 22:
            self.anchors = anchors
            # 111
            query_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_src,
                                         training=self.training, name='query_src',
                                         reuse=tf.AUTO_REUSE)
            query_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_trg,
                                         training=self.training, name='query_trg',
                                         reuse=tf.AUTO_REUSE)
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [1, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [1, anchor_user_num]

            # 222
            query_len_src = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(query_src), axis=-1)), [-1, 1])
            query_len_trg = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(query_trg), axis=-1)), [-1, 1])
            anchors_len = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(anchors), axis=-1)), [1, -1])
            denominator_src = tf.clip_by_value(query_len_src * anchors_len, 1e-6,
                                               tf.reduce_max(query_len_src * anchors_len))
            denominator_trg = tf.clip_by_value(query_len_trg * anchors_len, 1e-6,
                                               tf.reduce_max(query_len_trg * anchors_len))
            score_src = score_src / denominator_src
            score_trg = score_trg / denominator_trg

            anchor_src = tf.matmul(score_src, anchors)  # [1, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
        elif anchor_user == 23:
            self.anchors = anchors
            # 111
            score_src, _, _, _ = bin_mlp([64, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_src,
                                         training=self.training, name='query_src',
                                         reuse=tf.AUTO_REUSE)
            score_trg, _, _, _ = bin_mlp([64, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_trg,
                                         training=self.training, name='query_trg',
                                         reuse=tf.AUTO_REUSE)

            anchor_src = tf.matmul(score_src, anchors)  # [1, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
        elif anchor_user == 24:
            self.anchors = anchors
            domain_quries = get_variable(init, name='domain_quries', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src, dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg, dim=-1)

            head_embeddings_src = tf.matmul(head_scores_src, anchors)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, head_scores_src)
            anchor_trg = tf.matmul(score_trg, head_scores_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 25:  # 24`
            self.anchors = anchors
            domain_quries = get_variable(init, name='domain_quries', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            head_embeddings_src = tf.matmul(head_scores_src, anchors)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 26:  # 24`
            self.anchors = anchors
            domain_quries = get_variable(init, name='domain_quries', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            head_embeddings_src = tf.matmul(head_scores_src, anchors)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors)  # [head_num, embed_size]

            # head_embeddings_src_matrix = tf.matmul(head_embeddings_src,
            #                                        tf.transpose(head_embeddings_src))  # [head_num, head_num]
            # head_embeddings_trg_matrix = tf.matmul(head_embeddings_trg,
            #                                        tf.transpose(head_embeddings_trg))  # [head_num, head_num]
            # head_embeddings_src_matrix = tf.nn.softmax(head_embeddings_src_matrix / np.sqrt(head_num))
            # head_embeddings_trg_matrix = tf.nn.softmax(head_embeddings_trg_matrix / np.sqrt(head_num))

            # # anchor_src = tf.matmul(score_src, head_scores_src)
            # # anchor_trg = tf.matmul(score_trg, head_scores_trg)
            # anchor_src = tf.matmul(head_embeddings_src_matrix, head_embeddings_src)  # [head_num, embed_size]
            # anchor_trg = tf.matmul(head_embeddings_trg_matrix, head_embeddings_trg)  # [head_num, embed_size]

            anchor_src = tf.reshape(head_embeddings_src, [1, -1])
            anchor_trg = tf.reshape(head_embeddings_trg, [1, -1])
            anchor_src = tf.tile(anchor_src, [tf.shape(xv1_src)[0], 1])
            anchor_trg = tf.tile(anchor_trg, [tf.shape(xv1_trg)[0], 1])

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 27:  # 25 -> xv2
            self.anchors = anchors
            domain_quries = get_variable(init, name='domain_quries', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            head_embeddings_src = tf.matmul(head_scores_src, anchors)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 28:  # 25 -> xv2
            self.anchors = anchors
            domain_quries = get_variable(init, name='domain_quries', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            head_embeddings_src = tf.matmul(head_scores_src, anchors)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = (xv2_src + anchor_src) / 2
            xv2_trg = (xv2_trg + anchor_trg) / 2
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 29:  # 28 -> xv1 + xv2
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 30:  # 28 -> xv1 + xv2
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            _, top_k_indices_src = tf.nn.top_k(head_scores_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(head_scores_trg, k=k)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)

            head_scores_src = head_scores_src * mask_src
            head_scores_trg = head_scores_trg * mask_trg
            head_scores_src = head_scores_src / tf.reduce_sum(head_scores_src, axis=-1, keep_dims=True)
            head_scores_trg = head_scores_trg / tf.reduce_sum(head_scores_trg, axis=-1, keep_dims=True)

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 31:  # 28 -> xv1 + xv2
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            _, top_k_indices_src = tf.nn.top_k(head_scores_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(head_scores_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)

            head_scores_src = head_scores_src * mask_src
            head_scores_trg = head_scores_trg * mask_trg
            head_scores_src = head_scores_src / tf.reduce_sum(head_scores_src, axis=-1, keep_dims=True)
            head_scores_trg = head_scores_trg / tf.reduce_sum(head_scores_trg, axis=-1, keep_dims=True)

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 32:  # 28 -> xv1 + xv2
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            _, top_k_indices_src = tf.nn.top_k(head_scores_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(head_scores_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)
            head_scores_src = head_scores_src * mask_src
            head_scores_trg = head_scores_trg * mask_trg

            head_scores_src = head_scores_src / tf.reduce_sum(head_scores_src, axis=-1, keep_dims=True)
            head_scores_trg = head_scores_trg / tf.reduce_sum(head_scores_trg, axis=-1, keep_dims=True)

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            _, top_k_indices_src = tf.nn.top_k(head_scores_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(head_scores_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)
            head_scores_src = head_scores_src * mask_src
            head_scores_trg = head_scores_trg * mask_trg

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 33:  # 28 -> xv1 + xv2
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            _, top_k_indices_src = tf.nn.top_k(head_scores_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(head_scores_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)
            head_scores_src = head_scores_src * mask_src
            head_scores_trg = head_scores_trg * mask_trg

            # head_scores_src = head_scores_src / tf.reduce_sum(head_scores_src, axis=-1, keep_dims=True)
            # head_scores_trg = head_scores_trg / tf.reduce_sum(head_scores_trg, axis=-1, keep_dims=True)

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            _, top_k_indices_src = tf.nn.top_k(head_scores_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(head_scores_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)
            head_scores_src = head_scores_src * mask_src
            head_scores_trg = head_scores_trg * mask_trg

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 34:  # 28 -> xv1 + xv2
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            _, top_k_indices_src = tf.nn.top_k(head_scores_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(head_scores_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)
            head_scores_src = head_scores_src * mask_src
            head_scores_trg = head_scores_trg * mask_trg

            head_scores_src = head_scores_src / tf.reduce_sum(head_scores_src, axis=-1, keep_dims=True)
            head_scores_trg = head_scores_trg / tf.reduce_sum(head_scores_trg, axis=-1, keep_dims=True)

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size), dim=-1)  # [head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size), dim=-1)

            _, top_k_indices_src = tf.nn.top_k(head_scores_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(head_scores_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)
            head_scores_src = head_scores_src * mask_src
            head_scores_trg = head_scores_trg * mask_trg

            head_scores_src = head_scores_src / tf.reduce_sum(head_scores_src, axis=-1, keep_dims=True)
            head_scores_trg = head_scores_trg / tf.reduce_sum(head_scores_trg, axis=-1, keep_dims=True)

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 35:
            self.anchors = anchors
            # 111
            query_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_src,
                                         training=self.training, name='query_src',
                                         reuse=tf.AUTO_REUSE)
            query_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_trg,
                                         training=self.training, name='query_trg',
                                         reuse=tf.AUTO_REUSE)
            score_src = tf.matmul(query_src, tf.transpose(anchors))  # [1, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors))  # [1, anchor_user_num]

            # 222
            score_src = tf.nn.softmax(score_src, dim=-1)
            score_trg = tf.nn.softmax(score_trg, dim=-1)

            anchor_src = tf.matmul(score_src, anchors)  # [1, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
        elif anchor_user == 36:  # 28 -> xv1 + xv2
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.sigmoid(head_scores_src)
            head_scores_trg = tf.nn.sigmoid(head_scores_trg)

            _, top_k_indices_src = tf.nn.top_k(head_scores_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(head_scores_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)
            head_scores_src = head_scores_src * mask_src
            head_scores_trg = head_scores_trg * mask_trg

            # head_scores_src = head_scores_src / tf.reduce_sum(head_scores_src, axis=-1, keep_dims=True)
            # head_scores_trg = head_scores_trg / tf.reduce_sum(head_scores_trg, axis=-1, keep_dims=True)

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.sigmoid(head_scores_src)
            head_scores_trg = tf.nn.sigmoid(head_scores_trg)

            _, top_k_indices_src = tf.nn.top_k(head_scores_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(head_scores_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)
            head_scores_src = head_scores_src * mask_src
            head_scores_trg = head_scores_trg * mask_trg

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 37:  # 28 -> xv1 + xv2
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.sigmoid(head_scores_src)
            head_scores_trg = tf.nn.sigmoid(head_scores_trg)

            _, top_k_indices_src = tf.nn.top_k(head_scores_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(head_scores_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)
            mask_src = head_scores_src + tf.stop_gradient(mask_src - head_scores_src)
            mask_trg = head_scores_trg + tf.stop_gradient(mask_trg - head_scores_trg)

            head_scores_src = head_scores_src * mask_src
            head_scores_trg = head_scores_trg * mask_trg

            # head_scores_src = head_scores_src / tf.reduce_sum(head_scores_src, axis=-1, keep_dims=True)
            # head_scores_trg = head_scores_trg / tf.reduce_sum(head_scores_trg, axis=-1, keep_dims=True)

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.sigmoid(head_scores_src)
            head_scores_trg = tf.nn.sigmoid(head_scores_trg)

            _, top_k_indices_src = tf.nn.top_k(head_scores_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(head_scores_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)
            mask_src = head_scores_src + tf.stop_gradient(mask_src - head_scores_src)
            mask_trg = head_scores_trg + tf.stop_gradient(mask_trg - head_scores_trg)

            head_scores_src = head_scores_src * mask_src
            head_scores_trg = head_scores_trg * mask_trg

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 38:
            self.anchors1 = anchors1
            self.anchors2 = anchors2
            # 111
            query_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_src,
                                         training=self.training, name='query_src',
                                         reuse=tf.AUTO_REUSE)
            query_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_trg,
                                         training=self.training, name='query_trg',
                                         reuse=tf.AUTO_REUSE)
            score_src = tf.matmul(query_src, tf.transpose(anchors1))  # [1, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors1))  # [1, anchor_user_num]

            # 222
            score_src = tf.nn.sigmoid(score_src)
            score_trg = tf.nn.sigmoid(score_trg)

            _, top_k_indices_src = tf.nn.top_k(score_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(score_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)
            mask_src = score_src + tf.stop_gradient(mask_src - score_src)
            mask_trg = score_trg + tf.stop_gradient(mask_trg - score_trg)

            score_src = score_src * mask_src
            score_trg = score_trg * mask_trg

            anchor_src = tf.matmul(score_src, anchors1)  # [1, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors1)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            # 111
            query_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv2_src,
                                         training=self.training, name='query_src',
                                         reuse=tf.AUTO_REUSE)
            query_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv2_trg,
                                         training=self.training, name='query_trg',
                                         reuse=tf.AUTO_REUSE)
            score_src = tf.matmul(query_src, tf.transpose(anchors2))  # [1, anchor_user_num]
            score_trg = tf.matmul(query_trg, tf.transpose(anchors2))  # [1, anchor_user_num]

            # 222
            score_src = tf.nn.sigmoid(score_src)
            score_trg = tf.nn.sigmoid(score_trg)

            _, top_k_indices_src = tf.nn.top_k(score_src, k=k)
            _, top_k_indices_trg = tf.nn.top_k(score_trg, k=k2)
            mask_src = tf.one_hot(top_k_indices_src, depth=anchor_num)  # [head_num, anchor_num]
            mask_trg = tf.one_hot(top_k_indices_trg, depth=anchor_num)  # [head_num, anchor_num]
            mask_src = tf.reduce_sum(mask_src, axis=1)
            mask_trg = tf.reduce_sum(mask_trg, axis=1)
            mask_src = score_src + tf.stop_gradient(mask_src - score_src)
            mask_trg = score_trg + tf.stop_gradient(mask_trg - score_trg)

            score_src = score_src * mask_src
            score_trg = score_trg * mask_trg

            anchor_src = tf.matmul(score_src, anchors2)  # [1, embed_size]
            anchor_trg = tf.matmul(score_trg, anchors2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
        elif anchor_user == 39:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.sigmoid(head_scores_src / tau)
            head_scores_trg = tf.nn.sigmoid(head_scores_trg / tau)

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.sigmoid(head_scores_src / tau)
            head_scores_trg = tf.nn.sigmoid(head_scores_trg / tau)

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 40:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / tau)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau)

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / tau)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau)

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 41:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            tau_a = get_variable(init, name='tau', shape=[1])
            tau_a = tf.nn.sigmoid(tau_a)

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / tau_a)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau_a)

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / tau_a)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau_a)

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 42:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            tau_a = get_variable(init, name='tau', shape=[2])
            tau_a = tf.nn.sigmoid(tau_a)

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / tau_a[0])
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau_a[0])

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / tau_a[1])
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau_a[1])

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif anchor_user == 43:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            tau_a = get_variable(init, name='tau', shape=[2])
            tau_a = tf.nn.sigmoid(tau_a)

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors1), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / tau_a[0])
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau_a[0])

            head_embeddings_src = tf.matmul(head_scores_src, anchors1)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors1)  # [head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.softmax(head_scores_src / tau_a[1])
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau_a[1])

            head_embeddings_src = tf.matmul(head_scores_src, anchors2)  # [head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, anchors2)  # [head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.matmul(score_src, head_embeddings_src)
            anchor_trg = tf.matmul(score_trg, head_embeddings_trg)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg

        if anchor_user > 0 and anchor_user not in [26]:
            self.score_src = score_src
            self.score_trg = score_trg

        self.xv1_src = xv1_src
        self.xv1_trg = xv1_trg
        hist1 = tf.gather(v_item, self.history1)
        hist2 = tf.gather(v_item, self.history2)

        if hist_type == 1:
            # source
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history1 = user_history1 / tf.expand_dims(tf.cast(self.history_len1, tf.float32), 1)

            # target
            user_history2 = tf.reduce_sum(hist2, axis=-2)
            user_history2 = user_history2 / tf.expand_dims(tf.cast(self.history_len2, tf.float32), 1)
        elif hist_type == 2:
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history2 = tf.reduce_sum(hist2, axis=-2)
        elif hist_type == 3:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)
        elif hist_type == 4:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention1', self.training), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention2', self.training), 1)

        if hist_type > 0:
            user_feat1 = tf.concat([xv1_src, user_history1], axis=-1)
            user_feat2 = tf.concat([xv1_trg, user_history2], axis=-1)
            self.concat_shape += embed_size * 2
        else:
            user_feat1 = xv1_src
            user_feat2 = xv1_trg
            self.concat_shape += embed_size * 1

        h1 = tf.concat([user_feat1, xv2_src], axis=-1)
        h2 = tf.concat([user_feat2, xv2_trg], axis=-1)
        self.h1 = h1
        self.h2 = h2
        self.concat_shape += embed_size

        if cotrain:
            h1, self.layer_kernels, self.layer_biases, nn_h = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                      training=self.training, name='mlp',
                                                                      reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, nn_h = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                      training=self.training, name='mlp',
                                                                      reuse=tf.AUTO_REUSE)
        elif cotrain == 2:
            h1, self.layer_kernels, self.layer_biases, nn_h = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h1,
                                                                        training=self.training, name='mlp',
                                                                        reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, nn_h = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h2,
                                                                        training=self.training, name='mlp',
                                                                        reuse=tf.AUTO_REUSE)
        elif cotrain == 3:
            h1, self.layer_kernels, self.layer_biases, nn_h = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                      training=self.training, name='mlp1',
                                                                      reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, nn_h = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                      training=self.training, name='mlp2',
                                                                      reuse=tf.AUTO_REUSE)
        elif cotrain == 4:
            h1, self.layer_kernels, self.layer_biases, nn_h = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h1,
                                                                        training=self.training, name='mlp1',
                                                                        reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, nn_h = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h2,
                                                                        training=self.training, name='mlp2',
                                                                        reuse=tf.AUTO_REUSE)

        h1 = tf.squeeze(h1)
        h2 = tf.squeeze(h2)

        self.logits1, self.outputs1 = output([h1, b])
        self.logits2, self.outputs2 = output([h2, b])

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.entropy1 = loss(logits=self.logits1, targets=self.labels_src, pos_weight=pos_weight)
                self.origin_loss1 = tf.reduce_mean(self.entropy1)
                self.loss1 = self.origin_loss1

                self.entropy2 = loss(logits=self.logits2, targets=self.labels_trg, pos_weight=pos_weight)
                self.origin_loss2 = tf.reduce_mean(self.entropy2)
                self.loss2 = self.origin_loss2

                if self.anchor_user_orth == 1:
                    grad_vec1 = tf.gradients(self.origin_loss1, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    grad_vec2 = tf.gradients(self.origin_loss2, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    inners = tf.reduce_sum(grad_vec1 * grad_vec2, axis=-1)  # [anchor_user_num]
                    norms1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec1), axis=-1)))  # [anchor_user_num]
                    norms2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec2), axis=-1)))  # [anchor_user_num]
                    self.user_orth_loss = inners / tf.clip_by_value(norms1 * norms2, 1e-8,
                                                                    tf.reduce_max(norms1 * norms2))
                    self.user_orth_loss = tf.reduce_sum(tf.nn.relu(-self.user_orth_loss))
                    self.user_orth_loss = self.user_orth_loss * self.anchor_user_orth_alpha
                    self.loss1 += self.user_orth_loss * 0.5
                    self.loss2 += self.user_orth_loss * 0.5
                elif self.anchor_user_orth == 2:
                    grad_vec1 = tf.gradients(self.origin_loss1, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    grad_vec2 = tf.gradients(self.origin_loss2, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    inners = tf.reduce_sum(grad_vec1 * grad_vec2, axis=-1)  # [anchor_user_num]
                    norms1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec1), axis=-1)))  # [anchor_user_num]
                    norms2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec2), axis=-1)))  # [anchor_user_num]
                    self.user_orth_loss = inners / tf.clip_by_value(norms1 * norms2, 1e-8,
                                                                    tf.reduce_max(norms1 * norms2))
                    self.user_orth_loss = tf.reduce_mean(tf.nn.relu(-self.user_orth_loss))
                    self.user_orth_loss = self.user_orth_loss * self.anchor_user_orth_alpha
                    self.loss1 += self.user_orth_loss * 0.5
                    self.loss2 += self.user_orth_loss * 0.5
                elif self.anchor_user_orth == 3:
                    grad_vec1 = tf.gradients(self.origin_loss1, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    grad_vec2 = tf.gradients(self.origin_loss2, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    inners = tf.reduce_sum(grad_vec1 * grad_vec2, axis=-1)  # [anchor_user_num]
                    norms1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec1), axis=-1)))  # [anchor_user_num]
                    norms2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec2), axis=-1)))  # [anchor_user_num]
                    self.user_orth_loss = inners / tf.clip_by_value(norms1 * norms2, 1e-8,
                                                                    tf.reduce_max(norms1 * norms2))
                    self.user_orth_loss = tf.reduce_mean(tf.abs(self.user_orth_loss))
                    self.user_orth_loss = self.user_orth_loss * self.anchor_user_orth_alpha
                    self.loss1 += self.user_orth_loss * 0.5
                    self.loss2 += self.user_orth_loss * 0.5
                elif self.anchor_user_orth == 4:
                    grad_vec1 = tf.gradients(self.origin_loss1, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    grad_vec2 = tf.gradients(self.origin_loss2, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    norms1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec1), axis=-1)))  # [anchor_user_num]
                    norms2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec2), axis=-1)))  # [anchor_user_num]
                    self.user_orth_loss = tf.reduce_sum(tf.abs(norms1 - norms2))
                    self.user_orth_loss = self.user_orth_loss * self.anchor_user_orth_alpha
                    self.loss1 += self.user_orth_loss * 0.5
                    self.loss2 += self.user_orth_loss * 0.5
                elif self.anchor_user_orth == 5:
                    grad_vec1 = tf.gradients(self.origin_loss1, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    grad_vec2 = tf.gradients(self.origin_loss2, self.anchors)[0]  # [anchor_user_num, anchor_dim]
                    inners = tf.reduce_sum(grad_vec1 * grad_vec2, axis=-1)  # [anchor_user_num]
                    norms1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec1), axis=-1)))  # [anchor_user_num]
                    norms2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(grad_vec2), axis=-1)))  # [anchor_user_num]
                    self.user_orth_loss = inners / tf.clip_by_value(norms1 * norms2, 1e-8,
                                                                    tf.reduce_max(norms1 * norms2))
                    self.user_orth_loss = tf.reduce_mean(tf.nn.relu(self.user_orth_loss))
                    self.user_orth_loss = self.user_orth_loss * self.anchor_user_orth_alpha
                    self.loss1 += self.user_orth_loss * 0.5
                    self.loss2 += self.user_orth_loss * 0.5

                if self.anchor_score_reg == 1:
                    score_reg = tf.sqrt(tf.reduce_sum(tf.square(self.score_src - self.score_trg), axis=-1))
                    score_reg = tf.reduce_mean(score_reg)
                    self.loss1 += score_reg * 0.5
                    self.loss2 += score_reg * 0.5
                elif self.anchor_score_reg == 2:
                    score_reg = tf.sqrt(tf.reduce_sum(tf.square(self.head_scores_src - self.head_scores_trg), axis=-1))
                    score_reg = tf.reduce_mean(score_reg)
                    self.loss1 += score_reg * 0.5
                    self.loss2 += score_reg * 0.5
                elif self.anchor_score_reg == 3:
                    score_reg = tf.sqrt(tf.reduce_sum(tf.square(self.head_scores_src - self.head_scores_trg), axis=-1))
                    score_reg = tf.reduce_mean(-score_reg)
                    self.loss1 += score_reg * 0.5
                    self.loss2 += score_reg * 0.5
                elif self.anchor_score_reg == 4:
                    score_reg = tf.reduce_sum(tf.abs(self.head_scores_src - self.head_scores_trg), axis=-1)
                    score_reg = tf.reduce_mean(-score_reg)
                    self.loss1 += score_reg * 0.5
                    self.loss2 += score_reg * 0.5
                elif self.anchor_score_reg == 5:
                    score_reg = []
                    for i in range(self.head_num):
                        for j in range(self.head_num):
                            head_src = self.head_scores_src[i]
                            head_trg = self.head_scores_trg[j]
                            score_reg.append(-tf.sqrt(tf.reduce_sum(tf.square(head_src - head_trg), axis=-1)))
                    score_reg = sum(score_reg) / len(score_reg)
                    self.loss1 += score_reg * 0.5
                    self.loss2 += score_reg * 0.5
                elif self.anchor_score_reg == 6:
                    score_reg_cross = []
                    for i in range(self.head_num):
                        for j in range(self.head_num):
                            head_src = self.head_scores_src[i]
                            head_trg = self.head_scores_trg[j]
                            score_reg_cross.append(tf.sqrt(tf.reduce_sum(tf.square(head_src - head_trg), axis=-1)))
                    score_reg_cross = sum(score_reg_cross) / len(score_reg_cross)

                    score_reg_inner_src = []
                    for i in range(self.head_num):
                        for j in range(i + 1, self.head_num):
                            head1 = self.head_scores_src[i]
                            head2 = self.head_scores_src[j]
                            score_reg_inner_src.append(tf.sqrt(tf.reduce_sum(tf.square(head1 - head2), axis=-1)))
                    score_reg_inner_src = sum(score_reg_inner_src) / len(score_reg_inner_src)

                    score_reg_inner_trg = []
                    for i in range(self.head_num):
                        for j in range(i + 1, self.head_num):
                            head1 = self.head_scores_trg[i]
                            head2 = self.head_scores_trg[j]
                            score_reg_inner_trg.append(tf.sqrt(tf.reduce_sum(tf.square(head1 - head2), axis=-1)))
                    score_reg_inner_trg = sum(score_reg_inner_trg) / len(score_reg_inner_trg)

                    score_reg = tf.nn.relu(score_reg_inner_src - score_reg_cross) + tf.nn.relu(
                        score_reg_inner_trg - score_reg_cross)

                    self.loss1 += score_reg * 0.5
                    self.loss2 += score_reg * 0.5
                elif self.anchor_score_reg == 7:
                    score_reg_cross = []
                    for i in range(self.head_num):
                        for j in range(self.head_num):
                            head_src = self.head_scores_src[i]
                            head_trg = self.head_scores_trg[j]
                            score_reg_cross.append(tf.sqrt(tf.reduce_sum(tf.square(head_src - head_trg), axis=-1)))
                    score_reg_cross = sum(score_reg_cross) / len(score_reg_cross)

                    score_reg_inner_src = []
                    for i in range(self.head_num):
                        for j in range(i + 1, self.head_num):
                            head1 = self.head_scores_src[i]
                            head2 = self.head_scores_src[j]
                            score_reg_inner_src.append(tf.sqrt(tf.reduce_sum(tf.square(head1 - head2), axis=-1)))
                    score_reg_inner_src = sum(score_reg_inner_src) / len(score_reg_inner_src)

                    score_reg_inner_trg = []
                    for i in range(self.head_num):
                        for j in range(i + 1, self.head_num):
                            head1 = self.head_scores_trg[i]
                            head2 = self.head_scores_trg[j]
                            score_reg_inner_trg.append(tf.sqrt(tf.reduce_sum(tf.square(head1 - head2), axis=-1)))
                    score_reg_inner_trg = sum(score_reg_inner_trg) / len(score_reg_inner_trg)

                    score_reg = score_reg_inner_src - score_reg_cross + score_reg_inner_trg - score_reg_cross

                    self.loss1 += score_reg * 0.5
                    self.loss2 += score_reg * 0.5
                elif self.anchor_score_reg == 8:
                    score_reg_cross = []
                    for i in range(self.head_num):
                        for j in range(self.head_num):
                            head_src = self.head_scores_src[i]
                            head_trg = self.head_scores_trg[j]
                            score_reg_cross.append(tf.sqrt(tf.reduce_sum(tf.square(head_src - head_trg), axis=-1)))
                    score_reg_cross = sum(score_reg_cross) / len(score_reg_cross)

                    score_reg_inner_src = []
                    for i in range(self.head_num):
                        for j in range(i + 1, self.head_num):
                            head1 = self.head_scores_src[i]
                            head2 = self.head_scores_src[j]
                            score_reg_inner_src.append(tf.sqrt(tf.reduce_sum(tf.square(head1 - head2), axis=-1)))
                    score_reg_inner_src = sum(score_reg_inner_src) / len(score_reg_inner_src)

                    score_reg_inner_trg = []
                    for i in range(self.head_num):
                        for j in range(i + 1, self.head_num):
                            head1 = self.head_scores_trg[i]
                            head2 = self.head_scores_trg[j]
                            score_reg_inner_trg.append(tf.sqrt(tf.reduce_sum(tf.square(head1 - head2), axis=-1)))
                    score_reg_inner_trg = sum(score_reg_inner_trg) / len(score_reg_inner_trg)

                    score_reg = tf.maximum(score_reg_inner_src, score_reg_inner_trg) - 0.2 * (
                            score_reg_inner_src + score_reg_inner_trg) - score_reg_cross

                    self.loss1 += score_reg * 0.5
                    self.loss2 += score_reg * 0.5
                elif self.anchor_score_reg == 9:
                    score_reg_cross = []
                    for i in range(self.head_num):
                        for j in range(self.head_num):
                            head1 = self.head_scores_src[i]
                            head2 = self.head_scores_trg[j]

                            inner = tf.reduce_sum(head1 * head2, axis=-1)
                            len1 = tf.sqrt(tf.reduce_sum(tf.square(head1), axis=-1))
                            len2 = tf.sqrt(tf.reduce_sum(tf.square(head2), axis=-1))
                            dist = inner / tf.clip_by_value(len1 * len2, 1e-6, tf.reduce_max(len1 * len2))
                            score_reg_cross.append(1 - dist)
                    score_reg_cross = sum(score_reg_cross) / len(score_reg_cross)

                    score_reg_inner_src = []
                    for i in range(self.head_num):
                        for j in range(i + 1, self.head_num):
                            head1 = self.head_scores_src[i]
                            head2 = self.head_scores_src[j]

                            inner = tf.reduce_sum(head1 * head2, axis=-1)
                            len1 = tf.sqrt(tf.reduce_sum(tf.square(head1), axis=-1))
                            len2 = tf.sqrt(tf.reduce_sum(tf.square(head2), axis=-1))
                            dist = inner / tf.clip_by_value(len1 * len2, 1e-6, tf.reduce_max(len1 * len2))
                            score_reg_inner_src.append(1 - dist)
                    score_reg_inner_src = sum(score_reg_inner_src) / len(score_reg_inner_src)

                    score_reg_inner_trg = []
                    for i in range(self.head_num):
                        for j in range(i + 1, self.head_num):
                            head1 = self.head_scores_trg[i]
                            head2 = self.head_scores_trg[j]

                            inner = tf.reduce_sum(head1 * head2, axis=-1)
                            len1 = tf.sqrt(tf.reduce_sum(tf.square(head1), axis=-1))
                            len2 = tf.sqrt(tf.reduce_sum(tf.square(head2), axis=-1))
                            dist = inner / tf.clip_by_value(len1 * len2, 1e-6, tf.reduce_max(len1 * len2))
                            score_reg_inner_trg.append(1 - dist)
                    score_reg_inner_trg = sum(score_reg_inner_trg) / len(score_reg_inner_trg)

                    score_reg = tf.maximum(score_reg_inner_src, score_reg_inner_trg) - 0.2 * (
                            score_reg_inner_src + score_reg_inner_trg) - score_reg_cross

                    self.loss1 += score_reg * 0.5
                    self.loss2 += score_reg * 0.5

                if self.l2 > 0:
                    _loss1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(self.h1), axis=-1)))
                    _loss2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(self.h2), axis=-1)))
                    self.loss1 += self.l2 * tf.reduce_mean(_loss1)
                    self.loss2 += self.l2 * tf.reduce_mean(_loss2)

                _loss_ = self.loss
                self.optimizer1 = optimizer.minimize(loss=self.loss1,
                                                     global_step=global_step)
                self.optimizer2 = optimizer.minimize(loss=self.loss2,
                                                     global_step=global_step)


class Model_Sharespecific(Model):
    def __init__(self, init='xavier', user_max_id=None, src_item_max_id=None, trg_item_max_id=None,
                 embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, batch_norm=False, layer_norm=False,
                 l1_w=None, l1_v=None, layer_l1=None, user_his_len=None, hist_type=None,
                 cotrain=None, input_num_share=None, anchor_input=None,
                 input_num_specific=None, input_head_num=None, anchor_net=None, net_num_share=None,
                 net_num_specific=None, net_head_num=None, l2=None):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.layer_l2 = layer_l2
        self.l1_w = l1_w
        self.layer_l1 = layer_l1
        self.l1_v = l1_v
        self.hist_type = hist_type
        self.embed_size = embed_size
        self.user_his_len = user_his_len
        self.layer_sizes = layer_sizes
        self.layer_acts = layer_acts
        self.layer_keeps = layer_keeps
        self.cotrain = cotrain
        self.concat_shape = 0
        self.l2 = l2

        with tf.name_scope('input'):
            self.user_id_src = tf.placeholder(tf.int32, [None], name='user_id_src')
            self.item_id_src = tf.placeholder(tf.int32, [None], name='item_id_src')
            self.user_id_trg = tf.placeholder(tf.int32, [None], name='user_id_trg')
            self.item_id_trg = tf.placeholder(tf.int32, [None], name='item_id_trg')
            self.history1 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items1')
            self.history_len1 = tf.placeholder(tf.int32, [None], name='history_items_len1')
            self.history2 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items2')
            self.history_len2 = tf.placeholder(tf.int32, [None], name='history_items_len2')
            self.labels_src = tf.placeholder(tf.float32, [None], name='label_src')
            self.labels_trg = tf.placeholder(tf.float32, [None], name='label_trg')
            self.training = tf.placeholder(dtype=tf.bool, name='training')

        v_user = get_variable(init, name='v_user', shape=[user_max_id, embed_size])
        v_item = get_variable(init, name='v_item', shape=[src_item_max_id + trg_item_max_id, embed_size])
        b = get_variable('zero', name='b', shape=[1])
        self.v_user = v_user
        self.v_item = v_item

        xv1_src = tf.gather(v_user, self.user_id_src)
        xv2_src = tf.gather(v_item, self.item_id_src)
        xv1_trg = tf.gather(v_user, self.user_id_trg)
        xv2_trg = tf.gather(v_item, self.item_id_trg)

        input_share_anchors1 = get_variable(init, name='input_share_anchors1', shape=[input_num_share, embed_size])
        input_share_anchors2 = get_variable(init, name='input_share_anchors2', shape=[input_num_share, embed_size])

        input_specific_anchors1 = get_variable(init, name='input_specific_anchors1',
                                               shape=[2, input_num_specific, embed_size])
        input_specific_anchors2 = get_variable(init, name='input_specific_anchors2',
                                               shape=[2, input_num_specific, embed_size])

        if anchor_input == 1:
            weights_src_ss, _, _, _ = bin_mlp([64, 32, 2], ['tanh', 'tanh', None], layer_keeps,
                                              tf.concat([xv1_src, xv2_src], axis=-1),
                                              training=self.training, name='share_spec_src',
                                              reuse=tf.AUTO_REUSE)
            weights_trg_ss, _, _, _ = bin_mlp([64, 32, 2], ['tanh', 'tanh', None], layer_keeps,
                                              tf.concat([xv1_trg, xv2_trg], axis=-1),
                                              training=self.training, name='share_spec_trg',
                                              reuse=tf.AUTO_REUSE)

            weights_src_ss = tf.expand_dims(tf.nn.softmax(weights_src_ss), axis=-1)  # [N, 2, 1]
            weights_trg_ss = tf.expand_dims(tf.nn.softmax(weights_trg_ss), axis=-1)  # [N, 2, 1]

            # user !!!!
            # share$$$$$
            # 111: common queries || unique queries
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, input_head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [input_head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors_share = tf.expand_dims(tf.transpose(input_share_anchors1),
                                            axis=0)  # [1, embed_size, input_num_share]
            _anchors_share = tf.tile(_anchors_share,
                                     [input_head_num, 1, 1])  # [input_head_num, embed_size, input_num_share]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors_share),
                                         [input_head_num, input_num_share])  # [input_head_num, input_num_share]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors_share),
                                         [input_head_num, input_num_share])  # [input_head_num, input_num_share]
            # 222: softmax; sigmoid
            head_scores_src = tf.nn.softmax(head_scores_src)
            head_scores_trg = tf.nn.softmax(head_scores_trg)

            head_embeddings_src = tf.matmul(head_scores_src, input_share_anchors1)  # [input_head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, input_share_anchors1)  # [input_head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, input_head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, input_head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src_share = tf.expand_dims(tf.matmul(score_src, head_embeddings_src), axis=1)
            anchor_trg_share = tf.expand_dims(tf.matmul(score_trg, head_embeddings_trg), axis=1)

            # specific$$$$$
            _anchors_spec_src = tf.expand_dims(tf.transpose(input_specific_anchors1[0]),
                                               axis=0)  # [1, embed_size, input_num_specific]
            _anchors_spec_src = tf.tile(_anchors_spec_src,
                                        [input_head_num, 1, 1])  # [input_head_num, embed_size, input_num_specific]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors_spec_src),
                                         [input_head_num, input_num_specific])  # [input_head_num, input_num_specific]

            _anchors_spec_trg = tf.expand_dims(tf.transpose(input_specific_anchors1[1]),
                                               axis=0)  # [1, embed_size, input_num_specific]
            _anchors_spec_trg = tf.tile(_anchors_spec_trg,
                                        [input_head_num, 1, 1])  # [input_head_num, embed_size, input_num_specific]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors_spec_trg),
                                         [input_head_num, input_num_specific])  # [input_head_num, input_num_specific]
            # 222: softmax; sigmoid
            head_scores_src = tf.nn.softmax(head_scores_src)
            head_scores_trg = tf.nn.softmax(head_scores_trg)

            head_embeddings_src = tf.matmul(head_scores_src, input_specific_anchors1[0])  # [input_head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, input_specific_anchors1[1])  # [input_head_num, embed_size]

            score_src = tf.matmul(xv1_src, tf.transpose(head_embeddings_src))  # [B, input_head_num]
            score_trg = tf.matmul(xv1_trg, tf.transpose(head_embeddings_trg))  # [B, input_head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src_spec = tf.expand_dims(tf.matmul(score_src, head_embeddings_src), axis=1)  # [B, 1, embed_size]
            anchor_trg_spec = tf.expand_dims(tf.matmul(score_trg, head_embeddings_trg), axis=1)

            anchor_src = tf.reduce_sum(weights_src_ss * tf.concat([anchor_src_share, anchor_src_spec], axis=1), axis=1)
            anchor_trg = tf.reduce_sum(weights_trg_ss * tf.concat([anchor_trg_share, anchor_trg_spec], axis=1), axis=1)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            # item !!!!!!
            # share$$$$$
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, input_head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [input_head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors_share = tf.expand_dims(tf.transpose(input_share_anchors2),
                                            axis=0)  # [1, embed_size, input_num_share]
            _anchors_share = tf.tile(_anchors_share,
                                     [input_head_num, 1, 1])  # [input_head_num, embed_size, input_num_share]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors_share),
                                         [input_head_num, input_num_share])  # [input_head_num, input_num_share]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors_share),
                                         [input_head_num, input_num_share])  # [input_head_num, input_num_share]
            head_scores_src = tf.nn.sigmoid(head_scores_src)
            head_scores_trg = tf.nn.sigmoid(head_scores_trg)

            head_embeddings_src = tf.matmul(head_scores_src, input_share_anchors2)  # [input_head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, input_share_anchors2)  # [input_head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, input_head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, input_head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src_share = tf.expand_dims(tf.matmul(score_src, head_embeddings_src), axis=1)
            anchor_trg_share = tf.expand_dims(tf.matmul(score_trg, head_embeddings_trg), axis=1)

            # specific$$$$$
            _anchors_spec_src = tf.expand_dims(tf.transpose(input_specific_anchors2[0]),
                                               axis=0)  # [1, embed_size, input_num_specific]
            _anchors_spec_src = tf.tile(_anchors_spec_src,
                                        [input_head_num, 1, 1])  # [input_head_num, embed_size, input_num_specific]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors_spec_src),
                                         [input_head_num, input_num_specific])  # [input_head_num, input_num_specific]

            _anchors_spec_trg = tf.expand_dims(tf.transpose(input_specific_anchors2[1]),
                                               axis=0)  # [1, embed_size, input_num_specific]
            _anchors_spec_trg = tf.tile(_anchors_spec_trg,
                                        [input_head_num, 1, 1])  # [input_head_num, embed_size, input_num_specific]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors_spec_trg),
                                         [input_head_num, input_num_specific])  # [input_head_num, input_num_specific]
            # 222: softmax; sigmoid
            head_scores_src = tf.nn.softmax(head_scores_src)
            head_scores_trg = tf.nn.softmax(head_scores_trg)

            head_embeddings_src = tf.matmul(head_scores_src, input_specific_anchors2[0])  # [input_head_num, embed_size]
            head_embeddings_trg = tf.matmul(head_scores_trg, input_specific_anchors2[1])  # [input_head_num, embed_size]

            score_src = tf.matmul(xv2_src, tf.transpose(head_embeddings_src))  # [B, input_head_num]
            score_trg = tf.matmul(xv2_trg, tf.transpose(head_embeddings_trg))  # [B, input_head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src_spec = tf.expand_dims(tf.matmul(score_src, head_embeddings_src), axis=1)  # [B, 1, embed_size]
            anchor_trg_spec = tf.expand_dims(tf.matmul(score_trg, head_embeddings_trg), axis=1)

            anchor_src = tf.reduce_sum(weights_src_ss * tf.concat([anchor_src_share, anchor_src_spec], axis=1), axis=1)
            anchor_trg = tf.reduce_sum(weights_trg_ss * tf.concat([anchor_trg_share, anchor_trg_spec], axis=1), axis=1)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg

        if anchor_input > 0 and anchor_input not in [26]:
            self.score_src = score_src
            self.score_trg = score_trg

        self.xv1_src = xv1_src
        self.xv1_trg = xv1_trg
        hist1 = tf.gather(v_item, self.history1)
        hist2 = tf.gather(v_item, self.history2)

        if hist_type == 1:
            # source
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history1 = user_history1 / tf.expand_dims(tf.cast(self.history_len1, tf.float32), 1)

            # target
            user_history2 = tf.reduce_sum(hist2, axis=-2)
            user_history2 = user_history2 / tf.expand_dims(tf.cast(self.history_len2, tf.float32), 1)
        elif hist_type == 2:
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history2 = tf.reduce_sum(hist2, axis=-2)
        elif hist_type == 3:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)
        elif hist_type == 4:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention1', self.training), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention2', self.training), 1)

        if hist_type > 0:
            user_feat1 = tf.concat([xv1_src, user_history1], axis=-1)
            user_feat2 = tf.concat([xv1_trg, user_history2], axis=-1)
            self.concat_shape += embed_size * 2
        else:
            user_feat1 = xv1_src
            user_feat2 = xv1_trg
            self.concat_shape += embed_size * 1

        h1 = tf.concat([user_feat1, xv2_src], axis=-1)
        h2 = tf.concat([user_feat2, xv2_trg], axis=-1)
        self.h1 = h1
        self.h2 = h2
        self.concat_shape += embed_size

        if cotrain:
            h1, self.layer_kernels, self.layer_biases, nn_h = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                      training=self.training, name='mlp',
                                                                      reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, nn_h = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                      training=self.training, name='mlp',
                                                                      reuse=tf.AUTO_REUSE)
        elif cotrain == 2:
            h1, self.layer_kernels, self.layer_biases, nn_h = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h1,
                                                                        training=self.training, name='mlp',
                                                                        reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, nn_h = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h2,
                                                                        training=self.training, name='mlp',
                                                                        reuse=tf.AUTO_REUSE)
        elif cotrain == 3:
            h1, self.layer_kernels, self.layer_biases, nn_h = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                      training=self.training, name='mlp1',
                                                                      reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, nn_h = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                      training=self.training, name='mlp2',
                                                                      reuse=tf.AUTO_REUSE)
        elif cotrain == 4:
            h1, self.layer_kernels, self.layer_biases, nn_h = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h1,
                                                                        training=self.training, name='mlp1',
                                                                        reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, nn_h = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h2,
                                                                        training=self.training, name='mlp2',
                                                                        reuse=tf.AUTO_REUSE)

        h1 = tf.squeeze(h1)
        h2 = tf.squeeze(h2)

        self.logits1, self.outputs1 = output([h1, b])
        self.logits2, self.outputs2 = output([h2, b])

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.entropy1 = loss(logits=self.logits1, targets=self.labels_src, pos_weight=pos_weight)
                self.origin_loss1 = tf.reduce_mean(self.entropy1)
                self.loss1 = self.origin_loss1

                self.entropy2 = loss(logits=self.logits2, targets=self.labels_trg, pos_weight=pos_weight)
                self.origin_loss2 = tf.reduce_mean(self.entropy2)
                self.loss2 = self.origin_loss2

                if self.l2 > 0:
                    _loss1 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(self.h1), axis=-1)))
                    _loss2 = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(self.h2), axis=-1)))
                    self.loss1 += self.l2 * tf.reduce_mean(_loss1)
                    self.loss2 += self.l2 * tf.reduce_mean(_loss2)

                _loss_ = self.loss
                self.optimizer1 = optimizer.minimize(loss=self.loss1,
                                                     global_step=global_step)
                self.optimizer2 = optimizer.minimize(loss=self.loss2,
                                                     global_step=global_step)


class Model_Hardset(Model):
    def __init__(self, init='xavier', user_max_id=None, src_item_max_id=None, trg_item_max_id=None,
                 embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, batch_norm=False, layer_norm=False,
                 l1_w=None, l1_v=None, layer_l1=None, user_his_len=None, hist_type=None, anchor_num=5,
                 cotrain=None, cluster_embeddings=None, user_embeddings=None, l2=None, anchor_user_orth=None,
                 anchor_user_orth_alpha=None, anchor_score_reg=None, head_num=None, k_ratio=None, k_ratio2=None,
                 tau=None, hardness=None, ae=1, calc_pattern=None, thres=0.5):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.layer_l2 = layer_l2
        self.l1_w = l1_w
        self.layer_l1 = layer_l1
        self.l1_v = l1_v
        self.hist_type = hist_type
        self.embed_size = embed_size
        self.user_his_len = user_his_len
        self.layer_sizes = layer_sizes
        self.layer_acts = layer_acts
        self.layer_keeps = layer_keeps
        self.cotrain = cotrain
        self.anchor_num = anchor_num
        self.l2 = l2
        self.anchor_user_orth = anchor_user_orth
        self.anchor_user_orth_alpha = anchor_user_orth_alpha
        self.concat_shape = 0
        self.anchor_score_reg = anchor_score_reg
        self.head_num = head_num
        k = math.ceil(k_ratio * anchor_num)
        k2 = math.ceil(k_ratio2 * anchor_num)
        self.tau = tau
        self.hardness = hardness
        self.ae = ae
        self.calc_pattern = calc_pattern

        with tf.name_scope('input'):
            self.user_id_src = tf.placeholder(tf.int32, [None], name='user_id_src')
            self.item_id_src = tf.placeholder(tf.int32, [None], name='item_id_src')
            self.user_id_trg = tf.placeholder(tf.int32, [None], name='user_id_trg')
            self.item_id_trg = tf.placeholder(tf.int32, [None], name='item_id_trg')
            self.history1 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items1')
            self.history_len1 = tf.placeholder(tf.int32, [None], name='history_items_len1')
            self.history2 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items2')
            self.history_len2 = tf.placeholder(tf.int32, [None], name='history_items_len2')
            self.labels_src = tf.placeholder(tf.float32, [None], name='label_src')
            self.labels_trg = tf.placeholder(tf.float32, [None], name='label_trg')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.hardness_score_src = tf.placeholder(tf.float32, [None], name='hardness_score_src')
            self.hardness_score_trg = tf.placeholder(tf.float32, [None], name='hardness_score_trg')
            self.hardness_score_src2 = tf.placeholder(tf.float32, [None], name='hardness_score_src2')
            self.hardness_score_trg2 = tf.placeholder(tf.float32, [None], name='hardness_score_trg2')
            self.hardness_num = tf.placeholder(tf.int32, [None], name='hardness_num')

        if user_embeddings is not None:
            v_user = tf.Variable(user_embeddings)
        else:
            v_user = get_variable(init, name='v_user', shape=[user_max_id, embed_size])
        v_item = get_variable(init, name='v_item', shape=[src_item_max_id + trg_item_max_id, embed_size])
        b = get_variable('zero', name='b', shape=[1])
        self.v_user = v_user
        self.v_item = v_item

        # if hardness not in [31]:
        if hardness not in [30]:
            xv1_src = tf.gather(v_user, self.user_id_src)
            xv2_src = tf.gather(v_item, self.item_id_src)
            xv1_trg = tf.gather(v_user, self.user_id_trg)
            xv2_trg = tf.gather(v_item, self.item_id_trg)
        else:
            v_user2 = get_variable(init, name='v_user_trg', shape=[user_max_id, embed_size])
            xv1_src = tf.gather(v_user, self.user_id_src)
            xv2_src = tf.gather(v_item, self.item_id_src)
            xv1_trg = tf.gather(v_user2, self.user_id_trg)
            xv2_trg = tf.gather(v_item, self.item_id_trg)
        self.xv1_src_origin = xv1_src
        self.xv2_src_origin = xv2_src
        self.xv1_trg_origin = xv1_trg
        self.xv2_trg_origin = xv2_trg

        if ae == 1:
            self.ae_xv1_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.xv1_src_origin, training=self.training, name='generator_encoder_src1',
                                                                      reuse=tf.AUTO_REUSE)

            self.ae_xv1_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.xv1_trg_origin, training=self.training, name='generator_encoder_trg1',
                                                                      reuse=tf.AUTO_REUSE)

            self.ae_xv1_src_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.xv1_src_origin, training=self.training, name='decoder_src1',
                                                                      reuse=tf.AUTO_REUSE)

            self.ae_xv1_trg_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.xv1_trg_origin, training=self.training, name='decoder_trg1',
                                                                      reuse=tf.AUTO_REUSE)


            h_dis_src, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.ae_xv1_src, training=self.training, name='discriminator',
                                                                      reuse=tf.AUTO_REUSE)
            self.h_dis_src = tf.squeeze(h_dis_src)
            h_dis_trg, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                       self.ae_xv1_trg, training=self.training, name='discriminator',
                                       reuse=tf.AUTO_REUSE)
            self.h_dis_trg = tf.squeeze(h_dis_trg)
        elif ae == 2:
            self.ae_xv1_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.xv1_src_origin, training=self.training,
                                               name='generator_encoder1',
                                               reuse=tf.AUTO_REUSE)

            self.ae_xv1_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.xv1_trg_origin, training=self.training,
                                               name='generator_encoder1',
                                               reuse=tf.AUTO_REUSE)

            self.ae_xv1_src_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'],
                                                       layer_keeps,
                                                       self.xv1_src_origin, training=self.training,
                                                       name='decoder_src1',
                                                       reuse=tf.AUTO_REUSE)

            self.ae_xv1_trg_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'],
                                                       layer_keeps,
                                                       self.xv1_trg_origin, training=self.training,
                                                       name='decoder_trg1',
                                                       reuse=tf.AUTO_REUSE)

            h_dis_src, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                         self.ae_xv1_src, training=self.training, name='discriminator',
                                         reuse=tf.AUTO_REUSE)
            self.h_dis_src = tf.squeeze(h_dis_src)
            h_dis_trg, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                         self.ae_xv1_trg, training=self.training, name='discriminator',
                                         reuse=tf.AUTO_REUSE)
            self.h_dis_trg = tf.squeeze(h_dis_trg)
        elif ae == 3:
            self.ae_xv1_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv1_src_origin, training=self.training, name='generator1_encoder_src',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv1_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv1_trg_origin, training=self.training, name='generator1_encoder_trg',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv1_src_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.xv1_src_origin, training=self.training, name='decoder1_src',
                                                                      reuse=tf.AUTO_REUSE)

            self.ae_xv1_trg_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.xv1_trg_origin, training=self.training, name='decoder1_trg',
                                                                      reuse=tf.AUTO_REUSE)

            h_dis_src, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.ae_xv1_src, training=self.training, name='discriminator1',
                                                                      reuse=tf.AUTO_REUSE)
            self.h_dis_src1 = tf.squeeze(h_dis_src)
            h_dis_trg, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                       self.ae_xv1_trg, training=self.training, name='discriminator1',
                                       reuse=tf.AUTO_REUSE)
            self.h_dis_trg1 = tf.squeeze(h_dis_trg)



            self.ae_xv2_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv2_src_origin, training=self.training, name='generator2_encoder_src',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv2_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv2_trg_origin, training=self.training, name='generator2_encoder_trg',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv2_src_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv2_src_origin, training=self.training, name='decoder2_src',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv2_trg_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv2_trg_origin, training=self.training, name='decoder2_trg',
                                                                  reuse=tf.AUTO_REUSE)

            h_dis_src, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.ae_xv1_src, training=self.training, name='discriminator2',
                                                                      reuse=tf.AUTO_REUSE)
            self.h_dis_src2 = tf.squeeze(h_dis_src)
            h_dis_trg, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                       self.ae_xv1_trg, training=self.training, name='discriminator2',
                                       reuse=tf.AUTO_REUSE)
            self.h_dis_trg2 = tf.squeeze(h_dis_trg)
        elif ae == 4:
            self.ae_xv1_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv1_src_origin, training=self.training, name='generator1_encoder_src',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv1_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv1_trg_origin, training=self.training, name='generator1_encoder_trg',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv1_src_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', None], layer_keeps,
                                               self.xv1_src_origin, training=self.training, name='decoder1_src',
                                                                      reuse=tf.AUTO_REUSE)

            self.ae_xv1_trg_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', None], layer_keeps,
                                               self.xv1_trg_origin, training=self.training, name='decoder1_trg',
                                                                      reuse=tf.AUTO_REUSE)

            h_dis_src, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.ae_xv1_src, training=self.training, name='discriminator1',
                                                                      reuse=tf.AUTO_REUSE)
            self.h_dis_src1 = tf.squeeze(h_dis_src)
            h_dis_trg, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                       self.ae_xv1_trg, training=self.training, name='discriminator1',
                                       reuse=tf.AUTO_REUSE)
            self.h_dis_trg1 = tf.squeeze(h_dis_trg)



            self.ae_xv2_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv2_src_origin, training=self.training, name='generator2_encoder_src',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv2_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv2_trg_origin, training=self.training, name='generator2_encoder_trg',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv2_src_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', None], layer_keeps,
                                           self.xv2_src_origin, training=self.training, name='decoder2_src',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv2_trg_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', None], layer_keeps,
                                           self.xv2_trg_origin, training=self.training, name='decoder2_trg',
                                                                  reuse=tf.AUTO_REUSE)

            h_dis_src, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.ae_xv1_src, training=self.training, name='discriminator2',
                                                                      reuse=tf.AUTO_REUSE)
            self.h_dis_src2 = tf.squeeze(h_dis_src)
            h_dis_trg, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                       self.ae_xv1_trg, training=self.training, name='discriminator2',
                                       reuse=tf.AUTO_REUSE)
            self.h_dis_trg2 = tf.squeeze(h_dis_trg)
        elif ae == 5:
            self.ae_xv1_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv1_src_origin, training=self.training, name='generator1_encoder_src',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv1_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv1_trg_origin, training=self.training, name='generator1_encoder_trg',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv1_src_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'tanh'], layer_keeps,
                                               self.xv1_src_origin, training=self.training, name='decoder1_src',
                                                                      reuse=tf.AUTO_REUSE)

            self.ae_xv1_trg_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'tanh'], layer_keeps,
                                               self.xv1_trg_origin, training=self.training, name='decoder1_trg',
                                                                      reuse=tf.AUTO_REUSE)

            h_dis_src, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.ae_xv1_src, training=self.training, name='discriminator1',
                                                                      reuse=tf.AUTO_REUSE)
            self.h_dis_src1 = tf.squeeze(h_dis_src)
            h_dis_trg, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                       self.ae_xv1_trg, training=self.training, name='discriminator1',
                                       reuse=tf.AUTO_REUSE)
            self.h_dis_trg1 = tf.squeeze(h_dis_trg)



            self.ae_xv2_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv2_src_origin, training=self.training, name='generator2_encoder_src',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv2_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                           self.xv2_trg_origin, training=self.training, name='generator2_encoder_trg',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv2_src_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'tanh'], layer_keeps,
                                           self.xv2_src_origin, training=self.training, name='decoder2_src',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv2_trg_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'tanh'], layer_keeps,
                                           self.xv2_trg_origin, training=self.training, name='decoder2_trg',
                                                                  reuse=tf.AUTO_REUSE)

            h_dis_src, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.ae_xv1_src, training=self.training, name='discriminator2',
                                                                      reuse=tf.AUTO_REUSE)
            self.h_dis_src2 = tf.squeeze(h_dis_src)
            h_dis_trg, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                       self.ae_xv1_trg, training=self.training, name='discriminator2',
                                       reuse=tf.AUTO_REUSE)
            self.h_dis_trg2 = tf.squeeze(h_dis_trg)
        elif ae in [6, 7]:
            self.ae_xv1_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'tanh'], layer_keeps,
                                           self.xv1_src_origin, training=self.training, name='generator1_encoder_src',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv1_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'tanh'], layer_keeps,
                                           self.xv1_trg_origin, training=self.training, name='generator1_encoder_trg',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv1_src_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'tanh'], layer_keeps,
                                               self.xv1_src_origin, training=self.training, name='decoder1_src',
                                                                      reuse=tf.AUTO_REUSE)

            self.ae_xv1_trg_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'tanh'], layer_keeps,
                                               self.xv1_trg_origin, training=self.training, name='decoder1_trg',
                                                                      reuse=tf.AUTO_REUSE)

            h_dis_src, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.ae_xv1_src, training=self.training, name='discriminator1',
                                                                      reuse=tf.AUTO_REUSE)
            self.h_dis_src1 = tf.squeeze(h_dis_src)
            h_dis_trg, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                       self.ae_xv1_trg, training=self.training, name='discriminator1',
                                       reuse=tf.AUTO_REUSE)
            self.h_dis_trg1 = tf.squeeze(h_dis_trg)



            self.ae_xv2_src, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'tanh'], layer_keeps,
                                           self.xv2_src_origin, training=self.training, name='generator2_encoder_src',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv2_trg, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'tanh'], layer_keeps,
                                           self.xv2_trg_origin, training=self.training, name='generator2_encoder_trg',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv2_src_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'tanh'], layer_keeps,
                                           self.xv2_src_origin, training=self.training, name='decoder2_src',
                                                                  reuse=tf.AUTO_REUSE)

            self.ae_xv2_trg_decoded, _, _, _ = bin_mlp([64, 64, embed_size], ['tanh', 'tanh', 'tanh'], layer_keeps,
                                           self.xv2_trg_origin, training=self.training, name='decoder2_trg',
                                                                  reuse=tf.AUTO_REUSE)

            h_dis_src, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                               self.ae_xv1_src, training=self.training, name='discriminator2',
                                                                      reuse=tf.AUTO_REUSE)
            self.h_dis_src2 = tf.squeeze(h_dis_src)
            h_dis_trg, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'], layer_keeps,
                                       self.ae_xv1_trg, training=self.training, name='discriminator2',
                                       reuse=tf.AUTO_REUSE)
            self.h_dis_trg2 = tf.squeeze(h_dis_trg)

        anchors1 = get_variable(init, name='anchors1', shape=[anchor_num, embed_size])
        anchors2 = get_variable(init, name='anchors2', shape=[anchor_num, embed_size])

        if hardness == 1:
            # mask_scores_src, _, _, _ = bin_mlp([64, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_src,
            #                                                        training=self.training, name='gate1',
            #                                                        reuse=tf.AUTO_REUSE)
            # mask_scores_trg, _, _, _ = bin_mlp([64, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'], layer_keeps, xv1_trg,
            #                               training=self.training, name='gate2',
            #                               reuse=tf.AUTO_REUSE)
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_num_src = tf.gather(self.hardness_num, self.user_id_src)  # [B,]
            gate_num_trg = tf.gather(self.hardness_num, self.user_id_trg)  # [B,]

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, embed_size]
            queries_trg = domain_quries1[1]

            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(queries_src, _anchors)      # [head_num, anchor_num]
            head_scores_trg = tf.matmul(queries_trg, _anchors)      # [head_num, anchor_num]
            head_scores_src = tf.nn.sigmoid(head_scores_src)
            head_scores_trg = tf.nn.sigmoid(head_scores_trg)

            # sorted anchor-embeddings
            sorted_indices_src = tf.argsort(head_scores_src, axis=-1)  # [head_num, anchor_num]
            sorted_indices_trg = tf.argsort(head_scores_trg, axis=-1)
            # sorted_indices_src = tf.nn.top_k(head_scores_src, anchor_num, sorted=True).indices        # [head_num, anchor_num]
            # sorted_indices_trg = tf.nn.top_k(head_scores_trg, anchor_num, sorted=True).indices
            sorted_indices_src = tf.expand_dims(sorted_indices_src, axis=0)  # [1, head_num, anchor_num]
            sorted_indices_trg = tf.expand_dims(sorted_indices_trg, axis=0)  # [1, head_num, anchor_num]
            sorted_anchors_src = tf.gather(anchors1, sorted_indices_src)  # [1, head_num, anchor_num, embed_size]
            sorted_anchors_trg = tf.gather(anchors1, sorted_indices_trg)  # [1, head_num, anchor_num, embed_size]

            self.a = head_scores_src
            self.b = sorted_indices_src

            # sequence_mask
            mask_src = tf.cast(tf.sequence_mask(gate_num_src, maxlen=anchor_num), tf.float32)  # [B, anchor_num]
            mask_trg = tf.cast(tf.sequence_mask(gate_num_trg, maxlen=anchor_num), tf.float32)  # [B, anchor_num]
            mask_src = tf.reshape(mask_src, [-1, 1, anchor_num, 1])  # [B, 1, anchor_num, 1]
            mask_trg = tf.reshape(mask_trg, [-1, 1, anchor_num, 1])  # [B, 1, anchor_num, 1]
            # sorted_head_scores_src = tf.nn.top_k(head_scores_src, anchor_num, sorted=True).values
            # sorted_head_scores_trg = tf.nn.top_k(head_scores_trg, anchor_num, sorted=True).values
            sorted_head_scores_src = tf.sort(head_scores_src, axis=-1)      # [head_num, anchor_num]
            sorted_head_scores_trg = tf.sort(head_scores_trg, axis=-1)
            sorted_head_scores_src = tf.reshape(sorted_head_scores_src, [1, head_num, anchor_num, 1])
            sorted_head_scores_trg = tf.reshape(sorted_head_scores_trg, [1, head_num, anchor_num, 1])

            # !!!!!
            mask_src = sorted_head_scores_src + tf.stop_gradient(mask_src - sorted_head_scores_src)
            mask_trg = sorted_head_scores_trg + tf.stop_gradient(mask_trg - sorted_head_scores_trg)

            head_embeddings_src = sorted_anchors_src * mask_src  # [B, head_num, anchor_num, embed_size]
            head_embeddings_trg = sorted_anchors_trg * mask_trg  # [B, head_num, anchor_num, embed_size]
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.expand_dims(tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1),
                                       axis=-1)  # [B, head_num, 1]
            score_trg = tf.expand_dims(tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1),
                                       axis=-1)  # [B, head_num, 1]

            anchor_src = tf.reduce_sum(score_src * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(score_trg * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.expand_dims(tf.transpose(anchors2), axis=0)  # [1, embed_size, anchor_num]
            _anchors = tf.tile(_anchors, [head_num, 1, 1])  # [head_num, embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [head_num, anchor_num])  # [head_num, anchor_num]
            head_scores_src = tf.nn.sigmoid(head_scores_src)
            head_scores_trg = tf.nn.sigmoid(head_scores_trg)

            # sorted anchor-embeddings
            sorted_indices_src = tf.argsort(head_scores_src, axis=-1)  # [head_num, anchor_num]
            sorted_indices_trg = tf.argsort(head_scores_trg, axis=-1)
            # sorted_indices_src = tf.nn.top_k(head_scores_src, anchor_num, sorted=True).indices        # [head_num, anchor_num]
            # sorted_indices_trg = tf.nn.top_k(head_scores_trg, anchor_num, sorted=True).indices
            sorted_indices_src = tf.expand_dims(sorted_indices_src, axis=0)  # [1, head_num, anchor_num]
            sorted_indices_trg = tf.expand_dims(sorted_indices_trg, axis=0)  # [1, head_num, anchor_num]
            sorted_anchors_src = tf.gather(anchors2, sorted_indices_src)  # [1, head_num, anchor_num, embed_size]
            sorted_anchors_trg = tf.gather(anchors2, sorted_indices_trg)  # [1, head_num, anchor_num, embed_size]

            # sequence_mask
            mask_src = tf.cast(tf.sequence_mask(gate_num_src, maxlen=anchor_num), tf.float32)  # [B, anchor_num]
            mask_trg = tf.cast(tf.sequence_mask(gate_num_trg, maxlen=anchor_num), tf.float32)  # [B, anchor_num]
            mask_src = tf.reshape(mask_src, [-1, 1, anchor_num, 1])  # [B, 1, anchor_num, 1]
            mask_trg = tf.reshape(mask_trg, [-1, 1, anchor_num, 1])  # [B, 1, anchor_num, 1]
            # sorted_head_scores_src = tf.nn.top_k(head_scores_src, anchor_num, sorted=True).values
            # sorted_head_scores_trg = tf.nn.top_k(head_scores_trg, anchor_num, sorted=True).values
            sorted_head_scores_src = tf.sort(head_scores_src, axis=-1)      # [head_num, anchor_num]
            sorted_head_scores_trg = tf.sort(head_scores_trg, axis=-1)
            sorted_head_scores_src = tf.reshape(sorted_head_scores_src, [1, head_num, anchor_num, 1])
            sorted_head_scores_trg = tf.reshape(sorted_head_scores_trg, [1, head_num, anchor_num, 1])

            # !!!!!
            mask_src = sorted_head_scores_src + tf.stop_gradient(mask_src - sorted_head_scores_src)
            mask_trg = sorted_head_scores_trg + tf.stop_gradient(mask_trg - sorted_head_scores_trg)

            head_embeddings_src = sorted_anchors_src * mask_src  # [B, head_num, anchor_num, embed_size]
            head_embeddings_trg = sorted_anchors_trg * mask_trg  # [B, head_num, anchor_num, embed_size]
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.expand_dims(tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1),
                                       axis=-1)  # [B, head_num, 1]
            score_trg = tf.expand_dims(tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1),
                                       axis=-1)  # [B, head_num, 1]

            anchor_src = tf.reduce_sum(score_src * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(score_trg * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness == 2:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_num_src = tf.gather(self.hardness_num, self.user_id_src)  # [B,]
            gate_num_trg = tf.gather(self.hardness_num, self.user_id_trg)  # [B,]

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_src, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_trg, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_src = tf.reshape(gate_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate_trg = tf.reshape(gate_trg, [-1, head_num, anchor_num, 1])

            self.gate_src1 = tf.squeeze(gate_src)
            self.gate_trg1 = tf.squeeze(gate_trg)

            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(queries_src, _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(queries_trg, _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])


            '''
                2, get `mask label`
            '''
            # sorted anchor-embeddings
            sorted_indices_src = tf.argsort(head_scores_src, axis=-2)  # [1, head_num, anchor_num, 1]
            sorted_indices_trg = tf.argsort(head_scores_trg, axis=-2)
            # sequence_mask
            mask_src = tf.cast(tf.sequence_mask(gate_num_src, maxlen=anchor_num), tf.float32)  # [B, anchor_num]
            mask_trg = tf.cast(tf.sequence_mask(gate_num_trg, maxlen=anchor_num), tf.float32)  # [B, anchor_num]
            mask_src = tf.reshape(mask_src, [-1, 1, anchor_num, 1])  # [B, 1, anchor_num, 1]
            mask_trg = tf.reshape(mask_trg, [-1, 1, anchor_num, 1])  # [B, 1, anchor_num, 1]
            sorted_indices_src = tf.cast(mask_src, tf.int32) * (sorted_indices_src + 1) - 1
            sorted_indices_trg = tf.cast(mask_trg, tf.int32) * (sorted_indices_trg + 1) - 1
            # sorted_indices_src = tf.cast(mask_src, tf.int32) * (
            #             sorted_indices_src + tf.ones_like(sorted_indices_src, dtype=tf.int32)) - 1
            # sorted_indices_trg = tf.cast(mask_trg, tf.int32) * (
            #             sorted_indices_trg + tf.ones_like(sorted_indices_trg, dtype=tf.int32)) - 1
            self.mask_label_src1 = tf.one_hot(sorted_indices_src, depth=anchor_num, axis=2)
            self.mask_label_src1 = tf.squeeze(tf.reduce_sum(self.mask_label_src1, axis=-2))      # [B, head_num, anchor_num, 1]
            self.mask_label_trg1 = tf.one_hot(sorted_indices_trg, depth=anchor_num, axis=2)
            self.mask_label_trg1 = tf.squeeze(tf.reduce_sum(self.mask_label_trg1, axis=-2))      # [B, head_num, anchor_num, 1]



            # head_scores_src = tf.expand_dims(tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1),
            #                                  axis=-1)  # [B, head_num, anchor_num, 1]
            # head_scores_trg = tf.expand_dims(tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2), axis=-1)

            '''
                3, score(sigmoid) * gate(sigmoid)  && softmax
            '''
            head_scores_src = gate_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / tau1, dim=-2)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau2, dim=-2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_src, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate2_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_trg, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate2_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_src = tf.reshape(gate_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate_trg = tf.reshape(gate_trg, [-1, head_num, anchor_num, 1])

            self.gate_src2 = tf.squeeze(gate_src)   # [B, head_num, anchor_num]
            self.gate_trg2 = tf.squeeze(gate_trg)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(queries_src, _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(queries_trg, _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])

            '''
                2, get `mask label`
            '''
            # sorted anchor-embeddings
            sorted_indices_src = tf.argsort(head_scores_src, axis=-2)  # [1, head_num, anchor_num, 1]
            sorted_indices_trg = tf.argsort(head_scores_trg, axis=-2)
            # sequence_mask
            mask_src = tf.cast(tf.sequence_mask(gate_num_src, maxlen=anchor_num), tf.float32)  # [B, anchor_num]
            mask_trg = tf.cast(tf.sequence_mask(gate_num_trg, maxlen=anchor_num), tf.float32)  # [B, anchor_num]
            mask_src = tf.reshape(mask_src, [-1, 1, anchor_num, 1])  # [B, 1, anchor_num, 1]
            mask_trg = tf.reshape(mask_trg, [-1, 1, anchor_num, 1])  # [B, 1, anchor_num, 1]
            sorted_indices_src = tf.cast(mask_src, tf.int32) * (sorted_indices_src + 1) - 1
            sorted_indices_trg = tf.cast(mask_trg, tf.int32) * (sorted_indices_trg + 1) - 1
            # sorted_indices_src = tf.cast(mask_src, tf.int32) * (
            #             sorted_indices_src + tf.ones_like(sorted_indices_src, dtype=tf.int32)) - 1
            # sorted_indices_trg = tf.cast(mask_trg, tf.int32) * (
            #             sorted_indices_trg + tf.ones_like(sorted_indices_trg, dtype=tf.int32)) - 1
            self.mask_label_src2 = tf.one_hot(sorted_indices_src, depth=anchor_num, axis=2)
            self.mask_label_src2 = tf.squeeze(tf.reduce_sum(self.mask_label_src2, axis=-2))  # [B, head_num, anchor_num]
            self.mask_label_trg2 = tf.one_hot(sorted_indices_trg, depth=anchor_num, axis=2)
            self.mask_label_trg2 = tf.squeeze(tf.reduce_sum(self.mask_label_trg2, axis=-2))  # [B, head_num, anchor_num]

            # head_scores_src = tf.expand_dims(tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1),
            #                                  axis=-1)  # [B, head_num, anchor_num, 1]
            # head_scores_trg = tf.expand_dims(tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2), axis=-1)

            '''
                3, score(sigmoid) * gate(sigmoid)  && softmax
            '''
            head_scores_src = gate_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / tau1, dim=-2)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau2, dim=-2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness == 3:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_num_src = tf.gather(self.hardness_num, self.user_id_src)  # [B,]
            gate_num_trg = tf.gather(self.hardness_num, self.user_id_trg)  # [B,]

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_src, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_trg, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_src = tf.reshape(gate_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate_trg = tf.reshape(gate_trg, [-1, head_num, anchor_num, 1])

            self.gate_src1 = tf.squeeze(gate_src)
            self.gate_trg1 = tf.squeeze(gate_trg)

            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(queries_src, _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(queries_trg, _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])


            # head_scores_src = tf.expand_dims(tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1),
            #                                  axis=-1)  # [B, head_num, anchor_num, 1]
            # head_scores_trg = tf.expand_dims(tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2), axis=-1)

            '''
                3, score(sigmoid) * gate(sigmoid)  && softmax
            '''
            head_scores_src = gate_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / tau1, dim=-2)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau2, dim=-2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_src, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate2_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_trg, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate2_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_src = tf.reshape(gate_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate_trg = tf.reshape(gate_trg, [-1, head_num, anchor_num, 1])

            self.gate_src2 = tf.squeeze(gate_src)   # [B, head_num, anchor_num]
            self.gate_trg2 = tf.squeeze(gate_trg)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(queries_src, _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(queries_trg, _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])

            # head_scores_src = tf.expand_dims(tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1),
            #                                  axis=-1)  # [B, head_num, anchor_num, 1]
            # head_scores_trg = tf.expand_dims(tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2), axis=-1)

            '''
                3, score(sigmoid) * gate(sigmoid)  && softmax
            '''
            head_scores_src = gate_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / tau1, dim=-2)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau2, dim=-2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness == 4:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_src, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_trg, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_src = tf.reshape(gate_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate_trg = tf.reshape(gate_trg, [-1, head_num, anchor_num, 1])

            self.gate_src1 = tf.squeeze(gate_src)
            self.gate_trg1 = tf.squeeze(gate_trg)

            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])


            # head_scores_src = tf.expand_dims(tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1),
            #                                  axis=-1)  # [B, head_num, anchor_num, 1]
            # head_scores_trg = tf.expand_dims(tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2), axis=-1)

            '''
                3, score(sigmoid) * gate(sigmoid)  && softmax
            '''
            head_scores_src = gate_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / tau1, dim=2)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_src, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate2_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_trg, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate2_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_src = tf.reshape(gate_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate_trg = tf.reshape(gate_trg, [-1, head_num, anchor_num, 1])

            self.gate_src2 = tf.squeeze(gate_src)   # [B, head_num, anchor_num]
            self.gate_trg2 = tf.squeeze(gate_trg)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])

            # head_scores_src = tf.expand_dims(tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1),
            #                                  axis=-1)  # [B, head_num, anchor_num, 1]
            # head_scores_trg = tf.expand_dims(tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2), axis=-1)

            '''
                3, score(sigmoid) * gate(sigmoid)  && softmax
            '''
            head_scores_src = gate_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / tau1, dim=2)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [5, 6, 7, 8]:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_src, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_trg, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_src = tf.reshape(gate_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate_trg = tf.reshape(gate_trg, [-1, head_num, anchor_num, 1])

            self.gate_src1 = tf.squeeze(gate_src)
            self.gate_trg1 = tf.squeeze(gate_trg)

            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])


            # head_scores_src = tf.expand_dims(tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1),
            #                                  axis=-1)  # [B, head_num, anchor_num, 1]
            # head_scores_trg = tf.expand_dims(tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2), axis=-1)

            '''
                3, score(sigmoid) * gate(sigmoid)  && softmax
            '''
            head_scores_src = gate_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / tau1, dim=2)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_src, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate2_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_trg, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate2_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_src = tf.reshape(gate_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate_trg = tf.reshape(gate_trg, [-1, head_num, anchor_num, 1])

            self.gate_src2 = tf.squeeze(gate_src)   # [B, head_num, anchor_num]
            self.gate_trg2 = tf.squeeze(gate_trg)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])

            # head_scores_src = tf.expand_dims(tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1),
            #                                  axis=-1)  # [B, head_num, anchor_num, 1]
            # head_scores_trg = tf.expand_dims(tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2), axis=-1)

            '''
                3, score(sigmoid) * gate(sigmoid) && softmax
            '''
            head_scores_src = gate_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / tau1, dim=2)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness == 9:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tf.exp(tau1), [-1, 1, 1, 1])
            tau2 = tf.reshape(tf.exp(tau2), [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_src, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_trg, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_src = tf.reshape(gate_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate_trg = tf.reshape(gate_trg, [-1, head_num, anchor_num, 1])

            self.gate_src1 = tf.squeeze(gate_src)
            self.gate_trg1 = tf.squeeze(gate_trg)

            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])


            # head_scores_src = tf.expand_dims(tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1),
            #                                  axis=-1)  # [B, head_num, anchor_num, 1]
            # head_scores_trg = tf.expand_dims(tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2), axis=-1)

            '''
                3, score(sigmoid) * gate(sigmoid)  && softmax
            '''
            head_scores_src = gate_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / tau1, dim=2)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_src, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate2_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_trg, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate2_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_src = tf.reshape(gate_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate_trg = tf.reshape(gate_trg, [-1, head_num, anchor_num, 1])

            self.gate_src2 = tf.squeeze(gate_src)   # [B, head_num, anchor_num]
            self.gate_trg2 = tf.squeeze(gate_trg)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])

            # head_scores_src = tf.expand_dims(tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1),
            #                                  axis=-1)  # [B, head_num, anchor_num, 1]
            # head_scores_trg = tf.expand_dims(tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2), axis=-1)

            '''
                3, score(sigmoid) * gate(sigmoid)  && softmax
            '''
            head_scores_src = gate_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / tau1, dim=2)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness == 10:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tf.exp(tau1), [-1, 1, 1, 1])
            tau2 = tf.reshape(tf.exp(tau2), [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_src, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_trg, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_src = tf.reshape(gate_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate_trg = tf.reshape(gate_trg, [-1, head_num, anchor_num, 1])

            self.gate_src1 = tf.squeeze(gate_src)
            self.gate_trg1 = tf.squeeze(gate_trg)

            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])


            # head_scores_src = tf.expand_dims(tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1),
            #                                  axis=-1)  # [B, head_num, anchor_num, 1]
            # head_scores_trg = tf.expand_dims(tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2), axis=-1)

            '''
                3, score(sigmoid) * gate(sigmoid)  && softmax
            '''
            head_scores_src = gate_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / tau1, dim=2)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_src, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate2_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_trg, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate2_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate_src = tf.reshape(gate_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate_trg = tf.reshape(gate_trg, [-1, head_num, anchor_num, 1])

            self.gate_src2 = tf.squeeze(gate_src)   # [B, head_num, anchor_num]
            self.gate_trg2 = tf.squeeze(gate_trg)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size)), [1, head_num, anchor_num, 1])

            # head_scores_src = tf.expand_dims(tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1),
            #                                  axis=-1)  # [B, head_num, anchor_num, 1]
            # head_scores_trg = tf.expand_dims(tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2), axis=-1)

            '''
                3, score(sigmoid) * gate(sigmoid)  && softmax
            '''
            head_scores_src = gate_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / tau1, dim=2)
            head_scores_trg = tf.nn.softmax(head_scores_trg / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness == 11:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tf.exp(tau1), [-1, 1, 1, 1])
            tau2 = tf.reshape(tf.exp(tau2), [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(head_scores_src, [1, head_num, anchor_num, 1])
            head_scores_trg = tf.reshape(head_scores_trg, [1, head_num, anchor_num, 1])
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size) / tau1), [-1, head_num, anchor_num, 1])  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size) / tau2), [-1, head_num, anchor_num, 1])

            self.head_scores_src1 = tf.squeeze(head_scores_src)   # [B, head_num, anchor_num]
            self.head_scores_trg1 = tf.squeeze(head_scores_trg)   # [B, head_num, anchor_num]

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            '''
                1, gate -> input: user, output: gate_vector
            '''
            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(head_scores_src, [1, head_num, anchor_num, 1])
            head_scores_trg = tf.reshape(head_scores_trg, [1, head_num, anchor_num, 1])
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size) / tau1), [-1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size) / tau2), [-1, head_num, anchor_num, 1])

            self.head_scores_src2 = tf.squeeze(head_scores_src)
            self.head_scores_trg2 = tf.squeeze(head_scores_trg)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [12, 13]:
            '''
                12: 将前面的gate删去，使用到的anchor比例全部通过温度系数tau获得
                13: 同12，但是计算hardset的Loss的时候不计算item部份
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            tau1, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            tau2, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            tau1 = tf.reshape(tf.exp(tau1), [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            tau2 = tf.reshape(tf.exp(tau2), [-1, head_num, anchor_num, 1])

            '''
                1, gate -> input: user, output: gate_vector
            '''
            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(head_scores_src, [1, head_num, anchor_num, 1])
            head_scores_trg = tf.reshape(head_scores_trg, [1, head_num, anchor_num, 1])
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size) / tau1), [-1, head_num, anchor_num, 1])  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size) / tau2), [-1, head_num, anchor_num, 1])

            self.head_scores_src1 = tf.squeeze(head_scores_src)   # [B, head_num, anchor_num]
            self.head_scores_trg1 = tf.squeeze(head_scores_trg)   # [B, head_num, anchor_num]

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            tau1, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate2_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            tau2, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate2_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            tau1 = tf.reshape(tf.exp(tau1), [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            tau2 = tf.reshape(tf.exp(tau2), [-1, head_num, anchor_num, 1])

            '''
                1, gate -> input: user, output: gate_vector
            '''
            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(head_scores_src, [1, head_num, anchor_num, 1])
            head_scores_trg = tf.reshape(head_scores_trg, [1, head_num, anchor_num, 1])
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size) / tau1), [-1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size) / tau2), [-1, head_num, anchor_num, 1])

            self.head_scores_src2 = tf.squeeze(head_scores_src)
            self.head_scores_trg2 = tf.squeeze(head_scores_trg)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [14]:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            tau1, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            tau2, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            tau1 = tf.reshape(tf.exp(tau1), [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            tau2 = tf.reshape(tf.exp(tau2), [-1, head_num, anchor_num, 1])

            '''
                1, gate -> input: user, output: gate_vector
            '''
            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(head_scores_src, [1, head_num, anchor_num, 1])
            head_scores_trg = tf.reshape(head_scores_trg, [1, head_num, anchor_num, 1])
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size) / tau1), [-1, head_num, anchor_num, 1])  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size) / tau2), [-1, head_num, anchor_num, 1])

            self.head_scores_src1 = tf.squeeze(head_scores_src)   # [B, head_num, anchor_num]
            self.head_scores_trg1 = tf.squeeze(head_scores_trg)   # [B, head_num, anchor_num]

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            tau1, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate2_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            tau2, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate2_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            tau1 = tf.reshape(tf.exp(tau1), [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            tau2 = tf.reshape(tf.exp(tau2), [-1, head_num, anchor_num, 1])

            '''
                1, gate -> input: user, output: gate_vector
            '''
            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(head_scores_src, [1, head_num, anchor_num, 1])
            head_scores_trg = tf.reshape(head_scores_trg, [1, head_num, anchor_num, 1])
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size) / tau1), [-1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size) / tau2), [-1, head_num, anchor_num, 1])

            self.head_scores_src2 = tf.squeeze(head_scores_src)
            self.head_scores_trg2 = tf.squeeze(head_scores_trg)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [15]:
            '''
                15: 
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            tau1, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            tau2, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            tau1 = tf.reshape(tf.exp(tau1), [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            tau2 = tf.reshape(tf.exp(tau2), [-1, head_num, anchor_num, 1])

            '''
                1, gate -> input: user, output: gate_vector
            '''
            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(head_scores_src, [1, head_num, anchor_num, 1])
            head_scores_trg = tf.reshape(head_scores_trg, [1, head_num, anchor_num, 1])
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size) / tau1), [-1, head_num, anchor_num, 1])  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size) / tau2), [-1, head_num, anchor_num, 1])

            self.head_scores_src1 = tf.squeeze(head_scores_src)   # [B, head_num, anchor_num]
            self.head_scores_trg1 = tf.squeeze(head_scores_trg)   # [B, head_num, anchor_num]

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            tau1, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate2_src',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            tau2, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate2_trg',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            tau1 = tf.reshape(tf.exp(tau1), [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            tau2 = tf.reshape(tf.exp(tau2), [-1, head_num, anchor_num, 1])

            '''
                1, gate -> input: user, output: gate_vector
            '''
            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.matmul(tf.squeeze(queries_src), _anchors)  # [head_num, anchor_num]
            head_scores_trg = tf.matmul(tf.squeeze(queries_trg), _anchors)  # [head_num, anchor_num]
            head_scores_src = tf.reshape(head_scores_src, [1, head_num, anchor_num, 1])
            head_scores_trg = tf.reshape(head_scores_trg, [1, head_num, anchor_num, 1])
            head_scores_src = tf.reshape(tf.nn.sigmoid(head_scores_src / np.sqrt(embed_size) / tau1), [-1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.nn.sigmoid(head_scores_trg / np.sqrt(embed_size) / tau2), [-1, head_num, anchor_num, 1])

            self.head_scores_src2 = tf.squeeze(head_scores_src)
            self.head_scores_trg2 = tf.squeeze(head_scores_trg)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [16]:
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            '''
                user embedding
            '''

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv2_src,
                                                                        training=self.training, name='alpha2_src',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv2_trg,
                                                                        training=self.training, name='alpha2_trg',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)


            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1_src',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha1_trg',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)


            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [17, 30]:
            '''
                3: no gate
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_num_src = tf.gather(self.hardness_num, self.user_id_src)  # [B,]
            gate_num_trg = tf.gather(self.hardness_num, self.user_id_trg)  # [B,]

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)


            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [18]:
            '''
                3: no gate
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_num_src = tf.gather(self.hardness_num, self.user_id_src)  # [B,]
            gate_num_trg = tf.gather(self.hardness_num, self.user_id_trg)  # [B,]

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)


            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [19]:
            '''
                3: no gate
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)


            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(self.ae_xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(self.ae_xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(self.ae_xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(self.ae_xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [20]:
            '''
                3: no gate
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)


            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(self.ae_xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(self.ae_xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, self.ae_xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, self.ae_xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(self.ae_xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(self.ae_xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, self.ae_xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, self.ae_xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [21]:
            '''
                3: no gate
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_num_src = tf.gather(self.hardness_num, self.user_id_src)  # [B,]
            gate_num_trg = tf.gather(self.hardness_num, self.user_id_trg)  # [B,]

            taus1, _, _, _ = bin_mlp([64, 32, 2], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            taus2, _, _, _ = bin_mlp([64, 32, 2], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            '''
                user embedding
            '''
            tau1 = tf.reshape(taus1[:, 0], [-1, 1, 1, 1])
            tau2 = tf.reshape(taus2[:, 0], [-1, 1, 1, 1])

            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)


            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            tau1 = tf.reshape(taus1[:, 1], [-1, 1, 1, 1])
            tau2 = tf.reshape(taus2[:, 1], [-1, 1, 1, 1])


            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [22]:
            '''
                3: no gate
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_num_src = tf.gather(self.hardness_num, self.user_id_src)  # [B,]
            gate_num_trg = tf.gather(self.hardness_num, self.user_id_trg)  # [B,]

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.tanh(head_scores_src / np.sqrt(embed_size) / tau1)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.tanh(head_scores_trg / np.sqrt(embed_size) / tau2)


            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.tanh(head_scores_src / np.sqrt(embed_size) / tau1)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.tanh(head_scores_trg / np.sqrt(embed_size) / tau2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [23]:
            '''
                17: multi-head scores from MLP
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)


            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            head_input_src = tf.concat([head_embeddings_src,
                                        tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=1), [1, head_num, 1]),
                                                   [-1, head_num, embed_size])], axis=-1)      # [B, head_num, 2 * embed_size]
            head_input_trg = tf.concat([head_embeddings_trg,
                                        tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=1), [1, head_num, 1]),
                                                   [-1, head_num, embed_size])], axis=-1)      # [B, head_num, 2 * embed_size]

            score_src, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
                                                                        layer_keeps, head_input_src,
                                                                        training=self.training, name='score_src1',
                                                                        reuse=tf.AUTO_REUSE)

            score_trg, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
                                                                        layer_keeps, head_input_trg,
                                                                        training=self.training, name='score_trg1',
                                                                        reuse=tf.AUTO_REUSE)

            score_src = tf.nn.softmax(score_src, dim=1)       # [B, head_num, 1]
            score_trg = tf.nn.softmax(score_trg, dim=1)

            anchor_src = tf.reduce_sum(score_src * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(score_trg * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            head_input_src = tf.concat([head_embeddings_src,
                                        tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=1), [1, head_num, 1]),
                                                   [-1, head_num, embed_size])], axis=-1)      # [B, head_num, 2 * embed_size]
            head_input_trg = tf.concat([head_embeddings_trg,
                                        tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=1), [1, head_num, 1]),
                                                   [-1, head_num, embed_size])], axis=-1)      # [B, head_num, 2 * embed_size]

            score_src, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
                                                                        layer_keeps, head_input_src,
                                                                        training=self.training, name='score_src2',
                                                                        reuse=tf.AUTO_REUSE)

            score_trg, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
                                                                        layer_keeps, head_input_trg,
                                                                        training=self.training, name='score_trg2',
                                                                        reuse=tf.AUTO_REUSE)

            score_src = tf.nn.softmax(score_src, dim=1)       # [B, head_num, 1]
            score_trg = tf.nn.softmax(score_trg, dim=1)

            anchor_src = tf.reduce_sum(score_src * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(score_trg * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [24]:
            '''
                K个head，每个head有N个anchor，总数为K*N
            '''
            k_anchors1 = get_variable(init, name='k_anchors1', shape=[head_num, anchor_num, embed_size])
            k_anchors2 = get_variable(init, name='k_anchors2', shape=[head_num, anchor_num, embed_size])

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries1[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries1[1], axis=1)

            _anchors = tf.transpose(k_anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.reduce_sum(queries_src * k_anchors1, axis=-1), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.reduce_sum(queries_trg * k_anchors1, axis=-1), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)


            head_embeddings_src = head_scores_src * tf.reshape(k_anchors1, [1, head_num, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(k_anchors1, [1, head_num, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            head_scores_src = tf.reshape(tf.reduce_sum(queries_src * k_anchors2, axis=-1), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.reduce_sum(queries_trg * k_anchors2, axis=-1), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)


            head_embeddings_src = head_scores_src * tf.reshape(k_anchors2, [1, head_num, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(k_anchors2, [1, head_num, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [25]:
            '''
                17: add twin network
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src)  # [B,]
            gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg)  # [B,]
            gate_score_src2 = tf.gather(self.hardness_score_src2, self.item_id_src)  # [B,]
            gate_score_trg2 = tf.gather(self.hardness_score_trg2, self.item_id_trg)  # [B,]

            gate_score_src = tf.reshape(gate_score_src, [-1, 1, 1])
            gate_score_trg = tf.reshape(gate_score_trg, [-1, 1, 1])
            gate_score_src2 = tf.reshape(gate_score_src2, [-1, 1, 1])
            gate_score_trg2 = tf.reshape(gate_score_trg2, [-1, 1, 1])

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]



            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate1_src_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B, head_num, anchor_num]

            gate1_src_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                    layer_keeps, gate_input_src,
                                    training=self.training, name='gate1_src_high',
                                    reuse=tf.AUTO_REUSE)  # [B, head_num, anchor_num]


            gate1_trg_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate1_trg_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                    layer_keeps, gate_input_trg,
                                    training=self.training, name='gate1_trg_high',
                                    reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate1_src = gate_score_src * gate1_src_low + (1 - gate_score_src) * gate1_src_high
            gate1_src = tf.nn.sigmoid(gate1_src)
            gate1_trg = gate_score_trg * gate1_trg_low + (1 - gate_score_trg) * gate1_trg_high
            gate1_trg = tf.nn.sigmoid(gate1_trg)

            gate1_src = 2 * tf.reshape(gate1_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate1_trg = 2 * tf.reshape(gate1_trg, [-1, head_num, anchor_num, 1])


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_scores_src = gate1_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate1_trg * head_scores_trg

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate2_src_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                             layer_keeps, gate_input_src,
                                             training=self.training, name='gate2_src_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                              layer_keeps, gate_input_src,
                                              training=self.training, name='gate2_src_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                             layer_keeps, gate_input_trg,
                                             training=self.training, name='gate2_trg_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                              layer_keeps, gate_input_trg,
                                              training=self.training, name='gate2_trg_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src = gate_score_src2 * gate2_src_low + (1 - gate_score_src2) * gate2_src_high
            gate2_src = tf.nn.sigmoid(gate2_src)
            gate2_trg = gate_score_trg2 * gate2_trg_low + (1 - gate_score_trg2) * gate2_trg_high
            gate2_trg = tf.nn.sigmoid(gate2_trg)

            gate2_src = 2 * tf.reshape(gate2_src, [-1, head_num, anchor_num, 1])  # [B, head_num, anchor_num, 1]
            gate2_trg = 2 * tf.reshape(gate2_trg, [-1, head_num, anchor_num, 1])

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1,
                                            dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_scores_src = gate2_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate2_trg * head_scores_trg

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [26]:
            '''
                17: add twin network, 先sigmoid后加权求和
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src)  # [B,]
            gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg)  # [B,]
            gate_score_src2 = tf.gather(self.hardness_score_src2, self.item_id_src)  # [B,]
            gate_score_trg2 = tf.gather(self.hardness_score_trg2, self.item_id_trg)  # [B,]

            gate_score_src = tf.reshape(gate_score_src, [-1, 1, 1])
            gate_score_trg = tf.reshape(gate_score_trg, [-1, 1, 1])
            gate_score_src2 = tf.reshape(gate_score_src2, [-1, 1, 1])
            gate_score_trg2 = tf.reshape(gate_score_trg2, [-1, 1, 1])

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]



            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate1_src_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B, head_num, anchor_num]

            gate1_src_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                    layer_keeps, gate_input_src,
                                    training=self.training, name='gate1_src_high',
                                    reuse=tf.AUTO_REUSE)  # [B, head_num, anchor_num]


            gate1_trg_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate1_trg_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                    layer_keeps, gate_input_trg,
                                    training=self.training, name='gate1_trg_high',
                                    reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate1_src = gate_score_src * gate1_src_low + (1 - gate_score_src) * gate1_src_high
            gate1_trg = gate_score_trg * gate1_trg_low + (1 - gate_score_trg) * gate1_trg_high

            gate1_src = 2 * tf.reshape(gate1_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate1_trg = 2 * tf.reshape(gate1_trg, [-1, head_num, anchor_num, 1])


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_scores_src = gate1_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate1_trg * head_scores_trg

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate2_src_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                             layer_keeps, gate_input_src,
                                             training=self.training, name='gate2_src_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                              layer_keeps, gate_input_src,
                                              training=self.training, name='gate2_src_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                             layer_keeps, gate_input_trg,
                                             training=self.training, name='gate2_trg_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                              layer_keeps, gate_input_trg,
                                              training=self.training, name='gate2_trg_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src = gate_score_src2 * gate2_src_low + (1 - gate_score_src2) * gate2_src_high
            gate2_trg = gate_score_trg2 * gate2_trg_low + (1 - gate_score_trg2) * gate2_trg_high

            gate2_src = 2 * tf.reshape(gate2_src, [-1, head_num, anchor_num, 1])  # [B, head_num, anchor_num, 1]
            gate2_trg = 2 * tf.reshape(gate2_trg, [-1, head_num, anchor_num, 1])

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1,
                                            dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_scores_src = gate2_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate2_trg * head_scores_trg

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [27]:
            '''
                17: add twin network, thershold
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src)  # [B,]
            gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg)  # [B,]
            gate_score_src2 = tf.gather(self.hardness_score_src2, self.item_id_src)  # [B,]
            gate_score_trg2 = tf.gather(self.hardness_score_trg2, self.item_id_trg)  # [B,]

            gate_score_src = tf.reshape(gate_score_src, [-1, 1, 1])
            gate_score_trg = tf.reshape(gate_score_trg, [-1, 1, 1])
            gate_score_src2 = tf.reshape(gate_score_src2, [-1, 1, 1])
            gate_score_trg2 = tf.reshape(gate_score_trg2, [-1, 1, 1])

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]



            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate1_src_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B, head_num, anchor_num]

            gate1_src_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                    layer_keeps, gate_input_src,
                                    training=self.training, name='gate1_src_high',
                                    reuse=tf.AUTO_REUSE)  # [B, head_num, anchor_num]


            gate1_trg_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate1_trg_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                    layer_keeps, gate_input_trg,
                                    training=self.training, name='gate1_trg_high',
                                    reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate1_src = gate_score_src * gate1_src_low + (1 - gate_score_src) * gate1_src_high
            gate1_src = tf.nn.sigmoid(gate1_src)
            gate1_trg = gate_score_trg * gate1_trg_low + (1 - gate_score_trg) * gate1_trg_high
            gate1_trg = tf.nn.sigmoid(gate1_trg)

            gate1_src_mask = tf.where(tf.greater(gate1_src, thres), tf.ones_like(gate1_src), tf.zeros_like(gate1_src))
            gate1_trg_mask = tf.where(tf.greater(gate1_trg, thres), tf.ones_like(gate1_trg), tf.zeros_like(gate1_trg))
            gate1_src_mask = gate1_src + tf.stop_gradient(gate1_src_mask - gate1_src)
            gate1_trg_mask = gate1_trg + tf.stop_gradient(gate1_trg_mask - gate1_trg)


            gate1_src_mask = tf.reshape(gate1_src_mask, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate1_trg_mask = tf.reshape(gate1_trg_mask, [-1, head_num, anchor_num, 1])


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_scores_src = gate1_src_mask * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate1_trg_mask * head_scores_trg

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate2_src_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                             layer_keeps, gate_input_src,
                                             training=self.training, name='gate2_src_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                              layer_keeps, gate_input_src,
                                              training=self.training, name='gate2_src_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                             layer_keeps, gate_input_trg,
                                             training=self.training, name='gate2_trg_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                              layer_keeps, gate_input_trg,
                                              training=self.training, name='gate2_trg_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src = gate_score_src2 * gate2_src_low + (1 - gate_score_src2) * gate2_src_high
            gate2_src = tf.nn.sigmoid(gate2_src)
            gate2_trg = gate_score_trg2 * gate2_trg_low + (1 - gate_score_trg2) * gate2_trg_high
            gate2_trg = tf.nn.sigmoid(gate2_trg)

            gate2_src_mask = tf.where(tf.greater(gate2_src, thres), tf.ones_like(gate2_src), tf.zeros_like(gate2_src))
            gate2_trg_mask = tf.where(tf.greater(gate2_trg, thres), tf.ones_like(gate2_trg), tf.zeros_like(gate2_trg))
            gate2_src_mask = gate2_src + tf.stop_gradient(gate2_src_mask - gate2_src)
            gate2_trg_mask = gate2_trg + tf.stop_gradient(gate2_trg_mask - gate2_trg)

            gate2_src_mask = tf.reshape(gate2_src_mask, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate2_trg_mask = tf.reshape(gate2_trg_mask, [-1, head_num, anchor_num, 1])

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1,
                                            dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_scores_src = gate2_src_mask * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate2_trg_mask * head_scores_trg

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [28]:
            '''
                17: add twin network, thershold, 先sigmoid后加权求和
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src)  # [B,]
            gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg)  # [B,]
            gate_score_src2 = tf.gather(self.hardness_score_src2, self.item_id_src)  # [B,]
            gate_score_trg2 = tf.gather(self.hardness_score_trg2, self.item_id_trg)  # [B,]

            gate_score_src = tf.reshape(gate_score_src, [-1, 1, 1])
            gate_score_trg = tf.reshape(gate_score_trg, [-1, 1, 1])
            gate_score_src2 = tf.reshape(gate_score_src2, [-1, 1, 1])
            gate_score_trg2 = tf.reshape(gate_score_trg2, [-1, 1, 1])

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]



            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate1_src_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B, head_num, anchor_num]

            gate1_src_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                    layer_keeps, gate_input_src,
                                    training=self.training, name='gate1_src_high',
                                    reuse=tf.AUTO_REUSE)  # [B, head_num, anchor_num]


            gate1_trg_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate1_trg_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                    layer_keeps, gate_input_trg,
                                    training=self.training, name='gate1_trg_high',
                                    reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate1_src = gate_score_src * gate1_src_low + (1 - gate_score_src) * gate1_src_high
            gate1_trg = gate_score_trg * gate1_trg_low + (1 - gate_score_trg) * gate1_trg_high

            gate1_src = 2 * tf.reshape(gate1_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate1_trg = 2 * tf.reshape(gate1_trg, [-1, head_num, anchor_num, 1])


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_scores_src = gate1_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate1_trg * head_scores_trg

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate2_src_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                             layer_keeps, gate_input_src,
                                             training=self.training, name='gate2_src_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                              layer_keeps, gate_input_src,
                                              training=self.training, name='gate2_src_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                             layer_keeps, gate_input_trg,
                                             training=self.training, name='gate2_trg_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', 'sigmoid'],
                                              layer_keeps, gate_input_trg,
                                              training=self.training, name='gate2_trg_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src = gate_score_src2 * gate2_src_low + (1 - gate_score_src2) * gate2_src_high
            gate2_trg = gate_score_trg2 * gate2_trg_low + (1 - gate_score_trg2) * gate2_trg_high

            gate2_src = 2 * tf.reshape(gate2_src, [-1, head_num, anchor_num, 1])  # [B, head_num, anchor_num, 1]
            gate2_trg = 2 * tf.reshape(gate2_trg, [-1, head_num, anchor_num, 1])

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1,
                                            dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_scores_src = gate2_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate2_trg * head_scores_trg

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [29]:
            '''
                17: add twin network, thershold, softmax -> sigmoid
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src)  # [B,]
            gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg)  # [B,]
            gate_score_src2 = tf.gather(self.hardness_score_src2, self.item_id_src)  # [B,]
            gate_score_trg2 = tf.gather(self.hardness_score_trg2, self.item_id_trg)  # [B,]

            gate_score_src = tf.reshape(gate_score_src, [-1, 1, 1])
            gate_score_trg = tf.reshape(gate_score_trg, [-1, 1, 1])
            gate_score_src2 = tf.reshape(gate_score_src2, [-1, 1, 1])
            gate_score_trg2 = tf.reshape(gate_score_trg2, [-1, 1, 1])

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]



            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate1_src_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B, head_num, anchor_num]

            gate1_src_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                    layer_keeps, gate_input_src,
                                    training=self.training, name='gate1_src_high',
                                    reuse=tf.AUTO_REUSE)  # [B, head_num, anchor_num]


            gate1_trg_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate1_trg_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                    layer_keeps, gate_input_trg,
                                    training=self.training, name='gate1_trg_high',
                                    reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate1_src = gate_score_src * gate1_src_low + (1 - gate_score_src) * gate1_src_high
            gate1_src = tf.nn.sigmoid(gate1_src)
            gate1_trg = gate_score_trg * gate1_trg_low + (1 - gate_score_trg) * gate1_trg_high
            gate1_trg = tf.nn.sigmoid(gate1_trg)

            gate1_src_mask = tf.where(tf.greater(gate1_src, thres), tf.ones_like(gate1_src), tf.zeros_like(gate1_src))
            gate1_trg_mask = tf.where(tf.greater(gate1_trg, thres), tf.ones_like(gate1_trg), tf.zeros_like(gate1_trg))
            gate1_src_mask = gate1_src + tf.stop_gradient(gate1_src_mask - gate1_src)
            gate1_trg_mask = gate1_trg + tf.stop_gradient(gate1_trg_mask - gate1_trg)


            gate1_src_mask = tf.reshape(gate1_src_mask, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate1_trg_mask = tf.reshape(gate1_trg_mask, [-1, head_num, anchor_num, 1])


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            soft_src = head_scores_src / np.sqrt(embed_size) / tau1
            soft_trg = head_scores_trg / np.sqrt(embed_size) / tau2
            soft_src = soft_src - tf.reduce_max(soft_src)
            soft_trg = soft_trg - tf.reduce_max(soft_trg)
            head_scores_src = tf.exp(soft_src)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.exp(soft_trg)

            head_scores_src = gate1_src_mask * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate1_trg_mask * head_scores_trg
            head_src_sum = tf.reduce_sum(head_scores_src, axis=-2, keep_dims=True)
            head_trg_sum = tf.reduce_sum(head_scores_trg, axis=-2, keep_dims=True)
            head_scores_src = head_scores_src / tf.clip_by_value(head_src_sum, 1e-6, tf.reduce_max(head_src_sum))
            head_scores_trg = head_scores_trg / tf.clip_by_value(head_trg_sum, 1e-6, tf.reduce_max(head_trg_sum))

            self.a = head_scores_src
            self.b = head_scores_trg

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate2_src_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                             layer_keeps, gate_input_src,
                                             training=self.training, name='gate2_src_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                              layer_keeps, gate_input_src,
                                              training=self.training, name='gate2_src_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                             layer_keeps, gate_input_trg,
                                             training=self.training, name='gate2_trg_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                              layer_keeps, gate_input_trg,
                                              training=self.training, name='gate2_trg_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src = gate_score_src2 * gate2_src_low + (1 - gate_score_src2) * gate2_src_high
            gate2_src = tf.nn.sigmoid(gate2_src)
            gate2_trg = gate_score_trg2 * gate2_trg_low + (1 - gate_score_trg2) * gate2_trg_high
            gate2_trg = tf.nn.sigmoid(gate2_trg)

            gate2_src_mask = tf.where(tf.greater(gate2_src, thres), tf.ones_like(gate2_src), tf.zeros_like(gate2_src))
            gate2_trg_mask = tf.where(tf.greater(gate2_trg, thres), tf.ones_like(gate2_trg), tf.zeros_like(gate2_trg))
            gate2_src_mask = gate2_src + tf.stop_gradient(gate2_src_mask - gate2_src)
            gate2_trg_mask = gate2_trg + tf.stop_gradient(gate2_trg_mask - gate2_trg)

            gate2_src_mask = tf.reshape(gate2_src_mask, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate2_trg_mask = tf.reshape(gate2_trg_mask, [-1, head_num, anchor_num, 1])

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            soft_src = head_scores_src / np.sqrt(embed_size) / tau1
            soft_trg = head_scores_trg / np.sqrt(embed_size) / tau2
            soft_src = soft_src - tf.reduce_max(soft_src)
            soft_trg = soft_trg - tf.reduce_max(soft_trg)
            head_scores_src = tf.exp(soft_src)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.exp(soft_trg)

            head_scores_src = gate2_src_mask * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate2_trg_mask * head_scores_trg
            head_src_sum = tf.reduce_sum(head_scores_src, axis=-2, keep_dims=True)
            head_trg_sum = tf.reduce_sum(head_scores_trg, axis=-2, keep_dims=True)
            head_scores_src = head_scores_src / tf.clip_by_value(head_src_sum, 1e-6, tf.reduce_max(head_src_sum))
            head_scores_trg = head_scores_trg / tf.clip_by_value(head_trg_sum, 1e-6, tf.reduce_max(head_trg_sum))

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [31]:

            '''
                17: add twin network, gate -> logit
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src)  # [B,]
            gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg)  # [B,]
            gate_score_src2 = tf.gather(self.hardness_score_src2, self.item_id_src)  # [B,]
            gate_score_trg2 = tf.gather(self.hardness_score_trg2, self.item_id_trg)  # [B,]

            gate_score_src = tf.reshape(gate_score_src, [-1, 1, 1])
            gate_score_trg = tf.reshape(gate_score_trg, [-1, 1, 1])
            gate_score_src2 = tf.reshape(gate_score_src2, [-1, 1, 1])
            gate_score_trg2 = tf.reshape(gate_score_trg2, [-1, 1, 1])

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]



            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate1_src_low, _, _, _ = bin_mlp([64, 32, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B, head_num, anchor_num]

            gate1_src_high, _, _, _ = bin_mlp([64, 32, anchor_num], ['tanh', 'tanh', None],
                                    layer_keeps, gate_input_src,
                                    training=self.training, name='gate1_src_high',
                                    reuse=tf.AUTO_REUSE)  # [B, head_num, anchor_num]


            gate1_trg_low, _, _, _ = bin_mlp([64, 32, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate1_trg_high, _, _, _ = bin_mlp([64, 32, anchor_num], ['tanh', 'tanh', None],
                                    layer_keeps, gate_input_trg,
                                    training=self.training, name='gate1_trg_high',
                                    reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate1_src = gate_score_src * gate1_src_low + (1 - gate_score_src) * gate1_src_high
            gate1_src = tf.nn.sigmoid(gate1_src)
            gate1_trg = gate_score_trg * gate1_trg_low + (1 - gate_score_trg) * gate1_trg_high
            gate1_trg = tf.nn.sigmoid(gate1_trg)

            gate1_src = 2 * tf.reshape(gate1_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate1_trg = 2 * tf.reshape(gate1_trg, [-1, head_num, anchor_num, 1])


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]

            head_scores_src = gate1_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate1_trg * head_scores_trg

            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate2_src_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                             layer_keeps, gate_input_src,
                                             training=self.training, name='gate2_src_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                              layer_keeps, gate_input_src,
                                              training=self.training, name='gate2_src_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                             layer_keeps, gate_input_trg,
                                             training=self.training, name='gate2_trg_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                              layer_keeps, gate_input_trg,
                                              training=self.training, name='gate2_trg_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src = gate_score_src2 * gate2_src_low + (1 - gate_score_src2) * gate2_src_high
            gate2_src = tf.nn.sigmoid(gate2_src)
            gate2_trg = gate_score_trg2 * gate2_trg_low + (1 - gate_score_trg2) * gate2_trg_high
            gate2_trg = tf.nn.sigmoid(gate2_trg)

            gate2_src = 2 * tf.reshape(gate2_src, [-1, head_num, anchor_num, 1])  # [B, head_num, anchor_num, 1]
            gate2_trg = 2 * tf.reshape(gate2_trg, [-1, head_num, anchor_num, 1])

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = gate2_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate2_trg * head_scores_trg
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1,
                                            dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [32]:
            '''
                17: add twin network, gate -> logit
            '''
            self.anchors1 = anchors1
            self.anchors2 = anchors2

            gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src)  # [B,]
            gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg)  # [B,]
            gate_score_src2 = tf.gather(self.hardness_score_src2, self.item_id_src)  # [B,]
            gate_score_trg2 = tf.gather(self.hardness_score_trg2, self.item_id_trg)  # [B,]

            gate_score_src = tf.reshape(gate_score_src, [-1, 1, 1])
            gate_score_trg = tf.reshape(gate_score_trg, [-1, 1, 1])
            gate_score_src2 = tf.reshape(gate_score_src2, [-1, 1, 1])
            gate_score_trg2 = tf.reshape(gate_score_trg2, [-1, 1, 1])

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            '''
                user embedding
            '''
            domain_quries1 = get_variable(init, name='domain_quries1', shape=[2, head_num, embed_size])
            queries_src = domain_quries1[0]  # [head_num, 1, embed_size]
            queries_trg = domain_quries1[1]



            tmp_xv1_src = tf.reshape(tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv1_trg = tf.reshape(tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries1[0], axis=0),
                                   [tf.shape(xv1_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries1[1], axis=0), [tf.shape(xv1_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv1_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv1_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate1_src_low, _, _, _ = bin_mlp([64, 32, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_src,
                                                                        training=self.training, name='gate1_src_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B, head_num, anchor_num]

            gate1_src_high, _, _, _ = bin_mlp([64, 32, anchor_num], ['tanh', 'tanh', None],
                                    layer_keeps, gate_input_src,
                                    training=self.training, name='gate1_src_high',
                                    reuse=tf.AUTO_REUSE)  # [B, head_num, anchor_num]


            gate1_trg_low, _, _, _ = bin_mlp([64, 32, anchor_num], ['tanh', 'tanh', None],
                                                                        layer_keeps, gate_input_trg,
                                                                        training=self.training, name='gate1_trg_low',
                                                                        reuse=tf.AUTO_REUSE)    # [B * head_num, anchor_num]

            gate1_trg_high, _, _, _ = bin_mlp([64, 32, anchor_num], ['tanh', 'tanh', None],
                                    layer_keeps, gate_input_trg,
                                    training=self.training, name='gate1_trg_high',
                                    reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate1_src = gate_score_src * gate1_src_low + (1 - gate_score_src) * gate1_src_high
            gate1_src = tf.nn.sigmoid(gate1_src)
            gate1_trg = gate_score_trg * gate1_trg_low + (1 - gate_score_trg) * gate1_trg_high
            gate1_trg = tf.nn.sigmoid(gate1_trg)

            gate1_src = 2 * tf.reshape(gate1_src, [-1, head_num, anchor_num, 1])   # [B, head_num, anchor_num, 1]
            gate1_trg = 2 * tf.reshape(gate1_trg, [-1, head_num, anchor_num, 1])


            _anchors = tf.transpose(anchors1)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors), [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]

            head_scores_src = gate1_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate1_trg * head_scores_trg

            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1, dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_embeddings_src = head_scores_src * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors1, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)    # [B, head_num]
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)    # [B, head_num]

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv1_src = tf.concat([xv1_src, anchor_src], axis=-1)
            xv1_trg = tf.concat([xv1_trg, anchor_trg], axis=-1)

            '''
                item embedding
            '''
            domain_quries2 = get_variable(init, name='domain_quries2', shape=[2, head_num, embed_size])
            queries_src = tf.expand_dims(domain_quries2[0], axis=1)  # [head_num, 1, embed_size]
            queries_trg = tf.expand_dims(domain_quries2[1], axis=1)

            tmp_xv2_src = tf.reshape(tf.tile(tf.expand_dims(xv2_src, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_xv2_trg = tf.reshape(tf.tile(tf.expand_dims(xv2_trg, axis=-2), [1, head_num, 1]),
                                     [-1, embed_size])  # [B * head_num, embed_size]
            tmp_head_src = tf.tile(tf.expand_dims(domain_quries2[0], axis=0),
                                   [tf.shape(xv2_src)[0], 1, 1])  # [B * head_num, embed_size]
            tmp_head_trg = tf.tile(tf.expand_dims(domain_quries2[1], axis=0), [tf.shape(xv2_trg)[0], 1, 1])
            tmp_head_src = tf.reshape(tmp_head_src, [-1, embed_size])
            tmp_head_trg = tf.reshape(tmp_head_trg, [-1, embed_size])

            gate_input_src = tf.concat([tmp_xv2_src, tmp_head_src], axis=-1)  # [B * head_num, embed_size * 2]
            gate_input_trg = tf.concat([tmp_xv2_trg, tmp_head_trg], axis=-1)

            gate_input_src = tf.reshape(gate_input_src, [-1, head_num, 2 * embed_size])
            gate_input_trg = tf.reshape(gate_input_trg, [-1, head_num, 2 * embed_size])

            gate2_src_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                             layer_keeps, gate_input_src,
                                             training=self.training, name='gate2_src_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                              layer_keeps, gate_input_src,
                                              training=self.training, name='gate2_src_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_low, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                             layer_keeps, gate_input_trg,
                                             training=self.training, name='gate2_trg_low',
                                             reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_trg_high, _, _, _ = bin_mlp([128, 64, anchor_num], ['tanh', 'tanh', None],
                                              layer_keeps, gate_input_trg,
                                              training=self.training, name='gate2_trg_high',
                                              reuse=tf.AUTO_REUSE)  # [B * head_num, anchor_num]

            gate2_src = gate_score_src2 * gate2_src_low + (1 - gate_score_src2) * gate2_src_high
            gate2_src = tf.nn.sigmoid(gate2_src)
            gate2_trg = gate_score_trg2 * gate2_trg_low + (1 - gate_score_trg2) * gate2_trg_high
            gate2_trg = tf.nn.sigmoid(gate2_trg)

            gate2_src = 2 * tf.reshape(gate2_src, [-1, head_num, anchor_num, 1])  # [B, head_num, anchor_num, 1]
            gate2_trg = 2 * tf.reshape(gate2_trg, [-1, head_num, anchor_num, 1])

            _anchors = tf.transpose(anchors2)  # [embed_size, anchor_num]
            head_scores_src = tf.reshape(tf.matmul(queries_src, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_trg = tf.reshape(tf.matmul(queries_trg, _anchors),
                                         [1, head_num, anchor_num, 1])  # [1, head_num, anchor_num, 1]
            head_scores_src = tf.nn.softmax(head_scores_src / np.sqrt(embed_size) / tau1,
                                            dim=2)  # [B, head_num, anchor_num, 1]
            head_scores_trg = tf.nn.softmax(head_scores_trg / np.sqrt(embed_size) / tau2, dim=2)

            head_scores_src = gate2_src * head_scores_src  # [B, head_num, anchor_num, 1]
            head_scores_trg = gate2_trg * head_scores_trg

            head_embeddings_src = head_scores_src * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_trg = head_scores_trg * tf.reshape(anchors2, [1, 1, anchor_num, embed_size])
            head_embeddings_src = tf.reduce_sum(head_embeddings_src, axis=-2)  # [B, head_num, embed_size]
            head_embeddings_trg = tf.reduce_sum(head_embeddings_trg, axis=-2)  # [B, head_num, embed_size]

            score_src = tf.reduce_sum(tf.expand_dims(xv2_src, axis=1) * head_embeddings_src, axis=-1)  # [B, head_num]
            score_trg = tf.reduce_sum(tf.expand_dims(xv2_trg, axis=1) * head_embeddings_trg, axis=-1)  # [B, head_num]
            score_src = tf.nn.softmax(score_src / np.sqrt(embed_size), dim=-1)
            score_trg = tf.nn.softmax(score_trg / np.sqrt(embed_size), dim=-1)

            anchor_src = tf.reduce_sum(tf.expand_dims(score_src, axis=-1) * head_embeddings_src, axis=-2)
            anchor_trg = tf.reduce_sum(tf.expand_dims(score_trg, axis=-1) * head_embeddings_trg, axis=-2)

            # 333
            xv2_src = tf.concat([xv2_src, anchor_src], axis=-1)
            xv2_trg = tf.concat([xv2_trg, anchor_trg], axis=-1)
            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg
        elif hardness in [33]:
            '''
                3: no gate
            '''

            self.new_embed_size = int(embed_size / 2)

            xv1_src = tf.reshape(xv1_src, [2, -1, self.new_embed_size])
            xv2_src = tf.reshape(xv2_src, [2, -1, self.new_embed_size])
            xv1_trg = tf.reshape(xv1_trg, [2, -1, self.new_embed_size])
            xv2_trg = tf.reshape(xv2_trg, [2, -1, self.new_embed_size])


            anchors1 = get_variable(init, name='anchors1_w', shape=[2, anchor_num, self.new_embed_size])
            anchors2 = get_variable(init, name='anchors2_w', shape=[2, anchor_num, self.new_embed_size])

            tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_src,
                                                                        training=self.training, name='alpha1',
                                                                        reuse=tf.AUTO_REUSE)

            tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                                                        layer_keeps, xv1_trg,
                                                                        training=self.training, name='alpha2',
                                                                        reuse=tf.AUTO_REUSE)

            tau1 = tf.reshape(tau1, [-1, 1, 1, 1])
            tau2 = tf.reshape(tau2, [-1, 1, 1, 1])

            queries = get_variable(init, name='queries',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])
            keys = get_variable(init, name='keys',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])
            values = get_variable(init, name='values',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])


            '''
                user embedding
            '''
            domain_emb_src1 = get_variable(init, name='domain_emb_src1', shape=[2, self.new_embed_size])
            domain_emb_trg1 = get_variable(init, name='domain_emb_trg1', shape=[2, self.new_embed_size])

            xv1_src_in_domain = xv1_src + tf.expand_dims(domain_emb_src1, axis=1)    # [2, B, self.new_embed_size]
            xv1_trg_in_domain = xv1_trg + tf.expand_dims(domain_emb_trg1, axis=1)

            xv1_src_sub_domain = tf.expand_dims(xv1_src, axis=2) + tf.expand_dims(anchors1,
                                                                    axis=1)  # [2, B, anchor_num, self.new_embed_size]
            xv1_trg_sub_domain = tf.expand_dims(xv1_trg, axis=2) + tf.expand_dims(anchors1,
                                                                    axis=1)  # [2, B, anchor_num, self.new_embed_size]

            xv1_src_in_domain_queries_mean, xv1_src_in_domain_keys_mean, xv1_src_in_domain_values_mean = \
                                self.get_qkv(queries[0], keys[0], values[0], xv1_src_in_domain[0])
            xv1_src_in_domain_queries_cov, xv1_src_in_domain_keys_cov, xv1_src_in_domain_values_cov = \
                                self.get_qkv(queries[1], keys[1], values[1], xv1_src_in_domain[1], True)

            xv1_trg_in_domain_queries_mean, xv1_trg_in_domain_keys_trg_mean, xv1_trg_in_domain_values_trg_mean = \
                                 self.get_qkv(queries[0], keys[0], values[0], xv1_trg_in_domain[0])
            xv1_trg_in_domain_queries_cov, xv1_trg_in_domain_keys_trg_cov, xv1_trg_in_domain_values_trg_cov = \
                                self.get_qkv(queries[1], keys[1], values[1], xv1_trg_in_domain[1], True)

            xv1_src_sub_domain_queries_mean, xv1_src_sub_domain_keys_mean, xv1_src_sub_domain_values_mean = \
                                self.get_qkv(queries[0], keys[0], values[0], xv1_src_sub_domain[0])
            xv1_src_sub_domain_queries_cov, xv1_src_sub_domain_keys_cov, xv1_src_sub_domain_values_cov = \
                                self.get_qkv(queries[1], keys[1], values[1], xv1_src_sub_domain[1], True)

            xv1_trg_sub_domain_queries_mean, xv1_trg_sub_domain_keys_mean, xv1_trg_sub_domain_values_mean = \
                                 self.get_qkv_anchor(queries[0], keys[0], values[0], xv1_trg_sub_domain[0])
            xv1_trg_sub_domain_queries_cov, xv1_trg_sub_domain_keys_cov, xv1_trg_sub_domain_values_cov = \
                                self.get_qkv_anchor(queries[1], keys[1], values[1], xv1_trg_sub_domain[1], True)
            # [head_num, B, anchor_num, embed_size]

            attention_src = -self.wasserstein_distance_matmul(xv1_src_in_domain_queries_mean,
                                                              xv1_src_in_domain_queries_cov,
                                                              xv1_src_sub_domain_keys_mean,
                                                              xv1_src_sub_domain_keys_cov)

            attention_trg = -self.wasserstein_distance_matmul(xv1_trg_in_domain_queries_mean,
                                                              xv1_trg_in_domain_queries_cov,
                                                              xv1_trg_sub_domain_keys_mean,
                                                              xv1_trg_sub_domain_keys_cov)      # [head_num, B, anchor_num]

            attention_src = attention_src / np.sqrt(self.new_embed_size)         # [head_num, B, anchor_num]
            attention_trg = attention_trg / np.sqrt(self.new_embed_size)
            head_scores_src = tf.nn.softmax(attention_src, dim=-1)       # [head_num, B, anchor_num]
            head_scores_trg = tf.nn.softmax(attention_trg, dim=-1)


            xv1_src_agg_mean = tf.matmul(tf.expand_dims(head_scores_src, axis=-2), xv1_src_sub_domain_values_mean)
            xv1_trg_agg_mean = tf.matmul(tf.expand_dims(head_scores_trg, axis=-2), xv1_trg_sub_domain_values_mean)

            xv1_src_agg_cov = tf.matmul(tf.expand_dims(tf.square(head_scores_src), axis=-2), xv1_src_sub_domain_values_cov)
            xv1_trg_agg_cov = tf.matmul(tf.expand_dims(tf.square(head_scores_trg), axis=-2), xv1_trg_sub_domain_values_cov)

            xv1_src_mean = (xv1_src_sub_domain[0] + xv1_src_agg_mean) / 2
            xv1_trg_mean = (xv1_trg_sub_domain[0] + xv1_trg_agg_mean) / 2

            xv1_src_cov = (xv1_src_sub_domain[1] + xv1_src_agg_cov) / 2
            xv1_trg_cov = (xv1_trg_sub_domain[1] + xv1_trg_agg_cov) / 2

            '''
                item embedding
            '''
            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_in_domain = xv2_src + tf.expand_dims(domain_emb_src2, axis=1)    # [2, B, self.new_embed_size]
            xv2_trg_in_domain = xv2_trg + tf.expand_dims(domain_emb_trg2, axis=1)

            xv2_src_sub_domain = tf.expand_dims(xv2_src, axis=2) + tf.expand_dims(anchors2,
                                                                    axis=1)  # [2, B, anchor_num, self.new_embed_size]
            xv2_trg_sub_domain = tf.expand_dims(xv1_trg, axis=2) + tf.expand_dims(anchors2,
                                                                    axis=1)  # [2, B, anchor_num, self.new_embed_size]

            # queries = get_variable(init, name='queries',
            #                               shape=[2, self.new_embed_size, head_num * self.new_embed_size])
            # keys = get_variable(init, name='keys',
            #                               shape=[2, self.new_embed_size, head_num * self.new_embed_size])
            # values = get_variable(init, name='values',
            #                               shape=[2, self.new_embed_size, head_num * self.new_embed_size])

            xv2_src_in_domain_queries_mean, xv2_src_in_domain_keys_mean, xv2_src_in_domain_values_mean = \
                                self.get_qkv(queries[0], keys[0], values[0], xv2_src_in_domain[0])
            xv2_src_in_domain_queries_cov, xv2_src_in_domain_keys_cov, xv2_src_in_domain_values_cov = \
                                self.get_qkv(queries[1], keys[1], values[1], xv2_src_in_domain[1], True)

            xv2_trg_in_domain_queries_mean, xv2_trg_in_domain_keys_trg_mean, xv2_trg_in_domain_values_trg_mean = \
                                 self.get_qkv(queries[0], keys[0], values[0], xv2_trg_in_domain[0])
            xv2_trg_in_domain_queries_cov, xv2_trg_in_domain_keys_trg_cov, xv2_trg_in_domain_values_trg_cov = \
                                self.get_qkv(queries[1], keys[1], values[1], xv2_trg_in_domain[1], True)

            xv2_src_sub_domain_queries_mean, xv2_src_sub_domain_keys_mean, xv2_src_sub_domain_values_mean = \
                                self.get_qkv(queries[0], keys[0], values[0], xv2_src_sub_domain[0])
            xv2_src_sub_domain_queries_cov, xv2_src_sub_domain_keys_cov, xv2_src_sub_domain_values_cov = \
                                self.get_qkv(queries[1], keys[1], values[1], xv2_src_sub_domain[1], True)

            xv2_trg_sub_domain_queries_mean, xv2_trg_sub_domain_keys_mean, xv2_trg_sub_domain_values_mean = \
                                 self.get_qkv_anchor(queries[0], keys[0], values[0], xv2_trg_sub_domain[0])
            xv2_trg_sub_domain_queries_cov, xv2_trg_sub_domain_keys_cov, xv2_trg_sub_domain_values_cov = \
                                self.get_qkv_anchor(queries[1], keys[1], values[1], xv2_trg_sub_domain[1], True)
            # [head_num, B, anchor_num, embed_size]

            attention_src = -self.wasserstein_distance_matmul(xv2_src_in_domain_queries_mean,
                                                              xv2_src_in_domain_queries_cov,
                                                              xv2_src_sub_domain_keys_mean,
                                                              xv2_src_sub_domain_keys_cov)

            attention_trg = -self.wasserstein_distance_matmul(xv2_trg_in_domain_queries_mean,
                                                              xv2_trg_in_domain_queries_cov,
                                                              xv2_trg_sub_domain_keys_mean,
                                                              xv2_trg_sub_domain_keys_cov)      # [head_num, B, anchor_num]

            attention_src = attention_src / np.sqrt(self.new_embed_size)         # [head_num, B, anchor_num]
            attention_trg = attention_trg / np.sqrt(self.new_embed_size)
            head_scores_src = tf.nn.softmax(attention_src, dim=-1)       # [head_num, B, anchor_num]
            head_scores_trg = tf.nn.softmax(attention_trg, dim=-1)


            xv2_src_agg_mean = tf.matmul(tf.expand_dims(head_scores_src, axis=-2), xv2_src_sub_domain_values_mean)
            xv2_trg_agg_mean = tf.matmul(tf.expand_dims(head_scores_trg, axis=-2), xv2_trg_sub_domain_values_mean)

            xv2_src_agg_cov = tf.matmul(tf.expand_dims(tf.square(head_scores_src), axis=-2), xv2_src_sub_domain_values_cov)
            xv2_trg_agg_cov = tf.matmul(tf.expand_dims(tf.square(head_scores_trg), axis=-2), xv2_trg_sub_domain_values_cov)

            xv2_src_mean = (xv2_src_sub_domain[0] + xv2_src_agg_mean) / 2
            xv2_trg_mean = (xv2_trg_sub_domain[0] + xv2_trg_agg_mean) / 2

            xv2_src_cov = (xv2_src_sub_domain[1] + xv2_src_agg_cov) / 2
            xv2_trg_cov = (xv2_trg_sub_domain[1] + xv2_trg_agg_cov) / 2


            self.concat_shape += embed_size
            self.head_scores_src = head_scores_src
            self.head_scores_trg = head_scores_trg




        self.xv1_src = xv1_src
        self.xv1_trg = xv1_trg
        self.xv2_src = xv2_src
        self.xv2_trg = xv2_trg
        hist1 = tf.gather(v_item, self.history1)
        hist2 = tf.gather(v_item, self.history2)

        if hist_type == 1:
            # source
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history1 = user_history1 / tf.expand_dims(tf.cast(self.history_len1, tf.float32), 1)

            # target
            user_history2 = tf.reduce_sum(hist2, axis=-2)
            user_history2 = user_history2 / tf.expand_dims(tf.cast(self.history_len2, tf.float32), 1)
        elif hist_type == 2:
            user_history1 = tf.reduce_sum(hist1, axis=-2)
            user_history2 = tf.reduce_sum(hist2, axis=-2)
        elif hist_type == 3:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention', self.training,
                          reuse=tf.AUTO_REUSE), 1)
        elif hist_type == 4:
            current_fengge_xv = xv2
            history_fengge_xv1 = hist1
            user_history1 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv1, self.history_len1, 'his_attention1', self.training), 1)

            history_fengge_xv2 = hist2
            user_history2 = tf.squeeze(
                attention(current_fengge_xv, history_fengge_xv2, self.history_len2, 'his_attention2', self.training), 1)

        if hist_type > 0:
            user_feat1 = tf.concat([xv1_src, user_history1], axis=-1)
            user_feat2 = tf.concat([xv1_trg, user_history2], axis=-1)
            self.concat_shape += embed_size * 2
        else:
            user_feat1 = xv1_src
            user_feat2 = xv1_trg
            self.concat_shape += embed_size * 1

        h1 = tf.concat([user_feat1, xv2_src], axis=-1)
        h2 = tf.concat([user_feat2, xv2_trg], axis=-1)
        self.h1 = h1
        self.h2 = h2
        self.concat_shape += embed_size

        if cotrain:
            h1, self.layer_kernels, self.layer_biases, nn_h = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                      training=self.training, name='mlp',
                                                                      reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, nn_h = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                      training=self.training, name='mlp',
                                                                      reuse=tf.AUTO_REUSE)
        elif cotrain == 2:
            h1, self.layer_kernels, self.layer_biases, nn_h = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h1,
                                                                        training=self.training, name='mlp',
                                                                        reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, nn_h = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h2,
                                                                        training=self.training, name='mlp',
                                                                        reuse=tf.AUTO_REUSE)
        elif cotrain == 3:
            h1, self.layer_kernels, self.layer_biases, nn_h = bin_mlp(layer_sizes, layer_acts, layer_keeps, h1,
                                                                      training=self.training, name='mlp1',
                                                                      reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, nn_h = bin_mlp(layer_sizes, layer_acts, layer_keeps, h2,
                                                                      training=self.training, name='mlp2',
                                                                      reuse=tf.AUTO_REUSE)
        elif cotrain == 4:
            h1, self.layer_kernels, self.layer_biases, nn_h = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h1,
                                                                        training=self.training, name='mlp1',
                                                                        reuse=tf.AUTO_REUSE)

            h2, self.layer_kernels, self.layer_biases, nn_h = bin_mlp_2(layer_sizes, layer_acts, layer_keeps, h2,
                                                                        training=self.training, name='mlp2',
                                                                        reuse=tf.AUTO_REUSE)

        h1 = tf.squeeze(h1)
        h2 = tf.squeeze(h2)

        self.logits1, self.outputs1 = output([h1, b])
        self.logits2, self.outputs2 = output([h2, b])

    def get_qkv(self, query, key, value, anchors, activate=False):
        new_queries = tf.transpose(tf.reshape(tf.matmul(anchors, query), [-1, self.head_num, self.new_embed_size]), (1, 0, 2))
        new_keys = tf.transpose(tf.reshape(tf.matmul(anchors, key), [-1, self.head_num, self.new_embed_size]), (1, 0, 2))
        new_values = tf.transpose(tf.reshape(tf.matmul(anchors, value), [-1, self.head_num, self.new_embed_size]), (1, 0, 2))
        # [head_num, B, embed_size]
        if activate:
            new_queries = tf.nn.elu(new_queries) + 1
            new_keys = tf.nn.elu(new_keys) + 1
            new_values = tf.nn.elu(new_values) + 1
        return new_queries, new_keys, new_values

    def get_qkv_anchor(self, query, key, value, anchors, activate=False):
        new_queries = tf.transpose(tf.reshape(tf.matmul(anchors, query), [-1, self.anchor_num, self.head_num, self.new_embed_size]), (2, 0, 1, 3))
        new_keys = tf.transpose(tf.reshape(tf.matmul(anchors, key), [-1, self.anchor_num, self.head_num, self.new_embed_size]), (2, 0, 1, 3))
        new_values = tf.transpose(tf.reshape(tf.matmul(anchors, value), [-1, self.anchor_num, self.head_num, self.new_embed_size]), (2, 0, 1, 3))
        # [head_num, B, anchor_num, embed_size]
        if activate:
            new_queries = tf.nn.elu(new_queries) + 1
            new_keys = tf.nn.elu(new_keys) + 1
            new_values = tf.nn.elu(new_values) + 1
        return new_queries, new_keys, new_values

    def wasserstein_distance_matmul(self, mean1, cov1, mean2, cov2):
        mean1_2 = tf.reduce_sum(tf.square(mean1), axis=-1, keepdims=True)   # [head_num, anchor_num, 1]
        mean2_2 = tf.reduce_sum(tf.square(mean2), axis=-1, keepdims=True)   # [head_num, anchor_num, 1]
        ret = -2 * tf.matmul(mean1, tf.transpose(mean2, (0, 2, 1))) + mean1_2 + tf.transpose(mean2_2, (0, 2, 1))
        # [head_num, anchor_num, anchor_num]

        cov1_2 = tf.reduce_sum(cov1, axis=-1, keepdims=True)
        cov2_2 = tf.reduce_sum(cov2, axis=-1, keepdims=True)
        cov_ret = -2 * tf.matmul(tf.sqrt(tf.clip_by_value(cov1, 1e-24, tf.reduce_max(cov1))),
                                    tf.sqrt(tf.transpose(tf.clip_by_value(cov2, 1e-24, tf.reduce_max(cov1)), (0, 2, 1))))\
                  + cov1_2 + tf.transpose(cov2_2, (0, 2, 1))

        return ret + cov_ret



    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.entropy1 = loss(logits=self.logits1, targets=self.labels_src, pos_weight=pos_weight)
                self.origin_loss1 = tf.reduce_mean(self.entropy1)
                self.loss1 = self.origin_loss1

                self.entropy2 = loss(logits=self.logits2, targets=self.labels_trg, pos_weight=pos_weight)
                self.origin_loss2 = tf.reduce_mean(self.entropy2)
                self.loss2 = self.origin_loss2


                if self.hardness in [2]:
                    gate_loss1 = tf.reduce_mean(tf.abs(self.mask_label_src1 - self.gate_src1))
                    gate_loss2 = tf.reduce_mean(tf.abs(self.mask_label_src2 - self.gate_src2))
                    gate_loss3 = tf.reduce_mean(tf.abs(self.mask_label_trg1 - self.gate_trg1))
                    gate_loss4 = tf.reduce_mean(tf.abs(self.mask_label_trg2 - self.gate_trg2))
                    self.gate_loss_src = gate_loss1 + gate_loss2
                    self.gate_loss_trg = gate_loss3 + gate_loss4
                    self.loss1 += 0.5 * self.gate_loss_src
                    self.loss2 += 0.5 * self.gate_loss_trg
                elif self.hardness in [4]:
                    gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src) * self.anchor_num  # [B,]
                    if self.calc_pattern < 9:
                        gate_score_trg = tf.gather(self.hardness_score_src, self.user_id_trg) * self.anchor_num  # [B,]
                    else:
                        gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg) * self.anchor_num  # [B,]
                    gate_score_src = tf.tile(tf.expand_dims(gate_score_src, axis=-1), [1, self.head_num])
                    gate_score_trg = tf.tile(tf.expand_dims(gate_score_trg, axis=-1), [1, self.head_num])
                    gate_src1 = tf.reduce_sum(self.gate_src1, axis=-1)  # [B, head_num]
                    gate_src2 = tf.reduce_sum(self.gate_src2, axis=-1)  # [B, head_num]
                    gate_trg1 = tf.reduce_sum(self.gate_trg1, axis=-1)  # [B, head_num]
                    gate_trg2 = tf.reduce_sum(self.gate_trg2, axis=-1)  # [B, head_num]
                    gate_loss1 = tf.reduce_mean(tf.abs(gate_src1 - gate_score_src))
                    gate_loss2 = tf.reduce_mean(tf.abs(gate_src2 - gate_score_src))
                    gate_loss3 = tf.reduce_mean(tf.abs(gate_trg1 - gate_score_trg))
                    gate_loss4 = tf.reduce_mean(tf.abs(gate_trg2 - gate_score_trg))
                    self.gate_loss_src = gate_loss1 + gate_loss2
                    self.gate_loss_trg = gate_loss3 + gate_loss4
                    self.loss1 += 0.5 * self.gate_loss_src
                    self.loss2 += 0.5 * self.gate_loss_trg
                elif self.hardness in [5]:
                    pos_num = tf.reduce_sum(tf.ones_like(self.labels_src))
                    neg_num = tf.reduce_sum(tf.ones_like(self.labels_trg))
                    sample_ratio = (neg_num - pos_num) / pos_num
                    # self.sample_ratio = sample_ratio

                    h_dis_src = self.h_dis_src / (self.h_dis_src + (1 - self.h_dis_src) / sample_ratio)
                    h_dis_trg = self.h_dis_trg / (self.h_dis_trg + (1 - self.h_dis_trg) / sample_ratio)

                    self.new_h_dis_src = h_dis_src
                    self.new_h_dis_trg = h_dis_trg

                    gate_score_src = h_dis_src * self.anchor_num  # [B,]
                    gate_score_trg = (1 - h_dis_trg) * self.anchor_num  # [B,]
                    gate_score_src = tf.tile(tf.expand_dims(gate_score_src, axis=-1), [1, self.head_num])
                    gate_score_trg = tf.tile(tf.expand_dims(gate_score_trg, axis=-1), [1, self.head_num])
                    gate_src1 = tf.reduce_sum(self.gate_src1, axis=-1)  # [B, head_num]
                    gate_src2 = tf.reduce_sum(self.gate_src2, axis=-1)  # [B, head_num]
                    gate_trg1 = tf.reduce_sum(self.gate_trg1, axis=-1)  # [B, head_num]
                    gate_trg2 = tf.reduce_sum(self.gate_trg2, axis=-1)  # [B, head_num]
                    gate_loss1 = tf.reduce_mean(tf.abs(gate_src1 - gate_score_src))
                    gate_loss2 = tf.reduce_mean(tf.abs(gate_src2 - gate_score_src))
                    gate_loss3 = tf.reduce_mean(tf.abs(gate_trg1 - gate_score_trg))
                    gate_loss4 = tf.reduce_mean(tf.abs(gate_trg2 - gate_score_trg))
                    self.gate_loss_src = gate_loss1 + gate_loss2
                    self.gate_loss_trg = gate_loss3 + gate_loss4
                    self.loss1 += 0.5 * self.gate_loss_src
                    self.loss2 += 0.5 * self.gate_loss_trg
                elif self.hardness in [6]:
                    pos_num = tf.reduce_sum(tf.ones_like(self.labels_src))
                    neg_num = tf.reduce_sum(tf.ones_like(self.labels_trg))
                    sample_ratio = (neg_num - pos_num) / pos_num

                    h_dis_src = self.h_dis_src / (self.h_dis_src + (1 - self.h_dis_src) / sample_ratio)
                    h_dis_trg = self.h_dis_trg / (self.h_dis_trg + (1 - self.h_dis_trg) / sample_ratio)

                    gate_score_src = (1 - h_dis_src) * self.anchor_num  # [B,]
                    gate_score_trg = h_dis_trg * self.anchor_num  # [B,]
                    gate_score_src = tf.tile(tf.expand_dims(gate_score_src, axis=-1), [1, self.head_num])
                    gate_score_trg = tf.tile(tf.expand_dims(gate_score_trg, axis=-1), [1, self.head_num])
                    gate_src1 = tf.reduce_sum(self.gate_src1, axis=-1)  # [B, head_num]
                    gate_src2 = tf.reduce_sum(self.gate_src2, axis=-1)  # [B, head_num]
                    gate_trg1 = tf.reduce_sum(self.gate_trg1, axis=-1)  # [B, head_num]
                    gate_trg2 = tf.reduce_sum(self.gate_trg2, axis=-1)  # [B, head_num]
                    gate_loss1 = tf.reduce_mean(tf.abs(gate_src1 - gate_score_src))
                    gate_loss2 = tf.reduce_mean(tf.abs(gate_src2 - gate_score_src))
                    gate_loss3 = tf.reduce_mean(tf.abs(gate_trg1 - gate_score_trg))
                    gate_loss4 = tf.reduce_mean(tf.abs(gate_trg2 - gate_score_trg))
                    self.gate_loss_src = gate_loss1 + gate_loss2
                    self.gate_loss_trg = gate_loss3 + gate_loss4
                    self.loss1 += 0.5 * self.gate_loss_src
                    self.loss2 += 0.5 * self.gate_loss_trg
                elif self.hardness in [7]:
                    pos_num = tf.reduce_sum(tf.ones_like(self.labels_src))
                    neg_num = tf.reduce_sum(tf.ones_like(self.labels_trg))
                    sample_ratio = (neg_num - pos_num) / pos_num

                    h_dis_src = self.h_dis_src / (self.h_dis_src + (1 - self.h_dis_src) / sample_ratio)
                    h_dis_trg = self.h_dis_trg / (self.h_dis_trg + (1 - self.h_dis_trg) / sample_ratio)



                    gate_score_src = tf.stop_gradient(h_dis_src) * self.anchor_num  # [B,]
                    gate_score_trg = (1 - tf.stop_gradient(h_dis_trg)) * self.anchor_num  # [B,]
                    gate_score_src = tf.tile(tf.expand_dims(gate_score_src, axis=-1), [1, self.head_num])
                    gate_score_trg = tf.tile(tf.expand_dims(gate_score_trg, axis=-1), [1, self.head_num])
                    gate_src1 = tf.reduce_sum(self.gate_src1, axis=-1)  # [B, head_num]
                    gate_src2 = tf.reduce_sum(self.gate_src2, axis=-1)  # [B, head_num]
                    gate_trg1 = tf.reduce_sum(self.gate_trg1, axis=-1)  # [B, head_num]
                    gate_trg2 = tf.reduce_sum(self.gate_trg2, axis=-1)  # [B, head_num]
                    gate_loss1 = tf.reduce_mean(tf.abs(gate_src1 - gate_score_src))
                    gate_loss2 = tf.reduce_mean(tf.abs(gate_src2 - gate_score_src))
                    gate_loss3 = tf.reduce_mean(tf.abs(gate_trg1 - gate_score_trg))
                    gate_loss4 = tf.reduce_mean(tf.abs(gate_trg2 - gate_score_trg))
                    self.gate_loss_src = gate_loss1 + gate_loss2
                    self.gate_loss_trg = gate_loss3 + gate_loss4
                    self.loss1 += 0.5 * self.gate_loss_src
                    self.loss2 += 0.5 * self.gate_loss_trg
                elif self.hardness in [8]:
                    pos_num = tf.reduce_sum(tf.ones_like(self.labels_src))
                    neg_num = tf.reduce_sum(tf.ones_like(self.labels_trg))
                    sample_ratio = (neg_num - pos_num) / pos_num

                    h_dis_src = self.h_dis_src / (self.h_dis_src + (1 - self.h_dis_src) / sample_ratio)
                    h_dis_trg = self.h_dis_trg / (self.h_dis_trg + (1 - self.h_dis_trg) / sample_ratio)

                    gate_score_src = (1 - tf.stop_gradient(h_dis_src)) * self.anchor_num  # [B,]
                    gate_score_trg = tf.stop_gradient(h_dis_trg) * self.anchor_num  # [B,]
                    gate_score_src = tf.tile(tf.expand_dims(gate_score_src, axis=-1), [1, self.head_num])
                    gate_score_trg = tf.tile(tf.expand_dims(gate_score_trg, axis=-1), [1, self.head_num])
                    gate_src1 = tf.reduce_sum(self.gate_src1, axis=-1)  # [B, head_num]
                    gate_src2 = tf.reduce_sum(self.gate_src2, axis=-1)  # [B, head_num]
                    gate_trg1 = tf.reduce_sum(self.gate_trg1, axis=-1)  # [B, head_num]
                    gate_trg2 = tf.reduce_sum(self.gate_trg2, axis=-1)  # [B, head_num]
                    gate_loss1 = tf.reduce_mean(tf.abs(gate_src1 - gate_score_src))
                    gate_loss2 = tf.reduce_mean(tf.abs(gate_src2 - gate_score_src))
                    gate_loss3 = tf.reduce_mean(tf.abs(gate_trg1 - gate_score_trg))
                    gate_loss4 = tf.reduce_mean(tf.abs(gate_trg2 - gate_score_trg))
                    self.gate_loss_src = gate_loss1 + gate_loss2
                    self.gate_loss_trg = gate_loss3 + gate_loss4
                    self.loss1 += 0.5 * self.gate_loss_src
                    self.loss2 += 0.5 * self.gate_loss_trg
                elif self.hardness in [10]:
                    gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src) * self.anchor_num  # [B,]
                    if self.calc_pattern < 9:
                        gate_score_trg = tf.gather(self.hardness_score_src, self.user_id_trg) * self.anchor_num  # [B,]
                    else:
                        gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg) * self.anchor_num  # [B,]
                    gate_score_src = tf.tile(tf.expand_dims(gate_score_src, axis=-1), [1, self.head_num])
                    gate_score_trg = tf.tile(tf.expand_dims(gate_score_trg, axis=-1), [1, self.head_num])
                    gate_src1 = tf.reduce_sum(self.gate_src1, axis=-1)  # [B, head_num]
                    gate_src2 = tf.reduce_sum(self.gate_src2, axis=-1)  # [B, head_num]
                    gate_trg1 = tf.reduce_sum(self.gate_trg1, axis=-1)  # [B, head_num]
                    gate_trg2 = tf.reduce_sum(self.gate_trg2, axis=-1)  # [B, head_num]
                    gate_loss1 = tf.reduce_mean(tf.abs(gate_src1 - gate_score_src))
                    gate_loss2 = tf.reduce_mean(tf.abs(gate_src2 - gate_score_src))
                    gate_loss3 = tf.reduce_mean(tf.abs(gate_trg1 - gate_score_trg))
                    gate_loss4 = tf.reduce_mean(tf.abs(gate_trg2 - gate_score_trg))
                    self.gate_loss_src = gate_loss1 + gate_loss2
                    self.gate_loss_trg = gate_loss3 + gate_loss4
                    self.loss1 += 0.5 * self.gate_loss_src
                    self.loss2 += 0.5 * self.gate_loss_trg
                elif self.hardness in [11, 12]:
                    if self.calc_pattern not in [15, 16, 17, 18]:
                        gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src) * self.anchor_num  # [B,]
                        if self.calc_pattern < 9:
                            gate_score_trg = tf.gather(self.hardness_score_src, self.user_id_trg) * self.anchor_num  # [B,]
                        else:
                            gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg) * self.anchor_num  # [B,]
                        gate_score_src = tf.tile(tf.expand_dims(gate_score_src, axis=-1), [1, self.head_num])
                        gate_score_trg = tf.tile(tf.expand_dims(gate_score_trg, axis=-1), [1, self.head_num])
                        gate_src1 = tf.reduce_sum(self.head_scores_src1, axis=-1)  # [B, head_num]
                        gate_src2 = tf.reduce_sum(self.head_scores_src2, axis=-1)  # [B, head_num]
                        gate_trg1 = tf.reduce_sum(self.head_scores_trg1, axis=-1)  # [B, head_num]
                        gate_trg2 = tf.reduce_sum(self.head_scores_trg2, axis=-1)  # [B, head_num]
                        gate_loss1 = tf.reduce_mean(tf.abs(gate_src1 - gate_score_src))
                        gate_loss2 = tf.reduce_mean(tf.abs(gate_src2 - gate_score_src))
                        gate_loss3 = tf.reduce_mean(tf.abs(gate_trg1 - gate_score_trg))
                        gate_loss4 = tf.reduce_mean(tf.abs(gate_trg2 - gate_score_trg))
                        self.gate_loss_src = gate_loss1 + gate_loss2
                        self.gate_loss_trg = gate_loss3 + gate_loss4
                        self.loss1 += 0.5 * self.gate_loss_src
                        self.loss2 += 0.5 * self.gate_loss_trg
                    else:
                        '''
                            redularization loss
                        '''
                        gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src)  # [B,]
                        gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg)  # [B,]
                        gate_src1 = tf.reduce_mean(self.head_scores_src1, axis=-1)  # [B, head_num]
                        gate_src2 = tf.reduce_mean(self.head_scores_src2, axis=-1)  # [B, head_num]
                        gate_trg1 = tf.reduce_mean(self.head_scores_trg1, axis=-1)  # [B, head_num]
                        gate_trg2 = tf.reduce_mean(self.head_scores_trg2, axis=-1)  # [B, head_num]
                        gate_loss1 = tf.reduce_mean(tf.reduce_mean(gate_src1, axis=-1) * gate_score_src)
                        gate_loss2 = tf.reduce_mean(tf.reduce_mean(gate_src2, axis=-1) * gate_score_src)
                        gate_loss3 = tf.reduce_mean(tf.reduce_mean(gate_trg1, axis=-1) * gate_score_trg)
                        gate_loss4 = tf.reduce_mean(tf.reduce_mean(gate_trg2, axis=-1) * gate_score_trg)
                        self.gate_loss_src = gate_loss1 + gate_loss2
                        self.gate_loss_trg = gate_loss3 + gate_loss4
                        self.loss1 += 0.5 * self.gate_loss_src
                        self.loss2 += 0.5 * self.gate_loss_trg
                elif self.hardness in [13]:
                    gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src) * self.anchor_num  # [B,]
                    if self.calc_pattern < 9:
                        gate_score_trg = tf.gather(self.hardness_score_src, self.user_id_trg) * self.anchor_num  # [B,]
                    else:
                        gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg) * self.anchor_num  # [B,]
                    gate_score_src = tf.tile(tf.expand_dims(gate_score_src, axis=-1), [1, self.head_num])
                    gate_score_trg = tf.tile(tf.expand_dims(gate_score_trg, axis=-1), [1, self.head_num])
                    gate_src1 = tf.reduce_sum(self.head_scores_src1, axis=-1)  # [B, head_num]
                    gate_trg1 = tf.reduce_sum(self.head_scores_trg1, axis=-1)  # [B, head_num]
                    gate_loss1 = tf.reduce_mean(tf.abs(gate_src1 - gate_score_src))
                    gate_loss3 = tf.reduce_mean(tf.abs(gate_trg1 - gate_score_trg))
                    self.gate_loss_src = gate_loss1
                    self.gate_loss_trg = gate_loss3
                    self.loss1 += 0.5 * self.gate_loss_src
                    self.loss2 += 0.5 * self.gate_loss_trg
                elif self.hardness in [15]:
                    gate_score_src = tf.gather(self.hardness_score_src, self.user_id_src) * self.anchor_num  # [B,]
                    gate_score_trg = tf.gather(self.hardness_score_trg, self.user_id_trg) * self.anchor_num  # [B,]
                    gate_score_src2 = tf.gather(self.hardness_score_src2, self.item_id_src) * self.anchor_num  # [B,]
                    gate_score_trg2 = tf.gather(self.hardness_score_trg2, self.item_id_trg) * self.anchor_num  # [B,]

                    gate_score_src = tf.tile(tf.expand_dims(gate_score_src, axis=-1), [1, self.head_num])
                    gate_score_trg = tf.tile(tf.expand_dims(gate_score_trg, axis=-1), [1, self.head_num])

                    gate_score_src2 = tf.tile(tf.expand_dims(gate_score_src2, axis=-1), [1, self.head_num])
                    gate_score_trg2 = tf.tile(tf.expand_dims(gate_score_trg2, axis=-1), [1, self.head_num])


                    gate_src1 = tf.reduce_sum(self.head_scores_src1, axis=-1)  # [B, head_num]
                    gate_src2 = tf.reduce_sum(self.head_scores_src2, axis=-1)  # [B, head_num]
                    gate_trg1 = tf.reduce_sum(self.head_scores_trg1, axis=-1)  # [B, head_num]
                    gate_trg2 = tf.reduce_sum(self.head_scores_trg2, axis=-1)  # [B, head_num]
                    gate_loss1 = tf.reduce_mean(tf.abs(gate_src1 - gate_score_src))
                    gate_loss2 = tf.reduce_mean(tf.abs(gate_src2 - gate_score_src2))
                    gate_loss3 = tf.reduce_mean(tf.abs(gate_trg1 - gate_score_trg))
                    gate_loss4 = tf.reduce_mean(tf.abs(gate_trg2 - gate_score_trg2))
                    self.gate_loss_src = gate_loss1 + gate_loss2
                    self.gate_loss_trg = gate_loss3 + gate_loss4
                    self.loss1 += 0.5 * self.gate_loss_src
                    self.loss2 += 0.5 * self.gate_loss_trg

                if self.ae in [1, 2]:
                    ae_loss_src = tf.sqrt(tf.reduce_sum(tf.square(self.ae_xv1_src - self.xv1_src_origin), axis=-1))
                    ae_loss_src = tf.reduce_mean(ae_loss_src)

                    ae_loss_trg = tf.sqrt(tf.reduce_sum(tf.square(self.ae_xv1_trg - self.xv1_trg_origin), axis=-1))
                    ae_loss_trg = tf.reduce_mean(ae_loss_trg)

                    self.loss1 += 0.5 * ae_loss_src
                    self.loss2 += 0.5 * ae_loss_trg

                    dis_labels = tf.concat([tf.ones_like(self.labels_src), tf.zeros_like(self.labels_trg)], axis=0)
                    dis_logits = tf.concat([self.h_dis_src, self.h_dis_trg], axis=0)
                    dis_loss = loss(logits=dis_logits, targets=dis_labels, pos_weight=pos_weight)
                    dis_loss = tf.reduce_mean(dis_loss)

                    t_vars = tf.trainable_variables()
                    # for var in t_vars:
                    #     print(var)
                    d_vars = [var for var in t_vars if 'discriminator' in var.name]
                    g_vars = [var for var in t_vars if 'generator' in var.name]
                    self.d_train_opt = optimizer.minimize(loss=dis_loss, global_step=global_step, var_list=d_vars)
                    self.g_train_opt = optimizer.minimize(loss=-dis_loss, global_step=global_step, var_list=g_vars)
                    self.dis_loss = dis_loss
                elif self.ae in [3, 4, 5, 6]:
                    ae_loss_src1 = tf.sqrt(tf.reduce_sum(tf.square(self.ae_xv1_src - self.xv1_src_origin), axis=-1))
                    ae_loss_src1 = tf.reduce_mean(ae_loss_src1)

                    ae_loss_trg1 = tf.sqrt(tf.reduce_sum(tf.square(self.ae_xv1_trg - self.xv1_trg_origin), axis=-1))
                    ae_loss_trg1 = tf.reduce_mean(ae_loss_trg1)

                    ae_loss_src2 = tf.sqrt(tf.reduce_sum(tf.square(self.ae_xv2_src - self.xv2_src_origin), axis=-1))
                    ae_loss_src2 = tf.reduce_mean(ae_loss_src2)

                    ae_loss_trg2 = tf.sqrt(tf.reduce_sum(tf.square(self.ae_xv2_trg - self.xv2_trg_origin), axis=-1))
                    ae_loss_trg2 = tf.reduce_mean(ae_loss_trg2)

                    self.loss1 += 0.5 * ae_loss_src1
                    self.loss2 += 0.5 * ae_loss_trg1
                    self.loss1 += 0.5 * ae_loss_src2
                    self.loss2 += 0.5 * ae_loss_trg2


                    dis_labels = tf.concat([tf.ones_like(self.labels_src), tf.zeros_like(self.labels_trg)], axis=0)
                    dis_logits1 = tf.concat([self.h_dis_src1, self.h_dis_trg1], axis=0)
                    dis_loss1 = loss(logits=dis_logits1, targets=dis_labels, pos_weight=pos_weight)
                    dis_loss1 = tf.reduce_mean(dis_loss1)
                    dis_logits2 = tf.concat([self.h_dis_src2, self.h_dis_trg2], axis=0)
                    dis_loss2 = loss(logits=dis_logits2, targets=dis_labels, pos_weight=pos_weight)
                    dis_loss2 = tf.reduce_mean(dis_loss2)

                    t_vars = tf.trainable_variables()
                    # for var in t_vars:
                    #     print(var)
                    d_vars = [var for var in t_vars if 'discriminator' in var.name]
                    g_vars = [var for var in t_vars if 'generator' in var.name]
                    self.d_train_opt = optimizer.minimize(loss=dis_loss1+dis_loss2, global_step=global_step, var_list=d_vars)
                    self.g_train_opt = optimizer.minimize(loss=-(dis_loss1+dis_loss2), global_step=global_step, var_list=g_vars)
                    self.dis_loss1 = dis_loss1
                    self.dis_loss2 = dis_loss2
                elif self.ae in [7]:

                    dis_labels = tf.concat([tf.ones_like(self.labels_src), tf.zeros_like(self.labels_trg)], axis=0)
                    dis_logits1 = tf.concat([self.h_dis_src1, self.h_dis_trg1], axis=0)
                    dis_loss1 = loss(logits=dis_logits1, targets=dis_labels, pos_weight=pos_weight)
                    dis_loss1 = tf.reduce_mean(dis_loss1)
                    dis_logits2 = tf.concat([self.h_dis_src2, self.h_dis_trg2], axis=0)
                    dis_loss2 = loss(logits=dis_logits2, targets=dis_labels, pos_weight=pos_weight)
                    dis_loss2 = tf.reduce_mean(dis_loss2)

                    t_vars = tf.trainable_variables()
                    # for var in t_vars:
                    #     print(var)
                    d_vars = [var for var in t_vars if 'discriminator' in var.name]
                    g_vars = [var for var in t_vars if 'generator' in var.name]
                    self.d_train_opt = optimizer.minimize(loss=dis_loss1+dis_loss2, global_step=global_step, var_list=d_vars)
                    self.g_train_opt = optimizer.minimize(loss=-(dis_loss1+dis_loss2), global_step=global_step, var_list=g_vars)
                    self.dis_loss1 = dis_loss1
                    self.dis_loss2 = dis_loss2


                _loss_ = self.loss
                self.optimizer1 = optimizer.minimize(loss=self.loss1,
                                                     global_step=global_step)
                self.optimizer2 = optimizer.minimize(loss=self.loss2,
                                                     global_step=global_step)

class Model_Wessertein(Model):
    def __init__(self, init='xavier', user_max_id=None, src_item_max_id=None, trg_item_max_id=None,
                 embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, batch_norm=False, layer_norm=False,
                 l1_w=None, l1_v=None, layer_l1=None, user_his_len=None, hist_type=None, anchor_num=5,
                 cotrain=None, l2=None, head_num=None, add_neg=None,
                 wessertein=None, ae=1, reg_alpha=None, drop_r=None, t1=None, t2=None, k_ratio=None, k_ratio2=None):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.layer_l2 = layer_l2
        self.l1_w = l1_w
        self.layer_l1 = layer_l1
        self.l1_v = l1_v
        self.hist_type = hist_type
        self.embed_size = embed_size
        self.user_his_len = user_his_len
        self.layer_sizes = layer_sizes
        self.layer_acts = layer_acts
        self.layer_keeps = layer_keeps
        self.cotrain = cotrain
        self.anchor_num = anchor_num
        self.l2 = l2
        self.head_num = head_num
        self.wessertein = wessertein
        self.ae = ae
        self.add_neg = add_neg
        self.reg_alpha = reg_alpha
        self.drop_r = drop_r
        self.k1 = int(anchor_num * k_ratio)
        self.k2 = int(anchor_num * k_ratio2)
        print('\n\n', self.k1, self.k2, '\n\n')

        with tf.name_scope('input'):
            self.user_id_src = tf.placeholder(tf.int32, [None], name='user_id_src')
            self.item_id_src = tf.placeholder(tf.int32, [None], name='item_id_src')
            self.user_id_trg = tf.placeholder(tf.int32, [None], name='user_id_trg')
            self.item_id_trg = tf.placeholder(tf.int32, [None], name='item_id_trg')
            self.history1 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items1')
            self.history_len1 = tf.placeholder(tf.int32, [None], name='history_items_len1')
            self.history2 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items2')
            self.history_len2 = tf.placeholder(tf.int32, [None], name='history_items_len2')
            self.neg_item_id_src = tf.placeholder(tf.int32, [None, add_neg], name='neg_item_id_src')
            self.neg_item_id_trg = tf.placeholder(tf.int32, [None, add_neg], name='neg_item_id_trg')
            self.labels_src = tf.placeholder(tf.float32, [None], name='label_src')
            self.labels_trg = tf.placeholder(tf.float32, [None], name='label_trg')
            self.training = tf.placeholder(dtype=tf.bool, name='training')

        v_user = get_variable(init, name='v_user', shape=[user_max_id, embed_size])
        v_user_2 = get_variable(init, name='v_user_2', shape=[user_max_id, embed_size])
        v_item = get_variable(init, name='v_item', shape=[src_item_max_id + trg_item_max_id, embed_size])
        b = get_variable('zero', name='b', shape=[1])
        self.v_user = v_user
        self.v_item = v_item


        xv1_src = tf.gather(v_user, self.user_id_src)
        xv2_src = tf.gather(v_item, self.item_id_src)
        neg_xv2_src = tf.gather(v_item, self.neg_item_id_src)   # [B, add_neg, embed_size]
        xv1_trg = tf.gather(v_user, self.user_id_trg)
        xv1_trg_spec = tf.gather(v_user_2, self.user_id_trg)
        xv2_trg = tf.gather(v_item, self.item_id_trg)
        neg_xv2_trg = tf.gather(v_item, self.neg_item_id_trg)
        self.xv1_src_origin = xv1_src
        self.xv2_src_origin = xv2_src
        self.xv2_src_neg_origin = neg_xv2_src
        self.xv1_trg_origin = xv1_trg
        self.xv2_trg_origin = xv2_trg
        self.xv2_trg_neg_origin = neg_xv2_trg

        self.new_embed_size = int(embed_size / 2)
        anchors1 = get_variable(init, name='anchors1_w', shape=[2, anchor_num, self.new_embed_size])
        anchors2 = get_variable(init, name='anchors2_w', shape=[2, anchor_num, self.new_embed_size])

        self.set_qkv(init=init)

        tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                layer_keeps, xv1_src,
                                training=self.training, name='alpha1',
                                reuse=tf.AUTO_REUSE)

        tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', 'sigmoid'],
                                layer_keeps, xv1_trg,
                                training=self.training, name='alpha2',
                                reuse=tf.AUTO_REUSE)

        tau1 = tf.reshape(tau1, [1, -1, 1, 1])
        tau2 = tf.reshape(tau2, [1, -1, 1, 1])

        xv1_src = tf.transpose(tf.reshape(xv1_src, [-1, 2, self.new_embed_size]), (1, 0, 2))  # [2, B, embed_size]
        xv2_src = tf.transpose(tf.reshape(xv2_src, [-1, 2, self.new_embed_size]), (1, 0, 2))
        xv1_trg = tf.transpose(tf.reshape(xv1_trg, [-1, 2, self.new_embed_size]), (1, 0, 2))
        xv2_trg = tf.transpose(tf.reshape(xv2_trg, [-1, 2, self.new_embed_size]), (1, 0, 2))
        neg_xv2_src = tf.transpose(tf.reshape(neg_xv2_src, [-1, add_neg, 2, self.new_embed_size]), (1, 2, 0, 3))
        neg_xv2_trg = tf.transpose(tf.reshape(neg_xv2_trg, [-1, add_neg, 2, self.new_embed_size]), (1, 2, 0, 3))
        # [K, 2, B, embed_size]

        xv1_trg_spec = tf.transpose(tf.reshape(xv1_trg_spec, [-1, 2, self.new_embed_size]), (1, 0, 2))


        if wessertein in [0]:
            v_user_trg = get_variable(init, name='v_user_trg', shape=[user_max_id, embed_size])
            xv1_trg = tf.gather(v_user_trg, self.user_id_trg)
            self.outputs1 = tf.reduce_sum(self.xv1_src_origin * self.xv2_src_origin, axis=-1)  # [B, ]
            self.outputs2 = tf.reduce_sum(xv1_trg * self.xv2_trg_origin, axis=-1)  # [B, ]
            self.neg_outputs1 = tf.reduce_sum(tf.expand_dims(self.xv1_src_origin, axis=1) * self.xv2_src_neg_origin, axis=-1)
            self.neg_outputs2 = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * self.xv2_trg_neg_origin, axis=-1)
        elif wessertein in [2]:
            self.tau1 = tau1
            self.tau2 = tau2


            '''
                user embedding
            '''
            domain_emb_src1 = get_variable(init, name='domain_emb_src1', shape=[2, self.new_embed_size])
            domain_emb_trg1 = get_variable(init, name='domain_emb_trg1', shape=[2, self.new_embed_size])

            xv1_src_mean, xv1_src_cov, xv1_trg_mean, xv1_trg_cov = self.get_emb(anchors1, domain_emb_src1,
                                                                    domain_emb_trg1, tf.expand_dims(xv1_src, axis=0)
                                                                , tf.expand_dims(xv1_trg, axis=0))
            # [1, B, head_num, embed_size]

            '''
                item embedding
            '''
            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_mean, xv2_src_cov, xv2_trg_mean, xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                    domain_emb_trg2, tf.expand_dims(xv2_src, axis=0)
                                                                , tf.expand_dims(xv2_trg, axis=0))
            # [1, B, head_num, embed_size]
            neg_xv2_src_mean, neg_xv2_src_cov, neg_xv2_trg_mean, neg_xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                    domain_emb_trg2, neg_xv2_src, neg_xv2_trg)
            # [K, B, head_num, embed_size]
            self.outputs1 = tf.squeeze(-self.wasserstein_distance(xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov))    # [B, 1])
            self.outputs2 = tf.squeeze(-self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov))
            self.neg_outputs1 = -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, neg_xv2_src_mean,
                                                                  neg_xv2_src_cov)    # [B, K]
            self.neg_outputs2 = -self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean,
                                                                  neg_xv2_trg_cov)
        elif wessertein in [3]:
            self.tau1 = 1
            self.tau2 = 1

            self.queries = get_variable(init, name='queries',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])
            self.keys = get_variable(init, name='keys',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])
            self.values = get_variable(init, name='values',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])


            '''
                user embedding
            '''
            domain_emb_src1 = get_variable(init, name='domain_emb_src1', shape=[2, self.new_embed_size])
            domain_emb_trg1 = get_variable(init, name='domain_emb_trg1', shape=[2, self.new_embed_size])

            xv1_src_mean, xv1_src_cov, xv1_trg_mean, xv1_trg_cov = self.get_emb(anchors1, domain_emb_src1,
                                                                    domain_emb_trg1, tf.expand_dims(xv1_src, axis=0)
                                                                , tf.expand_dims(xv1_trg, axis=0), self.quries,
                                                                                self.keys, self.values)
            # [1, B, head_num, embed_size]

            '''
                item embedding
            '''
            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_mean, xv2_src_cov, xv2_trg_mean, xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                    domain_emb_trg2, tf.expand_dims(xv2_src, axis=0)
                                                                , tf.expand_dims(xv2_trg, axis=0), self.quries,
                                                                                self.keys, self.values)
            # [1, B, head_num, embed_size]
            neg_xv2_src_mean, neg_xv2_src_cov, neg_xv2_trg_mean, neg_xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                    domain_emb_trg2, neg_xv2_src, neg_xv2_trg, self.quries,
                                                                                self.keys, self.values)
            # [K, B, head_num, embed_size]
            self.outputs1 = tf.squeeze(-tf.nn.sigmoid(self.wasserstein_distance(xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov)))    # [B, 1])
            self.outputs2 = tf.squeeze(-tf.nn.sigmoid(self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov)))
            self.neg_outputs1 = -tf.nn.sigmoid(self.wasserstein_distance(xv1_src_mean, xv1_src_cov, neg_xv2_src_mean,
                                                                  neg_xv2_src_cov))    # [B, K]
            self.neg_outputs2 = -tf.nn.sigmoid(self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean,
                                                                  neg_xv2_trg_cov))
        elif wessertein in [5]:
            self.tau1 = 1
            self.tau2 = 1

            self.quries1 = get_variable(init, name='quries1',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])
            self.keys1 = get_variable(init, name='keys1',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])
            self.values1 = get_variable(init, name='values1',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])


            '''
                user embedding
            '''
            domain_emb_src1 = get_variable(init, name='domain_emb_src1', shape=[2, self.new_embed_size])
            domain_emb_trg1 = get_variable(init, name='domain_emb_trg1', shape=[2, self.new_embed_size])

            xv1_src_mean, xv1_src_cov, xv1_trg_mean, xv1_trg_cov = self.get_emb(anchors1, domain_emb_src1,
                                                                    domain_emb_trg1, tf.expand_dims(xv1_src, axis=0)
                                                                , tf.expand_dims(xv1_trg, axis=0))
            # [1, B, head_num, embed_size]

            '''
                item embedding
            '''

            self.quries2 = get_variable(init, name='quries2',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])
            self.keys2 = get_variable(init, name='keys2',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])
            self.values2 = get_variable(init, name='values2',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])

            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_mean, xv2_src_cov, xv2_trg_mean, xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                    domain_emb_trg2, tf.expand_dims(xv2_src, axis=0)
                                                                , tf.expand_dims(xv2_trg, axis=0))
            # [1, B, head_num, embed_size]
            neg_xv2_src_mean, neg_xv2_src_cov, neg_xv2_trg_mean, neg_xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                domain_emb_trg2, neg_xv2_src, neg_xv2_trg)
            # [K, B, head_num, embed_size]

            self.outputs1 = tf.squeeze(
                -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov))  # [B, 1])
            self.outputs2 = tf.squeeze(-self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov))
            self.neg_outputs1 = -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, neg_xv2_src_mean,
                                                           neg_xv2_src_cov)  # [B, K]
            self.neg_outputs2 = -self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean,
                                                           neg_xv2_trg_cov)
        elif wessertein in [6]:
            self.tau1 = tau1
            self.tau2 = tau2

            '''
                user embedding
            '''
            domain_emb_src1 = get_variable(init, name='domain_emb_src1', shape=[2, self.new_embed_size])
            domain_emb_trg1 = get_variable(init, name='domain_emb_trg1', shape=[2, self.new_embed_size])

            xv1_src_mean, xv1_src_cov, xv1_trg_mean, xv1_trg_cov = self.get_emb(anchors1, domain_emb_src1,
                                                                    domain_emb_trg1, tf.expand_dims(xv1_src, axis=0)
                                                                , tf.expand_dims(xv1_trg, axis=0))
            # [1, B, head_num, embed_size]

            '''
                item embedding
            '''

            self.quries2 = get_variable(init, name='quries2',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])
            self.keys2 = get_variable(init, name='keys2',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])
            self.values2 = get_variable(init, name='values2',
                                          shape=[2, self.new_embed_size, head_num * self.new_embed_size])

            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_mean, xv2_src_cov, xv2_trg_mean, xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                    domain_emb_trg2, tf.expand_dims(xv2_src, axis=0)
                                                                , tf.expand_dims(xv2_trg, axis=0))
            # [1, B, head_num, embed_size]
            neg_xv2_src_mean, neg_xv2_src_cov, neg_xv2_trg_mean, neg_xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                domain_emb_trg2, neg_xv2_src, neg_xv2_trg)
            # [K, B, head_num, embed_size]

            self.outputs1 = tf.squeeze(
                -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov))  # [B, 1])
            self.outputs2 = tf.squeeze(-self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov))
            self.neg_outputs1 = -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, neg_xv2_src_mean,
                                                           neg_xv2_src_cov)  # [B, K]
            self.neg_outputs2 = -self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean,
                                                           neg_xv2_trg_cov)
        elif wessertein in [16]:
            self.tau1 = 1
            self.tau2 = 1

            '''
                user embedding
            '''
            domain_emb_src1 = get_variable(init, name='domain_emb_src1', shape=[2, self.new_embed_size])
            domain_emb_trg1 = get_variable(init, name='domain_emb_trg1', shape=[2, self.new_embed_size])

            xv1_src_mean, xv1_src_cov, xv1_trg_mean, xv1_trg_cov = self.get_emb(anchors1, domain_emb_src1,
                                                                    domain_emb_trg1, tf.expand_dims(xv1_src, axis=0)
                                                                , tf.expand_dims(xv1_trg_spec, axis=0), 1)
            # [1, B, head_num, embed_size]

            '''
                item embedding
            '''
            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_mean, xv2_src_cov, xv2_trg_mean, xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                    domain_emb_trg2, tf.expand_dims(xv2_src, axis=0)
                                                                , tf.expand_dims(xv2_trg, axis=0), 2)
            # [1, B, head_num, embed_size]
            neg_xv2_src_mean, neg_xv2_src_cov, neg_xv2_trg_mean, neg_xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                    domain_emb_trg2, neg_xv2_src, neg_xv2_trg, 2)
            # [K, B, head_num, embed_size]

            self.outputs1 = tf.squeeze(
                -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov))  # [B, 1])
            self.outputs2 = tf.squeeze(-self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov))
            self.neg_outputs1 = -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, neg_xv2_src_mean,
                                                           neg_xv2_src_cov)  # [B, K]
            self.neg_outputs2 = -self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean,
                                                           neg_xv2_trg_cov)
        elif wessertein in [26]:
            self.tau1 = t1
            self.tau2 = t2

            '''
                user embedding
            '''
            domain_emb_src1 = get_variable(init, name='domain_emb_src1', shape=[2, self.new_embed_size])
            domain_emb_trg1 = get_variable(init, name='domain_emb_trg1', shape=[2, self.new_embed_size])

            xv1_src_mean, xv1_src_cov, xv1_trg_mean, xv1_trg_cov = self.get_emb(anchors1, domain_emb_src1,
                                                                    domain_emb_trg1, tf.expand_dims(xv1_src, axis=0)
                                                                , tf.expand_dims(xv1_trg_spec, axis=0), 1)
            # [1, B, head_num, embed_size]

            '''
                item embedding
            '''
            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_mean, xv2_src_cov, xv2_trg_mean, xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                    domain_emb_trg2, tf.expand_dims(xv2_src, axis=0)
                                                                , tf.expand_dims(xv2_trg, axis=0), 2)
            # [1, B, head_num, embed_size]
            neg_xv2_src_mean, neg_xv2_src_cov, neg_xv2_trg_mean, neg_xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                    domain_emb_trg2, neg_xv2_src, neg_xv2_trg, 2)
            # [K, B, head_num, embed_size]

            self.outputs1 = tf.squeeze(
                -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov))  # [B, 1])
            self.outputs2 = tf.squeeze(-self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov))
            self.neg_outputs1 = -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, neg_xv2_src_mean,
                                                           neg_xv2_src_cov)  # [B, K]
            self.neg_outputs2 = -self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean,
                                                           neg_xv2_trg_cov)
        else:
            self.tau1 = 1
            self.tau2 = 1

            '''
                user embedding
            '''
            domain_emb_src1 = get_variable(init, name='domain_emb_src1', shape=[2, self.new_embed_size])
            domain_emb_trg1 = get_variable(init, name='domain_emb_trg1', shape=[2, self.new_embed_size])

            xv1_src_mean, xv1_src_cov, xv1_trg_mean, xv1_trg_cov = self.get_emb(anchors1, domain_emb_src1,
                                                                    domain_emb_trg1, tf.expand_dims(xv1_src, axis=0)
                                                                , tf.expand_dims(xv1_trg, axis=0), 1)
            # [1, B, head_num, embed_size]

            '''
                item embedding
            '''
            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_mean, xv2_src_cov, xv2_trg_mean, xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                    domain_emb_trg2, tf.expand_dims(xv2_src, axis=0)
                                                                , tf.expand_dims(xv2_trg, axis=0), 2)
            # [1, B, head_num, embed_size]
            neg_xv2_src_mean, neg_xv2_src_cov, neg_xv2_trg_mean, neg_xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                    domain_emb_trg2, neg_xv2_src, neg_xv2_trg, 2)
            # [K, B, head_num, embed_size]

            self.outputs1 = tf.squeeze(
                -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov))  # [B, 1])
            self.outputs2 = tf.squeeze(-self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov))
            self.neg_outputs1 = -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, neg_xv2_src_mean,
                                                           neg_xv2_src_cov)  # [B, K]
            self.neg_outputs2 = -self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean,
                                                           neg_xv2_trg_cov)

        if reg_alpha > 0:
            print('Use l2 regularization...\n')
            # 使用L2正则化
            # 获取所有可训练参数
            trainable_vars = tf.trainable_variables()

            # 对所有可训练参数加入L2正则化
            regularizer = tf.contrib.layers.l2_regularizer(scale=reg_alpha)
            self.reg_penalty = tf.contrib.layers.apply_regularization(regularizer, trainable_vars)
        else:
            self.reg_penalty = 0


    def get_emb(self, anchors, domain_emb_src, domain_emb_trg, xv_src, xv_trg, pat=None):
        '''
            anchors: [2, anchor_num, embed_size]
            domain_emb_src: [2, embed_size]
            domain_emb_trg: [2, embed_size]
            xv_src: [K, 2, B, embed_size]
            xv_trg: [K, 2, B, embed_size]
        '''
        if self.wessertein in [7]:
            if pat == 1:
                queries_src = self.queries_src1
                queries_trg = self.queries_trg1
                keys_src = self.keys_src1
                keys_trg = self.keys_trg1
                values_src = self.values_src1
                values_trg = self.values_trg1
            else:
                queries_src = self.queries_src2
                queries_trg = self.queries_trg2
                keys_src = self.keys_src2
                keys_trg = self.keys_trg2
                values_src = self.values_src2
                values_trg = self.values_trg2
        elif self.wessertein in [1, 2, 3, 4, 9, 10]:
            queries_src = self.queries_src1
            queries_trg = self.queries_src1
            keys_src = self.keys_src1
            keys_trg = self.keys_src1
            values_src = self.values_src1
            values_trg = self.values_src1
        else:
            queries_src = self.queries_src1
            queries_trg = self.queries_trg1
            keys_src = self.keys_src1
            keys_trg = self.keys_trg1
            values_src = self.values_src1
            values_trg = self.values_trg1
        if self.wessertein in [17]:
            xv_src_queries_mean, xv_src_keys_mean, xv_src_values_mean = \
                self.get_qkv(queries_src[0], keys_src[0], values_src[0], xv_src[:, 0])
            xv_src_queries_cov, xv_src_keys_cov, xv_src_values_cov = \
                self.get_qkv(queries_src[1], keys_src[1], values_src[1], xv_src[:, 1], True)
            # [K, B, head_num, embed_size]

            xv_trg_queries_mean, xv_trg_keys_mean, xv_trg_values_mean = \
                self.get_qkv(queries_trg[0], keys_trg[0], values_trg[0], xv_trg[:, 0])
            xv_trg_queries_cov, xv_trg_keys_cov, xv_trg_values_cov = \
                self.get_qkv(queries_trg[1], keys_trg[1], values_trg[1], xv_trg[:, 1], True)
            # [K, B, head_num, embed_size]

        elif self.wessertein in [18, 19]:
            xv_src_queries_mean, xv_src_keys_mean, xv_src_values_mean = \
                self.get_qkv(queries_src[0], keys_src[0], values_src[0], xv_src[:, 0])
            xv_src_queries_cov, xv_src_keys_cov, xv_src_values_cov = \
                self.get_qkv(queries_src[1], keys_src[1], values_src[1], xv_src[:, 1], True)
            # [K, B, head_num, embed_size]

            xv_trg_queries_mean, xv_trg_keys_mean, xv_trg_values_mean = \
                self.get_qkv(queries_trg[0], keys_trg[0], values_trg[0], xv_trg[:, 0])
            xv_trg_queries_cov, xv_trg_keys_cov, xv_trg_values_cov = \
                self.get_qkv(queries_trg[1], keys_trg[1], values_trg[1], xv_trg[:, 1], True)
            # [K, B, head_num, embed_size]

            in_domain_queries_mean, in_domain_keys_mean, in_domain_values_mean = \
                self.get_qkv(queries_src[0], keys_src[0], values_src[0], tf.reshape(domain_emb_src,
                                                                    [1, 2, 1, -1])[:, 0])
            in_domain_queries_cov, in_domain_keys_cov, in_domain_values_cov = \
                self.get_qkv(queries_src[1], keys_src[1], values_src[1], tf.reshape(domain_emb_src,
                                                                    [1, 2, 1, -1])[:, 1], True)
            # [1, 1, head_num, embed_size]

            xv_src_values =  tf.concat([tf.expand_dims(xv_src_values_mean, axis=1), tf.expand_dims(xv_src_values_cov, axis=1)], axis=1)
            in_domain_values =tf.concat([tf.expand_dims(in_domain_values_mean, axis=1), tf.expand_dims(in_domain_values_cov, axis=1)], axis=1)
            xv_src_in_domain_values = self.combine_dist(xv_src_values, in_domain_values)  # [K, 2, B, head_num, self.new_embed_size]

            xv_trg_values =  tf.concat([tf.expand_dims(xv_trg_values_mean, axis=1), tf.expand_dims(xv_trg_values_cov, axis=1)], axis=1)
            in_domain_values =tf.concat([tf.expand_dims(in_domain_values_mean, axis=1), tf.expand_dims(in_domain_values_cov, axis=1)], axis=1)
            xv_trg_in_domain_values = self.combine_dist(xv_trg_values, in_domain_values)  # [K, 2, B, head_num, self.new_embed_size]

            xv_src_in_domain_values_mean, xv_src_in_domain_values_cov = xv_src_in_domain_values[:,
                                                                          0], xv_src_in_domain_values[:, 1]
            xv_trg_in_domain_values_mean, xv_trg_in_domain_values_cov = xv_trg_in_domain_values[:,
                                                                          0], xv_trg_in_domain_values[:, 1]
            # [K, B, head_num, embed_size]

        elif self.wessertein in [14, 15, 16, 20, 21, 22, 23, 24, 25, 26]:
            xv_src_queries_mean, xv_src_keys_mean, xv_src_values_mean = \
                self.get_qkv(queries_src[0], keys_src[0], values_src[0], xv_src[:, 0])
            xv_src_queries_cov, xv_src_keys_cov, xv_src_values_cov = \
                self.get_qkv(queries_src[1], keys_src[1], values_src[1], xv_src[:, 1], True)
            # [K, B, head_num, embed_size]

            xv_trg_queries_mean, xv_trg_keys_mean, xv_trg_values_mean = \
                self.get_qkv(queries_trg[0], keys_trg[0], values_trg[0], xv_trg[:, 0])
            xv_trg_queries_cov, xv_trg_keys_cov, xv_trg_values_cov = \
                self.get_qkv(queries_trg[1], keys_trg[1], values_trg[1], xv_trg[:, 1], True)
            # [K, B, head_num, embed_size]

            in_domain_queries_mean, in_domain_keys_mean, in_domain_values_mean = \
                self.get_qkv(queries_src[0], keys_src[0], values_src[0], tf.reshape(domain_emb_src,
                                                                    [1, 2, 1, -1])[:, 0])
            in_domain_queries_cov, in_domain_keys_cov, in_domain_values_cov = \
                self.get_qkv(queries_src[1], keys_src[1], values_src[1], tf.reshape(domain_emb_src,
                                                                    [1, 2, 1, -1])[:, 1], True)
            # [1, 1, head_num, embed_size]
            # print('\nin_domain_queries_mean\n', in_domain_queries_mean.shape, '\n\n')

            sub_domain_queries_mean, sub_domain_keys_mean, sub_domain_values_mean = \
                self.get_qkv_anchor(queries_src[0], keys_src[0], values_src[0], tf.reshape(anchors, [1, 2, 1, self.anchor_num,
                                                                       -1])[:, 0])
            sub_domain_queries_cov, sub_domain_keys_cov, sub_domain_values_cov = \
                self.get_qkv_anchor(queries_src[1], keys_src[1], values_src[1], tf.reshape(anchors, [1, 2, 1, self.anchor_num,
                                                                       -1])[:, 1], True)
            # [K, B, head_num, anchor_num, embed_size]

            xv_src_queries = tf.concat([tf.expand_dims(xv_src_queries_mean, axis=1), tf.expand_dims(xv_src_queries_cov, axis=1)], axis=1)
            in_domain_queries =tf.concat([tf.expand_dims(in_domain_queries_mean, axis=1), tf.expand_dims(in_domain_queries_cov, axis=1)], axis=1)
            xv_src_in_domain_queries = self.combine_dist(xv_src_queries, in_domain_queries)  # [K, 2, B, head_num, self.new_embed_size]

            xv_trg_queries =  tf.concat([tf.expand_dims(xv_trg_queries_mean, axis=1), tf.expand_dims(xv_trg_queries_cov, axis=1)], axis=1)
            in_domain_queries =tf.concat([tf.expand_dims(in_domain_queries_mean, axis=1), tf.expand_dims(in_domain_queries_cov, axis=1)], axis=1)
            xv_trg_in_domain_queries = self.combine_dist(xv_trg_queries, in_domain_queries)  # [K, 2, B, head_num, self.new_embed_size]

            xv_src_in_domain_queries_mean, xv_src_in_domain_queries_cov = xv_src_in_domain_queries[:,
                                                                          0], xv_src_in_domain_queries[:, 1]
            xv_trg_in_domain_queries_mean, xv_trg_in_domain_queries_cov = xv_trg_in_domain_queries[:,
                                                                          0], xv_trg_in_domain_queries[:, 1]
            # [K, B, head_num, embed_size]

            xv_src_values =  tf.concat([tf.expand_dims(xv_src_values_mean, axis=1), tf.expand_dims(xv_src_values_cov, axis=1)], axis=1)
            in_domain_values =tf.concat([tf.expand_dims(in_domain_values_mean, axis=1), tf.expand_dims(in_domain_values_cov, axis=1)], axis=1)
            xv_src_in_domain_values = self.combine_dist(xv_src_values, in_domain_values)  # [K, 2, B, head_num, self.new_embed_size]

            xv_trg_values =  tf.concat([tf.expand_dims(xv_trg_values_mean, axis=1), tf.expand_dims(xv_trg_values_cov, axis=1)], axis=1)
            in_domain_values =tf.concat([tf.expand_dims(in_domain_values_mean, axis=1), tf.expand_dims(in_domain_values_cov, axis=1)], axis=1)
            xv_trg_in_domain_values = self.combine_dist(xv_trg_values, in_domain_values)  # [K, 2, B, head_num, self.new_embed_size]

            xv_src_in_domain_values_mean, xv_src_in_domain_values_cov = xv_src_in_domain_values[:,
                                                                          0], xv_src_in_domain_values[:, 1]
            xv_trg_in_domain_values_mean, xv_trg_in_domain_values_cov = xv_trg_in_domain_values[:,
                                                                          0], xv_trg_in_domain_values[:, 1]
            # [K, B, head_num, embed_size]


            xv_src_keys = tf.concat([tf.expand_dims(xv_src_keys_mean, axis=1), tf.expand_dims(xv_src_keys_cov, axis=1)], axis=1)
            xv_src_keys = tf.expand_dims(xv_src_keys, axis=-2)  # [K, 2, B, head_num, 1, embed_size]
            sub_domain_keys =tf.concat([tf.expand_dims(sub_domain_keys_mean, axis=1), tf.expand_dims(sub_domain_keys_cov, axis=1)], axis=1)
            xv_src_sub_domain_keys = self.combine_dist(xv_src_keys, sub_domain_keys)  # [K, 2, B, head_num, anchor_num, embed_size]

            xv_trg_keys = tf.concat([tf.expand_dims(xv_trg_keys_mean, axis=1), tf.expand_dims(xv_trg_keys_cov, axis=1)], axis=1)
            xv_trg_keys = tf.expand_dims(xv_trg_keys, axis=-2)
            sub_domain_keys = tf.concat([tf.expand_dims(sub_domain_keys_mean, axis=1), tf.expand_dims(sub_domain_keys_cov, axis=1)], axis=1)
            xv_trg_sub_domain_keys = self.combine_dist(xv_trg_keys, sub_domain_keys)  # [K, 2, B, head_num, anchor_num, embed_size]
            xv_src_sub_domain_keys_mean, xv_src_sub_domain_keys_cov = xv_src_sub_domain_keys[:,
                                                                          0], xv_src_sub_domain_keys[:, 1]
            xv_trg_sub_domain_keys_mean, xv_trg_sub_domain_keys_cov = xv_trg_sub_domain_keys[:,
                                                                          0], xv_trg_sub_domain_keys[:, 1]
            # [K, B, head_num, anchor_num, embed_size]

            xv_src_values = tf.concat([tf.expand_dims(xv_src_values_mean, axis=1), tf.expand_dims(xv_src_values_cov, axis=1)],
                                    axis=1)
            xv_src_values = tf.expand_dims(xv_src_values, axis=-2)  # [K, 2, B, 1, embed_size]
            sub_domain_values = tf.concat(
                [tf.expand_dims(sub_domain_values_mean, axis=1), tf.expand_dims(sub_domain_values_cov, axis=1)], axis=1)
            xv_src_sub_domain_values = self.combine_dist(xv_src_values, sub_domain_values)  # [K, 2, B, head_num, embed_size]

            xv_trg_values = tf.concat([tf.expand_dims(xv_trg_values_mean, axis=1), tf.expand_dims(xv_trg_values_cov, axis=1)],
                                    axis=1)
            xv_trg_values = tf.expand_dims(xv_trg_values, axis=-2)
            sub_domain_values = tf.concat(
                [tf.expand_dims(sub_domain_values_mean, axis=1), tf.expand_dims(sub_domain_values_cov, axis=1)], axis=1)
            xv_trg_sub_domain_values = self.combine_dist(xv_trg_values, sub_domain_values)  # [K, 2, B, head_num, embed_size]
            xv_src_sub_domain_values_mean, xv_src_sub_domain_values_cov = xv_src_sub_domain_values[:,
                                                                      0], xv_src_sub_domain_values[:, 1]
            xv_trg_sub_domain_values_mean, xv_trg_sub_domain_values_cov = xv_trg_sub_domain_values[:,
                                                                      0], xv_trg_sub_domain_values[:, 1]
            # [K, B, head_num, anchor_num, embed_size]


            attention_src = -self.wasserstein_distance_matmul(xv_src_in_domain_queries_mean,
                                                              xv_src_in_domain_queries_cov,
                                                              xv_src_sub_domain_keys_mean,
                                                              xv_src_sub_domain_keys_cov)  # [K, B, head_num, anchor_num]

            attention_trg = -self.wasserstein_distance_matmul(xv_trg_in_domain_queries_mean,
                                                              xv_trg_in_domain_queries_cov,
                                                              xv_trg_sub_domain_keys_mean,
                                                              xv_trg_sub_domain_keys_cov)  # [K, B, head_num, anchor_num]

            attention_src = attention_src / np.sqrt(self.new_embed_size)  # [K, B, head_num, anchor_num]
            attention_trg = attention_trg / np.sqrt(self.new_embed_size)
            head_scores_src = tf.nn.softmax(attention_src / self.tau1, dim=-1)  # [K, B, head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(attention_trg / self.tau2, dim=-1)

            if self.wessertein in [22]:
                random_masks_src = tf.random_uniform(shape=tf.shape(head_scores_src))
                random_masks_trg = tf.random_uniform(shape=tf.shape(head_scores_trg))
                max_vals = tf.reduce_max(random_masks_src, axis=-1, keepdims=True)
                rand_tensor_binary_src = tf.where(tf.less(tf.abs(random_masks_src - max_vals), 1e-5), tf.zeros_like(random_masks_src),
                                              tf.ones_like(random_masks_src))
                max_vals = tf.reduce_max(random_masks_trg, axis=-1, keepdims=True)
                rand_tensor_binary_trg = tf.where(tf.less(tf.abs(random_masks_trg - max_vals), 1e-5), tf.zeros_like(random_masks_trg),
                                                  tf.ones_like(random_masks_trg))
                rand_tensor_binary_src = tf.cond(self.training, lambda: rand_tensor_binary_src,
                                                 lambda: tf.ones_like(random_masks_src))        # training!!!!!!!!!
                rand_tensor_binary_trg = tf.cond(self.training, lambda: rand_tensor_binary_trg,
                                                 lambda: tf.ones_like(random_masks_trg))
                head_scores_src = head_scores_src * rand_tensor_binary_src
                head_scores_trg = head_scores_trg * rand_tensor_binary_trg
                head_scores_src = head_scores_src * self.anchor_num / tf.reduce_sum(rand_tensor_binary_src, axis=-1,
                                                                                    keep_dims=True)
                head_scores_trg = head_scores_trg * self.anchor_num / tf.reduce_sum(rand_tensor_binary_trg, axis=-1,
                                                                                    keep_dims=True)
            elif self.wessertein in [23]:
                random_masks_src = tf.random_uniform(shape=tf.shape(head_scores_src))
                random_masks_trg = tf.random_uniform(shape=tf.shape(head_scores_trg))
                rand_tensor_binary_src = tf.where(tf.less(random_masks_src, self.drop_r), tf.zeros_like(random_masks_src),
                                              tf.ones_like(random_masks_src))
                rand_tensor_binary_trg = tf.where(tf.less(random_masks_trg, self.drop_r), tf.zeros_like(random_masks_trg),
                                                  tf.ones_like(random_masks_trg))
                rand_tensor_binary_src = tf.cond(self.training, lambda: rand_tensor_binary_src,
                                                 lambda: tf.ones_like(random_masks_src))        # training!!!!!!!!!
                rand_tensor_binary_trg = tf.cond(self.training, lambda: rand_tensor_binary_trg,
                                                 lambda: tf.ones_like(random_masks_trg))
                head_scores_src = head_scores_src * rand_tensor_binary_src
                head_scores_trg = head_scores_trg * rand_tensor_binary_trg
                head_scores_src = head_scores_src * self.anchor_num / tf.reduce_sum(rand_tensor_binary_src, axis=-1,
                                                                                    keep_dims=True)
                head_scores_trg = head_scores_trg * self.anchor_num / tf.reduce_sum(rand_tensor_binary_trg, axis=-1,
                                                                                    keep_dims=True)
            elif self.wessertein in [24]:
                random_masks_src = tf.random_uniform(shape=tf.shape(head_scores_src))
                random_masks_trg = tf.random_uniform(shape=tf.shape(head_scores_trg))
                rand_tensor_binary_src = tf.where(tf.less(random_masks_src, self.drop_r), tf.zeros_like(random_masks_src),
                                              tf.ones_like(random_masks_src))
                rand_tensor_binary_trg = tf.where(tf.less(random_masks_trg, self.drop_r), tf.zeros_like(random_masks_trg),
                                                  tf.ones_like(random_masks_trg))
                rand_tensor_binary_src = tf.cond(self.training, lambda: rand_tensor_binary_src,
                                                 lambda: tf.ones_like(random_masks_src))        # training!!!!!!!!!
                rand_tensor_binary_trg = tf.cond(self.training, lambda: rand_tensor_binary_trg,
                                                 lambda: tf.ones_like(random_masks_trg))
                head_scores_src = head_scores_src * rand_tensor_binary_src
                head_scores_trg = head_scores_trg * rand_tensor_binary_trg
                ratio = tf.cond(self.training, lambda: 1.0, lambda: (1 - self.drop_r))
                head_scores_src = head_scores_src * ratio
                head_scores_trg = head_scores_trg * ratio

            elif self.wessertein in [25]:
                # top_k_vals, top_k_indices = tf.nn.top_k(head_scores_src, k=self.k1)
                # mask_binary_src = tf.one_hot(top_k_indices, depth=self.anchor_num, on_value=1, off_value=0,
                #                              axis=-1)
                # mask_binary_src = tf.reduce_sum(mask_binary_src, axis=-2)
                # top_k_vals, top_k_indices = tf.nn.top_k(head_scores_trg, k=self.k2)
                # mask_binary_trg = tf.one_hot(top_k_indices, depth=self.anchor_num, on_value=1, off_value=0,
                #                              axis=-1)
                # mask_binary_trg = tf.reduce_sum(mask_binary_trg, axis=-2)
                # head_scores_src = head_scores_src * tf.cast(mask_binary_src, tf.float32)
                # head_scores_trg = head_scores_trg * tf.cast(mask_binary_trg, tf.float32)
                # self.a = head_scores_src
                # self.b = head_scores_trg
                pass

            xv_src_agg_mean = tf.reduce_sum(tf.expand_dims(head_scores_src, axis=-1) * xv_src_sub_domain_values_mean,
                                            axis=-2)
            xv_trg_agg_mean = tf.reduce_sum(tf.expand_dims(head_scores_trg, axis=-1) * xv_trg_sub_domain_values_mean,
                                            axis=-2)  # [K, B, head_num, embed_size]

            xv_src_agg_cov = tf.reduce_sum(
                tf.expand_dims(tf.square(head_scores_src), axis=-1) * xv_src_sub_domain_values_cov, axis=-2)
            xv_trg_agg_cov = tf.reduce_sum(
                tf.expand_dims(tf.square(head_scores_trg), axis=-1) * xv_trg_sub_domain_values_cov, axis=-2)
            # [K, B, head_num, embed_size]

        elif self.wessertein in [27]:
            xv_src_queries_mean, xv_src_keys_mean, xv_src_values_mean = \
                self.get_qkv(queries_src[0], keys_src[0], values_src[0], xv_src[:, 0])
            xv_src_queries_cov, xv_src_keys_cov, xv_src_values_cov = \
                self.get_qkv(queries_src[1], keys_src[1], values_src[1], xv_src[:, 1], True)
            # [K, B, head_num, embed_size]

            xv_trg_queries_mean, xv_trg_keys_mean, xv_trg_values_mean = \
                self.get_qkv(queries_trg[0], keys_trg[0], values_trg[0], xv_trg[:, 0])
            xv_trg_queries_cov, xv_trg_keys_cov, xv_trg_values_cov = \
                self.get_qkv(queries_trg[1], keys_trg[1], values_trg[1], xv_trg[:, 1], True)
            # [K, B, head_num, embed_size]

            in_domain_queries_src_mean, in_domain_keys_src_mean, in_domain_values_src_mean = \
                self.get_qkv(queries_src[0], keys_src[0], values_src[0], tf.reshape(domain_emb_src,
                                                                    [1, 2, 1, -1])[:, 0])
            in_domain_queries_src_cov, in_domain_keys_src_cov, in_domain_values_src_cov = \
                self.get_qkv(queries_src[1], keys_src[1], values_src[1], tf.reshape(domain_emb_src,
                                                                    [1, 2, 1, -1])[:, 1], True)
            # [1, 1, head_num, embed_size]


            in_domain_queries_trg_mean, in_domain_keys_trg_mean, in_domain_values_trg_mean = \
                self.get_qkv(queries_trg[0], keys_trg[0], values_trg[0], tf.reshape(domain_emb_trg,
                                                                                    [1, 2, 1, -1])[:, 0])
            in_domain_queries_trg_cov, in_domain_keys_trg_cov, in_domain_values_trg_cov = \
                self.get_qkv(queries_trg[1], keys_trg[1], values_trg[1], tf.reshape(domain_emb_trg,
                                                                                    [1, 2, 1, -1])[:, 1], True)
            # [1, 1, head_num, embed_size]

            # print('\nin_domain_queries_mean\n', in_domain_queries_mean.shape, '\n\n')

            sub_domain_queries_mean, sub_domain_keys_mean, sub_domain_values_mean = \
                self.get_qkv_anchor(queries_src[0], keys_src[0], values_src[0], tf.reshape(anchors, [1, 2, 1, self.anchor_num,
                                                                       -1])[:, 0])
            sub_domain_queries_cov, sub_domain_keys_cov, sub_domain_values_cov = \
                self.get_qkv_anchor(queries_src[1], keys_src[1], values_src[1], tf.reshape(anchors, [1, 2, 1, self.anchor_num,
                                                                       -1])[:, 1], True)
            # [K, B, head_num, anchor_num, embed_size]

            xv_src_queries = tf.concat([tf.expand_dims(xv_src_queries_mean, axis=1), tf.expand_dims(xv_src_queries_cov, axis=1)], axis=1)
            in_domain_queries_src =tf.concat([tf.expand_dims(in_domain_queries_src_mean, axis=1), tf.expand_dims(in_domain_queries_src_cov, axis=1)], axis=1)
            xv_src_in_domain_queries = self.combine_dist(xv_src_queries, in_domain_queries_src)  # [K, 2, B, head_num, self.new_embed_size]

            xv_trg_queries =  tf.concat([tf.expand_dims(xv_trg_queries_mean, axis=1), tf.expand_dims(xv_trg_queries_cov, axis=1)], axis=1)
            in_domain_queries_trg =tf.concat([tf.expand_dims(in_domain_queries_trg_mean, axis=1), tf.expand_dims(in_domain_queries_trg_cov, axis=1)], axis=1)
            xv_trg_in_domain_queries = self.combine_dist(xv_trg_queries, in_domain_queries_trg)  # [K, 2, B, head_num, self.new_embed_size]

            xv_src_in_domain_queries_mean, xv_src_in_domain_queries_cov = xv_src_in_domain_queries[:,
                                                                          0], xv_src_in_domain_queries[:, 1]
            xv_trg_in_domain_queries_mean, xv_trg_in_domain_queries_cov = xv_trg_in_domain_queries[:,
                                                                          0], xv_trg_in_domain_queries[:, 1]
            # [K, B, head_num, embed_size]

            xv_src_values =  tf.concat([tf.expand_dims(xv_src_values_mean, axis=1), tf.expand_dims(xv_src_values_cov, axis=1)], axis=1)
            in_domain_values =tf.concat([tf.expand_dims(in_domain_values_src_mean, axis=1), tf.expand_dims(in_domain_values_src_cov, axis=1)], axis=1)
            xv_src_in_domain_values = self.combine_dist(xv_src_values, in_domain_values)  # [K, 2, B, head_num, self.new_embed_size]

            xv_trg_values =  tf.concat([tf.expand_dims(xv_trg_values_mean, axis=1), tf.expand_dims(xv_trg_values_cov, axis=1)], axis=1)
            in_domain_values =tf.concat([tf.expand_dims(in_domain_values_trg_mean, axis=1), tf.expand_dims(in_domain_values_trg_cov, axis=1)], axis=1)
            xv_trg_in_domain_values = self.combine_dist(xv_trg_values, in_domain_values)  # [K, 2, B, head_num, self.new_embed_size]

            xv_src_in_domain_values_mean, xv_src_in_domain_values_cov = xv_src_in_domain_values[:,
                                                                          0], xv_src_in_domain_values[:, 1]
            xv_trg_in_domain_values_mean, xv_trg_in_domain_values_cov = xv_trg_in_domain_values[:,
                                                                          0], xv_trg_in_domain_values[:, 1]
            # [K, B, head_num, embed_size]


            xv_src_keys = tf.concat([tf.expand_dims(xv_src_keys_mean, axis=1), tf.expand_dims(xv_src_keys_cov, axis=1)], axis=1)
            xv_src_keys = tf.expand_dims(xv_src_keys, axis=-2)  # [K, 2, B, head_num, 1, embed_size]
            sub_domain_keys =tf.concat([tf.expand_dims(sub_domain_keys_mean, axis=1), tf.expand_dims(sub_domain_keys_cov, axis=1)], axis=1)
            xv_src_sub_domain_keys = self.combine_dist(xv_src_keys, sub_domain_keys)  # [K, 2, B, head_num, anchor_num, embed_size]

            xv_trg_keys = tf.concat([tf.expand_dims(xv_trg_keys_mean, axis=1), tf.expand_dims(xv_trg_keys_cov, axis=1)], axis=1)
            xv_trg_keys = tf.expand_dims(xv_trg_keys, axis=-2)
            sub_domain_keys = tf.concat([tf.expand_dims(sub_domain_keys_mean, axis=1), tf.expand_dims(sub_domain_keys_cov, axis=1)], axis=1)
            xv_trg_sub_domain_keys = self.combine_dist(xv_trg_keys, sub_domain_keys)  # [K, 2, B, head_num, anchor_num, embed_size]
            xv_src_sub_domain_keys_mean, xv_src_sub_domain_keys_cov = xv_src_sub_domain_keys[:,
                                                                          0], xv_src_sub_domain_keys[:, 1]
            xv_trg_sub_domain_keys_mean, xv_trg_sub_domain_keys_cov = xv_trg_sub_domain_keys[:,
                                                                          0], xv_trg_sub_domain_keys[:, 1]
            # [K, B, head_num, anchor_num, embed_size]

            xv_src_values = tf.concat([tf.expand_dims(xv_src_values_mean, axis=1), tf.expand_dims(xv_src_values_cov, axis=1)],
                                    axis=1)
            xv_src_values = tf.expand_dims(xv_src_values, axis=-2)  # [K, 2, B, 1, embed_size]
            sub_domain_values = tf.concat(
                [tf.expand_dims(sub_domain_values_mean, axis=1), tf.expand_dims(sub_domain_values_cov, axis=1)], axis=1)
            xv_src_sub_domain_values = self.combine_dist(xv_src_values, sub_domain_values)  # [K, 2, B, head_num, embed_size]

            xv_trg_values = tf.concat([tf.expand_dims(xv_trg_values_mean, axis=1), tf.expand_dims(xv_trg_values_cov, axis=1)],
                                    axis=1)
            xv_trg_values = tf.expand_dims(xv_trg_values, axis=-2)
            sub_domain_values = tf.concat(
                [tf.expand_dims(sub_domain_values_mean, axis=1), tf.expand_dims(sub_domain_values_cov, axis=1)], axis=1)
            xv_trg_sub_domain_values = self.combine_dist(xv_trg_values, sub_domain_values)  # [K, 2, B, head_num, embed_size]
            xv_src_sub_domain_values_mean, xv_src_sub_domain_values_cov = xv_src_sub_domain_values[:,
                                                                      0], xv_src_sub_domain_values[:, 1]
            xv_trg_sub_domain_values_mean, xv_trg_sub_domain_values_cov = xv_trg_sub_domain_values[:,
                                                                      0], xv_trg_sub_domain_values[:, 1]
            # [K, B, head_num, anchor_num, embed_size]


            attention_src = -self.wasserstein_distance_matmul(xv_src_in_domain_queries_mean,
                                                              xv_src_in_domain_queries_cov,
                                                              xv_src_sub_domain_keys_mean,
                                                              xv_src_sub_domain_keys_cov)  # [K, B, head_num, anchor_num]

            attention_trg = -self.wasserstein_distance_matmul(xv_trg_in_domain_queries_mean,
                                                              xv_trg_in_domain_queries_cov,
                                                              xv_trg_sub_domain_keys_mean,
                                                              xv_trg_sub_domain_keys_cov)  # [K, B, head_num, anchor_num]

            attention_src = attention_src / np.sqrt(self.new_embed_size)  # [K, B, head_num, anchor_num]
            attention_trg = attention_trg / np.sqrt(self.new_embed_size)
            head_scores_src = tf.nn.softmax(attention_src / self.tau1, dim=-1)  # [K, B, head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(attention_trg / self.tau2, dim=-1)

            xv_src_agg_mean = tf.reduce_sum(tf.expand_dims(head_scores_src, axis=-1) * xv_src_sub_domain_values_mean,
                                            axis=-2)
            xv_trg_agg_mean = tf.reduce_sum(tf.expand_dims(head_scores_trg, axis=-1) * xv_trg_sub_domain_values_mean,
                                            axis=-2)  # [K, B, head_num, embed_size]

            xv_src_agg_cov = tf.reduce_sum(
                tf.expand_dims(tf.square(head_scores_src), axis=-1) * xv_src_sub_domain_values_cov, axis=-2)
            xv_trg_agg_cov = tf.reduce_sum(
                tf.expand_dims(tf.square(head_scores_trg), axis=-1) * xv_trg_sub_domain_values_cov, axis=-2)
            # [K, B, head_num, embed_size]


        else:
            xv_src_in_domain = self.combine_dist(xv_src, tf.reshape(domain_emb_src, [1, 2, 1, -1]))  # [K, 2, B, self.new_embed_size]
            xv_trg_in_domain = self.combine_dist(xv_trg, tf.reshape(domain_emb_trg, [1, 2, 1, -1]))

            xv_src_sub_domain = self.combine_dist(tf.expand_dims(xv_src, axis=-2),
                                                  tf.reshape(anchors, [1, 2, 1, self.anchor_num,
                                                                       -1]))  # [K, 2, B, anchor_num, self.new_embed_size]
            xv_trg_sub_domain = self.combine_dist(tf.expand_dims(xv_trg, axis=-2),
                                                  tf.reshape(anchors, [1, 2, 1, self.anchor_num,
                                                                       -1]))  # [K, 2, B, anchor_num, self.new_embed_size]

            xv_src_in_domain_queries_mean, xv_src_in_domain_keys_mean, xv_src_in_domain_values_mean = \
                self.get_qkv(queries_src[0], keys_src[0], values_src[0], xv_src_in_domain[:, 0])
            xv_src_in_domain_queries_cov, xv_src_in_domain_keys_cov, xv_src_in_domain_values_cov = \
                self.get_qkv(queries_src[1], keys_src[1], values_src[1], xv_src_in_domain[:, 1], True)

            xv_trg_in_domain_queries_mean, xv_trg_in_domain_keys_mean, xv_trg_in_domain_values_mean = \
                self.get_qkv(queries_trg[0], keys_trg[0], values_trg[0], xv_trg_in_domain[:, 0])
            xv_trg_in_domain_queries_cov, xv_trg_in_domain_keys_cov, xv_trg_in_domain_values_cov = \
                self.get_qkv(queries_trg[1], keys_trg[1], values_trg[1], xv_trg_in_domain[:, 1], True)
            # [K, B, head_num, embed_size]

            xv_src_sub_domain_queries_mean, xv_src_sub_domain_keys_mean, xv_src_sub_domain_values_mean = \
                self.get_qkv_anchor(queries_src[0], keys_src[0], values_src[0], xv_src_sub_domain[:, 0])
            xv_src_sub_domain_queries_cov, xv_src_sub_domain_keys_cov, xv_src_sub_domain_values_cov = \
                self.get_qkv_anchor(queries_src[1], keys_src[1], values_src[1], xv_src_sub_domain[:, 1], True)

            xv_trg_sub_domain_queries_mean, xv_trg_sub_domain_keys_mean, xv_trg_sub_domain_values_mean = \
                self.get_qkv_anchor(queries_trg[0], keys_trg[0], values_trg[0], xv_trg_sub_domain[:, 0])
            xv_trg_sub_domain_queries_cov, xv_trg_sub_domain_keys_cov, xv_trg_sub_domain_values_cov = \
                self.get_qkv_anchor(queries_trg[1], keys_trg[1], values_trg[1], xv_trg_sub_domain[:, 1], True)
            # [K, B, head_num, anchor_num, embed_size]

            attention_src = -self.wasserstein_distance_matmul(xv_src_in_domain_queries_mean,
                                                              xv_src_in_domain_queries_cov,
                                                              xv_src_sub_domain_keys_mean,
                                                              xv_src_sub_domain_keys_cov)  # [K, B, head_num, anchor_num]

            attention_trg = -self.wasserstein_distance_matmul(xv_trg_in_domain_queries_mean,
                                                              xv_trg_in_domain_queries_cov,
                                                              xv_trg_sub_domain_keys_mean,
                                                              xv_trg_sub_domain_keys_cov)  # [K, B, head_num, anchor_num]

            attention_src = attention_src / np.sqrt(self.new_embed_size)  # [K, B, head_num, anchor_num]
            attention_trg = attention_trg / np.sqrt(self.new_embed_size)
            head_scores_src = tf.nn.softmax(attention_src / self.tau1, dim=-1)  # [K, B, head_num, anchor_num]
            head_scores_trg = tf.nn.softmax(attention_trg / self.tau2, dim=-1)

            xv_src_agg_mean = tf.reduce_sum(tf.expand_dims(head_scores_src, axis=-1) * xv_src_sub_domain_values_mean,
                                            axis=-2)
            xv_trg_agg_mean = tf.reduce_sum(tf.expand_dims(head_scores_trg, axis=-1) * xv_trg_sub_domain_values_mean,
                                            axis=-2)  # [K, B, head_num, embed_size]

            xv_src_agg_cov = tf.reduce_sum(
                tf.expand_dims(tf.square(head_scores_src), axis=-1) * xv_src_sub_domain_values_cov, axis=-2)
            xv_trg_agg_cov = tf.reduce_sum(
                tf.expand_dims(tf.square(head_scores_trg), axis=-1) * xv_trg_sub_domain_values_cov, axis=-2)
            # [K, B, head_num, embed_size]

        if self.wessertein in [8]:
            xv_src_mean = tf.expand_dims(xv_src_in_domain[:, 0], axis=-2) + xv_src_agg_mean # [K, B, head_num, embed_size]
            xv_trg_mean = tf.expand_dims(xv_trg_in_domain[:, 0], axis=-2) + xv_trg_agg_mean

            xv_src_cov = tf.expand_dims(xv_src_in_domain[:, 1], axis=-2) + xv_src_agg_cov
            xv_trg_cov = tf.expand_dims(xv_trg_in_domain[:, 1], axis=-2) + xv_trg_agg_cov
        elif self.wessertein in [9]:
            xv_src_mean = tf.expand_dims(xv_src_in_domain[:, 0], axis=-2) * xv_src_agg_mean # [K, B, head_num, embed_size]
            xv_trg_mean = tf.expand_dims(xv_trg_in_domain[:, 0], axis=-2) * xv_trg_agg_mean

            xv_src_cov = tf.expand_dims(xv_src_in_domain[:, 1], axis=-2) * xv_src_agg_cov
            xv_trg_cov = tf.expand_dims(xv_trg_in_domain[:, 1], axis=-2) * xv_trg_agg_cov
        elif self.wessertein in [10, 11, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27]:
            xv_src_mean = xv_src_in_domain_values_mean + xv_src_agg_mean # [K, B, head_num, embed_size]
            xv_trg_mean = xv_trg_in_domain_values_mean + xv_trg_agg_mean

            xv_src_cov = tf.square(tf.sqrt(xv_src_in_domain_values_cov) + tf.sqrt(xv_src_agg_cov))
            xv_trg_cov = tf.square(tf.sqrt(xv_trg_in_domain_values_cov) + tf.sqrt(xv_trg_agg_cov))
        elif self.wessertein in [12]:
            xv_src_mean = (xv_src_in_domain_values_mean + xv_src_agg_mean) / 2  # [K, B, head_num, embed_size]
            xv_trg_mean = (xv_trg_in_domain_values_mean + xv_trg_agg_mean) / 2

            xv_src_cov = tf.square((tf.sqrt(xv_src_in_domain_values_cov) + tf.sqrt(xv_src_agg_cov)) / 2)
            xv_trg_cov = tf.square((tf.sqrt(xv_trg_in_domain_values_cov) + tf.sqrt(xv_trg_agg_cov)) / 2)
        elif self.wessertein in [15]:
            xv_src_mean = xv_src_in_domain_values_mean + xv_src_agg_mean # [K, B, head_num, embed_size]
            xv_trg_mean = xv_trg_in_domain_values_mean + xv_trg_agg_mean

            xv_src_cov = xv_src_in_domain_values_cov + xv_src_agg_cov
            xv_trg_cov = xv_trg_in_domain_values_cov + xv_trg_agg_cov
        elif self.wessertein in [17]:
            xv_src_mean = xv_src_values_mean
            xv_trg_mean = xv_trg_values_mean

            xv_src_cov = xv_src_values_cov
            xv_trg_cov = xv_trg_values_cov
        elif self.wessertein in [18, 19]:
            xv_src_mean = xv_src_in_domain_values_mean
            xv_trg_mean = xv_trg_in_domain_values_mean

            xv_src_cov = xv_src_in_domain_values_cov
            xv_trg_cov = xv_trg_in_domain_values_cov
        else:
            xv_src_mean = (tf.expand_dims(xv_src_in_domain[:, 0],
                                          axis=-2) + xv_src_agg_mean) / 2  # [K, B, head_num, embed_size]
            xv_trg_mean = (tf.expand_dims(xv_trg_in_domain[:, 0], axis=-2) + xv_trg_agg_mean) / 2

            xv_src_cov = (tf.expand_dims(xv_src_in_domain[:, 1], axis=-2) + xv_src_agg_cov) / 2
            xv_trg_cov = (tf.expand_dims(xv_trg_in_domain[:, 1], axis=-2) + xv_trg_agg_cov) / 2
        return xv_src_mean, xv_src_cov, xv_trg_mean, xv_trg_cov

    def combine_dist(self, xv, dist):
        xv_mean = xv[:, :1]
        xv_cov = xv[:, 1:]
        dist_mean = dist[:, :1]
        dist_cov = dist[:, 1:]

        if self.wessertein in [9]:
            combine_mean = xv_mean * dist_mean
            combine_cov = xv_cov * dist_cov
        elif self.wessertein in [14, 16, 20, 21, 22, 23, 24, 25, 26, 27]:
            combine_mean = xv_mean + dist_mean

            combine_cov = tf.square(tf.sqrt(tf.nn.relu(xv_cov)) + tf.sqrt(tf.nn.relu(dist_cov)))
        else:
            combine_mean = xv_mean + dist_mean
            combine_cov = xv_cov + dist_cov
        return tf.concat([combine_mean, combine_cov], axis=1)

    def set_qkv(self, init=None):
        self.queries_src1 = get_variable(init, name='queries_src1',
                                         shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.keys_src1 = get_variable(init, name='keys_src1',
                                      shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.values_src1 = get_variable(init, name='values_src1',
                                        shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])

        self.queries_trg1 = get_variable(init, name='queries_trg1',
                                         shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.keys_trg1 = get_variable(init, name='keys_trg1',
                                      shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.values_trg1 = get_variable(init, name='values_trg1',
                                        shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])

        self.queries_src2 = get_variable(init, name='queries_src2',
                                         shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.keys_src2 = get_variable(init, name='keys_src2',
                                      shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.values_src2 = get_variable(init, name='values_src2',
                                        shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])

        self.queries_trg2 = get_variable(init, name='queries_trg2',
                                         shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.keys_trg2 = get_variable(init, name='keys_trg2',
                                      shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.values_trg2 = get_variable(init, name='values_trg2',
                                        shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])

    def get_qkv(self, query, key, value, anchors, activate=False):
        # query: [embed_size, head_num * embed_size]
        # anchors: [K, B, embed_size]
        anchor_shape = anchors.shape.as_list()
        K = anchor_shape[0]
        new_queries = tf.reshape(tf.matmul(anchors, tf.expand_dims(query, axis=0)), [K, -1, self.head_num, self.new_embed_size])
        new_keys = tf.reshape(tf.matmul(anchors, tf.expand_dims(key, axis=0)), [K, -1, self.head_num, self.new_embed_size])
        new_values = tf.reshape(tf.matmul(anchors, tf.expand_dims(value, axis=0)), [K, -1, self.head_num, self.new_embed_size])
        # [K, B, head_num, embed_size]
        if activate:
            new_queries = tf.nn.elu(new_queries) + 1
            new_keys = tf.nn.elu(new_keys) + 1
            new_values = tf.nn.elu(new_values) + 1
        return new_queries, new_keys, new_values

    def get_qkv_anchor(self, query, key, value, anchors, activate=False):
        # query: [embed_size, head_num * embed_size]
        # anchors: [K, B, anchor_num, embed_size]
        anchor_shape = anchors.shape.as_list()
        K = anchor_shape[0]
        new_queries = tf.transpose(tf.reshape(tf.matmul(anchors, tf.reshape(query, [1, 1, self.new_embed_size, -1])),
                                              [K, -1, self.anchor_num, self.head_num, self.new_embed_size]), (0, 1, 3, 2, 4))
        new_keys = tf.transpose(tf.reshape(tf.matmul(anchors, tf.reshape(key, [1, 1, self.new_embed_size, -1])),
                                           [K, -1, self.anchor_num, self.head_num, self.new_embed_size]), (0, 1, 3, 2, 4))
        new_values = tf.transpose(tf.reshape(tf.matmul(anchors, tf.reshape(value, [1, 1, self.new_embed_size, -1])),
                                             [K, -1, self.anchor_num, self.head_num, self.new_embed_size]),
                                  (0, 1, 3, 2, 4))
        # [K, B, head_num, anchor_num, embed_size]
        if activate:
            new_queries = tf.nn.elu(new_queries) + 1
            new_keys = tf.nn.elu(new_keys) + 1
            new_values = tf.nn.elu(new_values) + 1
        return new_queries, new_keys, new_values

    def wasserstein_distance(self, mean1, cov1, mean2, cov2):
        # [K, B, head_num, embed_size]
        ret1 = tf.reduce_sum(tf.square(mean1 - mean2), axis=-1)        # [K, B, head_num]
        cov1_sq = tf.sqrt(tf.clip_by_value(cov1, 1e-12, tf.reduce_max(cov1)))
        cov2_sq = tf.sqrt(tf.clip_by_value(cov2, 1e-12, tf.reduce_max(cov2)))

        ret2 = tf.reduce_sum(tf.square(cov1_sq - cov2_sq), axis=-1)        # [K, B, head_num]
        ret = tf.reduce_sum(ret1 + ret2, axis=-1)
        return tf.transpose(ret)       # [B, K]

    def wasserstein_distance_matmul(self, mean1, cov1, mean2, cov2):
        mean1 = tf.expand_dims(mean1, axis=-2)        # [K, B, head_num, 1, embed_size]
        cov1 = tf.expand_dims(cov1, axis=-2)

        ret1 = tf.reduce_sum(tf.square(mean1 - mean2), axis=-1)        # [K, B, head_num, anchor_num]
        cov1_sq = tf.sqrt(tf.clip_by_value(cov1, 1e-12, tf.reduce_max(cov1)))
        cov2_sq = tf.sqrt(tf.clip_by_value(cov2, 1e-12, tf.reduce_max(cov2)))

        ret2 = tf.reduce_sum(tf.square(cov1_sq - cov2_sq), axis=-1)
        return ret1 + ret2

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                if self.wessertein in [4]:
                    self.entropy1 = -tf.log(tf.nn.sigmoid(
                        tf.expand_dims(tf.nn.sigmoid(self.outputs1), axis=-1) - tf.nn.sigmoid(self.neg_outputs1)))
                elif self.wessertein in [19, 20, 26]:
                    self.entropy1 = -tf.log(tf.expand_dims(self.outputs1, axis=-1) - self.neg_outputs1)
                else:
                    self.entropy1 = -tf.log(tf.nn.sigmoid(tf.expand_dims(self.outputs1, axis=-1) - self.neg_outputs1))
                self.origin_loss1 = tf.reduce_mean(self.entropy1)
                self.loss1 = self.origin_loss1

                if self.wessertein not in [4]:
                    self.entropy2 = -tf.log(tf.nn.sigmoid(
                        tf.expand_dims(tf.nn.sigmoid(self.outputs2), axis=-1) - tf.nn.sigmoid(self.neg_outputs2)))
                elif self.wessertein in [19, 20, 26]:
                    self.entropy2 = -tf.log(tf.expand_dims(self.outputs2, axis=-1) - self.neg_outputs2)
                else:
                    self.entropy2 = -tf.log(tf.nn.sigmoid(tf.expand_dims(self.outputs2, axis=-1) - self.neg_outputs2))
                self.origin_loss2 = tf.reduce_mean(self.entropy2)
                self.loss2 = self.origin_loss2

                if self.wessertein in [21]:
                    self.loss1 += self.reg_penalty * 0.5
                    self.loss2 += self.reg_penalty * 0.5

                _loss_ = self.loss
                self.optimizer1 = optimizer.minimize(loss=self.loss1,
                                                     global_step=global_step)
                self.optimizer2 = optimizer.minimize(loss=self.loss2,
                                                     global_step=global_step)

class Model_Wass(Model):
    def __init__(self, init='xavier', user_max_id=None, src_item_max_id=None, trg_item_max_id=None,
                 embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, batch_norm=False, layer_norm=False,
                 l1_w=None, l1_v=None, layer_l1=None, user_his_len=None, hist_type=None, anchor_num=5,
                 cotrain=None, l2=None, head_num=None, add_neg=None,
                 wessertein=None, ae=1, reg_alpha=None, drop_r=None, t1=None, t2=None, k_ratio=1.0, k_ratio2=1.0,
                 wloss=None, reg_a=0.0, m=0.0, pair_a=0.1, concat_emb=True):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.layer_l2 = layer_l2
        self.l1_w = l1_w
        self.layer_l1 = layer_l1
        self.l1_v = l1_v
        self.hist_type = hist_type
        self.embed_size = embed_size
        self.user_his_len = user_his_len
        self.layer_sizes = layer_sizes
        self.layer_acts = layer_acts
        self.layer_keeps = layer_keeps
        self.cotrain = cotrain
        self.anchor_num = anchor_num
        self.l2 = l2
        self.head_num = head_num
        self.wessertein = wessertein
        self.ae = ae
        self.add_neg = add_neg
        self.k1 = int(anchor_num * k_ratio)
        self.k2 = int(anchor_num * k_ratio2)
        self.wloss = wloss
        self.drop_r = drop_r
        self.reg_a = reg_a
        self.m = m
        self.pair_a = pair_a
        self.concat_emb = concat_emb

        with tf.name_scope('input'):
            self.user_id_src = tf.placeholder(tf.int32, [None], name='user_id_src')
            self.item_id_src = tf.placeholder(tf.int32, [None], name='item_id_src')
            self.user_id_trg = tf.placeholder(tf.int32, [None], name='user_id_trg')
            self.item_id_trg = tf.placeholder(tf.int32, [None], name='item_id_trg')
            self.history1 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items1')
            self.history_len1 = tf.placeholder(tf.int32, [None], name='history_items_len1')
            self.history2 = tf.placeholder(tf.int32, [None, user_his_len], name='history_items2')
            self.history_len2 = tf.placeholder(tf.int32, [None], name='history_items_len2')
            self.labels = tf.placeholder(tf.float32, [None], name='label')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.labels_src = tf.placeholder(tf.float32, [None], name='label_src')
            self.labels_trg = tf.placeholder(tf.float32, [None], name='label_trg')
            self.neg_item_id_src = tf.placeholder(tf.int32, [None, add_neg], name='neg_item_id_src')
            self.neg_item_id_trg = tf.placeholder(tf.int32, [None, add_neg], name='neg_item_id_trg')

        layer_keeps = drop_out(self.training, layer_keeps)
        v_user = get_variable(init, name='v_user', shape=[user_max_id, embed_size])
        v_item = get_variable(init, name='v_item', shape=[src_item_max_id + trg_item_max_id, embed_size])
        b = get_variable('zero', name='b', shape=[1])
        xv1_src = tf.gather(v_user, self.user_id_src)
        xv2_src = tf.gather(v_item, self.item_id_src)
        neg_xv2_src = tf.gather(v_item, self.neg_item_id_src)   # [B, add_neg, embed_size]
        xv1_trg = tf.gather(v_user, self.user_id_trg)
        xv2_trg = tf.gather(v_item, self.item_id_trg)
        neg_xv2_trg = tf.gather(v_item, self.neg_item_id_trg)

        self.new_embed_size = int(embed_size / 2)
        anchors1 = get_variable(init, name='anchors1_w', shape=[2, anchor_num, self.new_embed_size])
        anchors2 = get_variable(init, name='anchors2_w', shape=[2, anchor_num, self.new_embed_size])

        tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
                                layer_keeps, xv1_src,
                                training=self.training, name='alpha1',
                                reuse=tf.AUTO_REUSE)

        tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
                                layer_keeps, xv1_trg,
                                training=self.training, name='alpha1',
                                reuse=tf.AUTO_REUSE)

        # tau1, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
        #                         layer_keeps, tf.concat([xv1_src, xv2_src], axis=-1),
        #                         training=self.training, name='alpha1',
        #                         reuse=tf.AUTO_REUSE)
        # tau11, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
        #                          layer_keeps, tf.concat(
        #         [tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, add_neg, 1]), neg_xv2_src], axis=-1),
        #                          training=self.training, name='alpha1',
        #                          reuse=tf.AUTO_REUSE)
        #
        # tau2, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
        #                         layer_keeps, tf.concat([xv1_trg, xv2_trg], axis=-1),
        #                         training=self.training, name='alpha2',
        #                         reuse=tf.AUTO_REUSE)
        # tau21, _, _, _ = bin_mlp([64, 32, 1], ['tanh', 'tanh', None],
        #                          layer_keeps, tf.concat(
        #         [tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, add_neg, 1]), neg_xv2_trg], axis=-1),
        #                          training=self.training, name='alpha2',
        #                          reuse=tf.AUTO_REUSE)

        tau1 = tf.reshape(tau1, [1, -1, 1, 1])
        tau2 = tf.reshape(tau2, [1, -1, 1, 1])

        # tau11 = tf.transpose(tf.reshape(tau11, [-1, add_neg, 1, 1]), (1, 0, 2, 3))
        # tau21 = tf.transpose(tf.reshape(tau21, [-1, add_neg, 1, 1]), (1, 0, 2, 3))

        if self.wessertein in [-1]:
            pos_input_src = tf.concat([xv1_src, xv2_src], axis=-1)
            pos_input_trg = tf.concat([xv1_trg, xv2_trg], axis=-1)
            neg_input_src = tf.concat([tf.tile(tf.expand_dims(xv1_src, axis=-2), [1, add_neg, 1]), neg_xv2_src], axis=-1)
            neg_input_trg = tf.concat([tf.tile(tf.expand_dims(xv1_trg, axis=-2), [1, add_neg, 1]), neg_xv2_trg], axis=-1)

            if self.cotrain == 1:
                pos_outputs_src, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                                   layer_keeps, pos_input_src, training=self.training, name='mlp',
                                                   reuse=tf.AUTO_REUSE)
                neg_outputs_src, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                                   layer_keeps, neg_input_src, training=self.training, name='mlp',
                                                   reuse=tf.AUTO_REUSE)
                pos_outputs_src = tf.squeeze(pos_outputs_src, axis=-1)

                pos_outputs_trg, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                                   layer_keeps, pos_input_trg, training=self.training, name='mlp',
                                                   reuse=tf.AUTO_REUSE)
                neg_outputs_trg, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                                   layer_keeps, neg_input_trg, training=self.training, name='mlp',
                                                   reuse=tf.AUTO_REUSE)
                pos_outputs_trg = tf.squeeze(pos_outputs_trg, axis=-1)
            elif self.cotrain == 3:
                pos_outputs_src, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                                   layer_keeps, pos_input_src, training=self.training, name='mlp1',
                                                   reuse=tf.AUTO_REUSE)
                neg_outputs_src, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                                   layer_keeps, neg_input_src, training=self.training, name='mlp2',
                                                   reuse=tf.AUTO_REUSE)
                pos_outputs_src = tf.squeeze(pos_outputs_src, axis=-1)

                pos_outputs_trg, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                                   layer_keeps, pos_input_trg, training=self.training, name='mlp2',
                                                   reuse=tf.AUTO_REUSE)
                neg_outputs_trg, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                                   layer_keeps, neg_input_trg, training=self.training, name='mlp2',
                                                   reuse=tf.AUTO_REUSE)
                pos_outputs_trg = tf.squeeze(pos_outputs_trg, axis=-1)


            self.outputs1 = pos_outputs_src
            self.outputs2 = pos_outputs_trg
            self.neg_outputs1 = neg_outputs_src
            self.neg_outputs2 = neg_outputs_trg
        elif self.wessertein in [0]:
            # v_user_trg = get_variable(init, name='v_user_trg', shape=[user_max_id, embed_size])
            # xv1_trg = tf.gather(v_user_trg, self.user_id_trg)
            self.outputs1 = tf.reduce_sum(xv1_src * xv2_src, axis=-1)  # [B, ]
            self.outputs2 = tf.reduce_sum(xv1_trg * xv2_trg, axis=-1)  # [B, ]
            self.neg_outputs1 = tf.reduce_sum(tf.expand_dims(xv1_src, axis=1) * neg_xv2_src, axis=-1)
            self.neg_outputs2 = tf.reduce_sum(tf.expand_dims(xv1_trg, axis=1) * neg_xv2_trg, axis=-1)
        elif self.wessertein in [28, 29]:
            self.set_qkv(init=init)

            xv1_src = tf.transpose(tf.reshape(xv1_src, [-1, 2, self.new_embed_size]), (1, 0, 2))  # [2, B, embed_size]
            xv2_src = tf.transpose(tf.reshape(xv2_src, [-1, 2, self.new_embed_size]), (1, 0, 2))
            xv1_trg = tf.transpose(tf.reshape(xv1_trg, [-1, 2, self.new_embed_size]), (1, 0, 2))
            xv2_trg = tf.transpose(tf.reshape(xv2_trg, [-1, 2, self.new_embed_size]), (1, 0, 2))
            neg_xv2_src = tf.transpose(tf.reshape(neg_xv2_src, [-1, add_neg, 2, self.new_embed_size]), (1, 2, 0, 3))
            neg_xv2_trg = tf.transpose(tf.reshape(neg_xv2_trg, [-1, add_neg, 2, self.new_embed_size]), (1, 2, 0, 3))
            # [K, 2, B, embed_size]

            self.tau1 = 1
            self.tau2 = 1

            '''
                user embedding
            '''
            domain_emb_src1 = get_variable(init, name='domain_emb_src1', shape=[2, self.new_embed_size])
            domain_emb_trg1 = get_variable(init, name='domain_emb_trg1', shape=[2, self.new_embed_size])

            xv1_src_mean, xv1_src_cov, xv1_trg_mean, xv1_trg_cov = self.get_emb(anchors1, domain_emb_src1,
                                                                                domain_emb_trg1,
                                                                                tf.expand_dims(xv1_src, axis=0)
                                                                                , tf.expand_dims(xv1_trg, axis=0), pat=1,
                                                                                mul=False)
            # [1, B, head_num, embed_size]

            '''
                item embedding
            '''
            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_mean, xv2_src_cov, xv2_trg_mean, xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                                domain_emb_trg2,
                                                                                tf.expand_dims(xv2_src, axis=0)
                                                                                , tf.expand_dims(xv2_trg, axis=0), pat=2,
                                                                                mul=False)
            # [1, B, head_num, embed_size]
            neg_xv2_src_mean, neg_xv2_src_cov, neg_xv2_trg_mean, neg_xv2_trg_cov = self.get_emb(anchors2,
                                                                                                domain_emb_src2,
                                                                                                domain_emb_trg2,
                                                                                                neg_xv2_src,
                                                                                                neg_xv2_trg, pat=1,
                                                                                                mul=False)
            # [K, B, head_num, embed_size]

            self.outputs1 = tf.squeeze(
                -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov))  # [B, 1])
            self.outputs2 = tf.squeeze(-self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov))
            self.neg_outputs1 = -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, neg_xv2_src_mean,
                                                           neg_xv2_src_cov)  # [B, K]
            self.neg_outputs2 = -self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean,
                                                           neg_xv2_trg_cov)
        elif self.wessertein in [30]:
            self.set_qkv(init=init)

            xv1_src = tf.transpose(tf.reshape(xv1_src, [-1, 2, self.new_embed_size]), (1, 0, 2))  # [2, B, embed_size]
            xv2_src = tf.transpose(tf.reshape(xv2_src, [-1, 2, self.new_embed_size]), (1, 0, 2))
            xv1_trg = tf.transpose(tf.reshape(xv1_trg, [-1, 2, self.new_embed_size]), (1, 0, 2))
            xv2_trg = tf.transpose(tf.reshape(xv2_trg, [-1, 2, self.new_embed_size]), (1, 0, 2))
            neg_xv2_src = tf.transpose(tf.reshape(neg_xv2_src, [-1, add_neg, 2, self.new_embed_size]), (1, 2, 0, 3))
            neg_xv2_trg = tf.transpose(tf.reshape(neg_xv2_trg, [-1, add_neg, 2, self.new_embed_size]), (1, 2, 0, 3))
            # [K, 2, B, embed_size]

            # self.tau1 = 2 * tf.nn.sigmoid(tau1)
            # self.tau2 = 2 * tf.nn.sigmoid(tau2)
            # self.tau11 = 2 * tf.nn.sigmoid(tau11)
            # self.tau21 = 2 * tf.nn.sigmoid(tau21)

            self.tau1 = tf.exp(tau1)
            self.tau2 = tf.exp(tau2)
            self.tau11 = tf.exp(tau11)
            self.tau21 = tf.exp(tau21)


            '''
                user embedding
            '''
            domain_emb_src1 = get_variable(init, name='domain_emb_src1', shape=[2, self.new_embed_size])
            domain_emb_trg1 = get_variable(init, name='domain_emb_trg1', shape=[2, self.new_embed_size])

            xv1_src_mean, xv1_src_cov, xv1_trg_mean, xv1_trg_cov = self.get_emb(anchors1, domain_emb_src1,
                                                                                domain_emb_trg1,
                                                                                tf.expand_dims(xv1_src, axis=0)
                                                                                , tf.expand_dims(xv1_trg, axis=0),
                                                                                self.tau1, self.tau2, 1, mul=False)
            # [1, B, head_num, embed_size]

            '''
                item embedding
            '''
            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_mean, xv2_src_cov, xv2_trg_mean, xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                                domain_emb_trg2,
                                                                                tf.expand_dims(xv2_src, axis=0)
                                                                                , tf.expand_dims(xv2_trg, axis=0),
                                                                                self.tau1, self.tau2, 2,
                                                                                mul=False)
            # [1, B, head_num, embed_size]
            neg_xv2_src_mean, neg_xv2_src_cov, neg_xv2_trg_mean, neg_xv2_trg_cov = self.get_emb(anchors2,
                                                                                                domain_emb_src2,
                                                                                                domain_emb_trg2,
                                                                                                neg_xv2_src,
                                                                                                neg_xv2_trg,
                                                                                self.tau11, self.tau21, 2,
                                                                                                mul=False)
            # [K, B, head_num, embed_size]

            self.outputs1 = tf.squeeze(
                -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov))  # [B, 1])
            self.outputs2 = tf.squeeze(-self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov))
            self.neg_outputs1 = -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, neg_xv2_src_mean,
                                                           neg_xv2_src_cov)  # [B, K]
            self.neg_outputs2 = -self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean,
                                                           neg_xv2_trg_cov)
        elif self.wessertein in [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]:
            self.set_qkv(init=init)

            xv1_src = tf.transpose(tf.reshape(xv1_src, [-1, 2, self.new_embed_size]), (1, 0, 2))  # [2, B, embed_size]
            xv2_src = tf.transpose(tf.reshape(xv2_src, [-1, 2, self.new_embed_size]), (1, 0, 2))
            xv1_trg = tf.transpose(tf.reshape(xv1_trg, [-1, 2, self.new_embed_size]), (1, 0, 2))
            xv2_trg = tf.transpose(tf.reshape(xv2_trg, [-1, 2, self.new_embed_size]), (1, 0, 2))
            neg_xv2_src = tf.transpose(tf.reshape(neg_xv2_src, [-1, add_neg, 2, self.new_embed_size]), (1, 2, 0, 3))
            neg_xv2_trg = tf.transpose(tf.reshape(neg_xv2_trg, [-1, add_neg, 2, self.new_embed_size]), (1, 2, 0, 3))
            # [K, 2, B, embed_size]

            # self.tau1 = tf.nn.sigmoid(tau1)
            # self.tau2 = tf.nn.sigmoid(tau2)
            # self.tau11 = tf.nn.sigmoid(tau11)
            # self.tau21 = tf.nn.sigmoid(tau21)

            if self.wessertein in [37]:
                self.tau1 = 1
                self.tau2 = 1
                self.tau11 = 1
                self.tau21 = 1
            elif self.wessertein in [39, 41]:
                self.tau1 = 2*tf.nn.sigmoid(tau1)
                self.tau2 = 2*tf.nn.sigmoid(tau2)
                self.tau11 = 2*tf.nn.sigmoid(tau1)
                self.tau21 = 2*tf.nn.sigmoid(tau2)
            elif self.wessertein in [40]:
                self.tau1 = tf.exp(tau1)
                self.tau2 = tf.exp(tau2)
                self.tau11 = tf.exp(tau1)
                self.tau21 = tf.exp(tau2)
            else:
                self.tau1 = tf.nn.sigmoid(tau1)
                self.tau2 = tf.nn.sigmoid(tau2)
                self.tau11 = tf.nn.sigmoid(tau1)
                self.tau21 = tf.nn.sigmoid(tau2)

            if self.wessertein in [32, 33, 34, 36, 39, 40]:
                self.wessertein = 25
            elif self.wessertein in [38]:
                self.wessertein = 26

            '''
                user embedding
            '''
            domain_emb_src1 = get_variable(init, name='domain_emb_src1', shape=[2, self.new_embed_size])
            domain_emb_trg1 = get_variable(init, name='domain_emb_trg1', shape=[2, self.new_embed_size])

            xv1_src_mean, xv1_src_cov, xv1_trg_mean, xv1_trg_cov = self.get_emb(anchors1, domain_emb_src1,
                                                                                domain_emb_trg1,
                                                                                tf.expand_dims(xv1_src, axis=0)
                                                                                , tf.expand_dims(xv1_trg, axis=0),
                                                                                1, 1, self.tau1, self.tau2, pat=1,
                                                                                mul=False)
            # [1, B, head_num, embed_size]

            '''
                item embedding
            '''
            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_mean, xv2_src_cov, xv2_trg_mean, xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                                domain_emb_trg2,
                                                                                tf.expand_dims(xv2_src, axis=0)
                                                                                , tf.expand_dims(xv2_trg, axis=0),
                                                                                1, 1, self.tau1, self.tau2, pat=2,
                                                                                mul=False)
            # [1, B, head_num, embed_size]
            neg_xv2_src_mean, neg_xv2_src_cov, neg_xv2_trg_mean, neg_xv2_trg_cov = self.get_emb(anchors2,
                                                                                                domain_emb_src2,
                                                                                                domain_emb_trg2,
                                                                                                neg_xv2_src,
                                                                                                neg_xv2_trg,
                                                                                                1, 1, self.tau11, self.tau21, pat=2,
                                                                                                mul=False)
            # [K, B, head_num, embed_size]

            self.outputs1 = tf.squeeze(
                -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov))  # [B, 1])
            self.outputs2 = tf.squeeze(-self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov))
            self.neg_outputs1 = -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, neg_xv2_src_mean,
                                                           neg_xv2_src_cov)  # [B, K]
            self.neg_outputs2 = -self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean,
                                                           neg_xv2_trg_cov)
            if self.wessertein in [34, 35]:
                pos_input_src = tf.concat([xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov], axis=-1)
                pos_input_trg = tf.concat([xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov], axis=-1)
                neg_input_src = tf.concat([xv1_src_mean, xv1_src_cov, neg_xv2_src_mean, neg_xv2_src_cov], axis=-1)
                neg_input_trg = tf.concat([xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean, neg_xv2_trg_cov], axis=-1)

                w1 = get_variable(init, name='w1', shape=[1])
                w2 = get_variable(init, name='w2', shape=[1])

                if self.cotrain == 1:
                    pos_outputs_src, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                            layer_keeps, pos_input_src, training=self.training, name='mlp',
                                            reuse=tf.AUTO_REUSE)
                    neg_outputs_src, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                            layer_keeps, neg_input_src, training=self.training, name='mlp',
                                            reuse=tf.AUTO_REUSE)
                    neg_outputs_src = tf.squeeze(neg_outputs_src, axis=-1)

                    pos_outputs_trg, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                            layer_keeps, pos_input_trg, training=self.training, name='mlp',
                                            reuse=tf.AUTO_REUSE)
                    neg_outputs_trg, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                            layer_keeps, neg_input_trg, training=self.training, name='mlp',
                                            reuse=tf.AUTO_REUSE)
                    neg_outputs_trg = tf.squeeze(neg_outputs_trg, axis=-1)
                elif self.cotrain == 3:
                    pos_outputs_src, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                                       layer_keeps, pos_input_src, training=self.training, name='mlp1',
                                                       reuse=tf.AUTO_REUSE)
                    neg_outputs_src, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                                       layer_keeps, neg_input_src, training=self.training, name='mlp2',
                                                       reuse=tf.AUTO_REUSE)
                    neg_outputs_src = tf.squeeze(neg_outputs_src, axis=-1)


                    pos_outputs_trg, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                                       layer_keeps, pos_input_trg, training=self.training, name='mlp2',
                                                       reuse=tf.AUTO_REUSE)
                    neg_outputs_trg, _, _, _ = bin_mlp(layer_sizes, layer_acts,
                                                       layer_keeps, neg_input_trg, training=self.training, name='mlp2',
                                                       reuse=tf.AUTO_REUSE)
                    neg_outputs_trg = tf.squeeze(neg_outputs_trg, axis=-1)
                self.outputs1 = w1 * self.outputs1 + w2 * pos_outputs_src
                self.outputs2 = w1 * self.outputs2 + w2 * pos_outputs_trg
                self.neg_outputs1 = w1 * self.neg_outputs1 + w2 * neg_outputs_src
                self.neg_outputs2 = w1 * self.neg_outputs2 + w2 * neg_outputs_trg
        elif self.wessertein in [42]:
            self.set_qkv(init=init)

            xv1_src = tf.transpose(tf.reshape(xv1_src, [-1, 2, self.new_embed_size]), (1, 0, 2))  # [2, B, embed_size]
            xv2_src = tf.transpose(tf.reshape(xv2_src, [-1, 2, self.new_embed_size]), (1, 0, 2))
            xv1_trg = tf.transpose(tf.reshape(xv1_trg, [-1, 2, self.new_embed_size]), (1, 0, 2))
            xv2_trg = tf.transpose(tf.reshape(xv2_trg, [-1, 2, self.new_embed_size]), (1, 0, 2))
            neg_xv2_src = tf.transpose(tf.reshape(neg_xv2_src, [-1, add_neg, 2, self.new_embed_size]), (1, 2, 0, 3))
            neg_xv2_trg = tf.transpose(tf.reshape(neg_xv2_trg, [-1, add_neg, 2, self.new_embed_size]), (1, 2, 0, 3))
            # [K, 2, B, embed_size]

            self.tau1 = 2*tf.nn.sigmoid(tau1)
            self.tau2 = 2*tf.nn.sigmoid(tau2)
            self.tau11 = 2*tf.nn.sigmoid(tau1)
            self.tau21 = 2*tf.nn.sigmoid(tau2)

            '''
                user embedding
            '''
            domain_emb_src = get_variable(init, name='domain_emb_src', shape=[2, self.new_embed_size])

            xv1_src_mean, xv1_src_cov, xv1_trg_mean, xv1_trg_cov = self.get_emb(anchors1, domain_emb_src,
                                                                                domain_emb_src,
                                                                                tf.expand_dims(xv1_src, axis=0)
                                                                                , tf.expand_dims(xv1_trg, axis=0),
                                                                                1, 1, self.tau1, self.tau2, pat=1,
                                                                                mul=False)
            # [1, B, head_num, embed_size]

            '''
                item embedding
            '''
            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_mean, xv2_src_cov, xv2_trg_mean, xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                                domain_emb_trg2,
                                                                                tf.expand_dims(xv2_src, axis=0)
                                                                                , tf.expand_dims(xv2_trg, axis=0),
                                                                                1, 1, self.tau1, self.tau2, pat=2,
                                                                                mul=False)
            # [1, B, head_num, embed_size]
            neg_xv2_src_mean, neg_xv2_src_cov, neg_xv2_trg_mean, neg_xv2_trg_cov = self.get_emb(anchors2,
                                                                                                domain_emb_src2,
                                                                                                domain_emb_trg2,
                                                                                                neg_xv2_src,
                                                                                                neg_xv2_trg,
                                                                                                1, 1, self.tau11, self.tau21, pat=2,
                                                                                                mul=False)
            # [K, B, head_num, embed_size]

            self.outputs1 = tf.squeeze(
                -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov))  # [B, 1])
            self.outputs2 = tf.squeeze(-self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov))
            self.neg_outputs1 = -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, neg_xv2_src_mean,
                                                           neg_xv2_src_cov)  # [B, K]
            self.neg_outputs2 = -self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean,
                                                           neg_xv2_trg_cov)
        else:
            self.set_qkv(init=init)

            xv1_src = tf.transpose(tf.reshape(xv1_src, [-1, 2, self.new_embed_size]), (1, 0, 2))  # [2, B, embed_size]
            xv2_src = tf.transpose(tf.reshape(xv2_src, [-1, 2, self.new_embed_size]), (1, 0, 2))
            xv1_trg = tf.transpose(tf.reshape(xv1_trg, [-1, 2, self.new_embed_size]), (1, 0, 2))
            xv2_trg = tf.transpose(tf.reshape(xv2_trg, [-1, 2, self.new_embed_size]), (1, 0, 2))
            neg_xv2_src = tf.transpose(tf.reshape(neg_xv2_src, [-1, add_neg, 2, self.new_embed_size]), (1, 2, 0, 3))
            neg_xv2_trg = tf.transpose(tf.reshape(neg_xv2_trg, [-1, add_neg, 2, self.new_embed_size]), (1, 2, 0, 3))
            # [K, 2, B, embed_size]

            self.tau1 = 1
            self.tau2 = 1

            '''
                user embedding
            '''
            domain_emb_src1 = get_variable(init, name='domain_emb_src1', shape=[2, self.new_embed_size])
            domain_emb_trg1 = get_variable(init, name='domain_emb_trg1', shape=[2, self.new_embed_size])

            xv1_src_mean, xv1_src_cov, xv1_trg_mean, xv1_trg_cov = self.get_emb(anchors1, domain_emb_src1,
                                                                                domain_emb_trg1,
                                                                                tf.expand_dims(xv1_src, axis=0)
                                                                                , tf.expand_dims(xv1_trg, axis=0), pat=1)
            # [1, B, head_num, embed_size]

            '''
                item embedding
            '''
            domain_emb_src2 = get_variable(init, name='domain_emb_src2', shape=[2, self.new_embed_size])
            domain_emb_trg2 = get_variable(init, name='domain_emb_trg2', shape=[2, self.new_embed_size])

            xv2_src_mean, xv2_src_cov, xv2_trg_mean, xv2_trg_cov = self.get_emb(anchors2, domain_emb_src2,
                                                                                domain_emb_trg2,
                                                                                tf.expand_dims(xv2_src, axis=0)
                                                                                , tf.expand_dims(xv2_trg, axis=0), pat=2)
            # [1, B, head_num, embed_size]
            neg_xv2_src_mean, neg_xv2_src_cov, neg_xv2_trg_mean, neg_xv2_trg_cov = self.get_emb(anchors2,
                                                                                                domain_emb_src2,
                                                                                                domain_emb_trg2,
                                                                                                neg_xv2_src,
                                                                                                neg_xv2_trg, pat=2)
            # [K, B, head_num, embed_size]

            self.outputs1 = tf.squeeze(
                -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov))  # [B, 1])
            self.outputs2 = tf.squeeze(-self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov))
            self.neg_outputs1 = -self.wasserstein_distance(xv1_src_mean, xv1_src_cov, neg_xv2_src_mean,
                                                           neg_xv2_src_cov)  # [B, K]
            self.neg_outputs2 = -self.wasserstein_distance(xv1_trg_mean, xv1_trg_cov, neg_xv2_trg_mean,
                                                           neg_xv2_trg_cov)

        if self.reg_a > 0:
            self.reg = tf.reduce_mean(tf.reduce_sum(tf.square(tf.concat([xv1_src_mean, xv1_src_cov, xv2_src_mean, xv2_src_cov], axis=-1)), axis=-1)) \
                        + tf.reduce_mean(tf.reduce_sum(tf.square(tf.concat([xv1_trg_mean, xv1_trg_cov, xv2_trg_mean, xv2_trg_cov], axis=-1)), axis=-1))
            self.reg += tf.reduce_mean(tf.reduce_sum(tf.square(tf.concat([neg_xv2_src_mean, neg_xv2_src_cov], axis=-1)), axis=-1)) \
                        + tf.reduce_mean(tf.reduce_sum(tf.square(tf.concat([neg_xv2_trg_mean, neg_xv2_trg_cov], axis=-1)), axis=-1))

    def get_emb(self, anchors, domain_emb_src, domain_emb_trg, xv_src, xv_trg, tau1=1.0, tau2=1.0, agg_tau1=1.0, agg_tau2=1.0, pat=None, mul=True):
        '''
            anchors: [2, anchor_num, embed_size]
            domain_emb_src: [2, embed_size]
            domain_emb_trg: [2, embed_size]
            xv_src: [K, 2, B, embed_size]
            xv_trg: [K, 2, B, embed_size]
        '''

        if self.wessertein in [36]:
            if pat == 1:
                queries_src = self.queries_src1
                queries_trg = self.queries_src1
                keys_src = self.keys_src1
                keys_trg = self.keys_src1
                values_src = self.values_src1
                values_trg = self.values_src1
            else:
                queries_src = self.queries_src2
                queries_trg = self.queries_src2
                keys_src = self.keys_src2
                keys_trg = self.keys_src2
                values_src = self.values_src2
                values_trg = self.values_src2
        else:
            queries_src = self.queries_src1
            queries_trg = self.queries_trg1
            keys_src = self.keys_src1
            keys_trg = self.keys_trg1
            values_src = self.values_src1
            values_trg = self.values_trg1


        xv_src_queries_mean, xv_src_keys_mean, xv_src_values_mean = \
            self.get_qkv(queries_src[0], keys_src[0], values_src[0], xv_src[:, 0], mul=mul)
        xv_src_queries_cov, xv_src_keys_cov, xv_src_values_cov = \
            self.get_qkv(queries_src[1], keys_src[1], values_src[1], xv_src[:, 1], True, mul=mul)
        # [K, B, head_num, embed_size]

        xv_trg_queries_mean, xv_trg_keys_mean, xv_trg_values_mean = \
            self.get_qkv(queries_trg[0], keys_trg[0], values_trg[0], xv_trg[:, 0], mul=mul)
        xv_trg_queries_cov, xv_trg_keys_cov, xv_trg_values_cov = \
            self.get_qkv(queries_trg[1], keys_trg[1], values_trg[1], xv_trg[:, 1], True, mul=mul)
        # [K, B, head_num, embed_size]

        in_domain_queries_src_mean, in_domain_keys_src_mean, in_domain_values_src_mean = \
            self.get_qkv(queries_src[0], keys_src[0], values_src[0], tf.reshape(domain_emb_src,
                                                                                [1, 2, 1, -1])[:, 0], mul=mul)
        in_domain_queries_src_cov, in_domain_keys_src_cov, in_domain_values_src_cov = \
            self.get_qkv(queries_src[1], keys_src[1], values_src[1], tf.reshape(domain_emb_src,
                                                                                [1, 2, 1, -1])[:, 1], True, mul=mul)
        # [1, 1, head_num, embed_size]

        in_domain_queries_trg_mean, in_domain_keys_trg_mean, in_domain_values_trg_mean = \
            self.get_qkv(queries_trg[0], keys_trg[0], values_trg[0], tf.reshape(domain_emb_trg,
                                                                                [1, 2, 1, -1])[:, 0], mul=mul)
        in_domain_queries_trg_cov, in_domain_keys_trg_cov, in_domain_values_trg_cov = \
            self.get_qkv(queries_trg[1], keys_trg[1], values_trg[1], tf.reshape(domain_emb_trg,
                                                                                [1, 2, 1, -1])[:, 1], True, mul=mul)
        # [1, 1, head_num, embed_size]

        # print('\nin_domain_queries_mean\n', in_domain_queries_mean.shape, '\n\n')

        sub_domain_queries_mean, sub_domain_keys_mean, sub_domain_values_mean = \
            self.get_qkv_anchor(queries_src[0], keys_src[0], values_src[0],
                                tf.reshape(anchors, [1, 2, 1, self.anchor_num,
                                                     -1])[:, 0], mul=mul)
        sub_domain_queries_cov, sub_domain_keys_cov, sub_domain_values_cov = \
            self.get_qkv_anchor(queries_src[1], keys_src[1], values_src[1],
                                tf.reshape(anchors, [1, 2, 1, self.anchor_num,
                                                     -1])[:, 1], True, mul=mul)
        # [K, B, head_num, anchor_num, embed_size]

        '''
            get in_domain queries & values
        '''
        xv_src_queries = tf.concat(
            [tf.expand_dims(xv_src_queries_mean, axis=1), tf.expand_dims(xv_src_queries_cov, axis=1)], axis=1)
        in_domain_queries_src = tf.concat(
            [tf.expand_dims(in_domain_queries_src_mean, axis=1), tf.expand_dims(in_domain_queries_src_cov, axis=1)],
            axis=1)
        xv_src_in_domain_queries = self.combine_dist(xv_src_queries,
                                                     in_domain_queries_src)  # [K, 2, B, head_num, self.new_embed_size]

        xv_trg_queries = tf.concat(
            [tf.expand_dims(xv_trg_queries_mean, axis=1), tf.expand_dims(xv_trg_queries_cov, axis=1)], axis=1)
        in_domain_queries_trg = tf.concat(
            [tf.expand_dims(in_domain_queries_trg_mean, axis=1), tf.expand_dims(in_domain_queries_trg_cov, axis=1)],
            axis=1)
        xv_trg_in_domain_queries = self.combine_dist(xv_trg_queries,
                                                     in_domain_queries_trg)  # [K, 2, B, head_num, self.new_embed_size]

        xv_src_in_domain_queries_mean, xv_src_in_domain_queries_cov = xv_src_in_domain_queries[:,
                                                                      0], xv_src_in_domain_queries[:, 1]
        xv_trg_in_domain_queries_mean, xv_trg_in_domain_queries_cov = xv_trg_in_domain_queries[:,
                                                                      0], xv_trg_in_domain_queries[:, 1]
        # [K, B, head_num, embed_size]

        xv_src_values = tf.concat(
            [tf.expand_dims(xv_src_values_mean, axis=1), tf.expand_dims(xv_src_values_cov, axis=1)], axis=1)
        in_domain_values_src = tf.concat(
            [tf.expand_dims(in_domain_values_src_mean, axis=1), tf.expand_dims(in_domain_values_src_cov, axis=1)],
            axis=1)
        xv_src_in_domain_values = self.combine_dist(xv_src_values,
                                                    in_domain_values_src)  # [K, 2, B, head_num, self.new_embed_size]

        xv_trg_values = tf.concat(
            [tf.expand_dims(xv_trg_values_mean, axis=1), tf.expand_dims(xv_trg_values_cov, axis=1)], axis=1)
        in_domain_values_trg = tf.concat(
            [tf.expand_dims(in_domain_values_trg_mean, axis=1), tf.expand_dims(in_domain_values_trg_cov, axis=1)],
            axis=1)
        xv_trg_in_domain_values = self.combine_dist(xv_trg_values,
                                                    in_domain_values_trg)  # [K, 2, B, head_num, self.new_embed_size]

        xv_src_in_domain_values_mean, xv_src_in_domain_values_cov = xv_src_in_domain_values[:,
                                                                    0], xv_src_in_domain_values[:, 1]
        xv_trg_in_domain_values_mean, xv_trg_in_domain_values_cov = xv_trg_in_domain_values[:,
                                                                    0], xv_trg_in_domain_values[:, 1]
        # [K, B, head_num, embed_size]
        ''''
            get in_domain queries & values END!!!!!!!!
        '''


        xv_src_keys = tf.concat([tf.expand_dims(xv_src_keys_mean, axis=1), tf.expand_dims(xv_src_keys_cov, axis=1)],
                                axis=1)
        xv_src_keys = tf.expand_dims(xv_src_keys, axis=-2)  # [K, 2, B, head_num, 1, embed_size]
        sub_domain_keys = tf.concat(
            [tf.expand_dims(sub_domain_keys_mean, axis=1), tf.expand_dims(sub_domain_keys_cov, axis=1)], axis=1)
        xv_src_sub_domain_keys = self.combine_dist(xv_src_keys,
                                                   sub_domain_keys)  # [K, 2, B, head_num, anchor_num, embed_size]

        xv_trg_keys = tf.concat([tf.expand_dims(xv_trg_keys_mean, axis=1), tf.expand_dims(xv_trg_keys_cov, axis=1)],
                                axis=1)
        xv_trg_keys = tf.expand_dims(xv_trg_keys, axis=-2)
        sub_domain_keys = tf.concat(
            [tf.expand_dims(sub_domain_keys_mean, axis=1), tf.expand_dims(sub_domain_keys_cov, axis=1)], axis=1)
        xv_trg_sub_domain_keys = self.combine_dist(xv_trg_keys,
                                                   sub_domain_keys)  # [K, 2, B, head_num, anchor_num, embed_size]
        xv_src_sub_domain_keys_mean, xv_src_sub_domain_keys_cov = xv_src_sub_domain_keys[:,
                                                                  0], xv_src_sub_domain_keys[:, 1]
        xv_trg_sub_domain_keys_mean, xv_trg_sub_domain_keys_cov = xv_trg_sub_domain_keys[:,
                                                                  0], xv_trg_sub_domain_keys[:, 1]
        # [K, B, head_num, anchor_num, embed_size]

        xv_src_values = tf.concat(
            [tf.expand_dims(xv_src_values_mean, axis=1), tf.expand_dims(xv_src_values_cov, axis=1)],
            axis=1)
        xv_src_values = tf.expand_dims(xv_src_values, axis=-2)  # [K, 2, B, 1, embed_size]
        sub_domain_values = tf.concat(
            [tf.expand_dims(sub_domain_values_mean, axis=1), tf.expand_dims(sub_domain_values_cov, axis=1)], axis=1)
        xv_src_sub_domain_values = self.combine_dist(xv_src_values,
                                                     sub_domain_values)  # [K, 2, B, head_num, embed_size]

        xv_trg_values = tf.concat(
            [tf.expand_dims(xv_trg_values_mean, axis=1), tf.expand_dims(xv_trg_values_cov, axis=1)],
            axis=1)
        xv_trg_values = tf.expand_dims(xv_trg_values, axis=-2)
        sub_domain_values = tf.concat(
            [tf.expand_dims(sub_domain_values_mean, axis=1), tf.expand_dims(sub_domain_values_cov, axis=1)], axis=1)
        xv_trg_sub_domain_values = self.combine_dist(xv_trg_values,
                                                     sub_domain_values)  # [K, 2, B, head_num, embed_size]
        xv_src_sub_domain_values_mean, xv_src_sub_domain_values_cov = xv_src_sub_domain_values[:,
                                                                      0], xv_src_sub_domain_values[:, 1]
        xv_trg_sub_domain_values_mean, xv_trg_sub_domain_values_cov = xv_trg_sub_domain_values[:,
                                                                      0], xv_trg_sub_domain_values[:, 1]
        # [K, B, head_num, anchor_num, embed_size]

        attention_src = -self.wasserstein_distance_matmul(xv_src_in_domain_queries_mean,
                                                          xv_src_in_domain_queries_cov,
                                                          xv_src_sub_domain_keys_mean,
                                                          xv_src_sub_domain_keys_cov)  # [K, B, head_num, anchor_num]

        attention_trg = -self.wasserstein_distance_matmul(xv_trg_in_domain_queries_mean,
                                                          xv_trg_in_domain_queries_cov,
                                                          xv_trg_sub_domain_keys_mean,
                                                          xv_trg_sub_domain_keys_cov)  # [K, B, head_num, anchor_num]

        if not self.concat_emb:
            norm = np.sqrt(self.new_embed_size)
        else:
            norm = np.sqrt(self.new_embed_size * 2)
        attention_src = attention_src / norm  # [K, B, head_num, anchor_num]
        attention_trg = attention_trg / norm
        head_scores_src = tf.nn.softmax(attention_src / tau1, dim=-1)  # [K, B, head_num, anchor_num]
        head_scores_trg = tf.nn.softmax(attention_trg / tau2, dim=-1)

        if self.wessertein in [22]:
            random_masks_src = tf.random_uniform(shape=tf.shape(head_scores_src))
            random_masks_trg = tf.random_uniform(shape=tf.shape(head_scores_trg))
            max_vals = tf.reduce_max(random_masks_src, axis=-1, keepdims=True)
            rand_tensor_binary_src = tf.where(tf.less(tf.abs(random_masks_src - max_vals), 1e-5),
                                              tf.zeros_like(random_masks_src),
                                              tf.ones_like(random_masks_src))
            max_vals = tf.reduce_max(random_masks_trg, axis=-1, keepdims=True)
            rand_tensor_binary_trg = tf.where(tf.less(tf.abs(random_masks_trg - max_vals), 1e-5),
                                              tf.zeros_like(random_masks_trg),
                                              tf.ones_like(random_masks_trg))
            rand_tensor_binary_src = tf.cond(self.training, lambda: rand_tensor_binary_src,
                                             lambda: tf.ones_like(random_masks_src))  # training!!!!!!!!!
            rand_tensor_binary_trg = tf.cond(self.training, lambda: rand_tensor_binary_trg,
                                             lambda: tf.ones_like(random_masks_trg))
            head_scores_src = head_scores_src * rand_tensor_binary_src
            head_scores_trg = head_scores_trg * rand_tensor_binary_trg
            head_scores_src = head_scores_src * self.anchor_num / tf.reduce_sum(rand_tensor_binary_src, axis=-1,
                                                                                keep_dims=True)
            head_scores_trg = head_scores_trg * self.anchor_num / tf.reduce_sum(rand_tensor_binary_trg, axis=-1,
                                                                                keep_dims=True)
        elif self.wessertein in [23]:
            random_masks_src = tf.random_uniform(shape=tf.shape(head_scores_src))
            random_masks_trg = tf.random_uniform(shape=tf.shape(head_scores_trg))
            rand_tensor_binary_src = tf.where(tf.less(random_masks_src, self.drop_r), tf.zeros_like(random_masks_src),
                                              tf.ones_like(random_masks_src))
            rand_tensor_binary_trg = tf.where(tf.less(random_masks_trg, self.drop_r), tf.zeros_like(random_masks_trg),
                                              tf.ones_like(random_masks_trg))
            rand_tensor_binary_src = tf.cond(self.training, lambda: rand_tensor_binary_src,
                                             lambda: tf.ones_like(random_masks_src))  # training!!!!!!!!!
            rand_tensor_binary_trg = tf.cond(self.training, lambda: rand_tensor_binary_trg,
                                             lambda: tf.ones_like(random_masks_trg))
            head_scores_src = head_scores_src * rand_tensor_binary_src
            head_scores_trg = head_scores_trg * rand_tensor_binary_trg
            head_scores_src = head_scores_src * self.anchor_num / tf.reduce_sum(rand_tensor_binary_src, axis=-1,
                                                                                keep_dims=True)
            head_scores_trg = head_scores_trg * self.anchor_num / tf.reduce_sum(rand_tensor_binary_trg, axis=-1,
                                                                                keep_dims=True)
        elif self.wessertein in [24]:
            random_masks_src = tf.random_uniform(shape=tf.shape(head_scores_src))
            random_masks_trg = tf.random_uniform(shape=tf.shape(head_scores_trg))
            rand_tensor_binary_src = tf.where(tf.less(random_masks_src, self.drop_r), tf.zeros_like(random_masks_src),
                                              tf.ones_like(random_masks_src))
            rand_tensor_binary_trg = tf.where(tf.less(random_masks_trg, self.drop_r), tf.zeros_like(random_masks_trg),
                                              tf.ones_like(random_masks_trg))
            rand_tensor_binary_src = tf.cond(self.training, lambda: rand_tensor_binary_src,
                                             lambda: tf.ones_like(random_masks_src))  # training!!!!!!!!!
            rand_tensor_binary_trg = tf.cond(self.training, lambda: rand_tensor_binary_trg,
                                             lambda: tf.ones_like(random_masks_trg))
            head_scores_src = head_scores_src * rand_tensor_binary_src
            head_scores_trg = head_scores_trg * rand_tensor_binary_trg
            ratio = tf.cond(self.training, lambda: 1.0, lambda: (1 - self.drop_r))
            head_scores_src = head_scores_src * ratio
            head_scores_trg = head_scores_trg * ratio
        elif self.wessertein in [25, 30]:
            top_k_vals, top_k_indices = tf.nn.top_k(head_scores_src, k=self.k1)
            mask_binary_src = tf.one_hot(top_k_indices, depth=self.anchor_num, on_value=1, off_value=0,
                                         axis=-1)
            mask_binary_src = tf.reduce_sum(mask_binary_src, axis=-2)
            top_k_vals, top_k_indices = tf.nn.top_k(head_scores_trg, k=self.k2)
            mask_binary_trg = tf.one_hot(top_k_indices, depth=self.anchor_num, on_value=1, off_value=0,
                                         axis=-1)
            mask_binary_trg = tf.reduce_sum(mask_binary_trg, axis=-2)
            head_scores_src = head_scores_src * tf.cast(mask_binary_src, tf.float32)
            head_scores_trg = head_scores_trg * tf.cast(mask_binary_trg, tf.float32)
            self.a = head_scores_src
            self.b = head_scores_trg
        elif self.wessertein in [26]:
            top_k_vals, top_k_indices = tf.nn.top_k(head_scores_src, k=self.k1)
            mask_binary_src = tf.one_hot(top_k_indices, depth=self.anchor_num, on_value=1, off_value=0,
                                         axis=-1)
            mask_binary_src = tf.reduce_sum(mask_binary_src, axis=-2)
            top_k_vals, top_k_indices = tf.nn.top_k(head_scores_trg, k=self.k2)
            mask_binary_trg = tf.one_hot(top_k_indices, depth=self.anchor_num, on_value=1, off_value=0,
                                         axis=-1)
            mask_binary_trg = tf.reduce_sum(mask_binary_trg, axis=-2)
            ori_head_scores_src = head_scores_src
            ori_head_scores_trg = head_scores_trg

            head_scores_src = head_scores_src * tf.cast(mask_binary_src, tf.float32)
            head_scores_trg = head_scores_trg * tf.cast(mask_binary_trg, tf.float32)

            head_scores_src = ori_head_scores_src + tf.stop_gradient(head_scores_src - ori_head_scores_src)
            head_scores_trg = ori_head_scores_trg + tf.stop_gradient(head_scores_trg - ori_head_scores_trg)

            self.a = head_scores_src
            self.b = head_scores_trg
        elif self.wessertein in [41]:
            top_k_vals, top_k_indices = tf.nn.top_k(head_scores_src, k=self.k1)
            mask_binary_src = tf.one_hot(top_k_indices, depth=self.anchor_num, on_value=1, off_value=0,
                                         axis=-1)
            mask_binary_src = tf.reduce_sum(mask_binary_src, axis=-2)
            top_k_vals, top_k_indices = tf.nn.top_k(head_scores_trg, k=self.k2)
            mask_binary_trg = tf.one_hot(top_k_indices, depth=self.anchor_num, on_value=1, off_value=0,
                                         axis=-1)
            mask_binary_trg = tf.reduce_sum(mask_binary_trg, axis=-2)
            head_scores_src = head_scores_src * tf.cast(mask_binary_src, tf.float32)
            head_scores_trg = head_scores_trg * tf.cast(mask_binary_trg, tf.float32)
            head_scores_src = head_scores_src / tf.reduce_sum(head_scores_src, axis=-1, keep_dims=True)
            head_scores_trg = head_scores_trg / tf.reduce_sum(head_scores_trg, axis=-1, keep_dims=True)
            self.a = head_scores_src
            self.b = head_scores_trg

        xv_src_agg_mean = tf.reduce_sum(tf.expand_dims(head_scores_src, axis=-1) * xv_src_sub_domain_values_mean,
                                        axis=-2)
        xv_trg_agg_mean = tf.reduce_sum(tf.expand_dims(head_scores_trg, axis=-1) * xv_trg_sub_domain_values_mean,
                                        axis=-2)  # [K, B, head_num, embed_size]

        xv_src_agg_cov = tf.reduce_sum(
            tf.expand_dims(tf.square(head_scores_src), axis=-1) * xv_src_sub_domain_values_cov, axis=-2)
        xv_trg_agg_cov = tf.reduce_sum(
            tf.expand_dims(tf.square(head_scores_trg), axis=-1) * xv_trg_sub_domain_values_cov, axis=-2)
        # [K, B, head_num, embed_size]

        if self.wessertein in [27]:
            # aggregate
            xv_src_mean = (xv_src_in_domain_values_mean + xv_src_agg_mean) / 2  # [K, B, head_num, embed_size]
            xv_trg_mean = (xv_trg_in_domain_values_mean + xv_trg_agg_mean) / 2

            xv_src_cov = tf.square((tf.sqrt(xv_src_in_domain_values_cov) + tf.sqrt(xv_src_agg_cov)) / 2)
            xv_trg_cov = tf.square((tf.sqrt(xv_trg_in_domain_values_cov) + tf.sqrt(xv_trg_agg_cov)) / 2)
        elif self.wessertein in [17, 28]:
            xv_src_mean = xv_src_values_mean
            xv_trg_mean = xv_trg_values_mean

            xv_src_cov = xv_src_values_cov
            xv_trg_cov = xv_trg_values_cov
        elif self.wessertein in [31, 33]:
            # aggregate
            xv_src_mean = (1 - agg_tau1) * xv_src_in_domain_values_mean + agg_tau1 * xv_src_agg_mean  # [K, B, head_num, embed_size]
            xv_trg_mean = (1 - agg_tau2) * xv_trg_in_domain_values_mean + agg_tau2 * xv_trg_agg_mean

            xv_src_cov = tf.square((1 - agg_tau1) * tf.sqrt(xv_src_in_domain_values_cov) + agg_tau1 * tf.sqrt(xv_src_agg_cov))
            xv_trg_cov = tf.square((1 - agg_tau2) * tf.sqrt(xv_trg_in_domain_values_cov) + agg_tau2 * tf.sqrt(xv_trg_agg_cov))
        elif self.wessertein in [42]:
            xv_src_mean = xv_src_in_domain_values_mean
            xv_trg_mean = xv_trg_in_domain_values_mean

            xv_src_cov = xv_src_in_domain_values_cov
            xv_trg_cov = xv_trg_in_domain_values_cov
        else:
            # aggregate
            xv_src_mean = xv_src_in_domain_values_mean + xv_src_agg_mean  # [K, B, head_num, embed_size]
            xv_trg_mean = xv_trg_in_domain_values_mean + xv_trg_agg_mean

            xv_src_cov = tf.square(tf.sqrt(xv_src_in_domain_values_cov) + tf.sqrt(xv_src_agg_cov))
            xv_trg_cov = tf.square(tf.sqrt(xv_trg_in_domain_values_cov) + tf.sqrt(xv_trg_agg_cov))

        return xv_src_mean, xv_src_cov, xv_trg_mean, xv_trg_cov


    def set_qkv(self, init=None):
        self.queries_src1 = get_variable(init, name='queries_src1',
                                         shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.keys_src1 = get_variable(init, name='keys_src1',
                                      shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.values_src1 = get_variable(init, name='values_src1',
                                        shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])

        self.queries_trg1 = get_variable(init, name='queries_trg1',
                                         shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.keys_trg1 = get_variable(init, name='keys_trg1',
                                      shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.values_trg1 = get_variable(init, name='values_trg1',
                                        shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])

        self.queries_src2 = get_variable(init, name='queries_src2',
                                         shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.keys_src2 = get_variable(init, name='keys_src2',
                                      shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.values_src2 = get_variable(init, name='values_src2',
                                        shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])

        self.queries_trg2 = get_variable(init, name='queries_trg2',
                                         shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.keys_trg2 = get_variable(init, name='keys_trg2',
                                      shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])
        self.values_trg2 = get_variable(init, name='values_trg2',
                                        shape=[2, self.new_embed_size, self.head_num * self.new_embed_size])

    def get_qkv(self, query, key, value, anchors, activate=False, mul=True):
        # query: [embed_size, head_num * embed_size]
        # anchors: [K, B, embed_size]
        anchor_shape = anchors.shape.as_list()
        K = anchor_shape[0]
        if mul:
            new_queries = tf.reshape(tf.matmul(anchors, tf.expand_dims(query, axis=0)), [K, -1, self.head_num, self.new_embed_size])
            new_keys = tf.reshape(tf.matmul(anchors, tf.expand_dims(key, axis=0)), [K, -1, self.head_num, self.new_embed_size])
            new_values = tf.reshape(tf.matmul(anchors, tf.expand_dims(value, axis=0)), [K, -1, self.head_num, self.new_embed_size])
            # [K, B, head_num, embed_size]
        else:
            new_queries = tf.expand_dims(anchors, axis=-2)
            new_keys = tf.expand_dims(anchors, axis=-2)
            new_values = tf.expand_dims(anchors, axis=-2)
            # [K, B, head_num, embed_size]
        if activate:
            new_queries = tf.nn.elu(new_queries) + 1
            new_keys = tf.nn.elu(new_keys) + 1
            new_values = tf.nn.elu(new_values) + 1
        return new_queries, new_keys, new_values

    def get_qkv_anchor(self, query, key, value, anchors, activate=False, mul=True):
        # query: [embed_size, head_num * embed_size]
        # anchors: [K, B, anchor_num, embed_size]
        anchor_shape = anchors.shape.as_list()
        K = anchor_shape[0]
        if mul:
            new_queries = tf.transpose(tf.reshape(tf.matmul(anchors, tf.reshape(query, [1, 1, self.new_embed_size, -1])),
                                                  [K, -1, self.anchor_num, self.head_num, self.new_embed_size]), (0, 1, 3, 2, 4))
            new_keys = tf.transpose(tf.reshape(tf.matmul(anchors, tf.reshape(key, [1, 1, self.new_embed_size, -1])),
                                               [K, -1, self.anchor_num, self.head_num, self.new_embed_size]), (0, 1, 3, 2, 4))
            new_values = tf.transpose(tf.reshape(tf.matmul(anchors, tf.reshape(value, [1, 1, self.new_embed_size, -1])),
                                                 [K, -1, self.anchor_num, self.head_num, self.new_embed_size]),
                                      (0, 1, 3, 2, 4))
            # [K, B, head_num, anchor_num, embed_size]
        else:
            new_queries = tf.expand_dims(anchors, axis=-3)
            new_keys = tf.expand_dims(anchors, axis=-3)
            new_values = tf.expand_dims(anchors, axis=-3)
            # [K, B, head_num, anchor_num, embed_size]

        if activate:
            new_queries = tf.nn.elu(new_queries) + 1
            new_keys = tf.nn.elu(new_keys) + 1
            new_values = tf.nn.elu(new_values) + 1
        return new_queries, new_keys, new_values

    def wasserstein_distance(self, mean1, cov1, mean2, cov2):
        # [K, B, head_num, embed_size]
        ret1 = tf.reduce_sum(tf.square(mean1 - mean2), axis=-1)        # [K, B, head_num]
        cov1_sq = tf.sqrt(tf.clip_by_value(cov1, 1e-12, tf.reduce_max(cov1)))
        cov2_sq = tf.sqrt(tf.clip_by_value(cov2, 1e-12, tf.reduce_max(cov2)))

        ret2 = tf.reduce_sum(tf.square(cov1_sq - cov2_sq), axis=-1)        # [K, B, head_num]
        ret = tf.reduce_sum(ret1 + ret2, axis=-1)
        return tf.transpose(ret)       # [B, K]

    def wasserstein_distance_matmul(self, mean1, cov1, mean2, cov2):
        mean1 = tf.expand_dims(mean1, axis=-2)        # [K, B, head_num, 1, embed_size]
        cov1 = tf.expand_dims(cov1, axis=-2)

        ret1 = tf.reduce_sum(tf.square(mean1 - mean2), axis=-1)        # [K, B, head_num, anchor_num]
        cov1_sq = tf.sqrt(tf.clip_by_value(cov1, 1e-12, tf.reduce_max(cov1)))
        cov2_sq = tf.sqrt(tf.clip_by_value(cov2, 1e-12, tf.reduce_max(cov2)))

        ret2 = tf.reduce_sum(tf.square(cov1_sq - cov2_sq), axis=-1)
        return ret1 + ret2

    def combine_dist(self, xv, dist):
        xv_mean = xv[:, :1]
        xv_cov = xv[:, 1:]
        dist_mean = dist[:, :1]
        dist_cov = dist[:, 1:]

        if self.concat_emb:
            print('concat embeddings!')
            xv_mean = xv_mean + tf.zeros_like(dist_mean)
            xv_cov = xv_cov + tf.zeros_like(dist_cov)
            dist_mean = dist_mean + tf.zeros_like(xv_mean)
            dist_cov = dist_cov + tf.zeros_like(xv_cov)
            # aggregate
            combine_mean = tf.concat([xv_mean, dist_mean],  axis=-1)  # [K, B, head_num, embed_size]
            combine_cov = tf.concat([xv_cov, dist_cov], axis=-1)
        else:
            combine_mean = xv_mean + dist_mean
            combine_cov = tf.square(tf.sqrt(tf.nn.relu(xv_cov)) + tf.sqrt(tf.nn.relu(dist_cov)))
        # combine_cov = xv_cov + dist_cov

        return tf.concat([combine_mean, combine_cov], axis=1)

    def compile(self, loss=None, optimizer=None, global_step=None, pos_weight=1.0):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                with tf.name_scope('loss'):
                    if self.wloss in [0, 7]:
                        self.entropy1 = -tf.log(tf.nn.sigmoid(tf.expand_dims(self.outputs1, axis=-1) - self.neg_outputs1))
                    elif self.wloss == 1:
                        self.entropy1 = -tf.expand_dims(self.outputs1, axis=-1) + self.neg_outputs1
                    elif self.wloss == 2:
                        self.entropy1 = tf.nn.softplus(self.neg_outputs1 - tf.expand_dims(self.outputs1, axis=-1))
                    elif self.wloss == 3:
                        self.entropy1 = tf.nn.softplus(-tf.sqrt(tf.abs(self.neg_outputs1)) + tf.sqrt(
                            tf.abs(tf.expand_dims(self.outputs1, axis=-1))))
                    elif self.wloss == 4:
                        self.entropy1 = tf.log(tf.nn.sigmoid(-tf.sqrt(tf.abs(self.neg_outputs1)) + tf.sqrt(
                            tf.abs(tf.expand_dims(self.outputs1, axis=-1)))))
                    elif self.wloss == 5:
                        self.entropy1 = -tf.sqrt(tf.abs(self.neg_outputs1)) + tf.sqrt(tf.abs(tf.expand_dims(self.outputs1, axis=-1)))
                    elif self.wloss == 6:
                        self.entropy1 = tf.nn.relu(self.neg_outputs1 - tf.expand_dims(self.outputs1, axis=-1) + self.m)
                    elif self.wloss == 8:
                        all_outputs = tf.concat([self.outputs1, tf.reshape(self.neg_outputs1, [-1])], axis=-1)
                        all_labels = tf.concat(
                            [tf.ones_like(self.outputs1), tf.zeros_like(tf.reshape(self.neg_outputs1, [-1]))], axis=-1)
                        self.entropy1 = loss(logits=all_outputs, targets=all_labels, pos_weight=pos_weight)
                    elif self.wloss == 9:
                        all_logits = tf.concat([self.outputs1, tf.reshape(self.neg_outputs1, [-1])], axis=-1)
                        all_outputs = 2 * tf.nn.sigmoid(all_logits)
                        all_outputs = tf.clip_by_value(all_outputs, 1e-8, 1-1e-8)
                        all_labels = tf.concat(
                            [tf.ones_like(self.outputs1), tf.zeros_like(tf.reshape(self.neg_outputs1, [-1]))], axis=-1)
                        self.entropy1 = -1 * all_labels * tf.log(all_outputs) - (
                                    1 - all_labels) * tf.log(1 - all_outputs)
                    self.loss1 = tf.reduce_mean(self.entropy1)
                    if self.wloss == 7:
                        all_outputs = tf.concat([self.outputs1, tf.reshape(self.neg_outputs1, [-1])], axis=-1)
                        all_labels = tf.concat(
                            [tf.ones_like(self.outputs1), tf.zeros_like(tf.reshape(self.neg_outputs1, [-1]))], axis=-1)

                        logits0 = tf.expand_dims(all_outputs, 0)
                        logits1 = tf.expand_dims(all_outputs, 1)
                        logits_mat = logits0 - logits1
                        output_mat = tf.sigmoid(logits_mat)
                        max_value = 1 - 1e-5
                        min_value = 1e-5
                        output_mat = tf.clip_by_value(output_mat, min_value, max_value)

                        labels0 = tf.expand_dims(all_labels, 0)
                        labels1 = tf.expand_dims(all_labels, 1)
                        labels_mat = labels0 - labels1
                        s_mat = 0.5 * (labels_mat + 1)

                        loss_mat = -1 * (
                                tf.log(output_mat) * s_mat + tf.log(1 - output_mat) * (1 - s_mat))
                        self.loss1 = (1 - self.pair_a) * self.loss1 + self.pair_a * tf.reduce_mean(loss_mat)


                    if self.wloss in [0, 7]:
                        self.entropy2 = -tf.log(
                            tf.nn.sigmoid(tf.expand_dims(self.outputs2, axis=-1) - self.neg_outputs2))
                    elif self.wloss == 1:
                        self.entropy2 = -tf.expand_dims(self.outputs2, axis=-1) + self.neg_outputs2
                    elif self.wloss == 2:
                        self.entropy2 = tf.nn.softplus(self.neg_outputs2 - tf.expand_dims(self.outputs2, axis=-1))
                    elif self.wloss == 3:
                        self.entropy2 = tf.nn.softplus(-tf.sqrt(tf.abs(self.neg_outputs2)) + tf.sqrt(
                            tf.abs(tf.expand_dims(self.outputs2, axis=-1))))
                    elif self.wloss == 4:
                        self.entropy2 = tf.log(tf.nn.sigmoid(-tf.sqrt(tf.abs(self.neg_outputs2)) + tf.sqrt(
                            tf.abs(tf.expand_dims(self.outputs2, axis=-1)))))
                    elif self.wloss == 5:
                        self.entropy2 = -tf.sqrt(tf.abs(self.neg_outputs2)) + tf.sqrt(
                                            tf.abs(tf.expand_dims(self.outputs2, axis=-1)))
                    elif self.wloss == 6:
                        self.entropy2 = tf.nn.relu(self.neg_outputs2 - tf.expand_dims(self.outputs2, axis=-1) + self.m)
                    elif self.wloss == 8:
                        all_outputs = tf.concat([self.outputs2, tf.reshape(self.neg_outputs2, [-1])], axis=-1)
                        all_labels = tf.concat(
                            [tf.ones_like(self.outputs2), tf.zeros_like(tf.reshape(self.neg_outputs2, [-1]))], axis=-1)
                        self.entropy2 = loss(logits=all_outputs, targets=all_labels, pos_weight=pos_weight)
                    elif self.wloss == 9:
                        all_logits = tf.concat([self.outputs2, tf.reshape(self.neg_outputs2, [-1])], axis=-1)
                        all_outputs = 2 * tf.nn.sigmoid(all_logits)
                        all_outputs = tf.clip_by_value(all_outputs, 1e-8, 1-1e-8)

                        all_labels = tf.concat(
                            [tf.ones_like(self.outputs2), tf.zeros_like(tf.reshape(self.neg_outputs2, [-1]))], axis=-1)
                        self.entropy2 = -1 * all_labels * tf.log(all_outputs) - (
                                    1 - all_labels) * tf.log(1 - all_outputs)

                    self.loss2 = tf.reduce_mean(self.entropy2)

                    if self.wloss == 7:
                        all_outputs = tf.concat([self.outputs2, tf.reshape(self.neg_outputs2, [-1])], axis=-1)
                        all_labels = tf.concat(
                            [tf.ones_like(self.outputs2), tf.zeros_like(tf.reshape(self.neg_outputs2, [-1]))], axis=-1)

                        logits0 = tf.expand_dims(all_outputs, 0)
                        logits1 = tf.expand_dims(all_outputs, 1)
                        logits_mat = logits0 - logits1
                        output_mat = tf.sigmoid(logits_mat)
                        max_value = 1 - 1e-5
                        min_value = 1e-5
                        output_mat = tf.clip_by_value(output_mat, min_value, max_value)

                        labels0 = tf.expand_dims(all_labels, 0)
                        labels1 = tf.expand_dims(all_labels, 1)
                        labels_mat = labels0 - labels1
                        s_mat = 0.5 * (labels_mat + 1)

                        loss_mat = -1 * (
                                tf.log(output_mat) * s_mat + tf.log(1 - output_mat) * (1 - s_mat))
                        self.loss2 = (1 - self.pair_a) * self.loss2 + self.pair_a * tf.reduce_mean(loss_mat)

                    if self.reg_a > 0:
                        self.loss1 += 0.5 * self.reg_a * self.reg
                        self.loss2 += 0.5 * self.reg_a * self.reg

                    _loss_ = self.loss
                    self.optimizer1 = optimizer.minimize(loss=self.loss1,
                                                         global_step=global_step)
                    self.optimizer2 = optimizer.minimize(loss=self.loss2,
                                                         global_step=global_step)
