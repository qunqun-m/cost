from __future__ import division
from __future__ import print_function

import datetime
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, log_loss
from tf_utils import get_optimizer, get_loss, cal_group_auc, from_hdf
import pickle
from sklearn.cluster import KMeans

def mahalanobis_distance_center(x):
    x_mean = np.mean(x, axis=0, keepdims=True)

    _x = x - x_mean     # [user, embed_size]
    cov = np.cov(_x.T)       # [embed_size, embed_size]
    inv_cov = np.linalg.inv(cov)       # [embed_size, embed_size]

    m_dists_mat = np.sqrt(np.dot(np.dot(_x, inv_cov), _x.T))
    return m_dists_mat.diagonal()

def mahalanobis_distance_inter(x, pre_num):
    x = x.astype(np.float32)
    x1 = np.expand_dims(x, axis=1)      # [user, 1, embed]
    x2 = np.expand_dims(x, axis=0)      # [1, user, embed]
    _x = x1 - x2        # [user, user, embed]
    _x = _x[:pre_num, pre_num:]       # [user_src, user_trg, embed]
    x_norm = x - np.mean(x, axis=0, keepdims=True)

    cov = np.cov(x_norm.T)       # [embed_size, embed_size]
    inv_cov = np.linalg.inv(cov)       # [embed_size, embed_size]
    inv_cov = inv_cov.reshape([1, inv_cov.shape[0], inv_cov.shape[1]])       # [1, embed_size, embed_size]

    m_dists_mat = np.sqrt(np.matmul(np.matmul(_x, inv_cov), _x.transpose((0, 2, 1))))       # [user_src, user_trg, user_trg]
    coor_x = range(x.shape[0]-pre_num)
    coor_y = range(x.shape[0]-pre_num)
    return m_dists_mat[:, coor_x, coor_y]       # [user, user]

def fromDict(path):
    print('loading file:', path, end=' ')
    t = time.time()
    data = pickle.loads(open(path,'rb').read())
    print('Time spent:', time.time() - t)
    return data

class Trainer:
    logdir = None
    session = None
    dataset = None
    model = None
    saver = None
    learning_rate = None
    train_pos_ratio = None
    test_pos_ratio = None

    def __init__(self, model=None, train_gen1=None, train_gen2=None, test_dataset_src=None, test_dataset_trg=None, opt='adam',
                 epsilon=1e-8, initial_accumulator_value=1e-8, momentum=0.95, loss='weighted', pos_weight=1.0,
                 n_epoch=1, train_per_epoch=10000, test_per_epoch=10000, early_stop_epoch=5, batch_size1=2000,
                 batch_size2=2000, learning_rate=1e-2, decay_rate=0.95, test_every_epoch=1, save=False,
                 logdir=None, drop_epoch=1, resumedir=None, src_item_pre_sum=None, user_his_len=None, cotrain=False,
                 add_neg_sample_num=0, save_embedding=False, hardness=0, n_clusters=10, src_dataset=None,
                 trg_dataset=None, anchor_num=None, itemid_presum=None, alpha=None, calc_pattern=None,
                 calc_reverse=None):
        self.model = model
        self.train_gen1 = train_gen1
        self.train_gen2 = train_gen2
        self.src_dataset = src_dataset
        self.trg_dataset = trg_dataset
        self.test_dataset_src = test_dataset_src
        self.test_dataset_trg = test_dataset_trg
        optimizer = get_optimizer(opt)
        loss = get_loss(loss)
        self.pos_weight = pos_weight
        self.n_epoch = n_epoch
        self.train_per_epoch = train_per_epoch + 1
        self.early_stop_epoch = early_stop_epoch
        self.test_per_epoch = test_per_epoch
        self.batch_size1 = batch_size1
        self.batch_size2 = batch_size2
        self.cotrain = cotrain
        self._learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.call_auc = roc_auc_score
        self.call_loss = log_loss
        self.test_every_epoch = test_every_epoch
        self.user_his_len = user_his_len
        self.add_neg_sample_num = add_neg_sample_num
        self.save_embedding = save_embedding
        if save_embedding:
            self.saver_emb = tf.train.Saver(max_to_keep=25)
        self.itemid_presum = itemid_presum
        self.alpha = alpha
        self.calc_reverse = calc_reverse

        self.n_clusters = n_clusters
        self.hardness = hardness
        self.anchor_num = anchor_num
        self.calc_pattern = calc_pattern
        if hardness in [25, 26, 27, 28, 29, 31]:
            full_val = 0.5
        else:
            full_val = 1.0
        self.hard_r = np.full(self.src_dataset.feat_sizes[0], full_val)
        self.hard_r2 = np.full(self.src_dataset.feat_sizes[0], full_val)
        self.hard_r3 = np.full(self.src_dataset.feat_sizes[1] + self.trg_dataset.feat_sizes[1], full_val)
        self.hard_r4 = np.full(self.src_dataset.feat_sizes[1] + self.trg_dataset.feat_sizes[1], full_val)
        self.hardness_num = (self.hard_r * self.anchor_num).astype(int)

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.src_item_pre_sum = src_item_pre_sum
        self.learning_rate = tf.placeholder("float")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.logdir = logdir
        self.save = save
        self.drop_epoch = drop_epoch
        self.max_auc = 0.0
        if save:
            self.saver = tf.train.Saver(max_to_keep=25)
            self.saver2 = tf.train.Saver(max_to_keep=25)

        tf.summary.scalar('global_step', self.global_step)
        print(optimizer)
        if opt == 'adam':
            opt = optimizer(learning_rate=self.learning_rate,
                            epsilon=self.epsilon)
        elif opt == 'adagrad':
            opt = optimizer(learning_rate=self.learning_rate,
                            initial_accumulator_value=initial_accumulator_value)
        elif opt == 'moment':
            opt = optimizer(learning_rate=self.learning_rate,
                            momentum=momentum)
        else:
            opt = optimizer(learning_rate=self.learning_rate, )
        self.model.compile(loss=loss, optimizer=opt, global_step=self.global_step, pos_weight=pos_weight)
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        if resumedir and os.path.exists(resumedir):
            print('%s exists.' % resumedir)
            module_file = tf.train.latest_checkpoint(resumedir)
            print('Loading model...' + ' ' + str(module_file))
            self.optimistic_restore(self.session, module_file)

    def optimistic_restore(self, session, save_file):
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
                    saver = tf.train.Saver(restore_vars)
                    saver.restore(session, save_file)

    def _run(self, fetches, feed_dict):
        return self.session.run(fetches=fetches, feed_dict=feed_dict)

    def _train(self, user_id_src, item_id_src, user_history_src, y_src, user_id_trg, item_id_trg, user_history_trg,
               y_trg):
        user_history_src_len = np.sum(user_history_src >= 0, axis=-1)
        user_history_trg_len = np.sum(user_history_trg >= 0, axis=-1)
        feed_dict = {self.model.labels_src: y_src,
                     self.model.labels_trg: y_trg,
                     self.learning_rate: self._learning_rate,
                     self.model.user_id_src: user_id_src,
                     self.model.user_id_trg: user_id_trg,
                     self.model.item_id_src: item_id_src,
                     self.model.item_id_trg: item_id_trg+self.itemid_presum}
        feed_dict[self.model.history1] = user_history_src
        feed_dict[self.model.history_len1] = user_history_src_len
        feed_dict[self.model.history2] = user_history_trg
        feed_dict[self.model.history_len2] = user_history_trg_len

        if hasattr(self.model, 'training'):
            feed_dict[self.model.training] = True

        if self.hardness in [1, 2]:
            feed_dict[self.model.hardness_num] = self.hardness_num

        if self.hardness in [4, 10, 11, 12, 13, 15, 25, 26, 27, 28, 29, 31]:
            feed_dict[self.model.hardness_score_src] = self.hard_r
            feed_dict[self.model.hardness_score_trg] = self.hard_r2
            feed_dict[self.model.hardness_score_src2] = self.hard_r3
            feed_dict[self.model.hardness_score_trg2] = self.hard_r4

        # a, b = self._run([self.model.a, self.model.b], feed_dict)
        # a = np.array(a)
        # b = np.array(b)
        # print(a)
        # print(a.shape)
        # print(b)
        # print(b.shape)
        # print()
        # # assert 0

        _, _, _loss1, _loss2, outputs1, outputs2 = self._run(
            fetches=[self.model.optimizer1, self.model.optimizer2, self.model.loss1, self.model.loss2,
                     self.model.outputs1, self.model.outputs2], feed_dict=feed_dict)

        dis_loss = 0
        dis_loss1 = 0
        dis_loss2 = 0
        if hasattr(self.model, 'd_train_opt') and hasattr(self.model, 'g_train_opt'):
            # print('try')
            if hasattr(self.model, 'dis_loss'):
                _, _, dis_loss = self._run(
                fetches=[self.model.d_train_opt, self.model.g_train_opt, self.model.dis_loss], feed_dict=feed_dict)
            elif hasattr(self.model, 'dis_loss1') and hasattr(self.model, 'dis_loss2'):
                _, _, dis_loss1, dis_loss2 = self._run(
                fetches=[self.model.d_train_opt, self.model.g_train_opt, self.model.dis_loss1, self.model.dis_loss2],
                    feed_dict=feed_dict)

        # if is_src:
        #     _, _loss, outputs = self._run(
        #         fetches=[self.model.optimizer1, self.model.loss1,
        #                  self.model.outputs1], feed_dict=feed_dict)
        # else:
        #     _, _loss, outputs = self._run(
        #         fetches=[self.model.optimizer2, self.model.loss2,
        #                  self.model.outputs2], feed_dict=feed_dict)
        return _loss1, _loss2, outputs1, outputs2, dis_loss, dis_loss1, dis_loss2

    def _predict_ctr(self, user_ids, user_historys, neg_items, part):
        # print(user_ids)
        # print(user_historys)
        # print(neg_items)
        # print()
        is_src = True if part == 'src' else False
        user_history_len = np.sum(user_historys >= 0, axis=-1)
        if is_src:
            feed_dict = {self.model.user_id_src: user_ids,
                         self.model.item_id_src: neg_items}
            feed_dict[self.model.history1] = user_historys
            feed_dict[self.model.history_len1] = user_history_len
        else:
            feed_dict = {self.model.user_id_trg: user_ids,
                         self.model.item_id_trg: neg_items+self.itemid_presum}
            feed_dict[self.model.history2] = user_historys
            feed_dict[self.model.history_len2] = user_history_len


        if hasattr(self.model, 'training'):
            feed_dict[self.model.training] = False

        if self.hardness in [1, 2]:
            feed_dict[self.model.hardness_num] = self.hardness_num

        if self.hardness in [25, 26, 27, 28, 29, 31]:
            feed_dict[self.model.hardness_score_src] = self.hard_r
            feed_dict[self.model.hardness_score_trg] = self.hard_r2
            feed_dict[self.model.hardness_score_src2] = self.hard_r3
            feed_dict[self.model.hardness_score_trg2] = self.hard_r4

        if is_src:
            outputs = self._run(fetches=self.model.outputs1,
                             feed_dict=feed_dict)
        else:
            outputs = self._run(fetches=self.model.outputs2,
                                feed_dict=feed_dict)
        return outputs

    def test_ndcg_hr(self, part):
        def dcg_at_k(reward, k):
            reward = np.asfarray(reward)[:k]
            return np.sum(reward / np.log2(np.arange(2, len(reward) + 2)))
        def ndcg_at_k(reward, k):
            dcg_max = dcg_at_k(sorted(reward, reverse=True), k)
            if not dcg_max:
                return 0
            return dcg_at_k(reward, k) / dcg_max

        tic = time.time()
        domain = 'source' if part == 'src' else 'target'
        print("begin", domain)

        user_item_click_pred = []
        user_item_click_label = []
        config = {'batch_size': 5000, 'shuffle': False}
        if part == 'src':
            test_gen = self.test_dataset_src.batch_generator(config)
        else:
            test_gen = self.test_dataset_trg.batch_generator(config)
        cnt_user = 1
        for batch_data in test_gen:
            user_id, user_history, neg_items = batch_data
            for i in range(user_id.shape[0]):
                # if cnt_user % 200 == 0:
                #     print("user: " + str(cnt_user) + ' ' + "time:%s" % str(
                #         datetime.timedelta(seconds=int(time.time() - tic))))
                user = user_id[i]       # (1,)
                history = user_history[i]       # (50,)
                user_neg_items = neg_items[i]
                user_ids = np.full(user_neg_items.shape[0], user)        # (N=100,)
                user_historys = np.tile(history.reshape([1, -1]), [user_neg_items.shape[0], 1])      # (N, 50)
                batch_pred1 = self._predict_ctr(user_ids, user_historys, user_neg_items, part)      # (N,)
                # print(batch_pred1)
                labels = np.zeros_like(user_neg_items)
                labels[-1] = 1
                cnt_user += 1

                user_item_click_pred.append(batch_pred1)
                user_item_click_label.append(labels)
                # print(labels)
        user_item_click_pred = np.array(user_item_click_pred)  # [usernum, N]
        user_item_click_label = np.array(user_item_click_label)  # [usernum, N]
        print(user_item_click_pred)

        print("each user in %s click item" % domain)
        top_k_num_list = [5, 10, 20, 50]
        # precision = [[] for i in range(10)]
        # recall = [[] for i in range(10)]
        hit_rate = [[] for i in range(len(top_k_num_list)+1)]
        ndcg = [[] for i in range(len(top_k_num_list)+1)]

        for user_index in range(user_item_click_pred.shape[0]):
            preds = user_item_click_pred[user_index]
            labels = user_item_click_label[user_index]
            item_sort = np.argsort(-preds)
            # print(item_sort)
            N = labels.shape[0]
            ground_truth_item = len(item_sort) - 1
            item_sort_reward = (item_sort == ground_truth_item).astype(int)
            # print(item_sort)
            for i in range(len(top_k_num_list)+1):
                if i == len(top_k_num_list):
                    top_k_num = N
                else:
                    top_k_num = top_k_num_list[i]
                top_k_item_reward = item_sort_reward[:top_k_num]
                top_k_right_item_num = np.sum(top_k_item_reward)
                top_k_hit_rate = 0.0
                if top_k_right_item_num > 0:
                    top_k_hit_rate = 1.0

                hit_rate[i].append(top_k_hit_rate)
                ndcg[i].append(ndcg_at_k(top_k_item_reward, top_k_num))

        print("user in %s clicked item HR, NDCG at" % domain, top_k_num_list)
        # for i in range(10):
        #     print(round(sum(precision[i]) / len(precision[i]), 6))
        # print()
        # print("recall:")
        # for i in range(10):
        #     print(round(sum(recall[i]) / len(recall[i]), 6))
        # print()
        print("hit rate:")
        for i in range(len(top_k_num_list)+1):
            print(round(sum(hit_rate[i]) / len(hit_rate[i]), 6))
        print()
        print("NDCG:")
        for i in range(len(top_k_num_list)+1):
            print(round(sum(ndcg[i]) / len(ndcg[i]), 6))
        print()

    def _save(self):
        print('max saving checkpoint... '+os.path.join(self.logdir, 'checkpoints', 'model.ckpt')+' '+
              str(self.global_step.eval(self.session)))
        if not os.path.exists(os.path.join(self.logdir, 'checkpoints')):
            os.makedirs(os.path.join(self.logdir, 'checkpoints'))
        self.saver.save(self.session, os.path.join(self.logdir, 'checkpoints', 'model.ckpt'),
                        self.global_step.eval(self.session))

    def _last_save(self):
        print('last saving checkpoint... '+os.path.join(self.logdir, 'checkpoints2', 'model2.ckpt')+' '+
              str(self.global_step.eval(self.session)))
        if not os.path.exists(os.path.join(self.logdir, 'checkpoints2')):
            os.makedirs(os.path.join(self.logdir, 'checkpoints2'))
        self.saver2.save(self.session, os.path.join(self.logdir, 'checkpoints2', 'model2.ckpt'),
                         self.global_step.eval(self.session))

    def saveEmb(self):
        save_path = os.path.join(self.logdir, 'embedding.ckpt')
        print('saving embeddings... '+save_path+' ' + str(self.global_step.eval(self.session)))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.saver_emb.save(self.session, save_path, self.global_step.eval(self.session))

    def calc_hardness(self, user_src=None, user_trg=None):
        print('Begin calculate hardness....')
        if self.calc_pattern == 1:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.xv1_src_origin, self.model.xv1_trg_origin],
                                                 feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)
            dataset_ratio = len(user_id_src) / len(user_id_trg)

            user_src_num = len(user_id_src)
            embedding_matrix = np.concatenate([user_embed_src, user_embed_trg], axis=0)
            # 111 - cluster method
            kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=0).fit(embedding_matrix)

            cluster_labels = kmeans.labels_
            src_labels = cluster_labels[:user_src_num]
            trg_labels = cluster_labels[user_src_num:]

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)

            for i in range(self.n_clusters):
                keep_id1 = np.where(src_labels == i)[0]
                keep_id2 = np.where(trg_labels == i)[0]
                l1 = len(keep_id1)
                l2 = len(keep_id2)
                r = max(l1, 1) / max(l2, 1)
                cluster_r = dataset_ratio / r

                id_src = user_id_src[keep_id1]
                id_trg = user_id_trg[keep_id2]
                if cluster_r > 1:
                    cluster_r = 1 / cluster_r
                print(i, cluster_r)
                # 222 - [cluster_r] .... [1 - cluster_r]
                score_src[id_src] = cluster_r
                score_trg[id_trg] = cluster_r

            overlap_id = np.where(np.logical_and(score_src > 0, score_trg > 0))[0]
            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            scores[overlap_id] = 1.0
            self.hard_r = self.alpha * scores + (1 - self.alpha) * self.hard_r
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
        elif self.calc_pattern == 2:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.xv1_src_origin, self.model.xv1_trg_origin],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)
            dataset_ratio = len(user_id_src) / len(user_id_trg)

            user_src_num = len(user_id_src)
            embedding_matrix = np.concatenate([user_embed_src, user_embed_trg], axis=0)
            # 111 - cluster method
            kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=0).fit(embedding_matrix)

            cluster_labels = kmeans.labels_
            src_labels = cluster_labels[:user_src_num]
            trg_labels = cluster_labels[user_src_num:]

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)

            for i in range(self.n_clusters):
                keep_id1 = np.where(src_labels == i)[0]
                keep_id2 = np.where(trg_labels == i)[0]
                l1 = len(keep_id1)
                l2 = len(keep_id2)
                r = max(l1, 1) / max(l2, 1)
                cluster_r = dataset_ratio / r

                id_src = user_id_src[keep_id1]
                id_trg = user_id_trg[keep_id2]
                if cluster_r > 1:
                    cluster_r = 1 / cluster_r
                print(i, cluster_r)
                # 222 - [cluster_r] .... [1 - cluster_r]
                score_src[id_src] = 1 - cluster_r
                score_trg[id_trg] = 1 - cluster_r

            overlap_id = np.where(np.logical_and(score_src > 0, score_trg > 0))[0]
            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            scores[overlap_id] = 1.0
            self.hard_r = self.alpha * scores + (1 - self.alpha) * self.hard_r
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
        elif self.calc_pattern == 3:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                 feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)
            dataset_ratio = len(user_id_src) / len(user_id_trg)

            user_src_num = len(user_id_src)
            embedding_matrix = np.concatenate([user_embed_src, user_embed_trg], axis=0)
            # 111 - cluster method
            kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=0).fit(embedding_matrix)

            cluster_labels = kmeans.labels_
            src_labels = cluster_labels[:user_src_num]
            trg_labels = cluster_labels[user_src_num:]

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)

            for i in range(self.n_clusters):
                keep_id1 = np.where(src_labels == i)[0]
                keep_id2 = np.where(trg_labels == i)[0]
                l1 = len(keep_id1)
                l2 = len(keep_id2)
                r = max(l1, 1) / max(l2, 1)
                cluster_r = dataset_ratio / r

                id_src = user_id_src[keep_id1]
                id_trg = user_id_trg[keep_id2]
                if cluster_r > 1:
                    cluster_r = 1 / cluster_r
                print(i, cluster_r)
                # 222 - [cluster_r] .... [1 - cluster_r]
                score_src[id_src] = cluster_r
                score_trg[id_trg] = cluster_r

            overlap_id = np.where(np.logical_and(score_src > 0, score_trg > 0))[0]
            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            scores[overlap_id] = 1.0
            self.hard_r = self.alpha * scores + (1 - self.alpha) * self.hard_r
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
        elif self.calc_pattern == 4:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)
            dataset_ratio = len(user_id_src) / len(user_id_trg)

            user_src_num = len(user_id_src)
            embedding_matrix = np.concatenate([user_embed_src, user_embed_trg], axis=0)
            # 111 - cluster method
            kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=0).fit(embedding_matrix)

            cluster_labels = kmeans.labels_
            src_labels = cluster_labels[:user_src_num]
            trg_labels = cluster_labels[user_src_num:]

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)

            for i in range(self.n_clusters):
                keep_id1 = np.where(src_labels == i)[0]
                keep_id2 = np.where(trg_labels == i)[0]
                l1 = len(keep_id1)
                l2 = len(keep_id2)
                r = max(l1, 1) / max(l2, 1)
                cluster_r = dataset_ratio / r

                id_src = user_id_src[keep_id1]
                id_trg = user_id_trg[keep_id2]
                if cluster_r > 1:
                    cluster_r = 1 / cluster_r
                print(i, cluster_r)
                # 222 - [cluster_r] .... [1 - cluster_r]
                score_src[id_src] = 1 - cluster_r
                score_trg[id_trg] = 1 - cluster_r

            overlap_id = np.where(np.logical_and(score_src > 0, score_trg > 0))[0]
            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            scores[overlap_id] = 1.0
            self.hard_r = self.alpha * scores + (1 - self.alpha) * self.hard_r
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
        elif self.calc_pattern == 5:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)
            dataset_ratio = len(user_id_src) / len(user_id_trg)

            user_src_num = len(user_id_src)
            embedding_matrix = np.concatenate([user_embed_src, user_embed_trg], axis=0)
            # 111 - cluster method
            kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=0).fit(embedding_matrix)

            cluster_labels = kmeans.labels_
            src_labels = cluster_labels[:user_src_num]
            trg_labels = cluster_labels[user_src_num:]

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)

            for i in range(self.n_clusters):
                keep_id1 = np.where(src_labels == i)[0]
                keep_id2 = np.where(trg_labels == i)[0]
                l1 = len(keep_id1)
                l2 = len(keep_id2)
                r = max(l1, 1) / max(l2, 1)
                cluster_r = dataset_ratio / r

                id_src = user_id_src[keep_id1]
                id_trg = user_id_trg[keep_id2]
                if cluster_r > 1:
                    cluster_r = 1 / cluster_r
                print(i, cluster_r)
                # 222 - [cluster_r] .... [1 - cluster_r]
                score_src[id_src] = cluster_r
                score_trg[id_trg] = cluster_r

            overlap_id = np.where(np.logical_and(score_src > 0, score_trg > 0))[0]
            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            scores[overlap_id] = scores[overlap_id] * 0.5
            self.hard_r = self.alpha * scores + (1 - self.alpha) * self.hard_r
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
        elif self.calc_pattern == 6:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat = np.matmul(user_embed_src, user_embed_trg)
            mat_score_src = np.mean(user_score_mat, axis=1)
            mat_score_trg = np.mean(user_score_mat, axis=0)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = mat_score_src
            score_trg[user_id_trg] = mat_score_trg

            overlap_id = np.where(np.logical_and(score_src > 0, score_trg > 0))[0]
            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            scores[overlap_id] = scores[overlap_id] * 0.5
            self.hard_r = self.alpha * scores + (1 - self.alpha) * self.hard_r
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
        elif self.calc_pattern == 7:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat = np.matmul(user_embed_src, user_embed_trg.T)
            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([1, -1])
            user_len_mat = np.matmul(user_len_src, user_len_trg)
            user_score_mat = user_score_mat / user_len_mat
            user_score_mat = (user_score_mat + 1) * 0.5
            mat_score_src = np.mean(user_score_mat, axis=1)
            mat_score_trg = np.mean(user_score_mat, axis=0)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = mat_score_src
            score_trg[user_id_trg] = mat_score_trg

            overlap_id = np.where(np.logical_and(score_src > 0, score_trg > 0))[0]
            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            scores[overlap_id] = scores[overlap_id] * 0.5
            self.hard_r = self.alpha * scores + (1 - self.alpha) * self.hard_r
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
        elif self.calc_pattern == 8:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat = np.matmul(user_embed_src, user_embed_trg.T)
            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([1, -1])
            user_len_mat = np.matmul(user_len_src, user_len_trg)
            user_score_mat = user_score_mat / user_len_mat
            user_score_mat = (user_score_mat + 1) * 0.5
            user_score_mat = 1 - user_score_mat
            mat_score_src = np.mean(user_score_mat, axis=1)
            mat_score_trg = np.mean(user_score_mat, axis=0)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = mat_score_src
            score_trg[user_id_trg] = mat_score_trg

            overlap_id = np.where(np.logical_and(score_src > 0, score_trg > 0))[0]
            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            scores[overlap_id] = scores[overlap_id] * 0.5
            self.hard_r = self.alpha * scores + (1 - self.alpha) * self.hard_r
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(scores))
        elif self.calc_pattern == 9:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat = np.matmul(user_embed_src, user_embed_trg.T)
            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([1, -1])
            user_len_mat = np.matmul(user_len_src, user_len_trg)
            user_score_mat = user_score_mat / user_len_mat
            user_score_mat = (user_score_mat + 1) * 0.5
            mat_score_src = np.mean(user_score_mat, axis=1)
            mat_score_trg = np.mean(user_score_mat, axis=0)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = mat_score_src
            score_trg[user_id_trg] = mat_score_trg

            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(mat_score_src))
            print(np.mean(mat_score_trg))
        elif self.calc_pattern == 10:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat = np.matmul(user_embed_src, user_embed_trg.T)
            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([1, -1])
            user_len_mat = np.matmul(user_len_src, user_len_trg)
            user_score_mat = user_score_mat / user_len_mat
            user_score_mat = (user_score_mat + 1) * 0.5
            user_score_mat = 1 - user_score_mat
            mat_score_src = np.mean(user_score_mat, axis=1)
            mat_score_trg = np.mean(user_score_mat, axis=0)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = mat_score_src
            score_trg[user_id_trg] = mat_score_trg

            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(mat_score_src))
            print(np.mean(mat_score_trg))
        elif self.calc_pattern == 11:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat = np.matmul(user_embed_src, user_embed_trg.T)
            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([1, -1])
            user_len_mat = np.matmul(user_len_src, user_len_trg)
            user_score_mat = user_score_mat / user_len_mat
            user_score_mat = (user_score_mat + 1) * 0.5

            if self.calc_reverse in [0, 1]:
                mat_score_src = np.mean(user_score_mat / np.max(user_score_mat, axis=1, keepdims=True), axis=1)
                mat_score_trg = np.mean(user_score_mat / np.max(user_score_mat, axis=0, keepdims=True), axis=0)
                if self.calc_reverse == 1:
                    mat_score_src = 1 - mat_score_src
                    mat_score_trg = 1 - mat_score_trg

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = mat_score_src
            score_trg[user_id_trg] = mat_score_trg

            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(mat_score_src))
            print(np.mean(mat_score_trg))
        elif self.calc_pattern == 12:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat = np.matmul(user_embed_src, user_embed_trg.T)
            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([1, -1])
            user_len_mat = np.matmul(user_len_src, user_len_trg)
            user_score_mat = user_score_mat / user_len_mat
            user_score_mat = (user_score_mat + 1) * 0.5
            user_score_mat = user_score_mat / np.max(user_score_mat)
            user_score_mat = 1 - user_score_mat
            mat_score_src = np.mean(user_score_mat, axis=1)
            mat_score_trg = np.mean(user_score_mat, axis=0)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = mat_score_src
            score_trg[user_id_trg] = mat_score_trg

            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(mat_score_src))
            print(np.mean(mat_score_trg))
        elif self.calc_pattern in [12.5, 15]:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat = np.matmul(user_embed_src, user_embed_trg.T)
            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([1, -1])
            user_len_mat = np.matmul(user_len_src, user_len_trg)
            user_score_mat = user_score_mat / user_len_mat
            user_score_mat = (user_score_mat + 1) * 0.5
            user_score_mat = 1 - user_score_mat
            mat_score_src = np.mean(user_score_mat, axis=1)
            mat_score_trg = np.mean(user_score_mat, axis=0)
            mat_score_src = mat_score_src / np.max(mat_score_src)
            mat_score_trg = mat_score_trg / np.max(mat_score_trg)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = mat_score_src
            score_trg[user_id_trg] = mat_score_trg

            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(mat_score_src))
            print(np.mean(mat_score_trg))
        elif self.calc_pattern in [13, 16]:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat = np.matmul(user_embed_src, user_embed_trg.T)
            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([1, -1])
            user_len_mat = np.matmul(user_len_src, user_len_trg)
            user_score_mat = user_score_mat / user_len_mat
            user_score_mat = (user_score_mat + 1) * 0.5
            mat_score_src = np.mean(user_score_mat, axis=1)
            mat_score_trg = np.mean(user_score_mat, axis=0)
            mat_score_src = mat_score_src / np.max(mat_score_src)
            mat_score_trg = mat_score_trg / np.max(mat_score_trg)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = mat_score_src
            score_trg[user_id_trg] = mat_score_trg

            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(mat_score_src))
            print(np.mean(mat_score_trg))
        elif self.calc_pattern in [17]:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat = np.matmul(user_embed_src, user_embed_trg.T)
            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([1, -1])
            user_len_mat = np.matmul(user_len_src, user_len_trg)
            user_score_mat = user_score_mat / user_len_mat
            user_score_mat = (user_score_mat + 1) * 0.5
            user_score_mat = 1 - user_score_mat
            mat_score_src = np.mean(user_score_mat, axis=1)
            mat_score_trg = np.mean(user_score_mat, axis=0)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = mat_score_src
            score_trg[user_id_trg] = mat_score_trg

            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(mat_score_src))
            print(np.mean(mat_score_trg))
        elif self.calc_pattern in [18]:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat = np.matmul(user_embed_src, user_embed_trg.T)
            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([1, -1])
            user_len_mat = np.matmul(user_len_src, user_len_trg)
            user_score_mat = user_score_mat / user_len_mat
            user_score_mat = (user_score_mat + 1) * 0.5
            mat_score_src = np.mean(user_score_mat, axis=1)
            mat_score_trg = np.mean(user_score_mat, axis=0)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = mat_score_src
            score_trg[user_id_trg] = mat_score_trg

            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(mat_score_src))
            print(np.mean(mat_score_trg))
        elif self.calc_pattern == 14:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat = np.matmul(user_embed_src, user_embed_trg.T)
            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([1, -1])
            user_len_mat = np.matmul(user_len_src, user_len_trg)
            user_score_mat = user_score_mat / user_len_mat
            user_score_mat = (user_score_mat + 1) * 0.5
            mat_score_src = np.mean(user_score_mat / np.max(user_score_mat, axis=1, keepdims=True), axis=1)
            mat_score_trg = np.mean(user_score_mat / np.max(user_score_mat, axis=0, keepdims=True), axis=0)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = mat_score_src
            score_trg[user_id_trg] = mat_score_trg

            scores = score_src + score_trg
            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(mat_score_src))
            print(np.mean(mat_score_trg))
        elif self.calc_pattern == 19:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat_src = np.matmul(user_embed_src, user_embed_src.T)
            user_score_mat_trg = np.matmul(user_embed_trg, user_embed_trg.T)


            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([-1, 1])
            user_len_mat_src = np.matmul(user_len_src, user_len_src.T)
            user_len_mat_trg = np.matmul(user_len_trg, user_len_trg.T)

            user_score_mat_src = user_score_mat_src / user_len_mat_src
            user_score_mat_trg = user_score_mat_trg / user_len_mat_trg
            user_score_mat_src = (user_score_mat_src + 1) * 0.5
            user_score_mat_trg = (user_score_mat_trg + 1) * 0.5
            user_score_sim_src = np.mean(user_score_mat_src, axis=-1)
            user_score_sim_trg = np.mean(user_score_mat_trg, axis=-1)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = user_score_sim_src
            score_trg[user_id_trg] = user_score_sim_trg

            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(user_score_sim_src))
            print(np.mean(user_score_sim_trg))
        elif self.calc_pattern == 20:
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_mat_src = np.matmul(user_embed_src, user_embed_src.T)
            user_score_mat_trg = np.matmul(user_embed_trg, user_embed_trg.T)


            user_len_src = np.sqrt(np.sum(user_embed_src ** 2, axis=-1)).reshape([-1, 1])
            user_len_trg = np.sqrt(np.sum(user_embed_trg ** 2, axis=-1)).reshape([-1, 1])
            user_len_mat_src = np.matmul(user_len_src, user_len_src.T)
            user_len_mat_trg = np.matmul(user_len_trg, user_len_trg.T)

            user_score_mat_src = user_score_mat_src / user_len_mat_src
            user_score_mat_trg = user_score_mat_trg / user_len_mat_trg
            user_score_mat_src = (user_score_mat_src + 1) * 0.5
            user_score_mat_trg = (user_score_mat_trg + 1) * 0.5
            user_score_sim_src = np.mean(1 - user_score_mat_src, axis=-1)
            user_score_sim_trg = np.mean(1 - user_score_mat_trg, axis=-1)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = user_score_sim_src
            score_trg[user_id_trg] = user_score_sim_trg

            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(user_score_sim_src))
            print(np.mean(user_score_sim_trg))
        elif self.calc_pattern == 21:
            '''
                马氏距离，只有user，每个域单独计算马氏距离的均值
            '''
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_sim_src = mahalanobis_distance_center(user_embed_src)
            user_score_sim_trg = mahalanobis_distance_center(user_embed_trg)
            if self.calc_reverse in [0, 1]:
                user_score_sim_src = user_score_sim_src / np.max(user_score_sim_src)
                user_score_sim_trg = user_score_sim_trg / np.max(user_score_sim_trg)
                if self.calc_reverse == 1:
                    user_score_sim_src = 1 - user_score_sim_src
                    user_score_sim_trg = 1 - user_score_sim_trg
            elif self.calc_reverse == 2:
                user_score_sim_src = np.exp(-user_score_sim_src)
                user_score_sim_trg = np.exp(-user_score_sim_trg)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = user_score_sim_src
            score_trg[user_id_trg] = user_score_sim_trg
            print(user_score_sim_src)
            print(user_score_sim_trg)

            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(user_score_sim_src))
            print(np.mean(user_score_sim_trg))
        elif self.calc_pattern == 22:
            '''
                马氏距离，只有user，两个域一起计算均值
            '''
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_embed_mat = np.concatenate([user_embed_src, user_embed_trg], axis=0)
            user_src_num = user_embed_src.shape[0]

            user_score_sim = mahalanobis_distance_center(user_embed_mat)
            user_score_sim_src = user_score_sim[:user_src_num]
            user_score_sim_trg = user_score_sim[user_src_num:]
            if self.calc_reverse in [0, 1]:
                user_score_sim_src = user_score_sim_src / np.max(user_score_sim_src)
                user_score_sim_trg = user_score_sim_trg / np.max(user_score_sim_trg)
                if self.calc_reverse == 1:
                    user_score_sim_src = 1 - user_score_sim_src
                    user_score_sim_trg = 1 - user_score_sim_trg
            elif self.calc_reverse == 2:
                user_score_sim_src = np.exp(-user_score_sim_src)
                user_score_sim_trg = np.exp(-user_score_sim_trg)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = user_score_sim_src
            score_trg[user_id_trg] = user_score_sim_trg
            print(user_score_sim_src)
            print(user_score_sim_trg)

            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(user_score_sim_src))
            print(np.mean(user_score_sim_trg))
        elif self.calc_pattern == 23:
            '''
                21: 加入item部份的distance
            '''
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            item_id_trg = item_id_trg + self.itemid_presum
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.item_id_src: item_id_src,
                self.model.item_id_trg: item_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg, item_embed_src, item_embed_trg = self._run(fetches=[self.model.ae_xv1_src,
                                                                self.model.ae_xv1_trg, self.model.ae_xv2_src,
                                                                self.model.ae_xv2_trg], feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)
            item_embed_src, item_embed_trg = np.array(item_embed_src), np.array(item_embed_trg)

            user_score_sim_src = mahalanobis_distance_center(user_embed_src)
            user_score_sim_trg = mahalanobis_distance_center(user_embed_trg)
            item_score_sim_src = mahalanobis_distance_center(item_embed_src)
            item_score_sim_trg = mahalanobis_distance_center(item_embed_trg)
            if self.calc_reverse in [0, 1]:
                user_score_sim_src = user_score_sim_src / np.max(user_score_sim_src)
                user_score_sim_trg = user_score_sim_trg / np.max(user_score_sim_trg)
                item_score_sim_src = item_score_sim_src / np.max(item_score_sim_src)
                item_score_sim_trg = item_score_sim_trg / np.max(item_score_sim_trg)
                if self.calc_reverse == 1:
                    user_score_sim_src = 1 - user_score_sim_src
                    user_score_sim_trg = 1 - user_score_sim_trg
                    item_score_sim_src = 1 - item_score_sim_src
                    item_score_sim_trg = 1 - item_score_sim_trg
            elif self.calc_reverse == 2:
                user_score_sim_src = np.exp(-user_score_sim_src)
                user_score_sim_trg = np.exp(-user_score_sim_trg)
                item_score_sim_src = np.exp(-item_score_sim_src)
                item_score_sim_trg = np.exp(-item_score_sim_trg)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = user_score_sim_src
            score_trg[user_id_trg] = user_score_sim_trg

            item_max_id = self.src_dataset.feat_sizes[1] + self.trg_dataset.feat_sizes[1]
            score_src2 = np.zeros(item_max_id)
            score_trg2 = np.zeros(item_max_id)
            score_src2[item_id_src] = item_score_sim_src
            score_trg2[item_id_trg] = item_score_sim_trg
            print(user_score_sim_src)
            print(user_score_sim_trg)
            print(item_score_sim_src)
            print(item_score_sim_trg)

            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hard_r3 = self.alpha * score_src2 + (1 - self.alpha) * self.hard_r3
            self.hard_r4 = self.alpha * score_trg2 + (1 - self.alpha) * self.hard_r4
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(user_score_sim_src))
            print(np.mean(user_score_sim_trg))
            print(np.mean(item_score_sim_src))
            print(np.mean(item_score_sim_trg))
        elif self.calc_pattern == 24:
            '''
                21: 加入item部份的distance
            '''
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            item_id_trg = item_id_trg + self.itemid_presum
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.item_id_src: item_id_src,
                self.model.item_id_trg: item_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg, item_embed_src, item_embed_trg = self._run(fetches=[self.model.xv1_src_origin,
                                                                self.model.xv1_trg_origin, self.model.xv2_src_origin,
                                                                self.model.xv2_trg_origin], feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)
            item_embed_src, item_embed_trg = np.array(item_embed_src), np.array(item_embed_trg)

            user_score_sim_src = mahalanobis_distance_center(user_embed_src)
            user_score_sim_trg = mahalanobis_distance_center(user_embed_trg)
            item_score_sim_src = mahalanobis_distance_center(item_embed_src)
            item_score_sim_trg = mahalanobis_distance_center(item_embed_trg)
            if self.calc_reverse in [0, 1]:
                user_score_sim_src = user_score_sim_src / np.max(user_score_sim_src)
                user_score_sim_trg = user_score_sim_trg / np.max(user_score_sim_trg)
                item_score_sim_src = item_score_sim_src / np.max(item_score_sim_src)
                item_score_sim_trg = item_score_sim_trg / np.max(item_score_sim_trg)
                if self.calc_reverse == 1:
                    user_score_sim_src = 1 - user_score_sim_src
                    user_score_sim_trg = 1 - user_score_sim_trg
                    item_score_sim_src = 1 - item_score_sim_src
                    item_score_sim_trg = 1 - item_score_sim_trg
            elif self.calc_reverse == 2:
                user_score_sim_src = np.exp(-user_score_sim_src)
                user_score_sim_trg = np.exp(-user_score_sim_trg)
                item_score_sim_src = np.exp(-item_score_sim_src)
                item_score_sim_trg = np.exp(-item_score_sim_trg)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = user_score_sim_src
            score_trg[user_id_trg] = user_score_sim_trg

            item_max_id = self.src_dataset.feat_sizes[1] + self.trg_dataset.feat_sizes[1]
            score_src2 = np.zeros(item_max_id)
            score_trg2 = np.zeros(item_max_id)
            score_src2[item_id_src] = item_score_sim_src
            score_trg2[item_id_trg] = item_score_sim_trg
            print(user_score_sim_src)
            print(user_score_sim_trg)
            print(item_score_sim_src)
            print(item_score_sim_trg)

            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hard_r3 = self.alpha * score_src2 + (1 - self.alpha) * self.hard_r3
            self.hard_r4 = self.alpha * score_trg2 + (1 - self.alpha) * self.hard_r4
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(user_score_sim_src))
            print(np.mean(user_score_sim_trg))
            print(np.mean(item_score_sim_src))
            print(np.mean(item_score_sim_trg))
        elif self.calc_pattern == 25:
            '''
                22: 加入item部份的distance
            '''
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            item_id_trg = item_id_trg + self.itemid_presum
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.item_id_src: item_id_src,
                self.model.item_id_trg: item_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg, item_embed_src, item_embed_trg = self._run(fetches=[self.model.ae_xv1_src,
                                                                                                self.model.ae_xv1_trg,
                                                                                                self.model.ae_xv2_src,
                                                                                                self.model.ae_xv2_trg],
                                                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)
            item_embed_src, item_embed_trg = np.array(item_embed_src), np.array(item_embed_trg)

            user_embed_mat = np.concatenate([user_embed_src, user_embed_trg], axis=0)
            user_src_num = user_embed_src.shape[0]
            user_score_sim_mat = mahalanobis_distance_inter(user_embed_mat, user_src_num)     # [user, user]
            user_score_sim_src = np.mean(user_score_sim_mat, axis=1)
            user_score_sim_trg = np.mean(user_score_sim_mat, axis=0)

            item_embed_mat = np.concatenate([item_embed_src, item_embed_trg], axis=0)
            item_src_num = item_embed_src.shape[0]
            item_score_sim_mat = mahalanobis_distance_inter(item_embed_mat, item_src_num)  # [item, item]
            item_score_sim_src = np.mean(item_score_sim_mat, axis=1)
            item_score_sim_trg = np.mean(item_score_sim_mat, axis=0)
            if self.calc_reverse in [0, 1]:
                user_score_sim_src = user_score_sim_src / np.max(user_score_sim_src)
                user_score_sim_trg = user_score_sim_trg / np.max(user_score_sim_trg)
                item_score_sim_src = item_score_sim_src / np.max(item_score_sim_src)
                item_score_sim_trg = item_score_sim_trg / np.max(item_score_sim_trg)
                if self.calc_reverse == 1:
                    user_score_sim_src = 1 - user_score_sim_src
                    user_score_sim_trg = 1 - user_score_sim_trg
                    item_score_sim_src = 1 - item_score_sim_src
                    item_score_sim_trg = 1 - item_score_sim_trg
            elif self.calc_reverse == 2:
                user_score_sim_src = np.exp(-user_score_sim_src)
                user_score_sim_trg = np.exp(-user_score_sim_trg)
                item_score_sim_src = np.exp(-item_score_sim_src)
                item_score_sim_trg = np.exp(-item_score_sim_trg)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = user_score_sim_src
            score_trg[user_id_trg] = user_score_sim_trg

            item_max_id = self.src_dataset.feat_sizes[1] + self.trg_dataset.feat_sizes[1]
            score_src2 = np.zeros(item_max_id)
            score_trg2 = np.zeros(item_max_id)
            score_src2[item_id_src] = item_score_sim_src
            score_trg2[item_id_trg] = item_score_sim_trg
            print(user_score_sim_src)
            print(user_score_sim_trg)
            print(item_score_sim_src)
            print(item_score_sim_trg)

            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hard_r3 = self.alpha * score_src2 + (1 - self.alpha) * self.hard_r3
            self.hard_r4 = self.alpha * score_trg2 + (1 - self.alpha) * self.hard_r4
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(user_score_sim_src))
            print(np.mean(user_score_sim_trg))
            print(np.mean(item_score_sim_src))
            print(np.mean(item_score_sim_trg))
        elif self.calc_pattern == 26:
            '''
                22: 加入item部份的distance
            '''
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.ae_xv1_src, self.model.ae_xv1_trg],
                                                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_embed_mat = np.concatenate([user_embed_src, user_embed_trg], axis=0)
            user_src_num = user_embed_src.shape[0]
            user_score_sim_mat = mahalanobis_distance_inter(user_embed_mat, user_src_num)     # [user, user]
            user_score_sim_src = np.mean(user_score_sim_mat, axis=1)
            user_score_sim_trg = np.mean(user_score_sim_mat, axis=0)

            if self.calc_reverse in [0, 1]:
                user_score_sim_src = user_score_sim_src / np.max(user_score_sim_src)
                user_score_sim_trg = user_score_sim_trg / np.max(user_score_sim_trg)
                if self.calc_reverse == 1:
                    user_score_sim_src = 1 - user_score_sim_src
                    user_score_sim_trg = 1 - user_score_sim_trg
            elif self.calc_reverse == 2:
                user_score_sim_src = np.exp(-user_score_sim_src)
                user_score_sim_trg = np.exp(-user_score_sim_trg)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = user_score_sim_src
            score_trg[user_id_trg] = user_score_sim_trg

            print(user_score_sim_src)
            print(user_score_sim_trg)

            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(user_score_sim_src))
            print(np.mean(user_score_sim_trg))
        elif self.calc_pattern == 27:
            '''
                马氏距离，只有user，每个域单独计算马氏距离的均值，且使用origin的embedding
            '''
            user_id_src, item_id_src = self.src_dataset.get_whole_data()
            user_id_trg, item_id_trg = self.trg_dataset.get_whole_data()
            feed_dict = {
                self.model.user_id_src: user_id_src,
                self.model.user_id_trg: user_id_trg,
                self.model.training: False,
            }
            user_embed_src, user_embed_trg = self._run(fetches=[self.model.xv1_src_origin, self.model.xv1_trg_origin],
                                                       feed_dict=feed_dict)

            user_embed_src, user_embed_trg = np.array(user_embed_src), np.array(user_embed_trg)

            user_score_sim_src = mahalanobis_distance_center(user_embed_src)
            user_score_sim_trg = mahalanobis_distance_center(user_embed_trg)
            if self.calc_reverse in [0, 1]:
                user_score_sim_src = user_score_sim_src / np.max(user_score_sim_src)
                user_score_sim_trg = user_score_sim_trg / np.max(user_score_sim_trg)
                if self.calc_reverse == 1:
                    user_score_sim_src = 1 - user_score_sim_src
                    user_score_sim_trg = 1 - user_score_sim_trg
            elif self.calc_reverse == 2:
                user_score_sim_src = np.exp(-user_score_sim_src)
                user_score_sim_trg = np.exp(-user_score_sim_trg)

            user_max_id = self.src_dataset.feat_sizes[0]
            score_src = np.zeros(user_max_id)
            score_trg = np.zeros(user_max_id)
            score_src[user_id_src] = user_score_sim_src
            score_trg[user_id_trg] = user_score_sim_trg
            print(user_score_sim_src)
            print(user_score_sim_trg)

            # 222 - [cluster_r] .... [1 - cluster_r]
            self.hard_r = self.alpha * score_src + (1 - self.alpha) * self.hard_r
            self.hard_r2 = self.alpha * score_trg + (1 - self.alpha) * self.hard_r2
            self.hardness_num = (self.hard_r * self.anchor_num).astype(int)
            print(np.mean(user_score_sim_src))
            print(np.mean(user_score_sim_trg))


    def fit(self):
        num_of_batches = int(
            np.ceil(self.train_per_epoch / (self.batch_size1)))
        print('self.train_per_epoch', self.train_per_epoch)
        print('self.batch_size1', self.batch_size1)
        print('num_of_batches', num_of_batches)
        total_batches = self.n_epoch * num_of_batches
        print('total batches: %d\tbatch per epoch: %d' % (
        total_batches, num_of_batches))
        start_time = time.time()
        epoch = 1
        finished_batches = 0
        # self.calc_hardness()

        # self.test_ndcg_hr('src')
        # self.test_ndcg_hr('trg')

        # source
        avg_loss1 = 0
        label_list1 = []
        pred_list1 = []

        # target
        avg_loss2 = 0
        label_list2 = []
        pred_list2 = []

        avg_dis_loss = 0
        avg_dis_loss1 = 0
        avg_dis_loss2 = 0

        test_every_epoch = self.test_every_epoch

        train_gen1 = iter(self.train_gen1)
        train_gen2 = iter(self.train_gen2)

        while epoch <= self.n_epoch:
            print('======new iteration======')
            epoch_batches = 0

            while True:
                try:
                    batch_data = next(train_gen1)
                except:
                    train_gen1 = iter(self.train_gen1)
                    break
                user_id1, item_id1, user_history1, y1 = batch_data

                try:
                    batch_data = next(train_gen2)
                except:
                    train_gen2 = iter(self.train_gen2)
                    batch_data = next(train_gen2)
                user_id2, item_id2, user_history2, y2 = batch_data
                batch_loss1, batch_loss2, batch_pred1, batch_pred2, batch_dis_loss,\
                    batch_dis_loss1, batch_dis_loss2 = self._train(user_id1, item_id1, user_history1, y1, user_id2,
                                                                  item_id2, user_history2, y2)
                # print(batch_loss1, '\n', batch_loss2, '\n', batch_pred1, '\n', batch_pred2)
                # assert 0

                # source
                label_list1.append(y1)
                pred_list1.append(batch_pred1)
                avg_loss1 += batch_loss1

                # target
                label_list2.append(y2)
                pred_list2.append(batch_pred2)
                avg_loss2 += batch_loss2

                avg_dis_loss += batch_dis_loss
                avg_dis_loss1 += batch_dis_loss1
                avg_dis_loss2 += batch_dis_loss2

                finished_batches += 1
                epoch_batches += 1
                # print(epoch_batches)

                epoch_batch_num = 100
                if epoch_batches % epoch_batch_num == 0:
                    elapsed = int(time.time() - start_time)
                    eta = int((
                                      total_batches - finished_batches) / finished_batches * elapsed)
                    print("elapsed : %s, ETA : %s" % (str(datetime.timedelta(seconds=elapsed)),
                                                      str(datetime.timedelta(seconds=eta))))
                    avg_loss1 /= epoch_batch_num
                    label_list1 = np.concatenate(label_list1)
                    pred_list1 = np.concatenate(pred_list1)
                    pred_list1 = np.float64(pred_list1)
                    pred_list1 = np.clip(pred_list1, 1e-8, 1 - 1e-8)
                    moving_auc = self.call_auc(y_true=label_list1,
                                               y_score=pred_list1)
                    print('source dataset')
                    print('epoch %d / %d, batch %d / %d, global_step = %d, learning_rate = %e, loss = %f, '
                          'auc = %f' % (epoch, self.n_epoch, epoch_batches, num_of_batches,
                                        self.global_step.eval(self.session), self._learning_rate,
                                        avg_loss1, moving_auc))

                    avg_loss2 /= epoch_batch_num
                    label_list2 = np.concatenate(label_list2)
                    pred_list2 = np.concatenate(pred_list2)
                    pred_list2 = np.float64(pred_list2)
                    pred_list2 = np.clip(pred_list2, 1e-8, 1 - 1e-8)
                    moving_auc = self.call_auc(y_true=label_list2,
                                               y_score=pred_list2)
                    print('target dataset')
                    print('epoch %d / %d, batch %d / %d, global_step = %d, learning_rate = %e, loss = %f, '
                          'auc = %f' % (epoch, self.n_epoch, epoch_batches, num_of_batches,
                                        self.global_step.eval(self.session), self._learning_rate,
                                        avg_loss2, moving_auc))
                    if avg_dis_loss > 0:
                        print('gan_loss:', avg_dis_loss / epoch_batch_num)
                    elif avg_dis_loss1 > 0:
                        print('gan_loss1:', avg_dis_loss1 / epoch_batch_num)
                        print('gan_loss2:', avg_dis_loss2 / epoch_batch_num)
                    print()

                    # source
                    label_list1 = []
                    pred_list1 = []
                    avg_loss1 = 0

                    # target
                    label_list2 = []
                    pred_list2 = []
                    avg_loss2 = 0

                    avg_dis_loss = 0
                    avg_dis_loss1 = 0
                    avg_dis_loss2 = 0

                if epoch_batches % num_of_batches == 0:
                    if epoch % test_every_epoch == 0:
                        print("get source/target ndcg hr")
                        self.test_ndcg_hr('src')
                        self.test_ndcg_hr('trg')
                        if self.hardness not in [5,6,7,8]:
                            self.calc_hardness()
                        if self.save_embedding:
                            self.saveEmb()
                        if self.save:
                            self._last_save()
                    self._learning_rate *= self.decay_rate
                    epoch += 1
                    epoch_batches = 0
                    if epoch > self.n_epoch:
                        return

            if epoch_batches % num_of_batches != 0:
                if epoch % test_every_epoch == 0:
                    print("get source/target ndcg hr")
                    self.test_ndcg_hr('src')
                    self.test_ndcg_hr('trg')
                    if self.hardness not in [5,6,7,8]:
                        self.calc_hardness()
                    if self.save_embedding:
                        self.saveEmb()
                    if self.save:
                        self._last_save()
                self._learning_rate *= self.decay_rate
                epoch += 1
                epoch_batches = 0
                return
                if epoch > self.n_epoch:
                    pass
