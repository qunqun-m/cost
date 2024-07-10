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

    def __init__(self, model=None, train_gen1=None, train_gen2=None, test_dataset1=None, test_dataset2=None, opt='adam',
                 epsilon=1e-8, initial_accumulator_value=1e-8, momentum=0.95, loss='weighted', pos_weight=1.0,
                 n_epoch=1, train_per_epoch=10000, test_per_epoch=10000, early_stop_epoch=5, batch_size1=2000,
                 batch_size2=2000, learning_rate=1e-2, decay_rate=0.95, test_every_epoch=1, save=False,
                 logdir=None, drop_epoch=1, resumedir=None, src_item_pre_sum=None, user_his_len=None, cotrain=False,
                 add_neg_sample_num=0):
        self.model = model
        self.train_gen1 = train_gen1
        self.train_gen2 = train_gen2
        self.test_dataset1 = test_dataset1
        self.test_dataset2 = test_dataset2
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
                     self.model.item_id_trg: item_id_trg}
        feed_dict[self.model.history1] = user_history_src
        feed_dict[self.model.history_len1] = user_history_src_len
        feed_dict[self.model.history2] = user_history_trg
        feed_dict[self.model.history_len2] = user_history_trg_len

        if hasattr(self.model, 'training'):
            feed_dict[self.model.training] = True

        # a, b = self._run([self.model.c, self.model.b], feed_dict)
        # a = np.array(a)
        # print(a)
        # print(a.shape)
        # print()
        # # assert 0

        _, _, _loss1, _loss2, outputs1, outputs2 = self._run(
            fetches=[self.model.optimizer1, self.model.optimizer2, self.model.loss1, self.model.loss2,
                     self.model.outputs1, self.model.outputs2], feed_dict=feed_dict)

        # if is_src:
        #     _, _loss, outputs = self._run(
        #         fetches=[self.model.optimizer1, self.model.loss1,
        #                  self.model.outputs1], feed_dict=feed_dict)
        # else:
        #     _, _loss, outputs = self._run(
        #         fetches=[self.model.optimizer2, self.model.loss2,
        #                  self.model.outputs2], feed_dict=feed_dict)
        return _loss1, _loss2, outputs1, outputs2

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
                         self.model.item_id_trg: neg_items}
            feed_dict[self.model.history2] = user_historys
            feed_dict[self.model.history_len2] = user_history_len


        if hasattr(self.model, 'training'):
            feed_dict[self.model.training] = False
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
            test_gen = self.test_dataset1.batch_generator(config)
        else:
            test_gen = self.test_dataset2.batch_generator(config)
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
            ground_truth_item = 0
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

    def fit(self):
        num_of_batches = int(
            np.ceil(self.train_per_epoch / (self.batch_size1))) + 1
        print('self.train_per_epoch', self.train_per_epoch)
        print('self.batch_size1', self.batch_size1)
        print('num_of_batches', num_of_batches)
        total_batches = self.n_epoch * num_of_batches
        print('total batches: %d\tbatch per epoch: %d' % (
        total_batches, num_of_batches))
        start_time = time.time()
        epoch = 1
        finished_batches = 0

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

        test_every_epoch = self.test_every_epoch

        train_gen2 = iter(self.train_gen2)

        while epoch <= self.n_epoch:
            print('======new iteration======')
            epoch_batches = 0
            stop_flag = False

            for batch_data in self.train_gen1:
                user_id1, item_id1, user_history1, y1 = batch_data

                try:
                    batch_data = next(train_gen2)
                    user_id2, item_id2, user_history2, y2 = batch_data
                    batch_loss1, batch_loss2, batch_pred1, batch_pred2 = self._train(user_id1, item_id1, user_history1,
                                                                                     y1, user_id2, item_id2,
                                                                                     user_history2, y2)
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
                except:
                    stop_flag = True

                finished_batches += 1
                epoch_batches += 1
                # print(epoch_batches)

                epoch_batch_num = 100
                if not stop_flag and epoch_batches % epoch_batch_num == 0:
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
                    print()

                    # source
                    label_list1 = []
                    pred_list1 = []
                    avg_loss1 = 0

                    # target
                    label_list2 = []
                    pred_list2 = []
                    avg_loss2 = 0

                if epoch_batches % num_of_batches == 0:
                    if epoch % test_every_epoch == 0:
                        print("get source/target ndcg hr")
                        # self.test_ndcg_hr('src')
                        self.test_ndcg_hr('trg')
                        if self.save:
                            self._last_save()
                    self._learning_rate *= self.decay_rate
                    epoch += 1
                    epoch_batches = 0
                    if epoch > self.n_epoch:
                        return

            train_gen2 = iter(self.train_gen2)

            if epoch_batches % num_of_batches != 0:
                if epoch % test_every_epoch == 0:
                    print("get source/target ndcg hr")
                    self.test_ndcg_hr('src')
                    self.test_ndcg_hr('trg')
                    if self.save:
                        self._last_save()

                self._learning_rate *= self.decay_rate
                epoch += 1
                epoch_batches = 0
                if epoch > self.n_epoch:
                    return
