import os
from tf_model import Model_Hardset as Model
from tf_trainer_hardset import Trainer
from tf_utils import init_random_seed
import tensorflow as tf
import sys
backend = 'tf'
sys.path.append('./data/book/')
sys.path.append('./data/movie/')
sys.path.append('./data/music/')
from Book import Book
from Movie import Movie
from Music import Music
import random
import argparse
import math
from copy import deepcopy
from utils.data_utils import from_hdf


DATA_CONFIG = {
    'TRAIN_SRC': {
        'shuffle': True,
        # 'batch_size': 128,
        'batch_size': 256,
    },
    'TEST': {
        'shuffle': False,
        'batch_size': 512,
    },
}

TRAIN_CONFIG = {
    'learning_rate': 1e-3, 'epsilon': 1e-8,
    'decay_rate': 1, 'ep': 1
}

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='book', help='Source Dataset')
parser.add_argument('--trg', default='movie', help='Target Dataset')
parser.add_argument('--neg', type=int, default=5, help='Negative Sampling')
parser.add_argument('--name', default='anchors', help='Model Name')

def get_datasets(data_name, neg_num):
    if data_name == 'book':
        return Book('train', neg_num), Book('test')
    elif data_name == 'movie':
        return Movie('train', neg_num), Movie('test')
    elif data_name == 'music':
        return Music('train', neg_num), Music('test')

def run_one_model(src_dataset=None, trg_dataset=None, model=None, learning_rate=1e-3, decay_rate=1.0, save=False,
                  epsilon=1e-8, ep=5, drop_epoch=1, resumedir=None, test_dataset_src=None, user_his_len=None,
                  test_dataset_trg=None, src_item_pre_sum=None, logdir=None, cotrain=False, add_neg_sample_num=0,
                  save_embedding=None, hardness=None, n_clusters=None, anchor_num=None, itemid_presum=None,
                  alpha=None, calc_pattern=None, calc_reverse=None):
    n_ep = ep * 10
    train_param = {
        'opt': 'adam',
        'loss': 'weight',
        'pos_weight': 1.0,
        'n_epoch': n_ep,
        'train_per_epoch': src_dataset.train_size / ep,  # split training data
        # 'train_per_epoch': 10,
        'test_per_epoch': src_dataset.test_size,
        'src_item_pre_sum': src_item_pre_sum,
        'early_stop_epoch': math.ceil(0.5 * ep),
        'test_every_epoch': math.ceil(ep / 5),
        'batch_size1': DATA_CONFIG['TRAIN_SRC']['batch_size'],
        'batch_size2': DATA_CONFIG['TRAIN_TRG']['batch_size'],
        'learning_rate': learning_rate,
        'decay_rate': decay_rate,
        'epsilon': epsilon,
        'resumedir': resumedir,
        'user_his_len': user_his_len,
        'save': save,
        'drop_epoch': drop_epoch,
        'logdir': logdir,
        'cotrain': cotrain,
        'add_neg_sample_num': add_neg_sample_num,
        'save_embedding': save_embedding,
        'hardness': hardness,
        'n_clusters': n_clusters,
        'src_dataset': src_dataset,
        'trg_dataset': trg_dataset,
        'anchor_num': anchor_num,
        'itemid_presum': itemid_presum,
        'alpha': alpha,
        'calc_pattern': calc_pattern,
        'calc_reverse': calc_reverse
    }
    train_gen1 = src_dataset.batch_generator(DATA_CONFIG['TRAIN_SRC'])
    train_gen2 = trg_dataset.batch_generator(DATA_CONFIG['TRAIN_TRG'])
    trainer = Trainer(model=model, train_gen1=train_gen1, train_gen2=train_gen2,
                      test_dataset_src=test_dataset_src, test_dataset_trg=test_dataset_trg, **train_param)
    trainer.fit()
    trainer.session.close()

def train():
    args = parser.parse_args()
    # 2019, 2023, 2025, 2027
    for random_seed in [2019]:
        embed_size = 32
        hist_type = 0
        name = args.name
        logdir = 'save_models/%s' % name
        src = args.src
        trg = args.trg
        cotrain = 3


        name = '%s_src%s_trg%s_seed%d_hist%d' % (name, src, trg, random_seed, hist_type)

        # 0.7, 1
        dc = 1.0
        batch_norm = True
        layer_norm = False
        l1_w = 0.0
        l1_v = 0.0
        layer_l1 = 0.0
        layer_sizes = [128, 64, 1]
        layer_acts = ['relu', 'relu', None]
        layer_keeps = [1.0, 1.0, 1.0]
        drop_epoch = 1  # 1, 2, 5
        init_random_seed(random_seed)


        save_embedding = False
        restore_embedding = 0
        samebs = False

        # Model Parameters
        hardness = 0
        thres = 0.0

        anchor_num = 10
        head_num = 3
        k_ratio = 0.7
        k_ratio2 = 1.0
        tau = 0.2
        n_clusters = 10
        alpha = 1.0
        ae = 7

        calc_pattern = 0
        calc_reverse = 0

        print('dataset, source:', src, 'target:', trg)
        src_dataset, test_dataset_src = get_datasets(src, args.neg)
        trg_dataset, test_dataset_trg = get_datasets(trg, args.neg)

        itemid_presum = src_dataset.feat_sizes[1]

        DATA_CONFIG['TRAIN_TRG'] = deepcopy(DATA_CONFIG['TRAIN_SRC'])
        if not samebs:
            DATA_CONFIG['TRAIN_TRG']['batch_size'] = int(
                DATA_CONFIG['TRAIN_SRC']['batch_size'] * trg_dataset.train_size / src_dataset.train_size)
        print('source batch size:', DATA_CONFIG['TRAIN_SRC']['batch_size'], 'target batch size:',
              DATA_CONFIG['TRAIN_TRG']['batch_size'], 'batch size ratio of overall:',
              DATA_CONFIG['TRAIN_SRC']['batch_size'] / (
                          DATA_CONFIG['TRAIN_SRC']['batch_size'] + DATA_CONFIG['TRAIN_TRG']['batch_size']))

        print('test set')
        learning_rate = 1e-3
        split_epoch = 1
        resumedir = None

        save = False
        print("test ndcg, hr", "name:", name, "dc:", dc, "random_seed:", random_seed, "l1_w:", l1_w, "save:", save,
              "drop_epoch:", drop_epoch)

        if restore_embedding == 1:
            cluster_embeddings = from_hdf('process/cluster_%s.hdf5'%'alldata', 'data%d'%anchor_num)
            user_embeddings = from_hdf('process/embedding.hdf5', 'user')
            print('cluster_embeddings.shape, user_embeddings.shape', cluster_embeddings.shape, user_embeddings.shape)
        elif restore_embedding == 2:
            cluster_embeddings = from_hdf('process/cluster_%s.hdf5'%'alldata_cotrain2', 'data%d'%anchor_num)
            user_embeddings = from_hdf('process/embedding_%s_%d.hdf5'%('alldata_cotrain2', 6448), 'user')
            print('cluster_embeddings.shape, user_embeddings.shape', cluster_embeddings.shape, user_embeddings.shape)
        elif restore_embedding == 2.5:
            cluster_embeddings = from_hdf('process/cluster_%s.hdf5'%'alldata_cotrain2', 'data%d'%anchor_num)
            user_embeddings = None
            print('cluster_embeddings.shape', cluster_embeddings.shape)
        elif restore_embedding == 3:
            cluster_embeddings = from_hdf('process/cluster_%s.hdf5'%'alldata_rerun', 'data%d'%anchor_num)
            user_embeddings = from_hdf('process/embedding_%s_%d.hdf5'%('alldata_rerun', 4836), 'user')
            print('cluster_embeddings.shape, user_embeddings.shape', cluster_embeddings.shape, user_embeddings.shape)
        elif restore_embedding == 4:
            cluster_embeddings = from_hdf('process/cluster_%s.hdf5'%'alldata_emb32', 'data%d'%anchor_num)
            user_embeddings = from_hdf('process/embedding_%s_%d.hdf5'%('alldata_emb32', 3224), 'user')
            print('cluster_embeddings.shape, user_embeddings.shape', cluster_embeddings.shape, user_embeddings.shape)
        elif restore_embedding == 5:
            cluster_embeddings = from_hdf('process/cluster_%s.hdf5'%'alldata_emb16', 'data%d'%anchor_num)
            user_embeddings = from_hdf('process/embedding_%s_%d.hdf5'%('alldata_emb16', 3224), 'user')
            print('cluster_embeddings.shape, user_embeddings.shape', cluster_embeddings.shape, user_embeddings.shape)
        else:
            cluster_embeddings = None
            user_embeddings = None

        model = Model(init="xavier", user_max_id=src_dataset.feat_sizes[0],
                                       src_item_max_id=src_dataset.feat_sizes[1],
                                       trg_item_max_id=trg_dataset.feat_sizes[1],
                                       embed_size=embed_size, layer_sizes=layer_sizes, layer_acts=layer_acts,
                                       layer_keeps=layer_keeps, l1_w=l1_w, l1_v=l1_v, layer_l1=layer_l1,
                                       batch_norm=batch_norm, layer_norm=layer_norm,
                                      user_his_len=src_dataset.user_his_len, hist_type=hist_type,
                                      anchor_num=anchor_num, cotrain=cotrain,
                                    cluster_embeddings=cluster_embeddings, user_embeddings=user_embeddings,
                                    head_num=head_num, k_ratio=k_ratio, k_ratio2=k_ratio2, tau=tau, hardness=hardness,
                                    ae=ae, calc_pattern=calc_pattern, thres=thres)
        run_one_model(src_dataset=src_dataset, trg_dataset=trg_dataset, model=model, learning_rate=learning_rate,
                      epsilon=1e-8, user_his_len=src_dataset.user_his_len, src_item_pre_sum=src_dataset.feat_sizes[1],
                      decay_rate=dc, ep=split_epoch, save=save, drop_epoch=drop_epoch,
                      test_dataset_src=test_dataset_src, test_dataset_trg=test_dataset_trg,
                      resumedir=resumedir, logdir=logdir, cotrain=cotrain, add_neg_sample_num=args.neg,
                      save_embedding=save_embedding, hardness=hardness, n_clusters=n_clusters,
                    anchor_num=anchor_num, itemid_presum=itemid_presum, alpha=alpha, calc_pattern=calc_pattern,
                      calc_reverse=calc_reverse)
        tf.reset_default_graph()
        del src_dataset, test_dataset_src, trg_dataset, test_dataset_trg


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
