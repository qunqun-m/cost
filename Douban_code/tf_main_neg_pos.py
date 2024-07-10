import os
from tf_model import Model_NegSampling_positive
from tf_trainer_neg_pos import Trainer
from tf_utils import init_random_seed
import tensorflow as tf
import sys
backend = 'tf'
sys.path.append('/data/zxh/Douban/data/book/')
sys.path.append('/data/zxh/Douban/data/movie/')
sys.path.append('/data/zxh/Douban/data/music/')
from Book import Book
from Movie import Movie
from Music import Music
import random
import argparse
import math
from copy import deepcopy


DATA_CONFIG = {
    'TRAIN_SRC': {
        'shuffle': True,
        'batch_size': 128,
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
parser.add_argument('--neg', type=int, default=1, help='Negative Sampling')
parser.add_argument('--name', default='base', help='Model Name')

def get_datasets(data_name, neg_num):
    if data_name == 'book':
        return Book('train', neg_num), Book('test')
    elif data_name == 'movie':
        return Movie('train', neg_num), Movie('test')
    elif data_name == 'music':
        return Music('train', neg_num), Music('test')

def run_one_model(src_dataset=None, trg_dataset=None, model=None, learning_rate=1e-3, decay_rate=1.0, save=False,
                  epsilon=1e-8, ep=5, drop_epoch=1, resumedir=None, test_src_dataset=None, user_his_len=None,
                  test_trg_dataset=None, src_item_pre_sum=None, logdir=None, cotrain=False, add_neg_sample_num=0):
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
        'add_neg_sample_num': add_neg_sample_num
    }
    train_gen1 = src_dataset.batch_generator(DATA_CONFIG['TRAIN_SRC'])
    train_gen2 = trg_dataset.batch_generator(DATA_CONFIG['TRAIN_TRG'])
    trainer = Trainer(model=model, train_gen1=train_gen1, train_gen2=train_gen2,
                      test_dataset1=test_src_dataset, test_dataset2=test_trg_dataset, **train_param)
    trainer.fit()
    trainer.session.close()

def train():
    args = parser.parse_args()
    # 2019, 2023, 2025, 2027
    for random_seed in [2019]:
        embed_size = 64
        hist_type = 4
        name = args.name
        src = args.src
        trg = args.trg
        cotrain = False

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

        print('dataset, source:', src, 'target:', trg)
        src_dataset, test_src_dataset = get_datasets(src, args.neg)
        trg_dataset, test_trg_dataset = get_datasets(trg, args.neg)

        DATA_CONFIG['TRAIN_TRG'] = deepcopy(DATA_CONFIG['TRAIN_SRC'])
        DATA_CONFIG['TRAIN_TRG']['batch_size'] = int(
            DATA_CONFIG['TRAIN_SRC']['batch_size'] * trg_dataset.train_size / src_dataset.train_size)

        print('test set')
        learning_rate = 1e-3
        split_epoch = 1
        resumedir = None
        logdir = 'save_models/%s' % name

        save = True
        print("test ndcg, hr", "name:", name, "dc:", dc, "random_seed:", random_seed, "l1_w:", l1_w, "save:", save,
              "drop_epoch:", drop_epoch)
        model = SingleTower(init="xavier", user_max_id=src_dataset.feat_sizes[0],
                                       src_item_max_id=src_dataset.feat_sizes[1],
                                       trg_item_max_id=trg_dataset.feat_sizes[1],
                                       embed_size=embed_size, layer_sizes=layer_sizes, layer_acts=layer_acts,
                                       layer_keeps=layer_keeps, l1_w=l1_w, l1_v=l1_v, layer_l1=layer_l1,
                                       batch_norm=batch_norm, layer_norm=layer_norm,
                                       user_his_len=src_dataset.user_his_len, hist_type=hist_type, cotrain=cotrain)
        run_one_model(src_dataset=src_dataset, trg_dataset=trg_dataset, model=model, learning_rate=learning_rate,
                      epsilon=1e-8, user_his_len=src_dataset.user_his_len, src_item_pre_sum=src_dataset.feat_sizes[1],
                      decay_rate=dc, ep=split_epoch, save=save, drop_epoch=drop_epoch,
                      test_src_dataset=test_src_dataset, test_trg_dataset=test_trg_dataset,
                      resumedir=resumedir, logdir=logdir, cotrain=cotrain, add_neg_sample_num=args.neg)
        tf.reset_default_graph()
        del src_dataset, test_src_dataset, trg_dataset, test_trg_dataset


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
