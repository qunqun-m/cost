import os
from tf_model import Model_User_Anchors as Model
import tensorflow as tf
import numpy as np
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
from utils.data_utils import to_hdf, from_hdf, saveDict, fromDict

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='book', help='Source Dataset')
parser.add_argument('--trg', default='movie', help='Target Dataset')
parser.add_argument('--neg', type=int, default=1, help='Negative Sampling')
parser.add_argument('--name', default='alldata', help='Model Name')
parser.add_argument('--glb', type=int, default=6448, help='global step')
parser.add_argument('--K', type=int, default=10, help='cluster num')
parser.add_argument('--emb', type=int, default=64, help='cluster num')
args = parser.parse_args()

neg = args.neg
src = args.src
trg = args.trg
name = args.name
global_step = args.glb
K = args.K
embed_size = args.emb

def get_datasets(data_name, neg_num):
    if data_name == 'book':
        return Book('train', neg_num), Book('test')
    elif data_name == 'movie':
        return Movie('train', neg_num), Movie('test')
    elif data_name == 'music':
        return Music('train', neg_num), Music('test')

def get_embeddings():
    hist_type = 0
    logdir = 'save_models/%s' % name
    cotrain = 1

    print('dataset, source:', src, 'target:', trg)
    src_dataset, test_src_dataset = get_datasets(src, neg)
    trg_dataset, test_trg_dataset = get_datasets(trg, neg)

    # 0.7, 1
    batch_norm = True
    layer_norm = False
    l1_w = 0.0
    l1_v = 0.0
    layer_l1 = 0.0
    layer_sizes = [128, 64, 1]
    layer_acts = ['relu', 'relu', None]
    layer_keeps = [1.0, 1.0, 1.0]

    # Model Parameters
    anchor_user = 0
    anchor_num = 10
    model = Model(init="xavier", user_max_id=src_dataset.feat_sizes[0],
                                       src_item_max_id=src_dataset.feat_sizes[1],
                                       trg_item_max_id=trg_dataset.feat_sizes[1],
                                       embed_size=embed_size, layer_sizes=layer_sizes, layer_acts=layer_acts,
                                       layer_keeps=layer_keeps, l1_w=l1_w, l1_v=l1_v, layer_l1=layer_l1,
                                       batch_norm=batch_norm, layer_norm=layer_norm,
                                      user_his_len=src_dataset.user_his_len, hist_type=hist_type,
                                      anchor_user=anchor_user, anchor_num=anchor_num, cotrain=cotrain)

    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Create a saver object
    saver = tf.train.Saver(max_to_keep=25)
    save_path = os.path.join(logdir, 'embedding.ckpt')
    save_path = save_path + '-' + str(global_step)
    print('save_path: ', save_path)
    # Restore the embedding parameters
    saver.restore(sess, save_path)

    # Set the initial values of the embedding matrices
    uemb = sess.run(model.v_user)
    iemb = sess.run(model.v_item)
    to_path = 'process/embedding_%s_%d.hdf5' % (name, global_step)
    to_hdf(uemb, to_path, 'user', overwrite=True)       # (2666, 64)
    to_hdf(iemb, to_path, 'item', overwrite=True)       # (64212, 64)

def get_embedding_clusters():
    from sklearn.cluster import KMeans
    iemb = from_hdf('process/embedding_%s_%d.hdf5'%(name, global_step), 'user')
    # kmeans = KMeans(n_clusters=K, init='k-means++', random_state=40).fit(iemb)
    kmeans = KMeans(n_init=50, n_clusters=K, random_state=40).fit(iemb)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    print(cluster_labels)
    print(np.max(cluster_labels), np.min(cluster_labels))       # max: K-1, min: 0
    print(cluster_labels.shape)     # (2666,)

    for c in range(K):
        class_id = np.where(cluster_labels == c)[0]
        print(len(class_id))

    to_hdf(cluster_labels, 'process/cluster_%s.hdf5'%name, 'label%d'%K, overwrite=True)
    to_hdf(cluster_centers, 'process/cluster_%s.hdf5'%name, 'data%d'%K, overwrite=True)
    saveDict(cluster_labels, 'process/cluster_id_%s.pkl'%name)

def get_embedding_clusters_DBSCAN():
    from sklearn.cluster import DBSCAN
    iemb = from_hdf('process/embedding_%s_%d.hdf5' % (name, global_step), 'user')
    # Define your DBSCAN parameters
    eps = 2.0
    min_samples = 1

    # Create a DBSCAN object and fit it to the embedding vectors
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(iemb)
    print(labels)
    print(labels.shape)
    print(np.unique(labels).shape)

def get_embedding_clusters_GMM():
    from sklearn.mixture import GaussianMixture
    iemb = from_hdf('process/embedding_%s_%d.hdf5' % (name, global_step), 'user')

    # Define your GMM parameters
    n_components = K
    covariance_type = 'full'

    # Create a GMM object and fit it to the embedding vectors
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    gmm.fit(iemb)

    # Get the cluster labels and cluster centers
    labels = gmm.predict(iemb)
    for c in range(K):
        class_id = np.where(labels == c)[0]
        print(len(class_id))
    cluster_centers = gmm.means_
    print(cluster_centers.shape)

def test_model():

    pass

if __name__ == '__main__':
    get_embeddings()
    get_embedding_clusters()
    # get_embedding_clusters_DBSCAN()
    # get_embedding_clusters_GMM()
    pass
