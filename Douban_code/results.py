import os
import numpy as np
import sys
import random
import argparse
import math
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='log/default.txt', help='Default `txt` file')
args = parser.parse_args()

file = args.file

if __name__ == '__main__':
    np.set_printoptions(precision=6)
    k_list = [5, 10, 20, 50, 100000000]
    loop_epoch = 10
    max_K = 0
    source_hr = []
    source_ndcg = []
    target_hr = []
    target_ndcg = []
    with open(file, 'r') as f:
        results = f.readlines()
        start_hr1 = -1
        start_ndcg1 = -1
        start_hr2 = -1
        start_ndcg2 = -1
        begin = 0
        for i, l in enumerate(results):
            if 'source clicked item HR, NDCG' in l:
                start_hr1 = i+2
                start_ndcg1 = start_hr1 + len(k_list) + 2
                begin = 1
                tmp_hr = []
                tmp_ndcg = []
            if 'target clicked item HR, NDCG' in l:
                start_hr2 = i+2
                start_ndcg2 = start_hr2 + len(k_list) + 2
                begin = 2
                tmp_hr = []
                tmp_ndcg = []
            if begin == 1 and i >= start_hr1 and i < start_hr1 + len(k_list):
                tmp_hr.append(float(l))
            if begin == 1 and i >= start_ndcg1 and i < start_ndcg1 + len(k_list):
                tmp_ndcg.append(float(l))
            if begin == 1 and i == start_ndcg1 + len(k_list):
                source_hr.append(tmp_hr)
                source_ndcg.append(tmp_ndcg)


            if begin == 2 and i >= start_hr2 and i < start_hr2 + len(k_list):
                tmp_hr.append(float(l))
            if begin == 2 and i >= start_ndcg2 and i < start_ndcg2 + len(k_list):
                tmp_ndcg.append(float(l))
            if begin == 2 and i == start_ndcg2 + len(k_list):
                target_hr.append(tmp_hr)
                target_ndcg.append(tmp_ndcg)

    source_hr_final = []
    source_ndcg_final = []
    target_hr_final = []
    target_ndcg_final = []

    curr_epoch = 0
    tmp_shr = [0.0] * len(k_list)
    tmp_sndcg = [0.0] * len(k_list)
    tmp_thr = [0.0] * len(k_list)
    tmp_tndcg = [0.0] * len(k_list)
    for i in range(len(source_hr)):
        # if source_hr[i][max_K] > tmp_shr[max_K]:
        if target_hr[i][max_K] > tmp_thr[max_K]:
            tmp_shr = source_hr[i]
            tmp_sndcg = source_ndcg[i]
            tmp_thr = target_hr[i]
            tmp_tndcg = target_ndcg[i]
        curr_epoch += 1
        if curr_epoch == loop_epoch:
            source_hr_final.append(tmp_shr)
            source_ndcg_final.append(tmp_sndcg)
            target_hr_final.append(tmp_thr)
            target_ndcg_final.append(tmp_tndcg)

            tmp_shr = [0.0] * len(k_list)
            tmp_sndcg = [0.0] * len(k_list)
            tmp_thr = [0.0] * len(k_list)
            tmp_tndcg = [0.0] * len(k_list)

            curr_epoch = 0

    source_hr_final = np.array(source_hr_final)         # [N_loop, len(k_list)]
    source_ndcg_final = np.array(source_ndcg_final)
    target_hr_final = np.array(target_hr_final)
    target_ndcg_final = np.array(target_ndcg_final)

    source_hr_final = np.mean(source_hr_final, axis=0)
    source_ndcg_final = np.mean(source_ndcg_final, axis=0)
    target_hr_final = np.mean(target_hr_final, axis=0)
    target_ndcg_final = np.mean(target_ndcg_final, axis=0)

    print('source_hr_final\n', source_hr_final)
    print()
    print('source_ndcg_final\n', source_ndcg_final)
    print()
    print('target_hr_final\n', target_hr_final)
    print()
    print('target_ndcg_final\n', target_ndcg_final)
