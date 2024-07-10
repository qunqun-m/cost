from datetime import datetime
import time
import h5py
import _pickle as pickle
import os
import numpy as np
import pandas as pd

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

def is_in_set_pnb(a, b):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False, dtype=bool)
    set_b = set(b)
    for i in np.arange(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)

def saveDict(data, path):
    pickle.dump(data, open(path, 'wb'), protocol=4)

def fromDict(path):
    print('loading file:', path, end=' ')
    t = time.time()
    data = pickle.loads(open(path,'rb').read())
    print('Time spent:', time.time() - t)
    return data


def to_hdf(na, path, key, overwrite=False, lockfile='tmp.lock'):
    lockfile = os.path.join(os.path.dirname(path), lockfile)
    while os.path.exists(lockfile):
        print('waiting')
        sleep(5)
    with open(lockfile, 'w') as f:
        f.write('locking')
    try:
        with h5py.File(str(path), 'a') as hf:
            if overwrite and key in hf.keys():
                hf.__delitem__(key)
                print('\'' + key + '\'', 'has been overwrited.')
            hf.create_dataset(key, data=na)
    except:
        print('to_hdf Wrong!')
    try:
        os.remove(lockfile)
    except:
        pass

def from_hdf(path, key, start=0, stop='last', lockfile='tmp.lock'):
    lockfile = os.path.join(os.path.dirname(path), lockfile)
    while os.path.exists(lockfile):
        print('waiting')
        sleep(5)
    with h5py.File(str(path), 'r') as hf:
        if stop == 'last':
            data = hf[key][start:]
        else:
            data = hf[key][start:stop]
    try:
        os.remove(lockfile)
    except:
        pass
    return data

def parsedate(data):
    def func(s):
        s = str(int(s))
        s = s[:4]+'-'+s[4:6]+'-'+s[6:]
        return datetime.strptime(s, '%Y-%m-%d').weekday()
    return data.apply(func)

def parsetime(data):
    def func(s):
        s = str(int(s))
        s = s[8:10]
        return int(s)
    return data.apply(func)

# def parsetimeFromStamp(data):
#     def func(e):
#         timeArray = time.localtime(e)
#
#     pass

def checkdata(df):
    for col in df.columns:
        try:
            print('\'%s\' min:' % col, df[col].min(), 'max:', df[col].max())
        except:
            print('\'%s\'.' % col)
    print()
    for col in df.columns:
        print('\'%s\' number:' % col, len(df[col].unique()))
    print()
    for col in df.columns:
        print('\'%s\' has NaN:' % col, df[col].isnull().any(axis=0))

def merge(a, b, on):
    if type(b) != pd.Series:
        b = b.set_index(on).reindex(a[on].values)
    else:
        b = b.reindex(a[on].values)
    return pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1).reset_index(drop=True)

