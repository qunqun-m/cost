import numpy as np
from copy import deepcopy
import pandas as pd
class ProcessMultiField(object):
    '''
        Process the Multi-Value Colume.
        The getDataMap(self) function gets the encode value of the row data.
        If self.has_other is True, the row values that are equal or less than the self.threshold will be encode into 0.
        No matter self.has_other is True or False, the row values that are original 0 will be encoded into 0.

        :returns A series of columns. The length of the columns is the max class number of the data.
                 The column name is in the form <data.name>_<i>.
                 The last column stores the number of the multi-value in one row.
                 Other columns stores the encode value of the row data in the corresponding position.
    '''

    def __init__(self, data, number_map=None, data_map=None, has_other=False, threshold=20, has_nan=False, nan_val=-1,
                 sep=',', max_len=None, remove_v=None):
        self.data = data
        self.distribution = {}
        self.number_map = number_map if number_map is not None else {}
        self.data_map = data_map if data_map is not None else {}
        self.has_other = has_other
        self.threshold = threshold
        self.has_nan = has_nan
        self.nan_val = nan_val
        self.sep = sep
        self.max_len = max_len
        self.remove_v = remove_v if remove_v is not None else set()

    def getNumMap(self):
        if len(self.number_map) == 0:
            s = {}
            for multi in self.data:
                datalist = str(multi).split(self.sep)
                for data in datalist:
                    if s.get(data, None) is None:
                        s[data] = 1
                    else:
                        s[data] = s[data] + 1
            self.number_map = s

        print('The kinds of the data:', len(self.number_map))
        self.number_map = dict(sorted(self.number_map.items(), key=lambda e: e[1]))
        return self.number_map

    def getDist(self):
        if len(self.distribution) == 0:
            self.getNumMap()
            s = {}
            for key, value in self.number_map.items():
                if s.get(value, None) is None:
                    s[value] = 1
                else:
                    s[value] = s[value] + 1
            s = sorted(s.items(), key=lambda e: e[0])
            self.distribution = s
        return self.distribution

    def check(self, key, val):
        try:
            if key == val:
                return True
        except:
            pass
        return False

    def getDataMap(self):
        if len(self.data_map) == 0:
            self.getNumMap()
            sortmap = self.number_map
            cnt = 2
            for key, value in sortmap.items():
                if self.has_nan and self.check(key, self.nan_val):
                    self.data_map[key] = 1  # nan value
                elif self.has_other and value <= self.threshold:
                    self.data_map[key] = 0  # other value
                else:
                    self.data_map[key] = cnt
                    # if self.has_other and cnt == 2:
                    #     print(key)
                    #     print(self.data_map[key])
                    #     assert 0
                    cnt += 1
            print("feature number:", cnt)
        return self.data_map

    def setDataMap(self, input_dict):
        self.data_map = input_dict

    def setNumMap(self, input_dict):
        self.number_map = input_dict

    def process(self):
        '''
        :param
        :return:
            N+1 columns of values. The last column means the length of the multi-value field.
        '''
        self.getNumMap()
        self.getDataMap()
        if self.max_len is None:
            MAX_LEN = len(set(self.data_map.values())) + 1
        else:
            MAX_LEN = self.max_len
        print('MAX length:', MAX_LEN)
        columns = []
        for i in range(MAX_LEN-1):
            columns.append(self.data.name+'_%d'%(i))
        columns.append(self.data.name+'_len')
        arr = np.full((self.data.shape[0], MAX_LEN), -1, dtype=int)
        for i, multi in enumerate(self.data):
            s = {}
            datalist = str(multi).split(self.sep)
            for j, data in enumerate(datalist):
                if self.data_map[data] not in self.remove_v:
                    if s.get(self.data_map[data], None) is None:
                        s[self.data_map[data]] = 1
                    else:
                        s[self.data_map[data]] += 1
            s = sorted(s.items(), key=lambda e: e[1], reverse=True)

            for j, val in enumerate(s):
                if j < self.max_len - 1:
                    arr[i, j] = val[0]
                else:
                    break
            arr[i, -1] = min(len(s), MAX_LEN - 1)
        print('The columns are:\n', columns)
        return pd.DataFrame(arr, columns=columns, dtype=int)
