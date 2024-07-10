from copy import deepcopy
class ProcessOneField(object):
    def __init__(self, data, number_map=None, data_map=None, has_other=False, threshold=20, has_nan=False, nan_val=-1):
        self.data = data
        self.distribution = {}
        self.number_map = number_map if number_map is not None else {}
        self.data_map = data_map if data_map is not None else {}
        self.has_other = has_other
        self.threshold = threshold
        self.has_nan = has_nan
        self.nan_val = nan_val

    def getNumMap(self):
        if len(self.number_map) == 0:
            s = {}
            for data in self.data:
                if s.get(data, None) is None:
                    s[data] = 1
                else:
                    s[data] = s[data] + 1
            self.number_map = s

        print('The kinds of the data:', len(self.number_map))
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
            numMap = deepcopy(self.number_map)
            sortmap = dict(sorted(numMap.items(), key=lambda e: e[1]))
            cnt = 2
            for key, value in sortmap.items():
                if self.has_nan and self.check(key, self.nan_val):
                    self.data_map[key] = 1  # nan value
                elif self.has_other and value <= self.threshold:
                    self.data_map[key] = 0
                else:
                    self.data_map[key] = cnt
                    cnt += 1
        return self.data_map

    def setDataMap(self, input_dict):
        self.data_map = input_dict

    def setNumMap(self, input_dict):
        self.number_map = input_dict

    def process(self):
        def getmap(x):
            return self.data_map.get(x, float('nan'))
        self.getDataMap()
        self.data = self.data.map(getmap)
        return self.data

