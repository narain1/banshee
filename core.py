import json

class DictWrapper:
    def __init__(self, d):
        self.__dict__.update(d)

def dict2obj(x):
    return json.loads(json.dumps(d), object_hook=DictWrapper)

d = {'a': 1, 'b': {'c': 2}, 'd': ['hi', {'foo': 'bar'}]}

class attrdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
