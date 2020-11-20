import json
from functools import wraps

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

def add(cls):
    def _inner(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        setattr(cls, f.__name__, wrapper)
        return f
    return _inner
