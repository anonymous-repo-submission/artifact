"""Dictionary interface that just returns the original key"""

class DummyIdDict(dict):
    def __missing__(self, key):
        return key