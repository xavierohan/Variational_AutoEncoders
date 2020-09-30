
_MODEL = dict()


def register(name):

    def add_to_dict(fn):
        global _MODEL
        _MODEL[name] = fn
        #print(_MODEL)
        return fn

    return add_to_dict