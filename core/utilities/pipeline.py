from pypeln import thread as th
from tqdm import tqdm

class Handler:
    def __call__(self, *args, **kwargs):
        raise NotImplemented


class Compose(Handler):
    def __init__(self, handlers=[]):
        self.handlers = handlers

    def __call__(self, data):
        res = data
        for handle in self.handlers:
            res = handle(res)

        return res


class ForEach(Handler):
    def __init__(self, handlers, make_iter=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handlers = handlers
        self.pip = Compose(self.handlers)
        self.make_iter = make_iter

    def __call__(self, data):
        iter_data = data
        if self.make_iter is not None: iter_data = self.make_iter(data)
        res = list(map(self.pip, iter_data))
        return data


class ForEachApply(ForEach):
    def __call__(self, data):
        iter_data = data
        if self.make_iter is not None: iter_data = self.make_iter(data)
        res = list(map(self.pip, iter_data))
        return res



class MultiThreadWrapper():
    def __init__(self, n_workers, pip):
        self.n_workers = n_workers
        self.pip= pip
        
    def __call__(self, data):
        total = None
        if type(data) is list:
            total = len(data)
        return list(tqdm(th.map(self.pip, data, workers=self.n_workers), total=total))

class Combine():
    def __call__(self, *results):
        return zip(*results)

class Merge():
    def __init__(self, pip):
        self.pip = pip
    def __call__(self, *args):
        results = []
        for p, a in zip(self.pip, args):
            results.append(p(a))
        return zip(*results)

if __name__ == '__main__':
    import numpy as np


    def power(data):
        print('wee')
        return data ** 2


    class Sum(Handler):
        def __call__(self, data):
            return data.sum()


    def foo(data):
        print('data from foo {}'.format(data))

        return data


    pipeline = Compose([power, Compose([Sum(), foo])])

    pipeline(np.array([1, 2, 3, 4]))

    list(map(foo, {'wee': 'test'}.items()))
