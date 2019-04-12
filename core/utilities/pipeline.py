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

    pipeline(np.array([1,2,3,4]))

    list(map(foo, {'wee' : 'test'}.items()))