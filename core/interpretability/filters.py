class Shuffle():
    def __init__(self, f):
        self.f = f
        self.name = f.name

    def __call__(self, df):
        return self.f(df.sample(frac=1, random_state=0))

class All():
    name = 'all'

    def __call__(self, df):
        return df

class Best():
    name = 'best'

    def __call__(self, df):
        df = df.loc[df['label'] == 1]
        df.sort_values(['out_1'], ascending=False)
        return df

class Worst():
    name = 'worst'

    def __call__(self, df):
        df = df.loc[df['label'] == 0]
        df.sort_values(['out_0'], ascending=False)
        return df


class FalseNegative():
    name = 'false_negative'

    def __call__(self, df):
        return false_something(df, 0)


class FalsePositive():
    name = 'false_positive'

    def __call__(self, df):
        return false_something(df, 1)


def false_something(df, something):
    neg = df.loc[df['label'] == something]
    return neg.loc[neg['prediction'] != something]
