import numpy as np

def generate_random_value(df, column, prob=False):

    dist = (df[column].value_counts() / len(df)).to_dict()
    r, c = np.random.random(), 0
    for i, p in dist.items():
        c += p
        if c >= r:
            return i if not prob else i, dist