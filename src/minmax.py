
def min_max(df):
    mean = df.mean()
    std = df.std()

    return (df - mean) / std


def reverse_min_max(value, mean, std):
    return (value * std) + mean
