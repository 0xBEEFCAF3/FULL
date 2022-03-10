
def min_max(df, mean, std):
    return (df - mean) / std


def reverse_min_max(value, mean, std):
    return (value * std) + mean
