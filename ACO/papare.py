import numpy as np
def truncated_gaussian(mean, std_dev, size):
    data = []
    while len(data) < size:
        sample = np.random.normal(mean, std_dev)
        if 0 < sample <= 1:
            data.append(sample)
    return data