import numpy as np
import random

def cifar10_next_batch(num, x):
    """
    Return a total of `num` samples from x
    """
    idx = np.arange(0, len(x))     # get all possible indexes
    np.random.shuffle(idx)         # shuffle indexes
    idx = idx[0:num]               # use only `num` random indexes
    batch_x = [x[i] for i in idx]  # get list of `num` random samples
    batch_x = np.asarray(batch_x)  # get back numpy array
    batch_x = batch_x.reshape((num, 32*32*3))

    return batch_x