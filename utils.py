import torch
import random
import numpy as np


def accuracy_top_N(output, y, N=.01):
    k = int(y.shape[0]*N)
    arr1 = torch.argsort(output,0)[-k:]
    arr2 = torch.argsort(y,0)[-k:]
    inter = np.intersect1d(arr1.cpu(), arr2.cpu()).shape[0]
    return inter / k


def sample_edges(sample_ratio, n_node)
    count = 0
    pairs = []
    while count < arg.sample_ratio * args.n_node:
        pairs.append(random.sample(range(args.n_node),2))
        count += 1
    sample_edges = torch.LongTensor(pairs)
    
    return sample_edges