import numpy as np
import torch


class DefCollator(object):


    def __call__(self, DataLoaderBatch):

        batch_split = list(zip(*DataLoaderBatch))

        seqs, labels ,lengths = batch_split[0], batch_split[1],batch_split[2]

        return (seqs,labels,lengths)

