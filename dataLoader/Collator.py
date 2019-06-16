import numpy as np
import torch


class PadCollator(object):

    def __init__(self,PAD_idx):
        self.PAD_idx = PAD_idx

    def sort_batch(self,batch,  lengths):
        """
        Sort a minibatch by the length of the sequences with the longest sequences first
        return the sorted batch targes and sequence lengths.
        This way the output can be used by pack_padded_sequences(...)
        """
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        return torch.LongTensor(seq_tensor).cuda(),seq_lengths

    def __call__(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        seqs,  lengths = batch_split[0], batch_split[1]
        max_length = int(max(lengths))

        padded_seqs = np.zeros((batch_size, max_length))
        for i, l in enumerate(lengths):
            padded_seqs[i, 0:l] = seqs[i][0:l]
            padded_seqs[i,l:] = [self.PAD_idx]*(max_length-l)
        return self.sort_batch(padded_seqs,torch.LongTensor(lengths).cuda())

# def pad_and_sort_batch(DataLoaderBatch):
#     """
#     DataLoaderBatch should be a list of (sequence, target, length) tuples...
#     Returns a padded tensor of sequences sorted from longest to shortest,
#     """
#     batch_size = len(DataLoaderBatch)
#     batch_split = list(zip(*DataLoaderBatch))
#
#     seqs, targs, lengths = batch_split[0], batch_split[1], batch_split[2]
#     max_length = max(lengths)
#
#     padded_seqs = np.zeros((batch_size, max_length))
#     for i, l in enumerate(lengths):
#         padded_seqs[i, 0:l] = seqs[i][0:l]
#
#     return sort_batch(torch.tensor(padded_seqs), torch.tensor(targs).view(-1, 1), torch.tensor(lengths))
