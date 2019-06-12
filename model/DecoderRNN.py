import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np
from copy import deepcopy

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size,embeddings, n_layers=1):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.embedding = self.from_pretrained(embeddings)
        # init.normal_(self.embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=False)


    def from_pretrained(self,embeddings, freeze=True):
        working_matrix = deepcopy(embeddings)
        rows, cols = embeddings.shape
        added_rows = np.array([[rows]*cols,[rows+1]*cols])
        working_matrix = np.vstack((working_matrix,added_rows))
        embedding = torch.nn.Embedding(num_embeddings=rows+2, embedding_dim=cols)
        embedding.weight = torch.nn.Parameter(working_matrix)
        embedding.weight.requires_grad = not freeze
        return embedding

    def forward(self, word_inputs, hidden):
        # Note: we run this one by one
        # embedded (batch_size, 1, hidden_size)
        embedded = self.embedding(word_inputs).unsqueeze_(1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden