import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy

class DecoderRNN(nn.Module):
    def __init__(self, input_vector_size ,hidden_size,embeddings, PAD_idx,seed,p,device,n_layers=1):
        super(DecoderRNN, self).__init__()
        self.seed = seed
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.PAD_idx = PAD_idx
        self.embedding = self.from_pretrained(embeddings)
        # self.embedding = nn.DataParallel(self.embedding)
        self.lstm = nn.LSTM(input_vector_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(p)

    def from_pretrained(self,embeddings, freeze=True):
        device = torch.device("cuda" if torch.cudva.is_available() else "cpu")
        np.random.seed(self.seed)
        working_matrix = deepcopy(embeddings)
        rows, cols = embeddings.shape
        added_rows = np.array([np.random.rand(cols), np.random.rand(cols),np.random.rand(cols)])
        working_matrix=np.vstack((working_matrix, added_rows))
        working_matrix = torch.FloatTensor(working_matrix).to(device)
        embedding = torch.nn.Embedding(num_embeddings=rows +3 , embedding_dim=cols,padding_idx=self.PAD_idx)
        embedding.weight = torch.nn.Parameter(working_matrix)
        embedding.weight.requires_grad = not freeze
        return embedding

    def forward(self, word_inputs, hidden):
        # Note: we run this one by one
        # embedded (batch_size, 1, hidden_size)
        embedded = self.dropout(self.embedding(word_inputs)).unsqueeze_(1)
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(embedded, hidden)

        return output, hidden