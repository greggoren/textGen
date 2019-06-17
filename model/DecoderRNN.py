import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy

class DecoderRNN(nn.Module):
    def __init__(self, input_vector_size ,hidden_size,embeddings, PAD_idx,n_layers=1):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.PAD_idx = PAD_idx
        self.embedding = self.from_pretrained(embeddings)
        # self.embedding = nn.DataParallel(self.embedding)
        self.lstm = nn.LSTM(input_vector_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=False)


    def from_pretrained(self,embeddings, freeze=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        working_matrix = deepcopy(embeddings)
        rows, cols = embeddings.shape
        added_rows = np.array([[rows] * cols, [rows + 1] * cols,[rows+2]*cols])
        working_matrix=np.vstack((working_matrix, added_rows))
        working_matrix = torch.FloatTensor(working_matrix).to(device)
        embedding = torch.nn.Embedding(num_embeddings=rows +3 , embedding_dim=cols,padding_idx=self.PAD_idx)
        embedding.weight = torch.nn.Parameter(working_matrix)
        embedding.weight.requires_grad = not freeze
        return embedding

    def forward(self, word_inputs, hidden):
        # Note: we run this one by one
        # embedded (batch_size, 1, hidden_size)
        embedded = self.embedding(word_inputs).unsqueeze_(1)
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden