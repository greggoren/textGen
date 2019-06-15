import torch
from torch import nn
import torch.nn.init as init
from copy import deepcopy
import numpy as np
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size,embeddings,n_layers=1):
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.relu = nn.ReLU
        # self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.embedding = self.from_pretrained(embeddings)
        init.normal_(self.embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(
            embeddings.shape[1],
            int(hidden_size),
            num_layers=n_layers,
            batch_first=True,  # First dimension of input tensor will be treated as a batch dimension
            bidirectional=False
        )

    # word_inputs: (batch_size, seq_length), h: (h_or_c, layer_n_direction, batch, seq_length)
    def forward(self, word_inputs,input_lengths, hidden):
        embedded = self.embedding(word_inputs)
        lstm_input = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.lstm(lstm_input, hidden)
        encoded_out, hidden = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        encoded_out = self.relu(encoded_out)
        return encoded_out, hidden

    def init_hidden(self, batches):
        # hidden = torch.zeros(2, self.n_layers*2, batches, int(self.hidden_size/2))
        hidden = torch.zeros(1, self.n_layers, batches, int(self.hidden_size))
        # if USE_CUDA: hidden = hidden.cuda()
        return hidden

    def from_pretrained(self,embeddings, freeze=True):
        working_matrix = deepcopy(embeddings)
        rows, cols = embeddings.shape
        added_rows = np.array([[rows]*cols,[rows+1]*cols])
        working_matrix = torch.from_numpy(np.vstack((working_matrix,added_rows))).float()
        embedding = torch.nn.Embedding(num_embeddings=rows+2, embedding_dim=cols)
        embedding.weight = torch.nn.Parameter(working_matrix)
        embedding.weight.requires_grad = not freeze
        return embedding