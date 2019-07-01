import torch
from torch import nn
from copy import deepcopy
import numpy as np
from torch.autograd import Variable






class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size,embeddings,PAD_idx,seed,p,device,n_layers,bidirectional):
        super(EncoderGRU, self).__init__()
        self.seed = seed

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.PAD_idx = PAD_idx
        self.relu = nn.ReLU
        self.device = device
        self.embedding = self.from_pretrained(embeddings)
        self.dropout = nn.Dropout(p)
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            embeddings.shape[1],
            int(hidden_size),
            num_layers=n_layers,
            batch_first=True,  # First dimension of input tensor will be treated as a batch dimension
            bidirectional=self.bidirectional,
            dropout=0.5
        )

    # word_inputs: (batch_size, seq_length), h: (h_or_c, layer_n_direction, batch, seq_length)
    def forward(self, word_inputs,input_lengths, hidden):
        embedded = self.embedding(word_inputs)
        gru_input = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        self.gru.flatten_parameters()
        output, hidden = self.gru(gru_input, hidden)
        encoded_out, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return encoded_out, hidden

    def init_hidden(self, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden = Variable(torch.zeros(self.n_layers*(1+self.bidirectional),batch_size,self.hidden_size).double().to(device))
        return hidden

    def from_pretrained(self,embeddings, freeze=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(self.seed)
        working_matrix = deepcopy(embeddings)
        rows, cols = embeddings.shape
        added_rows = np.array([np.random.rand(cols), np.random.rand(cols),np.random.rand(cols)])
        working_matrix=np.vstack([working_matrix,added_rows])
        working_matrix = torch.FloatTensor(working_matrix).to(device)
        embedding = torch.nn.Embedding(num_embeddings=rows+3, embedding_dim=cols,padding_idx=self.PAD_idx)
        embedding.weight = torch.nn.Parameter(working_matrix)
        embedding.weight.requires_grad = not freeze
        return embedding