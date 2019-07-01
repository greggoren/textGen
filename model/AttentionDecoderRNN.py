from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,embeddings,seed,Pad_idx,n_layers,dropout_p,max_length,bidirectional):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.seed = seed
        self.PAD_idx = Pad_idx
        self.embedding,self.input_embedding_size = self.from_pretrained(embeddings)
        self.attn = nn.Linear(self.hidden_size+self.hidden_size*(1+bidirectional) , 1)
        self.attn_combined = nn.Linear(self.input_embedding_size+self.hidden_size*(1+bidirectional) ,self.input_embedding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(self.input_embedding_size, self.hidden_size, num_layers=n_layers, batch_first=True, bidirectional=False)

    def from_pretrained(self,embeddings, freeze=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(self.seed)
        working_matrix = deepcopy(embeddings)
        rows, cols = embeddings.shape
        added_rows = np.array([np.random.rand(cols), np.random.rand(cols),np.random.rand(cols)])
        working_matrix=np.vstack((working_matrix, added_rows))
        working_matrix = torch.FloatTensor(working_matrix).to(device)
        embedding = torch.nn.Embedding(num_embeddings=rows +3 , embedding_dim=cols,padding_idx=self.PAD_idx)
        embedding.weight = torch.nn.Parameter(working_matrix)
        embedding.weight.requires_grad = not freeze
        return embedding,cols

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        weights = []
        for i in range(len(encoder_outputs[0])):

            weights.append(self.attn(torch.cat((hidden[0],
                                                encoder_outputs[:,i]), dim=1)))

        normalized_weights = F.softmax(torch.cat(weights, 1), 1)

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                                 encoder_outputs)

        attn_comb = torch.cat((attn_applied.squeeze(1), embedded), dim=1)
        input_lstm = self.attn_combined(attn_comb).unsqueeze_(1)
        input_lstm = self.relu(input_lstm)
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(input_lstm, (hidden[0].unsqueeze(0),hidden[1].unsqueeze(0)))

        return output, hidden, normalized_weights

