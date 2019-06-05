import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
# import models

class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)

class rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        if hasattr(config, 'enc_num_layers'):
            num_layers = config.enc_num_layers
        else:
            num_layers = config.num_layers
        self.hidden_size = config.hidden_size

        if hasattr(config,'gru'):
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=num_layers, dropout=config.dropout, bidirectional=config.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=num_layers, dropout=config.dropout, bidirectional=config.bidirectional)

        self.config = config


    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, state





class rnn_decoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None, score_fn=None, extend_vocab_size=0):
        super(rnn_decoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        if hasattr(config,'gru'):
            self.rnn = StackedGRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)
        else:
            self.rnn = StackedLSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)

        self.score_fn = score_fn
        if self.score_fn.startswith('general'):
            self.linear = nn.Linear(config.hidden_size, config.emb_size)
        elif score_fn.startswith('concat'):
            self.linear_query = nn.Linear(config.hidden_size, config.hidden_size)
            self.linear_weight = nn.Linear(config.emb_size, config.hidden_size)
            self.linear_v = nn.Linear(config.hidden_size, 1)
        elif not self.score_fn.startswith('dot'):
            self.linear = nn.Linear(config.hidden_size, vocab_size)

        if self.score_fn.startswith('copy'):
            self.gen_linear = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Sigmoid())

        if hasattr(config, 'att_act'):
            activation = config.att_act
            print('use attention activation %s' % activation)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config
        self.extend_vocab_size = extend_vocab_size

    def forward(self, inputs, init_state, contexts, use_attention=True, compute_score=False, src=None, num_oovs=0):
        embs = self.embedding(inputs)
        outputs, state, attns = [], init_state, []
        self.attention.init_context(contexts)
        for emb in embs.split(1):
            x = emb.squeeze(0)
            output, state = self.rnn(x, state)
            if use_attention:
                output, attn_weights = self.attention(output, x, contexts)
                attns.append(attn_weights)
                if compute_score:
                    output = self.compute_score(output, src, attn_weights)
            output = self.dropout(output)
            outputs += [output]
        if not compute_score:
            outputs = torch.stack(outputs)
        return outputs, state