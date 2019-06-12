import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gensim
from copy import deepcopy


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda,path_to_embeddings):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.isCuda = isCuda
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()

        # initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))

        #load w2v model
        embeddings_model = self.load_embeddings_model(path_to_embeddings)
        self.embeddings = self.from_pretrained(embeddings_model.wv.vectors)


    def load_embeddings_model(self,path_to_embeddings):
        model = gensim.models.Word2Vec.load(path_to_embeddings)
        return model

    def from_pretrained(self,embeddings, freeze=True):
        working_matrix = deepcopy(embeddings)
        rows, cols = embeddings.shape
        added_rows = np.array([[rows]*cols,[rows+1]*cols])
        working_matrix = np.vstack((working_matrix,added_rows))
        embedding = torch.nn.Embedding(num_embeddings=rows+2, embedding_dim=cols)
        embedding.weight = torch.nn.Parameter(working_matrix)
        embedding.weight.requires_grad = not freeze
        return embedding



    def forward(self, input,hidden,input_lengths):
        tt = torch.cuda if self.isCuda else torch
        input = self.embeddings(input)
        input= torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        # h0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size))
        # c0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size))
        encoded_input, hidden = self.lstm(input, hidden)
        encoded_input, hidden = torch.nn.utils.rnn.pad_packed_sequence(encoded_input, batch_first=True)
        encoded_input = self.relu(encoded_input)
        return encoded_input

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.isCuda = isCuda
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax()
        # initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, input,initial_hidden_state):
        tt = torch.cuda if self.isCuda else torch
        # h0 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        # c0 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        decoded_output, hidden = self.lstm(input, initial_hidden_state)
        decoded_output = self.log_softmax(decoded_output)
        return decoded_output,hidden


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda,vocab_size,path_to_embeddings):
        super(LSTMAE, self).__init__()

        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda,path_to_embeddings)
        self.decoder = DecoderRNN(hidden_size, vocab_size, num_layers, isCuda)

    def save_encoder(self,fname):
        with open(fname,'wb') as f:
            torch.save(self.encoder,f)

    def save_decoder(self,fname):
        with open(fname,'wb') as f:
            torch.save(self.decoder,f)

    def save_model(self,fname):
        with open(fname, 'wb') as f:
            torch.save(self, f)


    def forward(self, input):
        encoded_input,hidden_enc = self.encoder(input)
        decoded_output,hidden_dec, = self.decoder(encoded_input,hidden_enc)
        return decoded_output,hidden_dec