from model.DecoderRNN import DecoderRNN
from model.EncoderRNN import EncoderRNN
import torch
from torch import nn







class Seq2seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size,SOS_idx,EOS_idx,PAD_idx,n_layers,embeddings,criterion):
        super(Seq2seq, self).__init__()
        self.SOS_idx,self.EOS_idx= SOS_idx,EOS_idx
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.criterion = criterion
        self.vocab_size = embeddings.shape[0]+3
        self.PAD_idx = PAD_idx
        self.encoder = EncoderRNN(input_vocab_size, hidden_size,embeddings,PAD_idx,self.n_layers)
        # self.encoder = self.encoder
        self.decoder = DecoderRNN(input_vocab_size,hidden_size,embeddings,PAD_idx,self.n_layers)
        # self.decoder = nn.DataParallel(self.decoder,dim=1)
        self.W = nn.Linear(hidden_size, output_vocab_size)

        self.softmax = nn.Softmax()

    def _forward_encoder(self, x,lengths):
        batch_size = x.shape[0]
        init_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(x, lengths,init_hidden)
        encoder_hidden_h, encoder_hidden_c = encoder_hidden

        self.decoder_hidden_h = encoder_hidden_h.permute(1,0,2).reshape(batch_size, self.n_layers, self.hidden_size).permute(1,0,2)
        self.decoder_hidden_c = encoder_hidden_c.permute(1,0,2).reshape(batch_size, self.n_layers, self.hidden_size).permute(1,0,2)
        return self.decoder_hidden_h, self.decoder_hidden_c


    def normalize_loss(self,loss,lengths):
        for i in range(loss.shape[0]):
            loss[i] = loss[i] / float(lengths[i])
        return loss.mean()


    def forward(self, x, y,lengths):
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x,lengths)
        loss = 0.0
        for i in range(y.shape[1]):
            input = y[:, i]
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (batch_size, vocab_size)
            h = self.W(decoder_output.squeeze(1)).squeeze(0)
            h = h.reshape((input.shape[0],self.vocab_size))
            loss+=self.criterion(h,input)
            # h: (batch_size, vocab_size, 1)
            # H.append(h.unsqueeze(2))
        # H: (batch_size, vocab_size, seq_len)
        # return torch.cat(H, dim=2)
        loss = self.normalize_loss(loss,lengths)
        return loss

    def forward_test(self, x):
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x)

        current_y = self.SOS_idx
        result = [current_y]
        counter = 0
        while current_y != self.EOS_idx and counter < 100:
            input = torch.tensor([current_y])
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (vocab_size)
            h = self.W(decoder_output.squeeze(1)).squeeze(0)
            y = self.softmax(h)
            _, current_y = torch.max(y, dim=0)
            current_y = current_y.item()
            result.append(current_y)
            counter += 1

        return result