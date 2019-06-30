from model.DecoderLSTM import DecoderLSTM
from model.EncoderLSTM import EncoderLSTM
import torch
from torch import nn







class Seq2seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size,SOS_idx,EOS_idx,PAD_idx,n_layers,embeddings,criterion,seed,p,device,bidirectional):
        super(Seq2seq, self).__init__()
        self.SOS_idx,self.EOS_idx= SOS_idx,EOS_idx
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.criterion = criterion
        self.vocab_size = embeddings.shape[0]+3
        self.PAD_idx = PAD_idx
        self.device = device
        self.bidirectional=bidirectional
        self.encoder = EncoderRNN(input_vocab_size,hidden_size,embeddings,PAD_idx,seed,p,device,self.n_layers,bidirectional)
        self.decoder = DecoderLSTM(input_vocab_size, hidden_size, embeddings, PAD_idx, seed, p, device, self.n_layers)
        self.W = nn.Linear(hidden_size, output_vocab_size)

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x,lengths)
        loss = 0.0
        init_token = self.SOS_idx
        #input of <SOS>
        input = torch.LongTensor([init_token]*x.shape[0]).to(device)
        decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
        decoder_hidden_h, decoder_hidden_c = decoder_hidden
        # Teacher forcing : input sequence
        for i in range(y.shape[1]):
            input = y[:, i]
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (batch_size, vocab_size)
            h = self.W(decoder_output.squeeze(1)).squeeze(0)
            h = h.reshape((input.shape[0],self.vocab_size))
            loss+=self.criterion(h,input)

        loss = self.normalize_loss(loss,lengths)
        return loss

