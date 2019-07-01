from model.AttentionDecoderLSTM import AttnDecoderLSTM
from model.EncoderRNN import EncoderRNN
import torch
from torch import nn
import random






class Seq2seqAttn(nn.Module):
    def __init__(self, input_vocab_size, hidden_size,SOS_idx,EOS_idx,PAD_idx,n_layers,embeddings,criterion,seed,p,device,max_length,bidirectional):
        super(Seq2seqAttn, self).__init__()
        self.SOS_idx,self.EOS_idx= SOS_idx,EOS_idx
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.criterion = criterion
        self.vocab_size = embeddings.shape[0]+3
        self.PAD_idx = PAD_idx
        self.device = device
        self.bidirectional = bidirectional

        self.encoder = EncoderRNN(input_vocab_size, hidden_size, embeddings, PAD_idx, seed, p, device, self.n_layers, bidirectional)
        self.decoder = AttnDecoderLSTM(hidden_size, self.vocab_size, embeddings, seed, PAD_idx, self.n_layers, p, max_length, self.bidirectional)
        self.W = nn.Linear(hidden_size, self.vocab_size)

    def _forward_encoder(self, x,lengths):
        batch_size = x.shape[0]
        init_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(x, lengths,init_hidden)
        encoder_hidden_h, encoder_hidden_c = encoder_hidden


        return encoder_hidden_h[0].unsqueeze_(0),encoder_hidden_c[0].unsqueeze_(0),encoder_outputs


    def normalize_loss(self,loss,lengths):
        for i in range(loss.shape[0]):
            loss[i] = loss[i] / float(lengths[i])
        return loss.mean()


    def forward(self, x, y,lengths):
        softmax = nn.Softmax(dim=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder_hidden_h, decoder_hidden_c,encoder_outputs = self._forward_encoder(x,lengths)
        loss = 0.0
        init_token = self.SOS_idx
        teacher_forcing = True if random.random() > 0.3 else False #decide if employ teacher forcing
        #input of <SOS>
        input = torch.LongTensor([init_token]*x.shape[0]).to(device)

        for i in range(y.shape[1]):
            decoder_output, decoder_hidden, _ = self.decoder(input,
                                                             (decoder_hidden_h.squeeze(0), decoder_hidden_c.squeeze(0)),
                                                             encoder_outputs, lengths)
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (batch_size, vocab_size)
            h = self.W(decoder_output.squeeze(1)).squeeze(0)
            h = h.reshape((input.shape[0], self.vocab_size))
            if teacher_forcing:
                input = y[:, i]
            else:
                dist = softmax(h)
                input = dist.max(1)[1]

            loss+=self.criterion(h,y[:,i])

        loss = self.normalize_loss(loss,lengths)
        return loss

