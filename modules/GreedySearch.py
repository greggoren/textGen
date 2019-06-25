import torch
from torch.nn.functional import log_softmax

def greedy_generation(model, x, lengths,max_generation_len,device):
    decoder_hidden_h, decoder_hidden_c = model._forward_encoder(x, lengths)

    current_y = model.SOS_idx
    result = [current_y]
    counter = 0
    while current_y != model.EOS_idx and counter < max_generation_len:
        input = torch.LongTensor([current_y]).to(device)
        decoder_output, decoder_hidden = model.decoder(input, (decoder_hidden_h, decoder_hidden_c))
        decoder_hidden_h, decoder_hidden_c = decoder_hidden
        # h: (vocab_size)
        h = model.W(decoder_output.squeeze(1)).squeeze(0)
        # y = softmax(h)
        y = log_softmax(h,dim=0)
        _, current_y = torch.max(y, dim=0)
        current_y = current_y.item()
        result.append(current_y)
        counter += 1

    return result