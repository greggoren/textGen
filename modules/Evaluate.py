import torch
from  nltk.translate.bleu_score import sentence_bleu
import numpy as np
from dataLoader.DataLoader import Loader
from torch.utils.data import DataLoader
from dataLoader.Collator import PadCollator
from dataLoader.DefCollate import DefCollator
import pandas as pd
import sys
import gensim
from os import listdir
import pickle
def get_words_from_indices_dict(model,SOS_idx,EOS_idx):
    indices_dict = {v.index:k for (k, v) in model.wv.vocab.items()}
    indices_dict[SOS_idx]="<SOS>"
    indices_dict[EOS_idx]="<EOS>"
    return indices_dict

def retrieve_sentence_from_indices(indices_dict,indices):
    # sentence = " ".join([indices_dict[i] for i in indices])
    return [indices_dict[i] for i in indices][1:]


def greedy_generation(model, x, lengths,max_generation_len,device):
    decoder_hidden_h, decoder_hidden_c = model._forward_encoder(x, lengths)
    softmax = torch.nn.Softmax(dim=0)
    current_y = model.SOS_idx
    result = [current_y]
    counter = 0
    while current_y != model.EOS_idx and counter < max_generation_len:
        input = torch.LongTensor([current_y]).to(device)
        decoder_output, decoder_hidden = model.decoder(input, (decoder_hidden_h, decoder_hidden_c))
        decoder_hidden_h, decoder_hidden_c = decoder_hidden
        h = model.W(decoder_output.squeeze(1).squeeze(0))
        y = softmax(h)
        current_y = y.max(0)[1].item()
        result.append(current_y)
        counter += 1

    return result

def greedy_generation_attn(model, x, lengths,max_generation_len,device):
    decoder_hidden_h, decoder_hidden_c,encoder_outputs = model._forward_encoder(x, lengths)
    softmax = torch.nn.Softmax(dim=0)
    current_y = model.SOS_idx
    result = [current_y]
    counter = 0
    while current_y != model.EOS_idx and counter < max_generation_len:
        input = torch.LongTensor([current_y]).to(device)
        decoder_output, decoder_hidden,_ = model.decoder(input, (decoder_hidden_h.squeeze(0), decoder_hidden_c.squeeze(0)),encoder_outputs)
        decoder_hidden_h, decoder_hidden_c = decoder_hidden
        h = model.W(decoder_output.squeeze(1).squeeze(0))
        y = softmax(h)
        current_y = y.max(0)[1].item()
        result.append(current_y)
        counter += 1

    return result


def calc_bleu(ref,cand):
    bleu1=sentence_bleu(ref, cand, weights=(1, 0, 0, 0))
    bleu2=sentence_bleu(ref, cand, weights=(0, 1, 0, 0))
    bleu3=sentence_bleu(ref, cand, weights=(0, 0, 1, 0))
    bleu4=sentence_bleu(ref, cand, weights=(0, 0, 0, 1))
    return bleu1,bleu2,bleu3,bleu4

def evaluate_attn(model, collator, indices_dict, device, eval_data):
    bleu1_stats,bleu2_stats,bleu3_stats,bleu4_stats=[],[],[],[]
    for i, batch in enumerate(eval_data):
        true,_,_ = batch
        print(true)
        ref = [retrieve_sentence_from_indices(indices_dict, true),]
        batch = collator(batch)
        sequence, label, length = batch
        result_indices = greedy_generation_attn(model,sequence,length,50,device)
        generated_sentece = retrieve_sentence_from_indices(indices_dict,result_indices)
        bleu1, bleu2, bleu3, bleu4 = calc_bleu(ref,generated_sentece)
        bleu1_stats.append(bleu1)
        bleu2_stats.append(bleu2)
        bleu3_stats.append(bleu3)
        bleu4_stats.append(bleu4)
    return np.mean(bleu1_stats),np.mean(bleu2_stats),np.mean(bleu3_stats),np.mean(bleu4_stats)


if __name__=="__main__":
    results = []
    data_set_file_path = sys.argv[1]
    w2v_model_path = sys.argv[2]
    models_folder = sys.argv[3]
    suffix = sys.argv[4]
    batch_size = 1
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path,binary=True,limit=5000)
    rows, cols = w2v_model.wv.vectors.shape
    SOS_idx = rows
    EOS_idx = rows + 1
    PAD_idx = rows + 2
    indices_dict = get_words_from_indices_dict(w2v_model,SOS_idx,EOS_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(data_set_file_path,delimiter=",",header=0)
    collator = PadCollator(PAD_idx, device)
    def_collator = DefCollator()
    data = Loader(df, w2v_model, PAD_idx, EOS_idx, SOS_idx)
    data_loading = DataLoader(data, num_workers=4, shuffle=True, batch_size=batch_size, collate_fn=def_collator)
    epochs = len(listdir(models_folder))
    for i in range(epochs):
        data_loading = DataLoader(data, num_workers=4, shuffle=True, batch_size=batch_size, collate_fn=def_collator)
        model_file_name = models_folder+"/model_"+str(i)
        model = torch.load(model_file_name, map_location=device)
        tmp_res = evaluate_attn(model, collator, indices_dict, device, data_loading)
        results.append(tmp_res)
    with open("eval_bleu_"+suffix+".pkl") as f:
        pickle.dump(results,f)


