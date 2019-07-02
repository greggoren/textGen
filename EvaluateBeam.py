import torch
from  nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
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
from modules.BeamSearch import beam_search_generation



def get_words_from_indices_dict(model,SOS_idx,EOS_idx):
    indices_dict = {v.index:k for (k, v) in model.wv.vocab.items()}
    indices_dict[SOS_idx]="<SOS>"
    indices_dict[EOS_idx]="<EOS>"
    indices_dict[EOS_idx+1]="<PAD>"
    return indices_dict

def retrieve_sentence_from_indices(indices_dict,results,EOS_idx):
    # sentence = " ".join([indices_dict[i] for i in indices])
    sentences= []
    for result in results:
        tmp = []
        for i in result:
            if i.item()==EOS_idx:
                tmp.append(indices_dict[EOS_idx])
                break
            tmp.append(indices_dict[i.item()])
        sentences.append(tmp)
    return sentences









def calc_bleu(references,candidates):
    res = []
    for i,icand in enumerate(candidates):
        try:
            cc = SmoothingFunction()
            cand = icand[1:]
            ref = [references[i],]
            bleu=sentence_bleu(ref, cand,smoothing_function=cc.method3)
            res.append(bleu)
        except:
            print("here")
    return res

def evaluate_attn_beam(model, collator, indices_dict, device, eval_data):
    total = []
    for i, batch in enumerate(eval_data):
        batch = collator(batch)
        sequence, label, length = batch
        ref = retrieve_sentence_from_indices(indices_dict, sequence,model.EOS_idx)
        result_indices = beam_search_generation(model, sequence, length, device)
        generated_senteces = retrieve_sentence_from_indices(indices_dict,result_indices,model.EOS_idx)
        res = calc_bleu(ref,generated_senteces)
        total.extend(res)
    result = np.mean(total)
    return result


if __name__=="__main__":
    results = []
    data_set_file_path = sys.argv[1]
    w2v_model_path = sys.argv[2]
    models_folder = sys.argv[3]
    suffix = sys.argv[4]
    print(" ".join(sys.argv))
    batch_size = 100
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
        # model_file_name = models_folder+"/model_"+str(i)
        model_file_name = models_folder+"/model_20"
        model = torch.load(model_file_name, map_location=device)
        model.eval()
        tmp_res = evaluate_attn_beam(model, collator, indices_dict, device, data_loading)
        results.append(tmp_res)
    with open("eval_bleu_beam_"+suffix+".pkl",'wb') as f:
        pickle.dump(results,f)


