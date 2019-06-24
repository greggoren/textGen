import torch
import sys
import gensim
from modules.GreedySearch import greedy_generation


def convert_to_indices(sentence,w2v_model,EOS_idx,SOS_idx,device):

    indices = [w2v_model.wv.vocab.get(token).index for token in sentence.split()]
    indices.append(EOS_idx)
    indices.insert(0,SOS_idx)
    return torch.LongTensor([indices]).to(device)


def get_words_from_indices_dict(model,SOS_idx,EOS_idx):
    indices_dict = {v.index:k for (k, v) in model.wv.vocab.items()}
    indices_dict[SOS_idx]="<SOS>"
    indices_dict[EOS_idx]="<EOS>"
    return indices_dict

def retrieve_sentence_from_indices(indices_dict,indices):
    sentence = " ".join([indices_dict[i] for i in indices])
    return sentence

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file_path = sys.argv[1]
    w2v_model_path = sys.argv[2]
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path,binary=True,limit=15000)
    rows, cols = w2v_model.wv.vectors.shape
    SOS_idx = rows
    EOS_idx = rows + 1
    model = torch.load(model_file_path,map_location=device)
    input_sentences = [('the first known use of this word was in',torch.LongTensor([11])),('karl marx became leading figure in the international and member of its general council',torch.LongTensor([15])),('in addition to playing cricket for the university he also played football for oxford university',torch.LongTensor([17])),('he was evacuated to malta where he died from his wounds on november',torch.LongTensor([15]))]
    greedy_output = open("GreedyReuslts.txt",'w')
    indices_dict = get_words_from_indices_dict(w2v_model,SOS_idx,EOS_idx)
    for i,inp in enumerate(input_sentences):
        x = convert_to_indices(inp[0],w2v_model,EOS_idx,SOS_idx,device)
        result_indices = greedy_generation(model,x,inp[1],50)
        generated_sentece = retrieve_sentence_from_indices(indices_dict,result_indices)
        greedy_output.write("Example:"+str(i+1)+"\n")
        greedy_output.write("Input: "+inp[0]+"\n")
        greedy_output.write("Generated Output: "+generated_sentece+"\n")
    greedy_output.close()






