import torch
import sys
import gensim
from modules.GreedySearch import greedy_generation
from modules.BeamSearch import beam_search_generation
from model.SequenceToSequence import Seq2seq

def convert_to_indices(sentence,w2v_model,EOS_idx,SOS_idx,device):

    indices = [w2v_model.wv.vocab.get(token).index for token in sentence.split()]
    indices.append(EOS_idx)
    # indices.insert(0,SOS_idx)
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
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path,binary=True,limit=5000)
    rows, cols = w2v_model.wv.vectors.shape
    SOS_idx = rows
    EOS_idx = rows + 1

    model = torch.load(model_file_path,map_location=device)
    model.eval()
    # input_sentences = [('the first known use of this word was in',torch.LongTensor([10]).to(device)),('karl marx became leading figure in the international and member of its general council',torch.LongTensor([14]).to(device)),('in addition to playing cricket for the university he also played football for oxford university',torch.LongTensor([16]).to(device)),('he was evacuated to malta where he died from his wounds on november',torch.LongTensor([14]).to(device))]
    input_sentences = [('the first known use of this word was in',torch.LongTensor([10]).to(device))
                       ,("due to its links to active workers movements the international became significant organisation",torch.LongTensor([14]).to(device))
                       ,("it developed mostly in the netherlands britain and the united states before and during the second world war",torch.LongTensor([19]).to(device))
                       ,("in the the number of people affected was estimated at per people worldwide",torch.LongTensor([14]).to(device))]




    greedy_output = open("GreedyReuslts.txt",'w')
    indices_dict = get_words_from_indices_dict(w2v_model,SOS_idx,EOS_idx)
    for i,inp in enumerate(input_sentences):
        x = convert_to_indices(inp[0],w2v_model,EOS_idx,SOS_idx,device)
        result_indices = greedy_generation(model,x,inp[1],50,device)
        generated_sentece = retrieve_sentence_from_indices(indices_dict,result_indices)
        greedy_output.write("Example:"+str(i+1)+"\n")
        greedy_output.write("Input: "+inp[0]+"\n")
        greedy_output.write("Generated Output: "+generated_sentece+"\n")
    greedy_output.close()

    beam_output = open("BeamResults.txt",'w')
    for i, inp in enumerate(input_sentences):
        x = convert_to_indices(inp[0], w2v_model, EOS_idx, SOS_idx, device)
        output_indices = beam_search_generation(model,x,inp[1],device)
        generated_sentence = retrieve_sentence_from_indices(indices_dict,output_indices[0][0])
        beam_output.write(("Example:"+str(i+1)+"\n"))
        beam_output.write("Input: "+inp[0]+"\n")
        beam_output.write("Generated Output: " + generated_sentence + "\n")
    beam_output.close()




