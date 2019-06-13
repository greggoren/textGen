from train import train_model
import gensim
import sys
import pandas as pd
if __name__=="__main__":

    w2v_model_file_path = sys.argv[1]
    data_set_file_path = sys.argv[2]
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_file_path,limit = 700000  ,binary=True)
    rows,cols = w2v_model.wv.vectors.shape
    SOS_idx = rows
    EOS_idx = rows+1
    PAD_idx = rows+2
    df = pd.read_csv(data_set_file_path,delimiter=",")
    n_layers = 1
    hidden_size = 200
    lr = 0.01
    batch_size = 20
    epochs = 5
    train_model(lr,batch_size,epochs,hidden_size,n_layers,w2v_model,SOS_idx,EOS_idx,PAD_idx,df)


