from train import train_model
import gensim
import sys
import pandas as pd
import logging
import os

if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    w2v_model_file_path = sys.argv[1]
    data_set_file_path = sys.argv[2]
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_file_path,binary=True)
    rows,cols = w2v_model.wv.vectors.shape
    SOS_idx = rows
    EOS_idx = rows+1
    PAD_idx = rows+2
    df = pd.read_csv(data_set_file_path,delimiter=",",nrows=5000,header=0)
    n_layers = 1
    hidden_size = 200
    lr = 0.01
    batch_size = 20
    epochs = 5
    train_model(lr,batch_size,epochs,hidden_size,n_layers,w2v_model,SOS_idx,EOS_idx,PAD_idx,df,logger)


