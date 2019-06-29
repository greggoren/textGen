from trainAttn import train_model
import gensim
import sys
import logging
import os

if __name__=="__main__":
    sys.setrecursionlimit(10000)
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    w2v_model_file_path = sys.argv[1]
    data_set_file_path = sys.argv[2]

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_file_path,binary=True,limit=5000)
    if len(sys.argv)>3:
        output_model = sys.argv[3]
        w2v_model.wv.save_word2vec_format(output_model,binary=True)

    rows,cols = w2v_model.wv.vectors.shape
    SOS_idx = rows
    EOS_idx = rows+1
    PAD_idx = rows+2
    n_layers = 1
    hidden_size = 512
    lrs = [0.1,]
    batch_sizes = [100]
    epochs = 100
    dropout = 0.5
    random_seed = 9001
    bidirectional = True
    for lr in lrs:
        for batch_size in batch_sizes:
            train_model(lr,batch_size,epochs,hidden_size,n_layers,w2v_model,SOS_idx,EOS_idx,PAD_idx,data_set_file_path,random_seed,dropout,bidirectional,logger)


