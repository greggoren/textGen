import gensim
import pandas as pd
import sys
import os
import logging
from multiprocessing.pool import Pool
from functools import partial
def f(sentence,model):
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

if __name__=="__main__":
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    df = pd.read_csv(data_path,delimiter=",",chunksize=100000)
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path,limit = 700000  ,binary=True)
    new_df = pd.DataFrame(columns=['article_uuid', 'proc_sentence', 'proc_len'])
    rnum=1
    for chunk in df:
        for index,row in chunk.iterrows():
            logger.info("Processing row:"+str(rnum))
            proc_sentence = row["proc_sentence"]
            del row['sentence']
            sentence = str(proc_sentence)
            write = True
            for token in sentence.split():
                if token not in model.wv:
                    write = False
                    break
            if write:
                new_df.append(row)
            rnum+=1
    new_df.to_csv("reduced_wiki_sentences1.csv")

