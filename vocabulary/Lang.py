from nltk import sent_tokenize,tokenize
import pickle
import sys
import pandas as pd
import logging
import os
import numpy as np
class LanguageHelper():
    def __init__(self):
        self.word2index = {"<PAD>":0,"<SOS>":1,"<EOS>":2}
        self.index2word = {0:"<PAD>",1:"<SOS>",2:"<EOS>"}
        self.new_word_index = 3
        self.word_count={}

    def retrieve_stats(self,df):
        program = os.path.basename(sys.argv[0])
        logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
        logging.root.setLevel(level=logging.INFO)
        logger.info("running %s" % ' '.join(sys.argv))
        rnum=0
        for chunk in df:
            for index,row in chunk.iterrows():
                if rnum%1000==0:
                    logger.info("in index "+str(rnum))
                sentence = row["proc_sentence"]
                sentence = str(sentence)
                if sentence=="":
                    continue
                # print(rnum,sentence)
                tokens = sentence.split()
                for token in tokens:
                    word = token.lower()
                    if word not in self.word2index:
                        self.word2index[word]=self.new_word_index
                        self.index2word[self.new_word_index]=word
                        self.new_word_index+=1
                        self.word_count[word]=1
                    else:
                        self.word_count[word] += 1
                rnum+=1
        logger.info("Finished run!")


    def save(self,filename):
        with open(filename,"wb") as f:
            pickle.dump(self,f)



if __name__=="__main__":
    df_filename = sys.argv[1]
    df = pd.read_csv(df_filename,sep=",",header=0,chunksize=100000)
    lang_helper = LanguageHelper()
    lang_helper.retrieve_stats(df)
    lang_helper.save("corpusStats.pkl")