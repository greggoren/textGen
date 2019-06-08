from nltk import sent_tokenize,tokenize
import pickle
import sys
import pandas as pd

class LanguageHelper():
    def __init__(self):
        self.word2index = {"<PAD>":0,"<SOS>":1,"<EOS>":2}
        self.index2word = {0:"<PAD>",1:"<SOS>",2:"<EOS>"}
        self.longest_sequence = 0
        self.new_word_index = 3

    def retrieve_stats(self,df):
        for index, row in df.iterrows():
            sentence = row["sentence"]
            tokens = tokenize(sentence)
            for token in tokens:
                word = token.lower()
                if word not in self.word2index:
                    self.word2index[word]=self.new_word_index
                    self.index2word[self.new_word_index]=word
                    self.new_word_index+=1
            if len(tokens)>self.longest_sequence:
                self.longest_sequence=len(tokens)

    def save(self,filename):
        with open(filename,"wb") as f:
            pickle.dump(self,f)



if __name__=="__main__":
    df_filename = sys.argv[1]
    df = pd.read_csv(df_filename)
    lang_helper = LanguageHelper()
    lang_helper.retrieve_stats(df)
    lang_helper.save("corpusStats.pkl")