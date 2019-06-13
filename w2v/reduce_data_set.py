import gensim
import pandas as pd
import sys
if __name__=="__main__":
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    df = pd.read_csv(data_path,delimiter=",",chunksize=100000)
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path,limit = 700000  ,binary=True)
    new_df = pd.DataFrame(columns=['article_uuid', 'sentence', 'proc_sentence', 'proc_len'])
    for chunk in df:
        for index,row in chunk.iterrows():
            proc_sentence = row["proc_sentence"]
            sentence = str(proc_sentence)
            write = True
            for token in sentence.split():
                if token not in model.wv:
                    write = False
                    break
            if write:
                new_df.append(row)
    new_df.to_csv("reduced_wiki_sentences.csv")

