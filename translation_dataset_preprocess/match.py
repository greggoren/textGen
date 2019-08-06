import os
from random import randrange
import pandas as pd
import sys

def read_file(fname):
    sentences = []
    with open(fname) as file:
        for line in file:
            sentences.append(line.split("\t")[1].rstrip())
    return sentences


def match(input_dir,target_dir,new_target_dir):
    if not os.path.exists(new_target_dir):
        os.makedirs(new_target_dir)
    for file in os.listdir(target_dir):
        rows= {}
        target_fname = target_dir+file
        input_fname = input_dir+file
        input_sentences = read_file(input_fname)
        target_sentences = read_file(target_fname)
        row = 0
        for index,sentence in enumerate(input_sentences):
            rand_index = randrange(len(target_sentences))
            target_sentence = target_sentences[rand_index]
            rows[row]={}
            rows[row]["query"]=file
            rows[row]["input_sentence"]=sentence
            rows[row]["target_sentence"]=target_sentence
            row+=1
        df = pd.DataFrame.from_dict(rows,orient="rows")
        df.to_csv(new_target_dir+file+".csv")

if __name__=="__main__":
    input_dir = sys.argv[1]
    target_dir = sys.argv[2]
    new_target_dir = sys.argv[3]
    match(input_dir,target_dir,new_target_dir)



