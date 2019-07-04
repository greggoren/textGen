import os
import pandas as pd
import sys
from multiprocessing import Pool


def retrieve_queries(fname):
    queries = {}
    with open(fname) as f:
        for line in f:
            query = line.split(":")[1].rstrip()
            queries[query]=0
    return queries


def get_appearance_indicator(sentence,query):
    res = set(sentence.split()).intersection(set(query.split()))
    return bool(res)

def write_file(args):
    query,df = args
    data_dir = "data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    fname = data_dir+"_".join([q.rstrip() for q in query.split()])
    f = open(fname,'w')
    seen=set([])
    for i,row in df.iterrows():
        sentence = row["proc_sentence"]
        if len(seen)>=100000:
            break
        if not get_appearance_indicator(sentence,query) and sentence not in seen:
            f.write(query+'\t'+sentence+"\n")
    return fname

def combine_results(results,final_file):
    for result in results:
        command= "cat "+result+" >> "+final_file
        os.popen(command)







if __name__=="__main__":
    sentences_file = sys.argv[1]
    queries_file = sys.argv[2]
    queries_stats = retrieve_queries(queries_file)
    queries = [q for q in queries_stats]
    df = pd.read_csv(sentences_file,delimiter=",",header=0,chunksize=250000)
    args = zip(queries,df[:len(queries)])
    final_file = "input_senteces.txt"
    with Pool(12) as pool:
        results = pool.map(write_file,args)
        combine_results(results,final_file)



