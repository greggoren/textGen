import pandas as pd
import sys
from multiprocessing import Pool
from functools import partial


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

def get_chunk_stats(queries_stats,df):
    queries = {q:0 for q in queries_stats}
    for i,row in df.iterrows():
        sentence = row["proc_sentence"]
        for query in queries:
            if get_appearance_indicator(sentence,query):
                queries[query]+=1
    return queries


def combine_results(queries,results):
    for result in results:
        for q in queries:
            queries[q]+=result[q]
    return queries

def write_stats(queries_stats,fname):
    with open(fname,'w') as f:
        for query in queries_stats:
            f.write(query+":"+str(queries_stats[query])+"\n")




if __name__=="__main__":
    sentences_file = sys.argv[1]
    queries_file = sys.argv[2]
    queries_stats = retrieve_queries(queries_file)
    results_fname = "query_appearance_histogram.txt"
    func = partial(get_chunk_stats,queries_stats)
    df = pd.read_csv(sentences_file,delimiter=",",header=0,chunksize=500000)
    with Pool(10) as pool:
        results = pool.map(func,df)
        queries_stats = combine_results(queries_stats,results)
        write_stats(queries_stats,results_fname)



