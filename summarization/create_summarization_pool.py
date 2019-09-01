import os
import pandas as pd
import sys
from multiprocessing import Pool
from threading import Lock



def retrieve_queries(fname):
    queries = {}
    with open(fname) as f:
        for line in f:
            query = line.split(":")[1].rstrip()
            queries[query] = 0
    return queries


def get_appearance_indicator(paragraph, query):
    res = set(paragraph.replace("\"","/").split()).intersection(set(query.split()))
    return bool(res)


def write_file(queries,df):
    data_dir = "summarization_pool/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    seen = set([])
    for i, row in df.iterrows():
        paragraph = row["proc_paragraph"]
        for query in queries:
            if get_appearance_indicator(paragraph, query) and paragraph not in seen:

                fname = data_dir + "_".join([q.rstrip() for q in query.split()])
                f = open(fname, 'a')
                lock.acquire()
                f.write(query + '\t' + paragraph + "\n")
                lock.release()
                f.close()

                seen.add(paragraph)


def combine_results(results, final_file):
    for result in results:
        command = "cat " + result + " >> " + final_file
        os.popen(command)

from functools import partial

def initializer():
    global lock
    lock = Lock()

if __name__ == "__main__":
    paragraphs_file = sys.argv[1]
    queries_file = sys.argv[2]
    queries_stats = retrieve_queries(queries_file)
    queries = [q for q in queries_stats]
    df = pd.read_csv(paragraphs_file, delimiter=",", header=0, chunksize=500000)
    func = partial(write_file,queries)
    with Pool(12,initializer,()) as pool:
        results = pool.map(func, df)




