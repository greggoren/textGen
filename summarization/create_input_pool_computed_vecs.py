import pandas as pd
from tqdm import tqdm
import nltk
from multiprocessing import Pool,cpu_count
from copy import deepcopy
import math
import sys
import gensim
from functools import partial
import os
import numpy as np
import pickle

def contain_digits(token):
    return any(char.isdigit() for char in token)


def initializer():
    global sw
    sw = set(nltk.corpus.stopwords.words('english'))
    sw.add("s")



def read_df(fname):
    df = pd.read_csv(fname,delimiter="\t",names=["query","input_paragraph"])
    return df


def list_multiprocessing(param_lst, func, **kwargs):
    workers = kwargs.pop('workers')
    with Pool(workers,initializer,()) as p:
        apply_lst = [([params], func, i, kwargs) for i, params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))
    return [_[1] for _ in result]


def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params, **kwargs)

def read_queries(fname):
    result=[]
    with open(fname) as f:
        for line in f:
            query = line.split(":")[1].rstrip()
            result.append("_".join(query.split()))
    return result

def get_args(translations_dir,queries):
    return [translations_dir+query for query in queries]




def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    if sumxx==0 or sumyy==0:
        return 0
    return sumxy/math.sqrt(sumxx*sumyy)


def is_in(query,text):
    res = set(query.split()).intersection(text.rstrip().split())
    return bool(res)


def load_vector(fname):
    with open(fname,"rb") as file:
        return pickle.load(file)


def calculate_similarities(query, centroid,sentence, fname):
    if is_in(query, sentence):
        return -float("inf")
    cent_sentence = load_vector(fname)
    if cent_sentence is None:
        return -float("inf")
    return cosine_similarity(centroid,cent_sentence)


def insert_to_queue(q,sim,paragraph,min_val):
    if len(q)<100:
        q.append((sim,paragraph))
        min_val = min(q,key=lambda x:x[0])[0]
        return q,min_val
    elif sim>min_val:
        q.append((sim,paragraph))
        q= sorted(q,key=lambda x:x[0])
        return q[1:],q[1][0]
    return q,min_val

def find_most_similar_sentences(input_vector_dir, input_file,cluster_dir, input_dir, query):
    df = pd.read_csv(input_file,delimiter = ",",header=0,chunksize=100000)
    centroid = load_vector(cluster_dir + query+".pkl")
    queue = []
    q = " ".join(query.split("_"))
    min_val = -float("inf")
    global_index =0
    for chunk in df:
        for row in chunk.itertuples():
            sentence =str(row[4])
            if sentence=="":
                continue
            fname = input_vector_dir+str(global_index%1000)+"/"+str(global_index)+".pkl"
            sim = calculate_similarities(q,centroid=centroid,sentence=sentence,fname=fname)
            queue,min_val=insert_to_queue(queue,sim,sentence,min_val)
            global_index+=1
    rows ={}
    i=0
    for item in queue:
        row={}
        row["query"]=q
        row["input_paragraph"] = item[1]
        rows[i]=row
        i+=1
    pd.DataFrame.from_dict(rows,orient="index").to_csv(input_dir+query)

def recovery_mode(queries,output_dir,target_dir):
    finished = [f.replace(".csv","") for f in os.listdir(output_dir)]
    updated_queries = [q for q in queries if q not in finished and os.path.isfile(target_dir+q)]
    return updated_queries


if __name__=="__main__":
    input_dir = sys.argv[1]
    target_dir = sys.argv[2]
    queries_file = sys.argv[3]
    model_file = sys.argv[4]
    input_file = sys.argv[5]
    input_vector_dir = sys.argv[6]
    cluster_dir = sys.argv[7]
    recovery = sys.argv[8]
    queries = read_queries(queries_file)
    print("there are ",str(len(queries)),"queries",flush=True)
    if recovery == "True":
        queries = recovery_mode(queries,input_dir,target_dir)
        print("Recovery mode detected, updated number of queries:" + str(len(queries)))
    func = partial(find_most_similar_sentences,input_vector_dir, input_file,cluster_dir,input_dir)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    workers = cpu_count()-1
    list_multiprocessing(queries,func,workers=workers)







