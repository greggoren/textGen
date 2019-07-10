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

def clean_sentence(sentence):
    return [token for token in sentence.rstrip().split() if token not in sw]


def initializer():
    global sw
    sw = set(nltk.corpus.stopwords.words('english'))


def read_df(fname):
    df = pd.read_csv(fname,delimiter="\t",names=["query","input_sentence"])
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


def get_sentence_centroid(sentence):
    sum_vector = None
    denom = 0
    for token in clean_sentence(sentence):
        if token not in model.wv:
            continue
        vector = model.wv[token]
        if sum_vector is None:
            sum_vector=deepcopy(vector)
        else:
            sum_vector+=vector
        denom+=1
    if sum_vector is None:
        return None
    return sum_vector/denom


def get_centroid_of_cluster(df):
    sum_vector = None
    denom = 0
    for idx,row in df.iterrows():
        vector = get_sentence_centroid(row["input_sentence"])
        if sum_vector is None:
            sum_vector = deepcopy(vector)
        else:
            sum_vector+=vector
        denom+=1
    return sum_vector/denom

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


def is_in(query,sentence):
    res = set(query.split()).intersection(sentence.rstrip().split())
    return bool(res)


def calculate_similarities(query, centroid, sentence):
    if is_in(query, sentence):
        return -float("inf")
    cent_sentence = get_sentence_centroid(sentence)
    if cent_sentence is None:
        return -float("inf")
    return cosine_similarity(centroid,cent_sentence)


def insert_to_queue(q,sim,sentence,min_val):
    if len(q)<10000:
        q.append((sim,sentence))
        min_val = min(q,key=lambda x:x[0])[0]
        return q,min_val
    elif sim>min_val:
        q.append((sim,sentence))
        q= sorted(q,key=lambda x:x[0])
        return q[1:],q[1][0]
    return q,min_val

def find_most_similar_sentences(input_file,translation_dir,input_dir,query):
    global model
    df = pd.read_csv(input_file,delimiter = ",",header=0,chunksize=100000)
    cluster = read_df(translation_dir+query)
    centroid = get_centroid_of_cluster(cluster)
    queue = []
    q = " ".join(query.split("_"))
    min_val = -float("inf")
    for chunk in df:
        for row in chunk.itertuples():
            sentence =str(row[4])
            if sentence=="":
                continue
            sim = calculate_similarities(q,centroid=centroid,sentence=sentence)
            queue,min_val=insert_to_queue(queue,sim,sentence,min_val)
    rows ={}
    i=0
    for item in queue:
        row={}
        row["query"]=q
        row["input_sentence"] = item[1]
        rows[i]=row
        i+=1
    pd.DataFrame.from_dict(rows,orient="index").to_csv(input_dir+query)


if __name__=="__main__":
    input_dir = sys.argv[1]
    target_dir = sys.argv[2]
    queries_file = sys.argv[3]
    model_file = sys.argv[4]
    input_file = sys.argv[5]
    queries = read_queries(queries_file)
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
    func = partial(find_most_similar_sentences,input_file, target_dir,input_dir)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    workers = cpu_count()
    list_multiprocessing(queries,func,workers=workers)







