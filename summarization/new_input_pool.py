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

def clean_text(text):
    return [token for token in text.rstrip().split() if token not in sw and not contain_digits(token)]


def contain_digits(token):
    return any(char.isdigit() for char in token)


def initializer():
    global sw
    sw = set(nltk.corpus.stopwords.words('english'))
    sw.add(".")
    sw.add(",")
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


def get_text_centroid(paragraph):
    sum_vector = None
    denom = 0
    for token in clean_text(paragraph):
        # if token not in model.wv:
        #     continue
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
        vector = get_text_centroid(row["input_paragraph"])
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


def is_in(query,text):
    res = set(query.split()).intersection(text.rstrip().split())
    return bool(res)


def calculate_similarities(query, centroid, sentence):
    if is_in(query, sentence):
        return -float("inf")
    cent_paragraph = get_text_centroid(sentence)
    if cent_paragraph is None:
        return -float("inf")
    return cosine_similarity(centroid,sentence)


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

def find_most_similar_sentences(input_file, translation_dir, input_dir, query):
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
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    target_dir = sys.argv[2]
    queries_file = sys.argv[3]
    model_file = sys.argv[4]
    input_file = sys.argv[5]
    recovery = bool(sys.argv[6])

    queries = read_queries(queries_file)
    print("there are ",str(len(queries)),"queries",flush=True)
    if recovery:
        queries = recovery_mode(queries,input_dir,target_dir)
        print("Recovery mode detected, updated number of queries:" + str(len(queries)))
    model = gensim.models.wrappers.FastText.load_fasttext_format(model_file)
    func = partial(find_most_similar_sentences, input_file, target_dir, input_dir)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    workers = cpu_count()
    list_multiprocessing(queries,func,workers=workers)







