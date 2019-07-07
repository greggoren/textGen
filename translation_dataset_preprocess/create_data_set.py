import gensim
import math
import nltk
import pandas as pd
from functools import partial,update_wrapper
import sys
from multiprocessing import Pool
from copy import deepcopy
import logging
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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

def get_sentence_centroid(sentence):
    sum_vector = None
    for token in sentence.rstrip().split():
        if token not in model.wv:
            continue
        vector = model.wv[token]
        if sum_vector is None:
            sum_vector=deepcopy(vector)
        else:
            sum_vector+=vector
    return sum_vector/len(sentence.split())

def centroid_similarity(s1,s2):
    centroid1 = get_sentence_centroid(s1)
    centroid2 = get_sentence_centroid(s2)
    return cosine_similarity(centroid1,centroid2)

def clean_sentence(sentence):
    return [token for token in sentence.split() if token not in sw]

def jaccard_similiarity(s1,s2):
    tokens1 = set(clean_sentence(s1))
    tokens2 = set(clean_sentence(s2))
    nominator = len(tokens1.intersection(tokens2))
    denominator = len(tokens1.union(tokens2))
    return float(nominator)/denominator

def minmax_query_token_similarity(maximum,sentence,query):
    query_tokens = set(query.split())
    centroid = get_sentence_centroid(sentence)
    similarities = [cosine_similarity(centroid,model.wv[token]) for token in query_tokens]
    if maximum:
        return max(similarities)
    return min(similarities)

def get_bigrams(sentence):
    tokens = sentence.rstrip().split()
    return list(nltk.bigrams(tokens))

def shared_bigrams_count(s1,s2):
    bigram1 = get_bigrams(s1)
    bigram2 = get_bigrams(s2)
    count =0
    for bigram in bigram1:
        if bigram in bigram2:
            count+=1
    return count


def read_sentences(fname):
    df = pd.read_csv(fname,delimiter="\t",names=["query","input_sentence"])
    return df

def return_subset_dataframes(df,col_name):
    substes = [sub_df for _,sub_df in df.groupby(col_name)]
    return substes

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

# def get_predictors_values(input_sentence, candidate_sentence, query):
#     result={}
#     max_query_token_sim = wrapped_partial(minmax_query_token_similarity,True)
#     min_query_token_sim = wrapped_partial(minmax_query_token_similarity,False)
#     funcs = [centroid_similarity,shared_bigrams_count,jaccard_similiarity,max_query_token_sim,min_query_token_sim]
#     for i,func in enumerate(funcs):
#         if func.__name__.__contains__("query"):
#             result[i]=func(candidate_sentence,query)
#         elif func.__name__.__contains__("centroid"):
#             result[i] = func(input_sentence,candidate_sentence)
#         else:
#             result[i] = func(input_sentence,candidate_sentence)
#     return result

def get_predictors_values(input_sentence, query,args):
    start = time()
    idx, candidate_sentence = args
    result={}
    max_query_token_sim = wrapped_partial(minmax_query_token_similarity,True)
    min_query_token_sim = wrapped_partial(minmax_query_token_similarity,False)
    funcs = [centroid_similarity,shared_bigrams_count,jaccard_similiarity,max_query_token_sim,min_query_token_sim]
    for i,func in enumerate(funcs):
        if func.__name__.__contains__("query"):
            result[i]=func(candidate_sentence,query)
        elif func.__name__.__contains__("centroid"):
            result[i] = func(input_sentence,candidate_sentence)
        else:
            result[i] = func(input_sentence,candidate_sentence)
    logger.info("took: "+str(time()-start))
    return idx,result

def indexes(res):
    result = {}
    for i,test in enumerate(res):
        result[i]={}
        for rank,idx in enumerate(test):
            result[i][idx]=rank
    return result

def  get_count(idx,result,len):
    return sum([len-1-result[t][idx] for t in result])

def apply_borda_in_dict(results):
    borda_counts = {}
    num_of_tests = len(results[list(results.keys())[0]])
    ranked_sentences = [sorted(list(results.keys()),key=lambda x:(results[x][j],x),reverse=True) for j in range(num_of_tests)]
    ranks = indexes(ranked_sentences)
    length = len(results)
    borda_counts ={idx:get_count(idx,ranks,length) for idx in results}
    chosen_cand = max(list(borda_counts.keys()),key=lambda x:(borda_counts[x],x))
    return chosen_cand

# def calculate_predictors(target_subset,row):
#     results={}
#     query = row["query"]
#     input_sentence = row["input_sentence"]
#
#     for idx,target_row in target_subset.iterrows():
#         target_sentence = target_row["input_sentence"]
#         results[idx] = get_predictors_values(input_sentence, target_sentence, query)
#     chosen_idx = apply_borda_in_dict(results)
#     return target_subset.ix[chosen_idx]["input_sentence"]

from time import time
def calculate_predictors(target_subset,row):
    results={}
    query = row["query"]
    input_sentence = row["input_sentence"]
    f = partial(get_predictors_values,input_sentence,query)
    arg_list = [(idx,target_row["input_sentence"]) for idx,target_row in target_subset.iterrows()]
    with ThreadPoolExecutor(max_workers=20) as executer:
        values = executer.map(f,arg_list)
        for idx,result in values:
            results[idx]=result
        chosen_idx = apply_borda_in_dict(results)
        return target_subset.ix[chosen_idx]["input_sentence"]



def get_true_subset(input_subset,target_subset):
    f = lambda x:calculate_predictors(target_subset,x)
    input_subset["target_sentence"] = input_subset.apply(f,axis=1)
    return input_subset

def apply_func_on_subset(input_dir,target_dir,query):
    global model
    global sw
    logger.info("Working on "+query)
    input_subset = read_sentences(input_dir+query)
    target_subset = read_sentences(target_dir+query)
    return get_true_subset(input_subset,target_subset)


def read_queries(fname):
    result=[]
    with open(fname) as f:
        for line in f:
            query = line.split(":")[1].rstrip()
            result.append("_".join(query.split()))
    return result

def initializer():
    global sw
    sw = set(nltk.corpus.stopwords.words('english'))




def list_multiprocessing(param_lst,
                         func,
                         **kwargs):
    workers = kwargs.pop('workers')

    with Pool(workers) as p:
        apply_lst = [([params], func, i, kwargs) for i, params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))

    return [_[1] for _ in result]


def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params, **kwargs)




if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    input_dir = sys.argv[1]
    target_dir = sys.argv[2]
    queries_file = sys.argv[3]
    model_file = sys.argv[4]
    queries = read_queries(queries_file)
    sw = set(nltk.corpus.stopwords.words('english'))
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file,binary=True)
    func = partial(apply_func_on_subset, input_dir, target_dir)
    results = list_multiprocessing(queries,func,workers=20)
    df = pd.concat(results).reset_index(drop=True)
    df.to_csv("query_ks_translation.csv")
