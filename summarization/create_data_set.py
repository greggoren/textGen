import gensim
import math
import nltk
import pandas as pd
from functools import partial,update_wrapper
import sys
from multiprocessing import Pool,cpu_count
from copy import deepcopy
import logging
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity as cs

from sklearn.feature_extraction.text import CountVectorizer





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
    denom = 0
    for token in clean_sentence(sentence):
        try:
            vector = model.wv[token]
        except KeyError:
            continue
        if sum_vector is None:
            sum_vector = np.zeros(vector.shape[0])
        sum_vector = sum_vector + vector
        denom += 1
    if sum_vector is None:
        return None
    return sum_vector / denom

def pos_overlap(s1,s2):
    tags1 = nltk.pos_tag(s1.split())
    tags2 = nltk.pos_tag(s2.split())
    return len(set(tags1).intersection(set(tags2)))


def get_term_frequency(text,term):
    return text.split().count(term)

def query_term_freq(mode,text,query):
    freqs = [get_term_frequency(text,q)/len(text.split()) for q in query.split()]
    if mode=="max":
        return max(freqs)
    if mode=="min":
        return min(freqs)
    if mode=="avg":
        return np.mean(freqs)
    if mode=="sum":
        return sum(freqs)


def centroid_similarity(s1,s2):
    centroid1 = get_sentence_centroid(s1)
    centroid2 = get_sentence_centroid(s2)
    if centroid1 is None or centroid2 is None:
        return 0
    return cosine_similarity(centroid1,centroid2)

def clean_sentence(sentence):
    return [token for token in sentence.rstrip().split() if token not in sw]


def tf_similarity(s1,s2):
    corpus = [" ".join(clean_sentence(s1))," ".join(clean_sentence(s2))]
    tf_matrix = vectorizer.fit_transform(corpus).toarray()
    if len(tf_matrix)<2:
        return 0
    return cosine_similarity(tf_matrix[0],tf_matrix[1])


def jaccard_similiarity(s1,s2):
    tokens1 = set(clean_sentence(s1))
    tokens2 = set(clean_sentence(s2))
    nominator = len(tokens1.intersection(tokens2))
    denominator = len(tokens1.union(tokens2))
    if nominator==0:
        return 0
    return float(nominator)/denominator

def minmax_query_token_similarity(maximum,sentence,query):
    query_tokens = set(query.split())
    centroid = get_sentence_centroid(sentence)
    if centroid is None:
        return 0
    similarities = [cosine_similarity(centroid,model.wv[token]) for token in query_tokens if token in model.wv]
    if not similarities:
        return 0
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

def read_texts(fname, inp=False):
    if not os.path.isfile(fname):
        return pd.DataFrame(columns=["query","input_paragraph"])
    if inp:
        df = pd.read_csv(fname,delimiter=",",header=0)
    else:
        df = pd.read_csv(fname, delimiter="\t", names=["query", "input_paragraph"])
    return df

def return_subset_dataframes(df,col_name):
    substes = [sub_df for _,sub_df in df.groupby(col_name)]
    return substes

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func



def get_predictors_values(input_sentence, query,args):
    idx, candidate_sentence = args
    result={}
    max_query_token_tf = wrapped_partial(query_term_freq,"max")
    avg_query_token_tf = wrapped_partial(query_term_freq,"avg")
    sum_query_token_tf = wrapped_partial(query_term_freq,"sum")
    funcs = [tf_similarity,centroid_similarity,jaccard_similiarity,max_query_token_tf,avg_query_token_tf,sum_query_token_tf]
    for i,func in enumerate(funcs):
        if func.__name__.__contains__("query"):
            result[i]=func(candidate_sentence,query)
        elif func.__name__.__contains__("centroid"):
            result[i] = func(input_sentence,candidate_sentence)
        else:
            result[i] = func(input_sentence,candidate_sentence)
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



def check_fit(series,s2):
    res = []
    for s1 in series:
        intersection = set(clean_sentence(s1)).intersection(set(clean_sentence(s2)))
        res.append(bool(intersection))
    return pd.Series(res)


def reduce_subset(df,row):
    s1 = row["input_paragraph"]
    result = df[check_fit(df["input_paragraph"],s1)]
    return result


def calculate_predictors(target_subset,row):
    reduced_subset = reduce_subset(target_subset,row)
    if reduced_subset.empty:
        reduced_subset = target_subset
    results={}
    query = row["query"]
    input_sentence = row["input_paragraph"]
    f = partial(get_predictors_values,input_sentence,query)
    arg_list = [(idx,target_row["input_paragraph"]) for idx,target_row in reduced_subset.iterrows()]
    with ThreadPoolExecutor(max_workers=1) as executer:
        values = executer.map(f,arg_list)
        for idx,result in values:
            results[idx]=result
        try:
            chosen_idx = apply_borda_in_dict(results)
        except:
            logger.error("problem in "+query+" and input sentence "+input_sentence)
            sys.exit(1)
        return reduced_subset.ix[chosen_idx]["input_paragraph"]

def parallelize(data, func,wrapper,name,translations_tmp_dir):
    if not os.path.exists(translations_tmp_dir):
        os.makedirs(translations_tmp_dir)
    tmp_fname = translations_tmp_dir+name+".csv"
    data_split = np.array_split(data, len(data))
    wrap = partial(wrapper,func)
    with ThreadPoolExecutor(max_workers=1) as pool:
        results = list(tqdm(pool.map(wrap, data_split),total=len(data_split)))
        data = pd.concat(results)
        data.to_csv(tmp_fname)
        return data


def warpper(f,df):
    df["target_paragraph"] = df.apply(f, axis=1)
    return df

def get_true_subset(target_subset,input_subset,output_dir,query):
    f = partial(calculate_predictors,target_subset)
    input_subset = parallelize(input_subset,f,warpper,name=query,translations_tmp_dir=output_dir)
    return input_subset

def apply_func_on_subset(input_dir,target_dir,output_dir,query):
    global model
    global logger
    input_subset = read_texts(input_dir + query,True)
    target_subset = read_texts(target_dir + query)
    return get_true_subset(target_subset,input_subset,output_dir,query)


def read_queries(fname):
    result=[]
    with open(fname) as f:
        for line in f:
            query = line.split(":")[1].rstrip()
            result.append("_".join(query.split()))
    return result

def initializer():
    global sw
    global vectorizer
    sw = set(nltk.corpus.stopwords.words('english'))
    vectorizer = CountVectorizer()


def list_multiprocessing(param_lst, func, **kwargs):
    workers = kwargs.pop('workers')
    with Pool(workers,initializer,()) as p:
        apply_lst = [([params], func, i, kwargs) for i, params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))
    return [_[1] for _ in result]


def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params, **kwargs)



def recovery_mode(queries,output_dir,target_dir):
    finished = [f.replace(".csv","") for f in os.listdir(output_dir)]
    updated_queries = [q for q in queries if q not in finished and os.path.isfile(target_dir+q)]
    return updated_queries


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
    output_dir = sys.argv[5]
    recovery = sys.argv[6]

    queries = read_queries(queries_file)

    logger.info("Number of queries:"+str(len(queries)))
    if recovery=="True":
        queries = recovery_mode(queries,output_dir,target_dir)
        logger.info("Recovery mode detected, updated number of queries:" + str(len(queries)))
        model = gensim.models.FastText.load_fasttext_format(model_file)
    func = partial(apply_func_on_subset, input_dir, target_dir,output_dir)
    workers = cpu_count()-1
    results = list_multiprocessing(queries,func,workers=workers)
    df = pd.concat(results).reset_index(drop=True)
    df.to_csv("query_biased_summarization.csv")
