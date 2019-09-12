import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial, update_wrapper
from multiprocessing import Pool, cpu_count
from summarization.seo_experiment.utils import clean_texts
import gensim
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


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

def get_sentence_centroid(sentence,model):
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
    freqs = [get_term_frequency(text,q)/len(text.split()) for q in query.split("_")]
    if mode=="max":
        return max(freqs)
    if mode=="min":
        return min(freqs)
    if mode=="avg":
        return np.mean(freqs)
    if mode=="sum":
        return sum(freqs)


def centroid_similarity(s1,s2,model):
    centroid1 = get_sentence_centroid(s1,model)
    centroid2 = get_sentence_centroid(s2,model)
    if centroid1 is None or centroid2 is None:
        return 0
    return cosine_similarity(centroid1,centroid2)

def clean_sentence(sentence):
    sw = set(nltk.corpus.stopwords.words('english'))
    return [token for token in sentence.rstrip().split() if token not in sw]


def tf_similarity(s1,s2):
    vectorizer = CountVectorizer()
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

def minmax_query_token_similarity(maximum,sentence,query,model):
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



def return_subset_dataframes(df,col_name):
    substes = [sub_df for _,sub_df in df.groupby(col_name)]
    return substes


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def get_predictors_values(input_sentence, query,args):
    idx, candidate_sentence,model = args
    result={}
    max_query_token_tf = wrapped_partial(query_term_freq,"max")
    avg_query_token_tf = wrapped_partial(query_term_freq,"avg")
    sum_query_token_tf = wrapped_partial(query_term_freq,"sum")
    funcs = [tf_similarity,centroid_similarity,jaccard_similiarity,max_query_token_tf,avg_query_token_tf,sum_query_token_tf]
    for i,func in enumerate(funcs):
        if func.__name__.__contains__("query"):
            result[i]=func(candidate_sentence,query)
        elif func.__name__.__contains__("centroid"):
            result[i] = func(input_sentence,candidate_sentence,model)
        else:
            result[i] = func(input_sentence,candidate_sentence)
    return result

def indexes(res):
    result = {}
    for i,test in enumerate(res):
        result[i]={}
        for rank,idx in enumerate(test):
            result[i][idx]=rank
    return result

def  get_count(idx,result,len):
    return sum([len-1-result[t][idx] for t in result])

def apply_borda_in_dict(results,k=10):
    borda_counts = {}
    num_of_tests = len(results[list(results.keys())[0]])
    ranked_sentences = [sorted(list(results.keys()),key=lambda x:(results[x][j],x),reverse=True) for j in range(num_of_tests)]
    ranks = indexes(ranked_sentences)
    length = len(results)
    borda_counts ={idx:get_count(idx,ranks,length) for idx in results}
    chosen_cands = sorted(list(borda_counts.keys()),key=lambda x:(borda_counts[x],x),reverse=True)[:k]
    return chosen_cands



def check_fit(series,s2):
    res = []
    for s1 in series:
        intersection = set(clean_sentence(s1)).intersection(set(clean_sentence(s2)))
        res.append(bool(intersection))
    return pd.Series(res)


def reduce_subset(df,row):
    s1 = row
    result = df[check_fit(df["input_paragraph"],s1)]
    return result


def calculate_predictors(target_subset, input_sentence, query,model):
    reduced_subset = reduce_subset(target_subset, input_sentence)
    if reduced_subset.empty:
        reduced_subset = target_subset
    results={}
    for idx,target_row in reduced_subset.iterrows():
        result = get_predictors_values(input_sentence,query,(idx,clean_texts(target_row["input_paragraph"].lower()),model))
        results[idx] = result
    chosen_idxs = apply_borda_in_dict(results)
    return "\n##\n".join([reduced_subset.ix[i]["input_paragraph"] for i in chosen_idxs])

def read_queries(fname):
    result=[]
    with open(fname) as f:
        for line in f:
            query = line.split(":")[1].rstrip()
            result.append("_".join(query.split()))
    return result










