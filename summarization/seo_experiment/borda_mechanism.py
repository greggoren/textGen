import logging
import math
import os
import sys

from functools import partial, update_wrapper
from summarization.seo_experiment.utils import clean_texts,get_java_object
from krovetzstemmer import Stemmer
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import javaobj
from tqdm import tqdm
def dict_norm(dict):
    sum=0
    for token in dict:
        sum+=(float(dict[token])**2)
    return sum




def dict_cosine_similarity(d1,d2):
    """
    cosine similarity between document vectors represented as dictionaries
    """
    sumxx = dict_norm(d1)
    sumyy = dict_norm(d2)
    if sumxx==0 or sumyy==0:
        return 0
    sumxy=0
    shared_token = set(d1.keys()).intersection(set(d2.keys()))
    for token in shared_token:
        tfidf1 = float(d1[token])
        tfidf2 = float(d2[token])
        sumxy+=tfidf1*tfidf2
    return sumxy/math.sqrt(sumyy*sumxx)

def add_dict(d1,d2):
    for token in d2:
        if token in d1:
            d1[token]=float(d1[token])+float(d2[token])
        else:
            d1[token]=float(d2[token])
    return d1

def normalize_dict(dict,n):
    for token in dict:
        dict[token] = float(dict[token])/n
    return dict

def document_centroid(document_vectors):
    centroid = {}
    for doc in document_vectors:
        centroid=add_dict(centroid,doc)
    return normalize_dict(centroid,len(document_vectors))

def cosine_similarity(v1,v2):
    if v1 is None or v2 is None:
        return 0
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    if sumxx==0 or sumyy==0:
        return 0
    return sumxy/math.sqrt(sumxx*sumyy)


def get_semantic_docs_centroid(doc_texts,doc_names,model,stemmer=None):
    sum_vector = None
    for doc in doc_names:
        text = doc_texts[doc]
        vector = get_text_centroid(clean_texts(text),model,stemmer)
        if sum_vector is None:
            sum_vector = np.zeros(vector.shape[0])
        sum_vector = sum_vector+vector
    if sum_vector is None:
        return None
    return sum_vector/len(doc_names)

def get_text_centroid(text, model,stemmer=None):
    sum_vector = None
    denom = 0
    if stemmer is not None:
        stem = Stemmer()
    for token in clean_sentence(text):
        if stemmer is not None:
            token = stem.stem(token)
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
    stemmer = Stemmer()
    return [stemmer.stem(token) for token in text.split()].count(term)


def query_term_occ(mode,text,query):
    freqs = [get_term_frequency(text, q)  for q in query.split()]
    if mode=="max":
        return max(freqs)
    if mode=="min":
        return min(freqs)
    if mode=="avg":
        return np.mean(freqs)
    if mode=="sum":
        return sum(freqs)


def query_term_freq(mode,text,query):
    if len(text.split())==0:
        print("PROBLEMATIC TEXT=",text)
        return 0
    if len(query.split("_"))>1:
        freqs = [get_term_frequency(text,q)/len(text.split()) for q in query.split("_")]
    else:
        freqs = [get_term_frequency(text, q) / len(text.split()) for q in query.split()]
    if mode=="max":
        return max(freqs)
    if mode=="min":
        return min(freqs)
    if mode=="avg":
        return np.mean(freqs)
    if mode=="sum":
        return sum(freqs)


def centroid_similarity(s1,s2,model,stemmer=None):
    centroid1 = get_text_centroid(s1, model,stemmer)
    centroid2 = get_text_centroid(s2, model,stemmer)
    if centroid1 is None or centroid2 is None:
        return 0
    return cosine_similarity(centroid1,centroid2)

def clean_sentence(sentence):
    # sw = set(nltk.corpus.stopwords.words('english'))
    sw = []
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
    centroid = get_text_centroid(sentence)
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


def calculate_similarity_to_docs_centroid_tf_idf(text_tfidf_fname, top_docs_tfidf):
    summary_tfidf=get_java_object(text_tfidf_fname)
    return dict_cosine_similarity(summary_tfidf,top_docs_tfidf)


def calculate_similarity_to_docs_centroid_tf_idf(text_tfidf_fname, top_docs_tfidf):
    summary_tfidf=get_java_object(text_tfidf_fname)
    return dict_cosine_similarity(summary_tfidf,top_docs_tfidf)


def calculate_similarity_tf_idf(text_tfidf, top_docs_tfidf):
    return dict_cosine_similarity(text_tfidf,top_docs_tfidf)



def calculate_semantic_similarity_to_top_docs(text, top_docs, doc_texts, model,stemmer=None):
    summary_vector = get_text_centroid(clean_texts(text), model,stemmer)
    top_docs_centroid_vector = get_semantic_docs_centroid(doc_texts,top_docs,model,stemmer)
    return cosine_similarity(summary_vector,top_docs_centroid_vector)

def calculate_semantic_similarity_to_centroid(text, centroid,model,stemmer=None):
    text_vector = get_text_centroid(clean_texts(text), model,stemmer)
    return cosine_similarity(text_vector,centroid)

def calculate_semantic_similarity(text_vector, centroid):
    return cosine_similarity(text_vector,centroid)


def context_similarity_precomputed_vec(reference, context_vectors, text_vector):
    context_vector = context_vectors[reference]
    try:
        return cosine_similarity(text_vector, context_vector)
    except:
        return 0

def context_similarity(reference,context_vectors,summary,model,stemmer=None):
    # document_sentences = nltk.sent_tokenize(document)
    summary_vector = get_text_centroid(clean_texts(summary), model,stemmer)
    context_vector = context_vectors[reference]

    try:
        return cosine_similarity(summary_vector,context_vector)
    except:
        return 0
    #     summary_vector = get_text_centroid(clean_texts(summary),model)
    #     sentence = document_sentences[replacement_index]
    #     sentence_vector = get_text_centroid(clean_texts(sentence),model)
    #     try:
    #         return cosine_similarity(summary_vector,sentence_vector)
    #     except:
    #         return 0
    # else:
    #     summary_sentences = nltk.sent_tokenize(summary.replace("<t>","").replace("</t>",""))
    #     summary_vectors = [get_text_centroid(clean_texts(s),model) for s in summary_sentences]
    #     if len(summary_vectors)==0:
    #         print("here")
    #     if reference == "pred":
    #         if replacement_index==0:
    #             real_index = replacement_index
    #         else:
    #             real_index = replacement_index-1
    #         pred_sentence = document_sentences[real_index]
    #         try:
    #             return cosine_similarity(get_text_centroid(clean_texts(pred_sentence),model),summary_vectors[0])
    #         except:
    #             return 0
    #     elif reference=="next":
    #         if replacement_index==len(document_sentences)-1:
    #             real_index = replacement_index
    #         else:
    #             real_index = replacement_index+1
    #         next_sentence = document_sentences[real_index]
    #         try:
    #             return cosine_similarity(get_text_centroid(clean_texts(next_sentence),model),summary_vectors[-1])
    #         except:
    #             return 0

def summary_len(summary):
    return len(clean_texts(summary).split())



def get_seo_predictors_values(summary, summary_tfidf_fname, context_vectors, query, top_documents_centroid_tf_idf, past_winners_centroid_tf_idf, documents_text, top_docs_centroid, past_winners_centroid, model):
    result={}
    avg_query_token_tf = wrapped_partial(query_term_freq,"avg")
    pred_context_similarity = wrapped_partial(context_similarity,"pred")
    next_context_similarity = wrapped_partial(context_similarity,"next")
    self_context_similarity = wrapped_partial(context_similarity,"self")
    funcs = [avg_query_token_tf, pred_context_similarity, next_context_similarity, self_context_similarity, calculate_similarity_to_docs_centroid_tf_idf, calculate_semantic_similarity_to_centroid]
    j = 0
    for i,func in enumerate(funcs):
        if func.__name__.__contains__("query"):
            result[j]=func(clean_texts(summary),query)
        elif func.__name__.__contains__("context"):
            result[j] = func(context_vectors,summary,model,True)
        elif func.__name__.__contains__("tf_idf"):
            result[j] = func(summary_tfidf_fname,top_documents_centroid_tf_idf)
            j+=1
            result[j] = func(summary_tfidf_fname, past_winners_centroid_tf_idf)
        elif func.__name__.__contains__("semantic"):
            result[j] = func(summary, top_docs_centroid, model,True)
            j += 1
            result[j] = func(summary, past_winners_centroid, model,True)
        j+=1
    return result




def get_seo_replacement_predictors_values(query,sentence,sentence_tfidf_fname,top_documents_centroid_tf_idf,past_winners_centroid_tf_idf,past_winners_centroid_vector,top_docs_centroid,model):
    result={}
    avg_query_token_tf = wrapped_partial(query_term_freq,"avg")
    funcs = [avg_query_token_tf, calculate_similarity_to_docs_centroid_tf_idf,calculate_semantic_similarity_to_centroid]
    j=0
    for i,func in enumerate(funcs):
        if func.__name__.__contains__("query"):
            result[j]=-func(clean_texts(sentence),query)
        elif func.__name__.__contains__("tf_idf"):
            result[j] =-func(sentence_tfidf_fname,top_documents_centroid_tf_idf)
            j+=1
            result[j] =-func(sentence_tfidf_fname,past_winners_centroid_tf_idf)
        elif func.__name__.__contains__("semantic"):
            result[j] = -func(clean_texts(sentence), top_docs_centroid, model,True)
            j += 1
            result[j] = -func(clean_texts(sentence), past_winners_centroid_vector, model,True)
        j+=1
    return result

def len_diff(source,update):
    return len(clean_texts(update).split()) - len(clean_texts(source).split())

def cover(text,query):
    numerator = 0
    for q in query.split():
        if q in clean_texts(text):
            numerator+=1
    return numerator/len(query.split())

# def `get_`predictors_values(input_sentence, query,args):
#     idx, candidate_sentence,model = args
#     result={}
#     max_query_token_tf = wrapped_partial(query_term_freq,"max")
#     avg_query_token_tf = wrapped_partial(query_term_freq,"avg")
#     sum_query_token_tf = wrapped_partial(query_term_freq,"sum")
#     funcs = [tf_similarity,centroid_similarity,jaccard_similiarity,max_query_token_tf,avg_query_token_tf,sum_query_token_tf,len_diff,cover]
#     for i,func in enumerate(funcs):
#         if func.__name__.__contains__("query") or func.__name__.__contains__("cover"):
#             result[i]=func(candidate_sentence,query)
#         elif func.__name__.__contains__("centroid"):
#             result[i] = func(input_sentence,candidate_sentence,model)
#         else:
#             result[i] = func(input_sentence,candidate_sentence)
#     return result


def get_past_winners_tfidf_centroid(past_winners,docuemnt_vectors_dir):
    result = {}
    decay_factors = [0.01 * math.exp(-0.01 * (len(past_winners) - i)) for i in range(len(past_winners))]
    denominator = sum(decay_factors)
    for i,doc in enumerate(past_winners):
        doc_tfidf = get_java_object(docuemnt_vectors_dir+doc)
        decay = decay_factors[i]/denominator
        normalized_vector = create_weighted_dict(doc_tfidf,decay)
        result=add_dict(result,normalized_vector)
    return result


def past_winners_centroid(past_winners,texts,model,stemmer=None):
    sum_vector = None
    decay_factors = [0.01*math.exp(-0.01*(len(past_winners)-i)) for i in range(len(past_winners))]
    denominator = sum(decay_factors)
    for i,doc in enumerate(past_winners):
        text = texts[doc]
        vector = get_text_centroid(clean_texts(text),model,stemmer)
        if sum_vector is None:
            sum_vector = np.zeros(vector.shape[0])
        sum_vector+=vector*decay_factors[i]/denominator
    return sum_vector


def create_weighted_dict(dict,weight):
    result={}
    for token in dict:
        result[token]=float(dict[token])*weight
    return result



def get_predictors_values(input_sentence, candidate_sentence_semantic_vector,context_vectors,query,candidate_sentence_tf_idf,top_documents_centroid_tf_idf,past_winners_centroid_tf_idf,top_docs_centroid,past_winners_centroid):
    result = {}
    avg_query_token_tf = wrapped_partial(query_term_freq, "avg")
    pred_context_similarity = wrapped_partial(context_similarity_precomputed_vec, "pred")
    next_context_similarity = wrapped_partial(context_similarity_precomputed_vec, "next")
    self_context_similarity = wrapped_partial(context_similarity_precomputed_vec, "self")
    funcs = [avg_query_token_tf, pred_context_similarity, next_context_similarity, self_context_similarity,
             calculate_similarity_tf_idf, calculate_semantic_similarity]
    j = 0
    for i, func in enumerate(funcs):
        if func.__name__.__contains__("query"):
            result[j] = func(clean_texts(input_sentence), query)
        elif func.__name__.__contains__("context"):
            result[j] = func(context_vectors,candidate_sentence_semantic_vector)
        elif func.__name__.__contains__("tf_idf"):
            result[j] = func(candidate_sentence_tf_idf, top_documents_centroid_tf_idf)
            j += 1
            result[j] = func(candidate_sentence_tf_idf, past_winners_centroid_tf_idf)
        elif func.__name__.__contains__("semantic"):
            result[j] = func(candidate_sentence_semantic_vector, top_docs_centroid)
            j += 1
            result[j] = func(candidate_sentence_semantic_vector, past_winners_centroid)
        j += 1
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
        intersection = set(clean_sentence(clean_texts(s1))).intersection(set(clean_sentence(s2)))
        res.append(bool(intersection))
    return pd.Series(res)


def reduce_subset(df,row):
    s1 = row
    result = df[check_fit(df["input_paragraph"],s1)]
    return result

def calculte_top_docs_centroid(top_docs,document_veoctors_dir):
    document_vectors = [get_java_object(document_veoctors_dir+top_doc) for top_doc in top_docs]
    return document_centroid(document_vectors)



def get_context_sentence_vectors(text, index, model,stemmer=None):
    result={}
    sentences = nltk.sent_tokenize(text)
    if index==0:
        result["pred"]=get_text_centroid(clean_texts(sentences[0]),model,stemmer)
        result["self"]=get_text_centroid(clean_texts(sentences[0]),model,stemmer)
        result["next"]=get_text_centroid(clean_texts(sentences[min(1,len(sentences)-1)]),model,stemmer)
    elif index==len(sentences)-1:
        result["pred"]=get_text_centroid(clean_texts(sentences[max(0,index-1)]),model,stemmer)
        result["self"]=get_text_centroid(clean_texts(sentences[index]),model,stemmer)
        result["next"]=get_text_centroid(clean_texts(sentences[index]),model,stemmer)
    else:
        result["pred"] = get_text_centroid(clean_texts(sentences[index - 1]), model,stemmer)
        result["self"] = get_text_centroid(clean_texts(sentences[index]), model,stemmer)
        result["next"] = get_text_centroid(clean_texts(sentences[index+1]), model,stemmer)
    return result




def calculate_summarization_predictors_for_former_docs(input_sentence, former_docs_for_summary, replacement_index, qid, queries_text, document_texts, ref_docs, top_docs, past_winners, document_vectors_dir, model):
    top_documents_centroid_tf_idf = calculte_top_docs_centroid(top_docs, document_vectors_dir)
    past_winners_centroid_tf_idf = get_past_winners_tfidf_centroid(past_winners, document_vectors_dir)
    ref_document_text = document_texts[ref_docs[qid]]
    past_winners_semantic_centroid_vector = past_winners_centroid(past_winners,document_texts,model)
    top_docs_centroid = get_semantic_docs_centroid(document_texts,top_docs,model,True)
    context_vectors = get_context_sentence_vectors(ref_document_text, replacement_index, model,True)
    results={}
    for idx,doc in enumerate(former_docs_for_summary):
        candidate_document_tf_idf_fname = document_vectors_dir+"/"+doc
        document_semantic_vector = get_text_centroid(clean_texts(document_texts[doc]),model,True)
        document_vector_tfidf = get_java_object(candidate_document_tf_idf_fname)
        result = get_predictors_values(input_sentence,document_semantic_vector,context_vectors,queries_text[qid],document_vector_tfidf,top_documents_centroid_tf_idf,past_winners_centroid_tf_idf,top_docs_centroid,past_winners_semantic_centroid_vector)
        results[idx] = result
    chosen_idxs = apply_borda_in_dict(results,30)
    return "\n##\n".join([document_texts[former_docs_for_summary[i]] for i in chosen_idxs])





def calculate_summarization_predictors(target_subset, input_sentence, replacement_index,qid,queries_text,document_texts,ref_docs,top_docs,past_winners,document_vectors_dir ,paragraphs_vector_dir,model):
    # reduced_subset = target_subset
    reduced_subset = reduce_subset(target_subset, input_sentence)
    top_documents_centroid_tf_idf = calculte_top_docs_centroid(top_docs, document_vectors_dir)
    past_winners_centroid_tf_idf = get_past_winners_tfidf_centroid(past_winners, document_vectors_dir)
    ref_document_text = document_texts[ref_docs[qid]]
    past_winners_semantic_centroid_vector = past_winners_centroid(past_winners,document_texts,model)
    top_docs_centroid = get_semantic_docs_centroid(document_texts,top_docs,model,True)
    context_vectors = get_context_sentence_vectors(ref_document_text, replacement_index, model,True)
    if reduced_subset.empty:
        reduced_subset = target_subset
    results={}
    for idx,target_row in reduced_subset.iterrows():
        candidate_paragraph_tf_idf_fname = paragraphs_vector_dir+"/"+"_".join(queries_text[qid].split())+"/"+str(idx%1000)+"/"+str(idx)
        paragraph_semantic_vector = get_text_centroid(clean_texts(target_row["input_paragraph"]),model,True)
        paragraph_vector_tfidf = get_java_object(candidate_paragraph_tf_idf_fname)
        result = get_predictors_values(input_sentence,paragraph_semantic_vector,context_vectors,queries_text[qid],paragraph_vector_tfidf,top_documents_centroid_tf_idf,past_winners_centroid_tf_idf,top_docs_centroid,past_winners_semantic_centroid_vector)
        results[idx] = result
    chosen_idxs = apply_borda_in_dict(results)
    return "\n##\n".join([reduced_subset.ix[i]["input_paragraph"] for i in chosen_idxs])






def calculate_seo_predictors(summaries,summary_tfidf_fname_index, replacement_index,query,document,document_vectors_dir,documents_text,top_docs,past_winners,model):
    top_documents_centroid_tf_idf = calculte_top_docs_centroid(top_docs,document_vectors_dir)
    past_winners_centroid_tf_idf = get_past_winners_tfidf_centroid(past_winners,document_vectors_dir)
    past_winners_semantic_centroid = past_winners_centroid(past_winners,documents_text,model,True)
    top_docs_centroid = get_semantic_docs_centroid(documents_text,top_docs,model,True)
    context_vectors = get_context_sentence_vectors(document,replacement_index,model,True)
    results={}
    for i,summary in enumerate(summaries):
        summary_tfidf_fname=summary_tfidf_fname_index[i]
        result = get_seo_predictors_values(summary,summary_tfidf_fname, context_vectors,query,top_documents_centroid_tf_idf,past_winners_centroid_tf_idf,documents_text,top_docs_centroid,past_winners_semantic_centroid,model)
        results[i] = result
    chosen_idxs = apply_borda_in_dict(results,30)
    return chosen_idxs


def calculate_seo_replacement_predictors(sentences, query, document_name, sentences_vectors_dir,document_vectors_dir, documents_text, top_docs, past_winners,model):
    top_documents_centroid_tf_idf = calculte_top_docs_centroid(top_docs,document_vectors_dir)
    past_winners_centroid_tf_idf = get_past_winners_tfidf_centroid(past_winners,document_vectors_dir)
    top_docs_semantic_centroid = get_semantic_docs_centroid(documents_text,top_docs,model,True)
    past_winners_semantic_centroid = past_winners_centroid(past_winners,documents_text,model,True)
    results={}
    for i,sentence in enumerate(sentences):
        if len(clean_texts(sentence).split())==0:
            continue
        sentence_tfidf_fname = sentences_vectors_dir+document_name+"_"+str(i)
        result = get_seo_replacement_predictors_values(query,sentence,sentence_tfidf_fname,top_documents_centroid_tf_idf,past_winners_centroid_tf_idf,past_winners_semantic_centroid,top_docs_semantic_centroid,model)
        results[i] = result
    chosen_idx = apply_borda_in_dict(results,1)[0]
    return chosen_idx


def read_queries(fname):
    result=[]
    with open(fname) as f:
        for line in f:
            query = line.split(":")[1].rstrip()
            result.append("_".join(query.split()))
    return result










