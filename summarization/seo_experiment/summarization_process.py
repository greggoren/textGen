from nltk import sent_tokenize
import os
import numpy as np
import pandas as pd
from functools import partial
from summarization.seo_experiment.borda_mechanism import calculate_predictors

def read_trec_file(trec_file):
    stats = {}
    with open(trec_file) as file:
        for line in file:
            query = line.split()[0]
            doc = line.split()[2]
            score = float(line.split()[4])
            if query not in stats:
                stats[query]={}
            stats[query][doc]=score
    return transform_stats(stats)

def transform_stats(stats):
    transformed = {}
    for query in stats:
        ranked_list = sorted(list(stats[query].keys()),key=lambda x:(stats[query][x],x),reverese = True)
        transformed[query]=ranked_list
    return transformed

def reference_docs_calculation(stats,ref_index):
    return {q:stats[q][ref_index] for q in stats}

def get_reference_doc(trec_file,index):
    stats = read_trec_file(trec_file)
    return reference_docs_calculation(stats,index)





def chosen_sentence_for_replacement(sentences, query):
    sentence_scores={}
    for i,sentence in enumerate(sentences):
        tokens = sentence.split()
        sentence_scores[i]=(-sum([tokens.count(w) for w in query.split()]),len(tokens))
    return sorted(list(sentence_scores.keys()),key=lambda x:(sentence_scores[x][0],sentence_scores[x][1],x),reverse=True)[0]




def get_sentences_for_replacement(doc_texts,reference_docs):
    replacements = {}
    for query in reference_docs:
        doc = reference_docs[query]
        text = doc_texts[doc]
        sentences = sent_tokenize(text)
        chosen_index = chosen_sentence_for_replacement(sentences=sentences, query=query)
        replacements[query]=chosen_index
    return replacements


def write_input_dataset_file(replacements,reference_docs,texts):
    input_dir = 'input_data/'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    with open(input_dir+"senteces_for_replace.txt",'w') as file:
        file.write("query\tdocname\tsentence_index\tsentence\n")
        for query in replacements:
            index = replacements[query]
            docname = reference_docs[query]
            text = texts[docname]
            sentence = sent_tokenize(text)[index]
            file.write("\t".join([query,docname,str(index),sentence])+'\n')
    return input_dir+'senteces_for_replace.txt'


def read_texts(fname,inp=False):
    if inp:
        df = pd.read_csv(fname, delimiter="\t", header =0)
    else:
        df = pd.read_csv(fname,delimiter="\t",names=["query", "input_paragraph"])
    return df


def write_files(**kwargs):
    for key,val in kwargs.items():
        val[0].write(val[1]+"\n")

def create_summarization_dataset(input_dataset_file,candidates_dir):
    input_df = read_texts(input_dataset_file,True)
    with open(os.path.dirname(input_dataset_file)+"/all_data.txt",'w') as complete:
        with open(os.path.dirname(input_dataset_file)+"/queries.txt",'w') as queries:
            with open(os.path.dirname(input_dataset_file)+"/source.txt",'w') as source:
                with open(os.path.dirname(input_dataset_file)+"/input_paragraphs.txt",'w') as inp_paragraphs:
                    header = "\t".join(input_df.columns)+"\tinput_paragraph\n"
                    complete.write(header)
                    for i,row in input_df.iterrows():
                        complete_data="\t".join([row[col] for col in input_df.columns])
                        query = row["query"]
                        query="_".join(query.split())
                        sentence = row["sentence"]
                        query_paragraph_df = read_texts(candidates_dir+query)
                        paragraphs = calculate_predictors(query_paragraph_df,sentence)
                        for paragraph in paragraphs.split("\n##\n"):
                            write_files(complete=(complete,complete_data+"\t"+paragraph),queries = (queries,query),source=(source,sentence),inp_paragraphs=(inp_paragraphs,paragraph))


