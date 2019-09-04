import math
from copy import deepcopy
import gensim
import sys
import numpy as np
def abstractive_fraction(document,output,segments):
    numerator =0
    for segment in segments:
        output = output.replace(segment,"")
    for i,out_part in enumerate(output.split(".")):
        if out_part.replace(".","") not in document:
            numerator+=1
    return numerator/(i+1)

def output_length(output,segments):
    for segment in segments:
        output = output.replace(segment,"")
    return len(output.split())

def query_coverage(query,output):
    numerator=0
    denominator=0
    for q in query.split():
        if q in output:
            numerator+=1
        denominator+=1
    return numerator/denominator

def query_coverage_recall(query,output,input):
    numerator=0
    denominator=0
    for q in query.split():
        if q in output and q in input:
            numerator+=1
        if q in input:
            denominator+=1
    return numerator/denominator



def is_query_included(query,output):
    for q in query.split():
        if q in output:
            return 1
    return 0

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
    for token in sentence.split():
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





def similarity_to_source(source,output,model,segments):
    for segment in segments:
        output = output.replace(segment,"")
    source_centroid = get_sentence_centroid(source,model)
    output_centroid = get_sentence_centroid(output,model)
    if source_centroid is None or output_centroid is None:
        return 0
    return cosine_similarity(source_centroid,output_centroid)

def read_file(fname):
    with open(fname) as file:
        return file.readlines()




def get_semantic_similarity_on_all_output(out_fname,source_fname,segments,model):
    out_lines = read_file(out_fname)
    source_lines = read_file(source_fname)
    numerator = 0
    for row,out_line in enumerate(out_lines):
        source_line = source_lines[row]
        similarity = similarity_to_source(source_line,out_line,model,segments)
        numerator+=similarity
    return numerator/(row+1)


def get_query_coverage_on_all_output(qfname,out_fname):
    out_lines = read_file(out_fname)
    query_lines = read_file(qfname)
    numerator = 0
    for row, out_line in enumerate(out_lines):
        query = query_lines[row]
        numerator+=query_coverage(query,out_line)
    return numerator/(row+1)

def get_query_coverage_recall_on_all_output(qfname,out_fname,in_fname):
    out_lines = read_file(out_fname)
    query_lines = read_file(qfname)
    in_lines = read_file(in_fname)
    numerator = 0
    for row, out_line in enumerate(out_lines):
        query = query_lines[row]
        in_line = in_lines[row]
        numerator+=query_coverage_recall(query,out_line,in_line)
    return numerator/(row+1)



def get_query_include_on_all_output(qfname,out_fname):
    out_lines = read_file(out_fname)
    query_lines = read_file(qfname)
    numerator = 0
    for row, out_line in enumerate(out_lines):
        query = query_lines[row]
        numerator+=is_query_included(query,out_line)
    return numerator/(row+1)

def get_abstractive_ratio_on_all_output(input_fname,out_fname,segments):
    out_lines = read_file(out_fname)
    input_lines = read_file(input_fname)
    numerator = 0
    for row, out_line in enumerate(out_lines):
        input_line = input_lines[row]
        numerator+=abstractive_fraction(input_line,out_line,segments)
    return numerator/(row+1)


def get_average_length_on_all_output(out_fname,segments):
    out_lines = read_file(out_fname)
    numerator = 0
    for row, out_line in enumerate(out_lines):
        numerator+=output_length(out_line,segments)
    return numerator/(row+1)




if __name__=="__main__":
    out_fname = sys.argv[1]
    qfname = sys.argv[2]
    input_fname = sys.argv[3]
    source_fname = sys.argv[4]
    model_file = sys.argv[5]
    metric_file = sys.argv[6]

    model = gensim.models.FastText.load_fasttext_format(model_file)
    segments = ["<t>","</t>"]
    with open(metric_file,'w') as mfile:
        mfile.write("similarity average to source: "+str(get_semantic_similarity_on_all_output(out_fname,source_fname,segments,model))+"\n")
        mfile.write("average output length: "+str(get_average_length_on_all_output(out_fname,segments))+"\n")
        mfile.write("average coverage ratio:"+str(get_query_coverage_on_all_output(qfname,out_fname))+"\n")
        mfile.write("average coverage recall ratio:"+str(get_query_coverage_recall_on_all_output(qfname,out_fname,input_fname))+"\n")
        mfile.write("average inclusion ratio:"+str(get_query_include_on_all_output(qfname,out_fname))+"\n")
        mfile.write("average abstractive ratio:"+str(get_abstractive_ratio_on_all_output(input_fname,out_fname,segments))+"\n")

