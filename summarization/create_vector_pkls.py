import numpy as np
from multiprocessing import Pool,cpu_count
from tqdm import tqdm
import pandas as pd
import pickle
import os
import logging
import sys
from optparse import OptionParser
import gensim
from functools import partial
from time import time
import nltk

def row_to_path(row_num,number_of_folders):
    return str(row_num%number_of_folders)+"/"

def clean_text(text,sw):
    text = text.replace("(","")
    text = text.replace(")","")
    text = text.replace("[","")
    text = text.replace("]","")
    text = text.replace("!","")
    text = text.replace(".","")
    text = text.replace(",","")
    text = text.replace("="," ")
    text = text.replace("#"," ")
    text = text.replace("$"," ")
    text = text.replace("%"," ")
    text = text.replace("&"," and ")
    text = text.replace("*","  ")
    text = text.replace("^","  ")
    text = text.replace("\\","")
    text = text.replace("-","  ")
    text = text.replace("+","  ")
    return [token for token in text.rstrip().split() if token not in sw]



def get_text_centroid(text,sw):
    sum_vector = None
    denom = 0
    for token in clean_text(text,sw):
        try:
            vector = model.wv[token]
        except KeyError:
            continue
        if sum_vector is None:
            sum_vector=np.zeros(vector.shape[0])
        sum_vector = sum_vector + vector
        denom+=1
    if sum_vector is None:
        return None
    return sum_vector/denom


def get_centroid_of_cluster(df,sw):
    sum_vector = None
    denom = 0
    for idx,row in df.iterrows():
        vector = get_text_centroid(row["input_text"],sw)
        if vector is None:
            logger.error("problem with text: "+row["input_text"])
            continue
        if sum_vector is None:
            sum_vector = np.zeros(vector.shape[0])
        sum_vector = sum_vector + vector
        denom+=1
    return sum_vector/denom






def save_vector(vector,number_of_folders,output_dir,rnum=None,query=None):
    if rnum is not None:
        folder = row_to_path(rnum,number_of_folders)
        path = output_dir + folder
        fname = path + str(rnum) + ".pkl"
        if os.path.exists(path):
            os.makedirs(path)
    elif query is not None:
        fname = output_dir+query+".pkl"
    else:
        sys.exit(1)
    with open(fname,'wb') as vector_file:
        pickle.dump(vector,vector_file)

def read_df(fname):
    df = pd.read_csv(fname,delimiter="\t",names=["query","input_text"])
    return df

def save_cluster_vector(cluster_dir,output_dir,query):
    global model
    df = read_df(cluster_dir+"/"+query)
    cluster_centroid = get_centroid_of_cluster(df,sw)
    save_vector(cluster_centroid,0,output_dir,query=query)

def initializer():
    global sw
    sw = set(nltk.corpus.stopwords.words('english'))
    sw.add('s')

def save_vectors(input_file,number_of_folders,output_dir):
    global model
    df = pd.read_csv(input_file, delimiter=",", header=0, chunksize=100000)
    global_index =0
    start = time()
    sw = set(nltk.corpus.stopwords.words('english'))

    for chunk in df:
        for row in chunk.itertuples():
            text = str(row[4])
            if text == "":
                continue
            text_vector = get_text_centroid(text,sw)
            save_vector(text_vector,number_of_folders,output_dir,rnum=global_index)
            global_index+=1
            if global_index%1000==0:
                logger.info("finished "+str(global_index)+" examples in "+str(time()-start)+" seconds")





def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params, **kwargs)

def list_multiprocessing(param_lst, func, **kwargs):
    workers = kwargs.pop('workers')
    with Pool(workers,initializer()) as p:
        apply_lst = [([params], func, i, kwargs) for i, params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))
    return [_[1] for _ in result]




if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    parser = OptionParser()
    parser.add_option("-m", "--mode", dest="mode",
                      help="set running mode")
    parser.add_option("-k", "--output_dir", dest="output_dir")
    parser.add_option("-a", "--embeddings_file", dest="embeddings_file")
    parser.add_option("-d", "--nfolders", dest="nfolders")
    parser.add_option("-t", "--input_file", dest="input_file")
    parser.add_option("-i", "--cluster_dir", dest="cluster_dir")
    (options, args) = parser.parse_args()
    model = gensim.models.FastText.load_fasttext_format(options.embeddings_file)
    output_dir = options.output_dir
    if options.mode == "file":
        input_file = options.input_file
        number_of_folders = options.nfolders
        save_vectors(input_file,number_of_folders,output_dir)
    elif options.mode == "cluster":
        cluster_dir = options.cluster_dir
        queries = [file for file in os.listdir(cluster_dir)]
        func = partial(save_cluster_vector,cluster_dir,output_dir)
        workers = cpu_count()-1
        list_multiprocessing(queries, func, workers=workers)
    else:
        logger.error("mode selection failure")


