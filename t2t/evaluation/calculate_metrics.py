import os,sys,logging
from t2t.utils import run_bash_command
import time
from optparse import OptionParser
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from copy import deepcopy
import math
import numpy as np
import gensim

def run_bleu(reference,script,translation):
    out = run_bash_command(script+" --translation="+translation+" --reference="+reference)

    for line in str(out).split("\n"):
        logger.info("line="+line+"\n")
        if "BLEU_uncased" in line:
            score=''
            score_raw = line.split()[-1].rstrip()
            for c in score_raw:
                if c.isdigit() or c=='.':
                    score+=c
            score = float(score)
            return score,translation


def get_accuracy_per_sequence(target_file,translation_file):
    translation_lines = open(translation_file).readlines()
    target_lines = open(target_file).readlines()
    total_sum = 0
    total_counter = 0
    for row,translated_sequence in enumerate(translation_lines):
        target_sequence = target_lines[row]
        current_sum = 0
        current_counter = 0
        transleted_tokens = translated_sequence.split()
        for index,target_token in enumerate(target_sequence.split()):
            if index>=len(transleted_tokens):
                current_counter+=1
                continue
            if transleted_tokens[index].rstrip()==target_token.rstrip():
                current_sum+=1
            current_counter+=1
        sequence_acc = current_sum/current_counter
        total_sum+=sequence_acc
        total_counter+=1
    return total_sum/total_counter

def get_accuracy_query_terms(target_file,queries_file,translation_file):
    translation_lines = open(translation_file).readlines()
    target_lines = open(target_file).readlines()
    queries_lines = open(queries_file).readlines()
    total_sum = 0
    total_counter = 0
    for row,translated_sequence in enumerate(translation_lines):
        target_sequence = target_lines[row]
        current_sum = 0
        current_counter = 0
        query = queries_lines[row]
        for q in query.split():
            if q in target_sequence and q in translated_sequence:
                current_sum+=1
                current_counter+=1
            elif q in target_sequence:
                current_counter+=1

        total_sum+=(current_sum/current_counter)
        total_counter+=1
    return total_sum/total_counter , translation_file



def calculate_seq_acc_multiprocess(translations_dir,reference_file,write_flag,ts,processes):
    files = [translation_dir+file for file in os.listdir(translations_dir)]
    if write_flag:
        bleu_results_file = open("SEQUENCE_POSITIONAL_ACCURACY_"+str(ts),'w')
    func = partial(get_accuracy_per_sequence,reference_file)
    with Pool(processes=processes) as pool:
        results = pool.map(func,files)
        if write_flag:
            for result in results:
                bleu_results_file.write(result[1]+"\t"+str(result[0])+"\n")
            bleu_results_file.close()
        return results

def calculate_seq_query_coverage_multiprocess(translations_dir,reference_file,write_flag,ts,query_file,processes):
    files = [translation_dir+file for file in os.listdir(translations_dir)]
    if write_flag:
        bleu_results_file = open("SEQUENCE_QUERY_TERMS_COVERAGE_"+str(ts),'w')
    func = partial(get_accuracy_query_terms,reference_file,query_file)
    with Pool(processes=processes) as pool:
        results = pool.map(func,files)
        if write_flag:
            for result in results:
                bleu_results_file.write(result[1]+"\t"+str(result[0])+"\n")
            bleu_results_file.close()
        return results
def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params, **kwargs)

def list_multiprocessing(param_lst, func, **kwargs):
    workers = kwargs.pop('workers')
    with Pool(workers) as p:
        apply_lst = [([params], func, i, kwargs) for i, params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))
    return [_[1] for _ in result]


def calculate_bleu_multiprocess(translations_dir,reference_file,bleu_script_bin,write_flag,ts,processes):
    files = [translation_dir+file for file in os.listdir(translations_dir)]
    if write_flag:
        bleu_results_file = open("BLEU_RESULTS_"+str(ts),'w')
    func = partial(run_bleu,reference_file,bleu_script_bin)
    with Pool(processes=processes) as pool:
        results = pool.map(func,files)
        if write_flag:
            for result in results:
                bleu_results_file.write(result[1]+"\t"+str(result[0])+"\n")
            bleu_results_file.close()
        return results

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

def get_similarity( target_file, queries_file,translation_file):
    global model
    translation_lines = open(translation_file).readlines()
    target_lines = open(target_file).readlines()
    queries_lines = open(queries_file).readlines()
    per_query_similarity={}
    for row,target_sequence in enumerate(target_lines):
        query = queries_lines[row]
        translated_sequence = translation_lines[row]
        centroid_translated = get_sentence_centroid(translated_sequence,model)
        query_centroid = get_sentence_centroid(query,model)
        similarity = cosine_similarity(centroid_translated,query_centroid)
        if query not in per_query_similarity:
            per_query_similarity[query]=[]
        per_query_similarity[query].append(similarity)
    for query in per_query_similarity:
        per_query_similarity[query]=np.mean(per_query_similarity[query])
    return np.mean([per_query_similarity[q] for q in per_query_similarity]),translation_file



def calculate_similarity_multiprocess(translations_dir,reference_file,write_flag,ts,query_file,processes):
    files = [translation_dir+file for file in os.listdir(translations_dir)]
    if write_flag:
        bleu_results_file = open("SIMILARITY_RESULTS_"+str(ts),'w')
    func = partial(get_similarity,reference_file,query_file)
    results = list_multiprocessing(files,func,workers=processes)
    if write_flag:
        for result in results:
            bleu_results_file.write(result[1]+"\t"+str(result[0])+"\n")
        bleu_results_file.close()
    return results



def calculate_bleu(translations_dir,reference_file,bleu_script_bin,write_flag):
    files = [file for file in os.listdir(translations_dir)]
    files = sorted(files)
    bleus = []
    if write_flag:
        ts = time.time()
        bleu_results_file = open("BLEU_RESULTS_"+str(ts),
                                 'w')
    for file in files:
        translation_file = translations_dir+file
        score,_ = run_bleu(reference_file,bleu_script_bin,translation_file)
        bleus.append(score)
        if write_flag:
            bleu_results_file.write(file+"\t"+str(score)+"\n")
    if write_flag:
        bleu_results_file.close()

    return bleus



if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    parser = OptionParser()
    parser.add_option("-m", "--metric", dest="metric",
                      help="set running mode")
    parser.add_option("-r", "--reference", dest="reference")
    parser.add_option("-t", "--translations_dir", dest="translations_dir")
    parser.add_option("-s", "--bleu_script", dest="bleu_script")
    parser.add_option("-w", "--write", dest="write",action="store_true")
    parser.add_option("-n", "--no_write", dest="write",action="store_false")
    parser.add_option("-k", "--multiproccess", dest="multi",default='0')
    parser.add_option("-q", "--query_file", dest="query_file")
    parser.add_option("-f", "--model_file", dest="model_file")
    (options, args) = parser.parse_args()
    metric = options.metric
    translation_dir = options.translations_dir
    bleu_script_path = options.bleu_script
    reference_file = options.reference
    write_flag = options.write
    ts = time.time()
    if metric.lower()=="bleu":
        if options.multi=='0':
            calculate_bleu(translation_dir,reference_file,bleu_script_path,write_flag)
        else:
            calculate_bleu_multiprocess(translation_dir,reference_file,bleu_script_path,write_flag,ts,int(options.multi))
    if metric.lower()=="accuracy":
        calculate_seq_acc_multiprocess(translation_dir,reference_file,write_flag,ts,int(options.multi))
    if metric.lower()=="coverage":
        query_file = options.query_file
        calculate_seq_query_coverage_multiprocess(translation_dir,reference_file,write_flag,ts,query_file,int(options.multi))
    if metric.lower()=="all":
        calculate_bleu_multiprocess(translation_dir, reference_file, bleu_script_path, write_flag, ts,
                                    int(options.multi))
        calculate_seq_acc_multiprocess(translation_dir, reference_file, write_flag, ts, int(options.multi))
        query_file = options.query_file
        calculate_seq_query_coverage_multiprocess(translation_dir, reference_file, write_flag, ts, query_file,
                                                  int(options.multi))
        model_file = options.model_file
        model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

        calculate_similarity_multiprocess(translation_dir,reference_file,write_flag,ts,query_file,
                                                  int(options.multi))
        