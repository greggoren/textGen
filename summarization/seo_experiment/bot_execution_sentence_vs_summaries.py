import logging
import sys
from optparse import OptionParser
import gensim
from copy import deepcopy
from gen_utils import run_bash_command
import os
from summarization.seo_experiment.borda_mechanism import query_term_freq,centroid_similarity,calculate_similarity_to_docs_centroid_tf_idf,past_winners_centroid, get_past_winners_tfidf_centroid,document_centroid,calculate_semantic_similarity_to_top_docs,get_text_centroid,cosine_similarity
from summarization.seo_experiment.workingset_creator import read_queries_file
from summarization.seo_experiment.utils import clean_texts,read_trec_file,load_file,get_java_object,create_trectext
from summarization.seo_experiment.summarization_process import transform_query_text
from summarization.seo_experiment.summarization_process import list_multiprocessing,read_texts

from nltk import sent_tokenize
import numpy as np
import math
from multiprocessing import cpu_count
from functools import partial

def run_summarization_model(script_file,model_file,kwargs):
    """
     cmd example:
     nohup python ~/OpenNMT-py/translate.py --replace_unk  -beam_size 10 --model ~/OpenNMT-py/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt
      --src input_transformer.txt --output transformer_real_par2.txt
      --batch_size 1  -min_length 1  -gpu 0 &
    """
    command = "python "+script_file+" --replace_unk  -beam_size 10 --model "+model_file+" --batch_size 1 -gpu 0 "
    for key, value in kwargs.items():
        command+="--"+key+" "+value+" "
    print("##Running summarization command: "+command+"##",flush=True)
    out = run_bash_command(command)
    print("Summarization output= "+str(out),flush=True)

def create_summaries(queries,options):
    logger.info("starting summarization")
    input_dir = options.input_summaries_dir
    output_dir = options.output_summaries_dir
    summarization_models = {"lstm": "summarizations_models/gigaword_copy_acc_51.78_ppl_11.71_e20.pt",
                            "transformer": "summarization_models/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt"}
    summary_kwargs = {"lstm": {"min_length": "10", "block_ngram_repeat": "2"}, "transformer": {"min_length": "3"}}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sum_model = options.sum_model
    args = []

    for query in queries:
        kwargs = deepcopy(summary_kwargs[sum_model])
        kwargs["src"] =input_dir+query
        kwargs["output"] =output_dir+query
        args.append(kwargs)

    summary_model = summarization_models[sum_model]
    func = partial(run_summarization_model,options.summary_script_file, summary_model)
    workers = 9 #GPU limited model space
    list_multiprocessing(args,func,workers=workers)
    logger.info("summarization completed")


def prepare_summarization_input_file(input_dir,output_dir,query):
    fname = input_dir+query
    df = read_texts(fname)
    with open(output_dir+query,'w') as out:
        for i,row in df.iterrows():
            paragraph = row["input_paragraph"]
            sentences = sent_tokenize(paragraph)
            fixed_paragraph = " ".join(["<t> " + s + " </t>" for s in sentences])
            out.write(fixed_paragraph+"\n")


def prepare_summarization_input(summarization_pool_dir,output_dir,queries):
    if os.path.exists(output_dir):
        logger.info("summaries input files exist already, no input creation needed")
        return
    else:
        os.makedirs(output_dir)
    func = partial(prepare_summarization_input_file,summarization_pool_dir,output_dir)
    workers = cpu_count()-1
    list_multiprocessing(queries,func,workers=workers)
    logger.info("summarization preperation done")





def read_raw_ds(raw_dataset):
    result={}
    with open(raw_dataset,encoding="utf-8") as ds:
        for line in ds:
            query = line.split("\t")[0]
            if query not in result:
                result[query]={}
            key = line.split("\t")[1]
            sentence_out = line.split("\t")[2]
            sentence_in = line.split("\t")[3].rstrip()
            result[query][key] = {"in":sentence_in,"out":sentence_out}
    return result



def reverese_query(qid):
    epoch = str(qid)[-2:]
    query = str(qid)[:-2].zfill(3)
    return epoch,query



def context_similarity(replacement_index,ref_sentences,sentence_compared,mode,model,stemmer=None):
    if mode=="own":
        ref_sentence = ref_sentences[replacement_index]
        return centroid_similarity(clean_texts(ref_sentence),clean_texts(sentence_compared),model,stemmer)
    if mode =="pred":
        if replacement_index+1==len(ref_sentences):
            sentence = ref_sentences[replacement_index]
        else:
            sentence = ref_sentences[replacement_index+1]
        return centroid_similarity(clean_texts(sentence), clean_texts(sentence_compared), model,stemmer)
    if mode=="prev":
        if replacement_index==0:
            sentence = ref_sentences[replacement_index]
        else:
            sentence = ref_sentences[replacement_index-1]
        return centroid_similarity(clean_texts(sentence), clean_texts(sentence_compared), model,stemmer)



def get_past_winners(ranked_lists,epoch,query):
    past_winners = []
    for iteration in range(int(epoch)):
        current_epoch = str(iteration+1).zfill(2)
        past_winners.append(ranked_lists[current_epoch][query][0])
    return past_winners




def write_files(feature_list, feature_vals, output_dir, qid):
    for feature in feature_list:
        with open(output_dir+"doc"+feature+"_"+qid,'w') as out:
            for pair in feature_vals[feature]:
                out.write(pair+" "+str(feature_vals[feature][pair])+"\n")




def create_features(summaries_dir,ranked_lists, doc_texts, top_doc_index, ref_doc_index, doc_tfidf_vectors_dir, tfidf_sentence_dir,summary_tfidf_dir, queries, output_dir, qid):
    global word_embd_model
    feature_vals = {}
    feature_list = ["FractionOfQueryWordsIn","FractionOfQueryWordsOut","CosineToCentroidIn","CosineToCentroidInVec","CosineToCentroidOut","CosineToCentroidOutVec","CosineToWinnerCentroidInVec","CosineToWinnerCentroidOutVec","CosineToWinnerCentroidIn","CosineToWinnerCentroidOut","SimilarityToPrev","SimilarityToRefSentence","SimilarityToPred","SimilarityToPrevRef","SimilarityToPredRef"]

    for feature in feature_list:
        feature_vals[feature]={}

    epoch,qid_original = reverese_query(qid)
    query = queries[qid]
    past_winners = get_past_winners(ranked_lists,epoch,qid_original)
    past_winners_semantic_centroid_vector = past_winners_centroid(past_winners,doc_texts,word_embd_model)
    past_winners_tfidf_centroid_vector = get_past_winners_tfidf_centroid(past_winners,doc_tfidf_vectors_dir)
    top_docs = ranked_lists[epoch][qid_original][:top_doc_index]
    ref_doc = ranked_lists[epoch][qid_original][ref_doc_index]
    ref_sentences = sent_tokenize(doc_texts[ref_doc])
    top_docs_tfidf_centroid = document_centroid([get_java_object(doc_tfidf_vectors_dir+doc) for doc in top_docs])
    with open(summaries_dir+"_".join(query.split())) as summaries:
        for i,sentence_out in enumerate(ref_sentences):
            for j,sentence_in in enumerate(summaries):
                pair = ref_doc+"_"+str(i)+"_"+str(j)
                replace_index = i
                in_vec = get_text_centroid(clean_texts(sentence_in), word_embd_model, True)
                out_vec = get_text_centroid(clean_texts(sentence_out), word_embd_model, True)
                feature_vals['FractionOfQueryWordsIn'][pair] = query_term_freq("avg", clean_texts(sentence_in),
                                                                               clean_texts(query))
                feature_vals['FractionOfQueryWordsOut'][pair] = query_term_freq("avg", clean_texts(sentence_out),
                                                                                clean_texts(query))

                feature_vals['CosineToCentroidIn'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
                    summary_tfidf_dir + pair, top_docs_tfidf_centroid)
                feature_vals['CosineToCentroidOut'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
                    tfidf_sentence_dir + pair.split("_")[0] + "_" + pair.split("_")[1], top_docs_tfidf_centroid)

                feature_vals["CosineToCentroidInVec"][pair] = calculate_semantic_similarity_to_top_docs(sentence_in,
                                                                                                        top_docs,
                                                                                                        doc_texts,
                                                                                                        word_embd_model,
                                                                                                        True)
                feature_vals["CosineToCentroidOutVec"][pair] = calculate_semantic_similarity_to_top_docs(sentence_out,
                                                                                                         top_docs,
                                                                                                         doc_texts,
                                                                                                         word_embd_model,
                                                                                                         True)

                feature_vals['CosineToWinnerCentroidInVec'][pair] = cosine_similarity(in_vec,
                                                                                      past_winners_semantic_centroid_vector)
                feature_vals['CosineToWinnerCentroidOutVec'][pair] = cosine_similarity(out_vec,
                                                                                       past_winners_semantic_centroid_vector)
                feature_vals['CosineToWinnerCentroidIn'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
                    summary_tfidf_dir + pair, past_winners_tfidf_centroid_vector)
                feature_vals['CosineToWinnerCentroidOut'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
                    tfidf_sentence_dir + pair.split("_")[0] + "_" + pair.split("_")[1],
                    past_winners_tfidf_centroid_vector)

                feature_vals['SimilarityToPrev'][pair] = context_similarity(replace_index, ref_sentences, sentence_in,
                                                                            "prev", word_embd_model, True)
                feature_vals['SimilarityToRefSentence'][pair] = context_similarity(replace_index, ref_sentences,
                                                                                   sentence_in, "own", word_embd_model,
                                                                                   True)
                feature_vals['SimilarityToPred'][pair] = context_similarity(replace_index, ref_sentences, sentence_in,
                                                                            "pred", word_embd_model, True)
                feature_vals['SimilarityToPrevRef'][pair] = context_similarity(replace_index, ref_sentences,
                                                                               sentence_out, "prev", word_embd_model,
                                                                               True)
                feature_vals['SimilarityToPredRef'][pair] = context_similarity(replace_index, ref_sentences,
                                                                               sentence_out, "pred", word_embd_model,
                                                                               True)



    write_files(feature_list,feature_vals,output_dir,qid)

def feature_creation_parallel(raw_dataset_file, ranked_lists, doc_texts, top_doc_index, ref_doc_index, doc_tfidf_vectors_dir, tfidf_sentence_dir, tfidf_summary_dir, queries, output_feature_files_dir, output_final_features_dir, workingset_file):
    global word_embd_model
    args = [qid for qid in queries]
    if not os.path.exists(output_feature_files_dir):
        os.makedirs(output_feature_files_dir)
    if not os.path.exists(output_final_features_dir):
        os.makedirs(output_final_features_dir)
    raw_ds = read_raw_ds(raw_dataset_file)
    create_ws(raw_ds,workingset_file)
    func = partial(create_features, raw_ds, ranked_lists, doc_texts, top_doc_index, ref_doc_index, doc_tfidf_vectors_dir, tfidf_sentence_dir, tfidf_summary_dir, queries, output_feature_files_dir)
    workers = cpu_count()-1
    list_multiprocessing(args,func,workers=workers)
    command = "perl generateSentences.pl " + output_feature_files_dir+" "+workingset_file
    run_bash_command(command)
    run_bash_command("mv features "+output_final_features_dir)


def run_svm_rank_model(test_file, model_file, predictions_folder):
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)
    predictions_file = predictions_folder + os.path.basename(model_file)
    command = "./svm_rank_classify " + test_file + " " + model_file + " " + predictions_file
    print("##Running command: "+command+"##")
    out = run_bash_command(command)
    print("Output of ranking command: "+str(out),flush=True)
    return predictions_file

def retrieve_scores(test_indices, score_file):
    with open(score_file) as scores:
        results = {test_indices[i]: score.rstrip() for i, score in enumerate(scores)}
        return results

def create_index_to_doc_name_dict(features):
    doc_name_index = {}
    index = 0
    with open(features) as ds:
        for line in ds:
            rec = line.split("# ")
            doc_name = rec[1].rstrip()
            doc_name_index[index] = doc_name
            index += 1
    return doc_name_index

def create_trec_eval_file(queries, results,fname):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    trec_file_access = open(fname, 'w')
    for doc in results:
        trec_file_access.write(queries[doc]+ " Q0 " + doc + " " + str(0) + " " + str(
                results[doc]) + " sentences\n")
    trec_file_access.close()

def order_trec_file(trec_file):
    final = trec_file.replace(".txt","_sorted.txt")
    command = "sort -k1,1n -k5nr -k2,1 "+trec_file+" > "+final
    print(command)
    run_bash_command(command)
    return final


def read_sentence_results(fname):
    result = {}
    with open(fname) as file:
        for line in file:
            query = line.split()[0]
            pair = line.split()[2]
            if query not in result:
                result[query]=[]
            result[query].append(pair)
    return result

def update_texts(doc_texts, pairs_ranked_lists, sentence_data):
    new_texts = {}
    for qid in pairs_ranked_lists:
        chosen_pair = pairs_ranked_lists[qid][0]
        ref_doc = chosen_pair.split("_")[0]
        replacement_index = int(chosen_pair.split("_")[1])
        sentence_in = sentence_data[qid][chosen_pair]["in"]
        sentences = sent_tokenize(doc_texts[ref_doc])
        sentences[replacement_index]=sentence_in
        new_text = "\n".join(sentences)
        new_texts[ref_doc]=new_text
    for doc in doc_texts:
        if doc not in new_texts:
            new_texts[doc]=doc_texts[doc]
    return new_texts



def create_ws(raw_ds,ws_fname):
    with open(ws_fname,'w') as ws:
        for qid in raw_ds:
            for i,pair in enumerate(raw_ds[qid]):
                ws.write(qid+" Q0 "+pair+" 0 "+str(i+1)+" pairs_seo\n")


if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    parser = OptionParser()

    parser.add_option("--mode", dest="mode")
    parser.add_option("--index_path", dest="index_path")
    parser.add_option("--raw_ds_out", dest="raw_ds_out")
    parser.add_option("--ref_index", dest="ref_index")
    parser.add_option("--input_summaries_dir", dest="input_summaries_dir")
    parser.add_option("--output_summaries_dir", dest="output_summaries_dir")
    parser.add_option("--input_summarization_pool_dir", dest="input_summarization_pool_dir")
    parser.add_option("--top_docs_index", dest="top_docs_index")

    parser.add_option("--doc_tfidf_dir", dest="doc_tfidf_dir")
    parser.add_option("--sentences_tfidf_dir", dest="sentences_tfidf_dir")
    parser.add_option("--summary_tfidf_dir", dest="summary_tfidf_dir")
    parser.add_option("--queries_file", dest="queries_file")
    parser.add_option("--scores_dir", dest="scores_dir")
    parser.add_option("--trec_file", dest="trec_file")
    parser.add_option("--sentence_trec_file", dest="sentence_trec_file")
    parser.add_option("--output_feature_files_dir", dest="output_feature_files_dir")
    parser.add_option("--output_final_feature_file_dir", dest="output_final_feature_file_dir")
    parser.add_option("--trectext_file", dest="trectext_file")
    parser.add_option("--new_trectext_file", dest="new_trectext_file")
    parser.add_option("--model_file", dest="model_file")
    parser.add_option("--workingset_file", dest="workingset_file")
    parser.add_option("--svm_model_file", dest="svm_model_file")
    parser.add_option("--sum_model", dest="sum_model")
    parser.add_option("--summary_script_file", dest="summary_script_file")
    (options, args) = parser.parse_args()
    ranked_lists = read_trec_file(options.trec_file)
    doc_texts = load_file(options.trectext_file)
    mode = options.mode

    if mode=="prepare":
        queries = read_queries_file(options.queries_file)
        queries = transform_query_text(queries)
        query_args = list(set(["_".join(queries[qid].split()) for qid in queries]))
        prepare_summarization_input(options.input_summarization_pool_dir,options.input_summaries_dir,query_args)
        create_summaries(query_args,options)

    if mode=="features":
        queries = read_queries_file(options.queries_file)
        queries = transform_query_text(queries)
        word_embd_model = gensim.models.KeyedVectors.load_word2vec_format(options.model_file,binary=True,limit=700000)
        feature_creation_parallel(options.raw_ds_out,ranked_lists,doc_texts,int(options.top_docs_index),int(options.ref_index),options.doc_tfidf_dir,
                                  options.sentences_tfidf_dir,options.summary_tfidf_dir,queries,options.output_feature_files_dir,options.output_final_feature_file_dir,options.workingset_file)

    if mode=="update":
        features_file = options.output_final_feature_file_dir+"features"
        name_index = create_index_to_doc_name_dict(features_file)
        scores_file = run_svm_rank_model(features_file,options.svm_model_file,options.scores_dir)
        results = retrieve_scores(name_index,scores_file)
        queries_index = {pair:str(int(pair.split("_")[0].split("-")[2]))+pair.split("$")[0].split("-")[1] for pair in results}
        create_trec_eval_file(queries_index,results,options.sentence_trec_file)
        final_trec = order_trec_file(options.sentence_trec_file)
        ranked_pairs = read_sentence_results(final_trec)
        new_texts = update_texts(doc_texts,ranked_pairs,read_raw_ds(options.raw_ds_out))
        create_trectext(new_texts,options.new_trectext_file,"data/ws_debug")

    if mode=='all':
        queries = read_queries_file(options.queries_file)
        queries = transform_query_text(queries)
        word_embd_model = gensim.models.KeyedVectors.load_word2vec_format(options.model_file, binary=True,limit=700000)
        feature_creation_parallel(options.raw_ds_out, ranked_lists, doc_texts, int(options.top_docs_index),
                                  int(options.ref_index), options.doc_tfidf_dir,
                                  options.sentences_tfidf_dir, options.summary_tfidf_dir, queries, options.output_feature_files_dir,
                                  options.output_final_feature_file_dir, options.workingset_file)
        features_file = options.output_final_feature_file_dir + "features"
        name_index = create_index_to_doc_name_dict(features_file)
        scores_file = run_svm_rank_model(features_file, options.svm_model_file, options.scores_dir)
        results = retrieve_scores(name_index, scores_file)
        queries_index = {pair: str(int(pair.split("_")[0].split("-")[2])) + pair.split("_")[0].split("-")[1] for pair in
                         results}
        create_trec_eval_file(queries_index, results, options.sentence_trec_file)
        final_trec = order_trec_file(options.sentence_trec_file)
        ranked_pairs = read_sentence_results(final_trec)
        new_texts = update_texts(doc_texts, ranked_pairs, read_raw_ds(options.raw_ds_out))
        create_trectext(new_texts, options.new_trectext_file, "data/ws_debug")
