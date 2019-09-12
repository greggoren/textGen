import sys
from optparse import OptionParser
from summarization.seo_experiment.utils import load_file,clean_texts,run_summarization_model
from nltk import sent_tokenize
import os,logging
import numpy as np
import pandas as pd
from functools import partial
from summarization.seo_experiment.workingset_creator import read_queries_file
from summarization.seo_experiment.borda_mechanism import calculate_predictors
import gensim


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
        ranked_list = sorted(list(stats[query].keys()),key=lambda x:(stats[query][x],x),reverse = True)
        transformed[query]=ranked_list
    return transformed

def reference_docs_calculation(stats,ref_index):
    return {q:stats[q][ref_index] for q in stats}

def get_reference_docs(trec_file, index):
    stats = read_trec_file(trec_file)
    return reference_docs_calculation(stats,index)





def chosen_sentence_for_replacement(sentences, query):
    sentence_scores={}
    for i,sentence in enumerate(sentences):
        tokens = clean_texts(sentence.lower()).split()
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
            sentence = clean_texts(sent_tokenize(text)[index])
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

def create_summarization_dataset(input_dataset_file, candidates_dir, queries_text, model, sum_model):
    input_df = read_texts(input_dataset_file,True)
    with open(os.path.dirname(input_dataset_file) +"/all_data_" + sum_model + ".txt", 'w') as complete:
        with open(os.path.dirname(input_dataset_file) +"/queries_" + sum_model + ".txt", 'w') as queries:
            with open(os.path.dirname(input_dataset_file) +"/source_" + sum_model + ".txt", 'w') as source:
                with open(os.path.dirname(input_dataset_file) +"/input_paragraphs_" + sum_model + ".txt", 'w') as inp_paragraphs:
                    header = "\t".join([str(col) for col in input_df.columns])+"\tinput_paragraph\n"
                    complete.write(header)
                    for i,row in input_df.iterrows():
                        complete_data="\t".join([str(row[str(col)]) for col in input_df.columns])
                        query = queries_text[str(row["query"])]
                        query="_".join(query.split())
                        sentence = row["sentence"]
                        query_paragraph_df = read_texts(candidates_dir+query)
                        paragraphs = calculate_predictors(query_paragraph_df,sentence,query,model)
                        for paragraph in paragraphs.split("\n##\n"):
                            if sum_model == 'transformer':
                                paragraph = "<t> "+ paragraph.replace(".",". </t> <t>").rstrip() +" </t>\n"
                                paragraph = paragraph.replace('</t> </t>','')
                            write_files(complete=(complete,complete_data+"\t"+paragraph),queries = (queries,query),source=(source,sentence),inp_paragraphs=(inp_paragraphs,paragraph))
    return os.path.dirname(input_dataset_file) +"/input_paragraphs_" + sum_model + ".txt"

def transform_query_text(queries_raw_text):
    transformed = {}
    for qid in queries_raw_text:
        transformed[qid]=queries_raw_text[qid].replace("#combine( ","").replace(" )","")
    return transformed

def summarization_ds(options):
    sum_model = options.sum_model
    raw_queries = read_queries_file(options.queries_file)
    queries = transform_query_text(raw_queries)
    doc_texts = load_file(options.trectext_file)
    reference_docs = get_reference_docs(options.trec_file, int(options.ref_index))
    senteces_for_replacement = get_sentences_for_replacement(doc_texts, reference_docs)
    input_file = write_input_dataset_file(senteces_for_replacement, reference_docs, doc_texts)
    model = gensim.models.FastText.load_fasttext_format(options.model_file)
    return create_summarization_dataset(input_file, options.candidate_dir, queries, model, sum_model)


if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    parser = OptionParser()
    parser.add_option("--mode", dest="mode")
    parser.add_option("--sum_model", dest="sum_model")
    parser.add_option("--trectext_file", dest="trectext_file")
    parser.add_option("--queries_file", dest="queries_file")
    parser.add_option("--trec_file", dest="trec_file")
    parser.add_option("--ref_index", dest="ref_index")
    parser.add_option("--candidate_dir", dest="candidate_dir")
    parser.add_option("--summary_script_file", dest="summary_script_file")
    parser.add_option("--model_file", dest="model_file")
    parser.add_option("--summary_output_file", dest="summary_output_file")
    parser.add_option("--summary_input_file", dest="summary_input_file")
    (options, args) = parser.parse_args()
    #TODO: make it more generic later
    summarization_models = {"lstm":"summarizations_models/gigaword_copy_acc_51.78_ppl_11.71_e20.pt","transformer":"summarization_models/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt"}
    summary_kwargs = {"lstm":{"min_length" :"10","block_ngram_repeat": "2"},"transformer":{"min_length" :"1"}}
    sum_model = options.sum_model
    if options.mode =="ds":
        summarization_ds(options)
    elif options.mode=="summary":
        summary_model = summarization_models[sum_model]
        input_file = options.summary_input_file
        run_summarization_model(options.summary_script_file, summary_model, input_file, options.summary_output_file,
                                **summary_kwargs[sum_model])
    elif options.mode=="all":
        input_file = summarization_ds(options)
        summary_model = summarization_models[sum_model]
        output_file = options.summary_output_file+"_"+sum_model+".txt"
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        run_summarization_model(options.summary_script_file,summary_model,input_file,,**summary_kwargs[sum_model])



