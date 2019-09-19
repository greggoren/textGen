import logging
import os
import sys
from optparse import OptionParser

import gensim
import nltk
from summarization.seo_experiment.borda_mechanism import calculate_seo_predictors
from summarization.seo_experiment.utils import create_trectext,load_file

def read_summaries_data(summaries_file, input_data_file, summaries_tfidf_dir,queries_file):
    summary_stats = {}
    summary_tfidf_fname_index = {}
    indexes = {}
    queries_text = {}
    reference_docs={}
    with open(input_data_file,encoding="utf-8") as input_data:
        inputs = input_data.readlines()
        with open(queries_file,encoding="utf-8") as queries_data:
            queries_lines = queries_data.readlines()
            with open(summaries_file,encoding="utf-8") as summaries_data:
                running_index = 0
                summary_index = 0
                last_query =None
                for i,summary in enumerate(summaries_data):
                    input = inputs[i+1]
                    doc = input.split("\t")[1]
                    index = input.split("\t")[2]
                    query = input.split("\t")[0]
                    if query!=last_query:
                        last_query=query
                        summary_index=0
                    reference_docs[query]=doc
                    queries_text[query] = queries_lines[i]
                    if query not in summary_tfidf_fname_index:
                        summary_tfidf_fname_index[query]={}
                    summary_tfidf_fname_index[query][summary_index]= summaries_tfidf_dir + doc + "_" + index + "_" + str(running_index)
                    running_index+=1
                    summary_index+=1
                    indexes[query]=int(index)
                    if query not in summary_stats:
                        summary_stats[query]=[]
                    summary_stats[query].append(summary)
    return summary_stats,summary_tfidf_fname_index,indexes,queries_text,reference_docs



def get_top_docs(trec_file,number_of_top_docs):
    """
    Relies on the fact trec file is sorted by qid,doc_score(reverse),doc_name
    """
    top_docs_per_query={}
    with open(trec_file) as file:
        for line in file:
            query = line.split()[0]
            if query not in top_docs_per_query:
                top_docs_per_query[query]=[]
            if len(top_docs_per_query[query])<number_of_top_docs:
                doc = line.split()[2]
                top_docs_per_query[query].append(doc)
    return top_docs_per_query

def update_text(text,summary,replacement_index):
    sentences = nltk.sent_tokenize(text)
    sentences[replacement_index]=summary.replace("\"","")
    return "\n".join(sentences)

def update_document_texts(updated_document_texts,document_texts):
    for doc in document_texts:
        if doc not in updated_document_texts:
            updated_document_texts[doc]=document_texts[doc]
    return updated_document_texts

def update_texts_with_replacement_summary(replacement_indexes,summaries_stats,document_vectors_dir,query_texts,document_texts,trec_file,number_of_top_docs,summary_tfidf_fname_index,reference_docs,model):
    updated_document_text={}
    top_docs_per_query=get_top_docs(trec_file,number_of_top_docs)
    """ Written only for analysis purposes!!!"""

    with open("summaries/summary_analysis.txt",'w') as analysis_file:
        """Production code:"""
        for query in summaries_stats:
            summaries = summaries_stats[query]
            query_text = " ".join(query_texts[query].split("_")).rstrip()
            document_text = document_texts[reference_docs[query]]
            top_docs = top_docs_per_query[query]
            summary_tfidf_fnames = summary_tfidf_fname_index[query]
            replacement_index = replacement_indexes[query]
            chosen_summary = calculate_seo_predictors(summaries,summary_tfidf_fnames,replacement_index,query_text,document_text,document_vectors_dir,document_texts,top_docs,model)
            summary = chosen_summary.replace("<t>","").replace("</t>","").rstrip()
            updated_text = update_text(document_text,summary,replacement_index)
            updated_document_text[reference_docs[query]]=updated_text
            """ Written only for analysis purposes!!!"""
            source_sentence = nltk.sent_tokenize(document_text)[replacement_index].rstrip().replace("\n","")
            analysis_file.write(query_text+"\t"+reference_docs[query]+source_sentence+"\t"+summary+"\n")
    return update_document_texts(updated_document_text,document_texts)


if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    parser = OptionParser()
    parser.add_option("--doc_tfidf_dir", dest="doc_tfidf_dir")
    parser.add_option("--summaries_tfidf_dir", dest="summaries_tfidf_dir")
    parser.add_option("--queries_file", dest="queries_file")
    parser.add_option("--summaries_file", dest="summaries_file")
    parser.add_option("--input_data_file", dest="input_data_file")
    parser.add_option("--trec_file", dest="trec_file")
    parser.add_option("--number_of_top_docs", dest="number_of_top_docs")
    parser.add_option("--trectext_file", dest="trectext_file")
    parser.add_option("--new_trectext_file", dest="new_trectext_file")
    parser.add_option("--new_ws_file", dest="new_ws_file")
    parser.add_option("--model_file", dest="model_file")
    (options, args) = parser.parse_args()
    summary_stats,summary_tfidf_fname_index,replacement_indexes,queries_text,reference_docs=read_summaries_data(options.summaries_file,options.input_data_file,options.summaries_tfidf_dir,options.queries_file)
    document_texts = load_file(options.trectext_file)
    model = gensim.models.FastText.load_fasttext_format(options.model_file)
    # model = gensim.models.KeyedVectors.load_word2vec_format("../../w2v/testW2V.txt"  ,binary=True)
    updated_texts = update_texts_with_replacement_summary(replacement_indexes,summary_stats,options.doc_tfidf_dir,queries_text,document_texts,options.trec_file,int(options.number_of_top_docs),summary_tfidf_fname_index,reference_docs,model)
    create_trectext(updated_texts,options.new_trectext_file,options.new_ws_file)