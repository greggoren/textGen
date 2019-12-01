import sys
from optparse import OptionParser
from summarization.seo_experiment.utils import load_file,clean_texts,run_summarization_model
from nltk import sent_tokenize
import os,logging
import pandas as pd
from summarization.seo_experiment.workingset_creator import read_queries_file
from summarization.seo_experiment.borda_mechanism import calculate_summarization_predictors_for_former_docs,calculate_seo_replacement_predictors
import gensim
from multiprocessing import Pool,cpu_count
from tqdm import tqdm
from functools import partial
from summarization.seo_experiment.utils\
    import get_past_winners,reverese_query
from summarization.seo_experiment.evaluation.analysis import read_trec_file as rtf


def reference_docs_calculation(stats,ref_index):
    return {q:stats[q][ref_index] for q in stats}

def get_reference_docs(trec_file, index):
    stats = read_trec_file(trec_file)
    return reference_docs_calculation(stats,index)

def read_trec_file(trec_file):
    stats = {}
    with open(trec_file) as file:
        for line in file:
            query = line.split()[0]
            doc = line.split()[2]
            if query not in stats:
                stats[query]=[]
            stats[query].append(doc)
    return stats





def chosen_sentence_for_replacement(sentences, query,document_name,sentences_vectors_dir,docuemnt_vectors_dir,document_texts,top_docs,past_winners,model):
    chosen_idx = calculate_seo_replacement_predictors(sentences,query,document_name,sentences_vectors_dir,docuemnt_vectors_dir,document_texts,top_docs,past_winners,model)
    return chosen_idx


# def chosen_sentence_for_replacement(sentences, query):
#     sentence_scores={}
#     for i,sentence in enumerate(sentences):
#         tokens = clean_texts(sentence.lower()).split()
#         sentence_scores[i]=(-sum([tokens.count(w) for w in query.split()]),len(tokens))
#     return sorted(list(sentence_scores.keys()),key=lambda x:(sentence_scores[x][0],sentence_scores[x][1],x),reverse=True)[0]

def get_sentences_for_replacement(doc_texts,reference_docs,query_text,sentences_vectors_dir,docuemnt_vectors_dir,top_docs_index,ranked_lists,starting_epoch,model):
    replacements = {}
    relevant_queries = [qid for qid in reference_docs if int(reverese_query(qid)[0])>=starting_epoch]
    for qid in relevant_queries:
        doc = reference_docs[qid]
        text = doc_texts[doc]
        sentences = sent_tokenize(text)
        query = query_text[qid]
        top_docs = top_docs_index[qid]
        epoch,real_qid = reverese_query(qid)
        past_winners = get_past_winners(ranked_lists,epoch,real_qid)
        chosen_index = chosen_sentence_for_replacement(sentences, query,doc,sentences_vectors_dir,docuemnt_vectors_dir,doc_texts,top_docs,past_winners,model)
        replacements[qid]=chosen_index
    return replacements


def write_input_dataset_file(replacements,reference_docs,texts,suffix):
    input_dir = 'input_data/'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    with open(input_dir+"senteces_for_replace_"+suffix+".txt",'w') as file:
        file.write("query\tdocname\tsentence_index\tsentence\n")
        for query in replacements:
            index = replacements[query]
            docname = reference_docs[query]
            text = texts[docname]
            sentence = clean_texts(sent_tokenize(text)[index])
            file.write("\t".join([query,docname,str(index),sentence])+'\n')
    return input_dir+"senteces_for_replace_"+suffix+".txt"


def read_texts(fname,inp=False):
    if inp:
        df = pd.read_csv(fname, delimiter="\t", header =0,encoding="unicode_escape")
    else:
        df = pd.read_csv(fname,delimiter="\t",names=["query", "input_paragraph"])
    return df


def write_files(dict,**kwargs):
    for key,val in kwargs.items():
        dict[val[0]].write(val[1]+"\n")


def calcualte_former_documents(current_epoch,qid,document_texts):
    former_docs = []
    seen_texts = []
    for doc in document_texts:
        query = doc.split("-")[2]
        epoch = int(doc.split("-")[1])
        if qid !=query:
            continue
        if epoch >= current_epoch:
            continue
        doc_text = document_texts[doc]
        cleaned_text = clean_texts(doc_text)
        cleaned_text = cleaned_text.replace(" ","")
        if cleaned_text in seen_texts:
            continue
        former_docs.append(doc)
        seen_texts.append(cleaned_text)
    return former_docs

def creaion_parrallel(queries_text,candidates_dir,input_df,files,document_texts,ref_docs,top_docs,document_vector_dir,paragraph_vector_dir,ranked_lists,start_epoch,row):
    global model
    complete_data = "\t".join([str(row[str(col)]).rstrip() for col in input_df.columns])
    results =[]
    query = queries_text[str(row["query"])]
    qid = str(row["query"])
    query = "_".join(query.split())
    sentence = str(row["sentence"]).rstrip()
    replacement_index = int(row["sentence_index"])
    query_paragraph_df = read_texts(candidates_dir + query)
    epoch,real_query = reverese_query(qid)
    past_winners = get_past_winners(ranked_lists,epoch,real_query)
    former_docs = calcualte_former_documents(epoch,real_query,document_texts)
    chosen_former_docs = calculate_summarization_predictors_for_former_docs(query_paragraph_df, former_docs,sentence,replacement_index,qid,queries_text,document_texts,ref_docs,top_docs[qid],past_winners,document_vector_dir,paragraph_vector_dir, model)
    for chosen_doc in chosen_former_docs.split("\n##\n"):
        if sum_model == 'transformer':
            sentences = sent_tokenize(chosen_doc)
            chosen_doc = " ".join(["<t> " + s + " </t>" for s in sentences])
            args = {"complete":(files[0],complete_data+"\t"+chosen_doc),
                                        "queries":  (files[1]," ".join(query.split("_"))),"source":(files[2],sentence),
                                        "inp_former_docs":(files[3],chosen_doc)}
            results.append(args)
    return results




def list_multiprocessing(param_lst, func, **kwargs):
    workers = kwargs.pop('workers')
    with Pool(workers) as p:
        apply_lst = [([params], func, i, kwargs) for i, params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))
    return [_[1] for _ in result]


def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params, **kwargs)



def parrallel_create_summarization_task(input_dataset_file, candidates_dir, queries_text, sum_model,document_texts,ref_docs,top_docs,document_vector_dir,paragraph_vector_dir,ranked_lists,suffix):
    global model
    input_df = read_texts(input_dataset_file, True)
    with open(os.path.dirname(input_dataset_file) + "/all_data_" + sum_model +"_"+suffix+ ".txt", 'w',
              encoding="utf-8") as complete:
        with open(os.path.dirname(input_dataset_file) + "/queries_" + sum_model +"_"+suffix+ ".txt", 'w',
                  encoding="utf-8") as queries:
            with open(os.path.dirname(input_dataset_file) + "/source_" + sum_model +"_"+suffix+ ".txt", 'w',
                      encoding="utf-8") as source:
                with open(os.path.dirname(input_dataset_file) + "/input_paragraphs_" + sum_model+"_"+suffix+  ".txt", 'w',
                          encoding="utf-8") as inp_paragraphs:
                    header = "\t".join([str(col) for col in input_df.columns]) + "\tinput_paragraph\n"
                    complete.write(header)
                    arguments = [row for i,row in input_df.iterrows()]
                    files = ["complete","queries","source","inp_former_docs"]
                    files_access = {"complete":complete,"queries":queries,"source":source,"inp_former_docs":inp_paragraphs}
                    func = partial(creaion_parrallel,queries_text,candidates_dir,input_df,files,document_texts,ref_docs,top_docs,document_vector_dir,paragraph_vector_dir,ranked_lists)
                    workers = cpu_count()-1
                    results = list_multiprocessing(arguments,func,workers=workers)
                    for result in results:
                        for writes in result:
                            write_files(files_access,**writes)
    return os.path.dirname(input_dataset_file) + "/input_paragraphs_" + sum_model +"_"+suffix +".txt"


def transform_query_text(queries_raw_text):
    transformed = {}
    for qid in queries_raw_text:
        transformed[qid]=queries_raw_text[qid].replace("#combine( ","").replace(" )","")
    return transformed

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


def summarization_ds(options):
    global model
    sum_model = options.sum_model
    logger.info("reading queries file")
    raw_queries = read_queries_file(options.queries_file)
    logger.info("reading trec file")
    ranked_lists = rtf(options.trec_file)
    logger.info("transforming queries")
    queries = transform_query_text(raw_queries)
    logger.info("reading trectext file")
    doc_texts = load_file(options.trectext_file)
    logger.info("calculating reference docs")
    reference_docs = get_reference_docs(options.trec_file, int(options.ref_index))
    logger.info("calculating top docs")
    top_docs = get_top_docs(options.trec_file,int(options.number_of_top_docs))
    logger.info("calculating sentences for replacement")
    senteces_for_replacement = get_sentences_for_replacement(doc_texts, reference_docs,queries,options.sentences_vectors_dir,options.documents_vectors_dir,top_docs,ranked_lists,int(options.starting_epoch),model)
    logger.info("writing input sentences file")
    input_file = write_input_dataset_file(senteces_for_replacement, reference_docs, doc_texts,options.suffix)
    logger.info("writing all files")
    return parrallel_create_summarization_task(input_file, options.candidate_dir, queries,  sum_model,doc_texts,reference_docs,top_docs,options.documents_vectors_dir,options.paragraph_vectors_dir,ranked_lists,options.suffix)




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
    parser.add_option("--starting_epoch", dest="starting_epoch")
    parser.add_option("--candidate_dir", dest="candidate_dir")
    parser.add_option("--summary_script_file", dest="summary_script_file")
    parser.add_option("--number_of_top_docs", dest="number_of_top_docs")
    parser.add_option("--sentences_vectors_dir", dest="sentences_vectors_dir")
    parser.add_option("--documents_vectors_dir", dest="documents_vectors_dir")
    parser.add_option("--model_file", dest="model_file")
    parser.add_option("--summary_output_file", dest="summary_output_file")
    parser.add_option("--summary_input_file", dest="summary_input_file")
    parser.add_option("--suffix", dest="suffix")
    (options, args) = parser.parse_args()
    #TODO: make it more generic later
    summarization_models = {"lstm":"summarizations_models/gigaword_copy_acc_51.78_ppl_11.71_e20.pt","transformer":"summarization_models/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt"}
    summary_kwargs = {"lstm":{"min_length" :"10","block_ngram_repeat": "2"},"transformer":{"min_length" :"3"}}
    sum_model = options.sum_model
    if options.mode =="ds":
        # model = gensim.models.FastText.load_fasttext_format(options.model_file)
        model = gensim.models.KeyedVectors.load_word2vec_format(options.model_file, binary=True, limit=700000)
        # model = gensim.models.KeyedVectors.load_word2vec_format("../../w2v/testW2V.txt"  ,binary=True)
        summarization_ds(options)
    elif options.mode=="summary":
        summary_model = summarization_models[sum_model]
        input_file = options.summary_input_file
        output_file = options.summary_output_file + "_" + sum_model+ "_"+options.suffix+ ".txt"
        logger.info("starting summarization")

        run_summarization_model(options.summary_script_file, summary_model, input_file, output_file,
                                **summary_kwargs[sum_model])
    elif options.mode=="all":

        # model = gensim.models.FastText.load_fasttext_format(options.model_file)
        model = gensim.models.KeyedVectors.load_word2vec_format(options.model_file, binary=True, limit=700000)
        input_file = summarization_ds(options)
        summary_model = summarization_models[sum_model]
        output_file = options.summary_output_file+"_"+sum_model+"_"+options.suffix+".txt"
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        logger.info("starting summarization")
        run_summarization_model(options.summary_script_file,summary_model,input_file,output_file,**summary_kwargs[sum_model])



