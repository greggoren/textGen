import sys
from summarization.seo_experiment.utils import load_file
import nltk
def read_data_file(fname):
    queries = {}
    reference_docs = {}
    with open(fname) as file:
        for i,line in enumerate(file):
            if i==0:
                continue
            query = line.split("\t")[0]
            doc = line.split("\t")[1]
            reference_docs[query]=doc
            queries[i-1]=query
    return queries,reference_docs

def read_summaries_file(fname):
    stats = {}
    with open(fname) as file:
        for i,line in enumerate(file):
            stats[i]=line.replace("<t>","").replace("</t>","")
    return stats


def fix_encoding(text):
    return bytes(text, 'cp1252', "ignore").decode('utf-8', 'ignore').replace("\n", " ").replace("\r", " ")

def write_raw_ds(queries, summaries, fname, document_texts, reference_docs):
    with open(fname,'w') as out:
         for i in range(len(summaries)):
            summary = summaries[i]
            query = queries[i]
            ref_doc = reference_docs[query]
            text = document_texts[ref_doc]
            sentences = nltk.sent_tokenize(text)
            for j,sentence in enumerate(sentences):
                new_line = "\t".join([query,ref_doc+"_"+str(j)+"_"+str(i),fix_encoding(sentence) ,fix_encoding(summary)])+"\n"
                out.write(new_line)

if __name__=="__main__":
    ref_index=sys.argv[1]
    queries, reference_docs = read_data_file("input_data/all_data_transformer_"+ref_index+".txt")
    document_texts = load_file("data/documents.trectext")
    summaries = read_summaries_file("summaries/output_transformer_"+ref_index+".txt")
    write_raw_ds(queries,summaries,"data/raw_bot_summary_"+ref_index+".txt",document_texts,reference_docs)
