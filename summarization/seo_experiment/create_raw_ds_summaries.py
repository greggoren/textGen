import sys
from summarization.seo_experiment.utils import load_file,read_trec_file
import nltk
def read_data_file(fname,ranked_list,index):
    queries = {}
    reference_docs={}
    summarized_docs = {}
    with open(fname) as file:
        for i,line in enumerate(file):
            doc = line.rstrip()
            summarized_docs[i]=doc
            epoch = doc.split("-")[1]
            if epoch=="00":
                continue
            query = doc.split("-")[2]
            ref_doc = ranked_list[epoch][query][index]
            reference_docs[qid]=ref_doc
            qid = str(int(doc.split("-")[2]))+doc.split("-")[1]
            queries[i]=qid
    return queries,reference_docs,summarized_docs



def read_summaries_file(fname):
    stats = {}
    with open(fname) as file:
        for i,line in enumerate(file):
            stats[i]=line.replace("<t>","").replace("</t>","")
    return stats


def fix_encoding(text):
    return bytes(text, 'cp1252', "ignore").decode('utf-8', 'ignore').replace("\n", " ").replace("\r", " ")

def write_raw_ds(queries, summaries, fname, document_texts, reference_docs,summarized_docs):
    with open(fname,'w') as out:
         for i in range(len(summaries)):
            summarized_doc = summarized_docs[i]
            summary = summaries[i]
            query = queries[i]
            ref_doc = reference_docs[query]
            if int(summarized_doc.split("-")[1])==0:
                continue
            if int(ref_doc.split("-")[1])<7:
                continue
            if int(summarized_doc.split("-")[1])>=int(ref_doc.split("-")[1]):
                continue
            text = document_texts[ref_doc]
            sentences = nltk.sent_tokenize(text)
            for j,sentence in enumerate(sentences):
                new_line = "\t".join([query,ref_doc+"_"+str(j)+"_"+summarized_doc,fix_encoding(sentence) ,fix_encoding(summary)])+"\n"
                out.write(new_line)

if __name__=="__main__":
    ref_index=sys.argv[1]
    ranked_lists = read_trec_file("trecs/trec_file_original_sorted.txt")
    queries, reference_docs,summarized_docs = read_data_file("docs.txt",ranked_lists,int(ref_index))
    document_texts = load_file("data/documents.trectext")
    summaries = read_summaries_file("competition_doc_summaries")
    write_raw_ds(queries,summaries,"data/raw_bot_summary_"+ref_index+".txt",document_texts,reference_docs,summarized_docs)
