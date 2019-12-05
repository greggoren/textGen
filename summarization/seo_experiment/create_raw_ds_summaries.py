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
            query = doc.split("-")[2]
            if epoch not in ranked_lists:
                reference_docs["0"] = doc
                queries[i] = "0"
                continue
            ref_doc = ranked_list[epoch][query][index]
            qid = str(int(doc.split("-")[2]))+doc.split("-")[1]
            reference_docs[qid]=ref_doc
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

def match_summaries(ref_doc,summarized_docs):
    query = ref_doc.split("-")[2]
    summaries_indexes = []
    if int(ref_doc.split("-")[1]) == 0:
        return []
    if int(ref_doc.split("-")[1]) < 7:
        return []
    for i in summarized_docs:
        summarized_doc = summarized_docs[i]
        if query != summarized_doc.split("-")[2]:
            continue
        if int(summarized_doc.split("-")[1]) == 0:
            continue
        if int(summarized_doc.split("-")[1]) >= int(ref_doc.split("-")[1]):
            continue
        summaries_indexes.append(i)
    return summaries_indexes



def write_raw_ds(queries, summaries, fname, document_texts, reference_docs,summarized_docs):
    with open(fname,'w') as out:
         for qid in reference_docs:
            ref_doc = reference_docs[qid]
            summary_indexes =match_summaries(ref_doc,summarized_docs)
            text = document_texts[ref_doc]
            sentences = nltk.sent_tokenize(text)
            for index in summary_indexes:
                summarized_doc = summarized_docs[index]
                summary = summaries[index]
                query = queries[index]
                for j,sentence in enumerate(sentences):
                        new_line = "\t".join([query,ref_doc+"_"+str(j)+"_"+summarized_doc,str(j),fix_encoding(sentence) ,fix_encoding(summary)])+"\n"
                        out.write(new_line)



if __name__=="__main__":
    ref_index=sys.argv[1]
    ranked_lists = read_trec_file("trecs/trec_file_original_sorted.txt")
    queries, reference_docs,summarized_docs = read_data_file("docs.txt",ranked_lists,int(ref_index))
    document_texts = load_file("data/documents.trectext")
    summaries = read_summaries_file("competition_doc_summaries")
    write_raw_ds(queries,summaries,"data/raw_bot_summary_"+ref_index+".txt",document_texts,reference_docs,summarized_docs)
