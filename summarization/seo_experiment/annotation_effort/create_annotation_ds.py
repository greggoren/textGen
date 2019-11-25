from summarization.seo_experiment.utils import read_trec_file,load_file
import nltk
from copy import deepcopy
from random import random,seed
import csv
def read_summaries(summary_data_file,summary_file):
    stats = {}
    summary_access = open(summary_file)
    summary_data_access = open(summary_data_file)
    summaries = summary_access.readlines()
    data_points = summary_data_access.readlines()
    for i,summary in enumerate(summaries):
        data = data_points[i]
        epoch = data.split("\t")[0]
        query = data.split("\t")[1]
        winner = data.split("\t")[2].rstrip()
        if query not in stats:
            stats[query]={}
        stats[query][epoch]={}
        stats[query][epoch][winner]=bytes(summary.replace("<t>","").replace("</t>","").replace(", .",".").replace(". .","."),"utf-8").decode('utf-8', 'ignore')
    summary_data_access.close()
    summary_access.close()
    return stats


def create_documet_identification_ds(summaries, ranked_lists, texts):

    seed(9001)
    rows = {}
    j=0
    for epoch in ranked_lists:
        if int(epoch)<6 or int(epoch)==8:
            continue
        keys = ["document_1","document_2"]
        for query in ranked_lists[epoch]:
            for rank in [1,4]:
                ref_doc = ranked_lists[epoch][query][rank]
                original_sentences = nltk.sent_tokenize(bytes(texts[ref_doc], 'utf-8').decode('utf-8', 'ignore'))
                for s_epoch in summaries[query]:
                    if s_epoch>=epoch:
                        continue
                    winner = list(summaries[query][s_epoch].keys())[0]
                    summary = summaries[query][s_epoch][winner]

                    for i,sentence in enumerate(original_sentences):
                        row = {}
                        copied_sentences = deepcopy(original_sentences)
                        copied_sentences_for_orig = deepcopy(original_sentences)
                        copied_sentences_for_orig[i] = "<b>"+sentence+"</b>"
                        copied_sentences[i]="<b> "+summary+ "</b>"
                        new_text="\n".join([bytes(s,"utf-8").decode("utf-8","ignore").replace("\n","") for s in copied_sentences])
                        old_text = "\n".join([bytes(s,"utf-8").decode("utf-8","ignore").replace("\n","") for s in copied_sentences_for_orig])
                        row["query"]=query
                        row["ref_doc"] = ref_doc
                        index = 0 if random()<0.5 else 1
                        row[keys[index]] = old_text
                        row[keys[1-index]] = new_text
                        row["golden_truth"] = keys[index]
                        row["original_rank"] = rank
                        row["replace_index"]=i
                        row["summary"]=summary
                        row["winner"]=winner
                        rows[j]=row
                        j+=1
    return rows



def create_sentence_identification_ds(summaries, ranked_lists, texts):

    seed(9001)
    rows = {}
    j=0
    for epoch in ranked_lists:
        if int(epoch)<6 or int(epoch)==8:
            continue
        for query in ranked_lists[epoch]:
            for rank in [1,4]:
                ref_doc = ranked_lists[epoch][query][rank]
                original_sentences = nltk.sent_tokenize(bytes(texts[ref_doc], 'utf-8').decode('utf-8', 'ignore'))
                for s_epoch in summaries[query]:
                    if s_epoch>=epoch:
                        continue
                    winner = list(summaries[query][s_epoch].keys())[0]
                    summary = summaries[query][s_epoch][winner]

                    for i,sentence in enumerate(original_sentences):
                        row = {}
                        copied_sentences = deepcopy(original_sentences)
                        copied_sentences[i]=summary
                        new_text="\n\n".join([str(k+1)+") "+bytes(s,"utf-8").decode("utf-8","ignore").replace("\n","") +" <br><br> "for k,s in enumerate(copied_sentences)])
                        row["query"]=query
                        row["ref_doc"] = ref_doc
                        row["text"] = new_text
                        row["golden_truth"] = i+1
                        row["original_rank"] = rank
                        row["replace_index"]=i
                        row["summary"]=summary
                        row["winner"]=winner
                        rows[j]=row
                        j+=1
    return rows



if __name__=="__main__":
    summaries = read_summaries("summarization_data.txt","top_docs_summaries.txt")
    trectext_file = "../data/documents.trectext"
    trec_file = "../trecs/trec_file_original_sorted.txt"
    ranked_lists = read_trec_file(trec_file)
    texts = load_file(trectext_file)
    rows = create_documet_identification_ds(summaries, ranked_lists, texts)
    with open("summaries_doc_identification_ds.csv","w",encoding="utf-8",newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for i in range(len(rows)):
            writer.writerow(rows[i])

    rows = create_sentence_identification_ds(summaries, ranked_lists, texts)
    with open("summaries_sentence_identification_ds.csv", "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for i in range(len(rows)):
            writer.writerow(rows[i])
