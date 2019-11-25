from summarization.seo_experiment.borda_mechanism import query_term_freq,query_term_occ
from summarization.seo_experiment.utils import clean_texts
from summarization.seo_experiment.workingset_creator import read_queries_file
from summarization.seo_experiment.summarization_process import transform_query_text
import nltk
import numpy as np

queries = read_queries_file("../data/queries.xml")
queries = transform_query_text(queries)

summary_access = open("top_docs_summaries.txt")
summary_data_access = open("summarization_data.txt")
summaries = summary_access.readlines()
data_points = summary_data_access.readlines()

freqs ={"all":[],"first":[]}
for i,summary in enumerate(summaries):
    data = data_points[i]
    qid = data.split("\t")[1]
    q_text = queries[qid]
    fixed_sum = summary.replace("<t>","").replace("</t>","").replace(", .",".").replace(". .",".")
    freqs["all"].append(query_term_occ("sum",clean_texts(fixed_sum),q_text))
    first = nltk.sent_tokenize(fixed_sum)[0]
    freqs["first"].append(query_term_occ("sum",clean_texts(first),q_text))

for k in freqs:
    freqs[k] = np.mean(freqs[k])

print(freqs)

