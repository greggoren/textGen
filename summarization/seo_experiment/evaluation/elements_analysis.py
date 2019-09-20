from summarization.seo_experiment.utils import load_file,clean_texts
from summarization.seo_experiment.borda_mechanism import query_term_freq
from summarization.seo_experiment.workingset_creator import read_queries_file
from summarization.seo_experiment.evaluation.analysis import read_trec_file
import numpy as np
import matplotlib.pyplot as plt

def doc_len(text):
    return len(text.split())


def cover(text,query):
    numerator = 0
    for q in query.split():
        if q in clean_texts(text):
            numerator+=1
    return numerator/len(query.split())



def doc_frequency_eval(lists,queries,texts):
    stats={"top":{},"self":{}}
    coverage={"top":{},"self":{}}

    docs_len = {"top":{},"self":{}}
    for epoch in lists:
        for k in stats:
            stats[k][epoch]=[]
            docs_len[k][epoch]=[]
            coverage[k][epoch]=[]
        for query in lists[epoch]:
            query_text = queries[query]
            top_docs = lists[epoch][query][:3]
            ref_doc = lists[epoch][query][-1]
            stats["top"][epoch].append(np.mean([query_term_freq("avg",clean_texts(texts[doc]),query_text) for doc in top_docs]))
            stats["self"][epoch].append(query_term_freq("avg",clean_texts(texts[ref_doc]),query_text))
            docs_len["top"][epoch].append(np.mean([doc_len(clean_texts(texts[doc])) for doc in top_docs]))
            docs_len["self"][epoch].append(doc_len(clean_texts(texts[ref_doc])))
            coverage["self"][epoch].append(cover(texts[ref_doc],query_text))
            coverage["top"][epoch].append(np.mean([cover(clean_texts(texts[doc]),query_text) for doc in top_docs]))
    for k in stats:
        for epoch in stats[k]:
            stats[k][epoch]=np.mean(stats[k][epoch])
            docs_len[k][epoch]=np.mean(docs_len[k][epoch])
            coverage[k][epoch]=np.mean(coverage[k][epoch])
    return stats,docs_len,coverage


def compare_frequecies(summary_stats_file):
    stats={"sentence":{},"summary":{},"success":{}}
    with open(summary_stats_file,encoding="utf-8") as file:
        for line in file:
            query = line.split("\t")[0]
            doc = line.split("\t")[1]
            epoch = int(doc.split("-")[1])
            for k in stats:
                if epoch not in stats[k]:
                    stats[k][epoch]=[]
            sentence = line.split("\t")[2]
            summary = line.split("\t")[3]
            sentence_qtf = query_term_freq("avg",clean_texts(sentence),query)
            stats["sentence"][epoch].append(sentence_qtf)
            summary_qtf = query_term_freq("avg",clean_texts(summary),query)
            stats["summary"][epoch].append(summary_qtf)
            stats["success"][epoch].append(1 if summary_qtf>sentence_qtf else 0)
    for k in stats:
        for epoch in stats[k]:
            stats[k][epoch] = np.mean(stats[k][epoch])
    return stats


def plot_metric(y,x,fname,y_label,x_label,legends=None,colors=None):
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (13, 8),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'small',
              'ytick.labelsize': 'x-large',
              'font.family': 'serif'}


    plt.rcParams.update(params)
    plt.figure()
    if legends is not None:
        for j,y_m in enumerate(y):
            plt.plot(x, y_m, color=colors[j], linewidth=5,markersize=5, mew=1,label=legends[j])
    else:

        plt.plot(x, y, color='b', linewidth=5,markersize=5, mew=1)

    plt.xticks(x,fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel(y_label, fontsize=30)
    plt.xlabel(x_label, fontsize=30)
    plt.legend(loc="best")
    plt.savefig(fname+".png")
    plt.clf()


if __name__=="__main__":
    trec="../trecs/trec_file_original_sorted.txt"
    queries_file="../data/queries.xml"
    queries = read_queries_file(queries_file)
    ranked_lists = read_trec_file(trec)
    doc_texts = load_file("../data/documents.trectext")
    updated_doc_texts = load_file("../data/updated_documents.trectext")
    summary_analysis_file = "../data/summary_analysis.txt"
    stats = compare_frequecies(summary_analysis_file)
    legends = ["sentence","summary","success"]
    colors = ['b','r','k']
    ys =[[stats[k][e] for e in sorted(list(stats[k].keys()))] for k in legends]
    x = sorted(list(stats["sentence"].keys()))
    plot_metric(ys,x,"plt/qtf_comp_summaries","Avg","Epochs",legends,colors)
    stats,docs_len,coverage = doc_frequency_eval(ranked_lists,queries,doc_texts)
    updated_stats,updated_docs_len,updated_coverage = doc_frequency_eval(ranked_lists,queries,updated_doc_texts)
    stats["after"]=updated_stats["self"]
    docs_len["after"]=updated_docs_len["self"]
    coverage["after"]=updated_coverage["self"]
    legends = ["self", "top","after"]
    colors = ['b', 'r','k']
    ys = [[stats[k][e] for e in sorted(list(stats[k].keys()))] for k in legends]
    x = sorted(list(stats["self"].keys()))
    plot_metric(ys, x, "plt/qtf_comp_docs", "Avg QTF", "Epochs", legends, colors)
    ys = [[docs_len[k][e] for e in sorted(list(docs_len[k].keys()))] for k in legends]
    x = sorted(list(docs_len["self"].keys()))
    plot_metric(ys, x, "plt/len_comp_docs", "Length", "Epochs", legends, colors)
    ys = [[coverage[k][e] for e in sorted(list(coverage[k].keys()))] for k in legends]
    x = sorted(list(coverage["self"].keys()))
    plot_metric(ys, x, "plt/coverage_comp_docs", "CoverRatio", "Epochs", legends, colors)
