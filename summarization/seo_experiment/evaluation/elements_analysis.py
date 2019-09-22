from summarization.seo_experiment.utils import load_file,clean_texts
from summarization.seo_experiment.borda_mechanism import query_term_freq,query_term_occ
from summarization.seo_experiment.workingset_creator import read_queries_file
from summarization.seo_experiment.evaluation.analysis import read_trec_scores
from summarization.seo_experiment.utils import read_trec_file

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


def compare_scores(scores, lists):
    stats = {"reference":{},"top":{},"next":{},"winner":{}}

    for epoch in lists:
        for k in stats:
            stats[k][epoch]=[]
        for query in lists[epoch]:
            top_docs = lists[epoch][query][:3]
            ref_doc = lists[epoch][query][-1]
            next_doc = lists[epoch][query][-2]
            winner_doc = lists[epoch][query][0]
            stats["reference"][epoch].append(scores[epoch][ref_doc])
            stats["next"][epoch].append(scores[epoch][next_doc])
            stats["winner"][epoch].append(scores[epoch][winner_doc])
            stats["top"][epoch].append(np.mean([scores[epoch][doc] for doc in top_docs]))
    for k in stats:
        for epoch in stats[k]:
            stats[k][epoch]=np.mean(stats[k][epoch])
    return stats



def doc_frequency_eval(lists,queries,texts):
    stats={"top":{},"reference":{},"next":{}}
    for epoch in lists:
        for k in stats:
            stats[k][epoch]=[]
        for query in lists[epoch]:
            query_text = queries[query]
            top_docs = lists[epoch][query][:3]
            ref_doc = lists[epoch][query][-1]
            next_doc = lists[epoch][query][-2]
            stats["top"][epoch].append(np.mean([query_term_freq("avg",clean_texts(texts[doc]),query_text) for doc in top_docs]))
            stats["reference"][epoch].append(query_term_freq("avg",clean_texts(texts[ref_doc]),query_text))
            stats["next"][epoch].append(query_term_freq("avg",clean_texts(texts[next_doc]),query_text))
    for k in stats:
        for epoch in stats[k]:
            stats[k][epoch]=np.mean(stats[k][epoch])
    return stats


def compare_frequecies(summary_stats_file):
    # stats={"sentence":{},"summary":{},"success":{}}
    stats={"sentence":{},"summary":{}}
    occ_stats={"sentence":{},"summary":{}}

    with open(summary_stats_file,encoding="utf-8") as file:
        for line in file:
            query = line.split("\t")[0]
            doc = line.split("\t")[1]
            epoch = int(doc.split("-")[1])
            for k in stats:
                if epoch not in stats[k]:
                    stats[k][epoch]=[]
                    if k in occ_stats:
                        occ_stats[k][epoch]=[]
            sentence = line.split("\t")[2]
            summary = line.split("\t")[3]
            sentence_qtf = query_term_freq("avg",clean_texts(sentence),query)
            stats["sentence"][epoch].append(sentence_qtf)
            occ_stats["sentence"][epoch].append(query_term_occ("sum",clean_texts(sentence),query))
            summary_qtf = query_term_freq("avg",clean_texts(summary),query)
            stats["summary"][epoch].append(summary_qtf)
            occ_stats["summary"][epoch].append(query_term_occ("sum", clean_texts(summary), query))
    for k in stats:
        for epoch in stats[k]:
            stats[k][epoch] = np.mean(stats[k][epoch])
            if k in occ_stats:
                occ_stats[k][epoch] = np.mean(occ_stats[k][epoch])
    return stats,occ_stats

def read_features_file(features_file):
    stats={}
    with open(features_file) as file:
        for line in file:
            features = line.split()[2:-2]
            doc = line.split(" # ")[1].rstrip()
            features = np.array([float(f.split(":")[1]) for f in features])
            stats[doc]=features
    return stats

def average_feature_vec(top_docs,features):
    sum_vec = None
    for doc in top_docs:
        feature_vec = features[doc]
        if sum_vec is None:
            sum_vec = np.zeros(feature_vec.shape[0])
        sum_vec+=feature_vec
    return sum_vec/len(top_docs)

def diff_sum(v1,v2):
    diff = [abs(i-j) for i,j in zip(v1,v2)]
    return sum(diff)

def diff_avg(v1,v2):
    diff = [abs(i-j) for i,j in zip(v1,v2)]
    return np.mean(diff)

def diff_std(v1,v2):
    diff = [abs(i-j) for i,j in zip(v1,v2)]
    return np.std(diff)


def analyze_fetures_diff(feature_stats, ranked_lists):
    stats={"next":{"sum":{},"avg":{},"std":{}},"top":{"sum":{},"avg":{},"std":{}}}
    func_index = {"sum":diff_sum,"avg":diff_avg,"std":diff_std}
    for epoch in ranked_lists:
        for k in stats:
            for k2 in stats[k]:
                stats[k][k2][epoch]=[]
        for query in ranked_lists[epoch]:
            ref_doc_vector= feature_stats[ranked_lists[epoch][query][-1]]
            top_docs_vector = average_feature_vec(ranked_lists[epoch][query][:3],feature_stats)
            next_doc_vector = feature_stats[ranked_lists[epoch][query][-2]]
            for k in stats["next"]:
                stats["next"][k][epoch].append(func_index[k](ref_doc_vector,next_doc_vector))
            for k in stats["top"]:
                stats["top"][k][epoch].append(func_index[k](ref_doc_vector,top_docs_vector))
    for k in stats:
        for k2 in stats[k]:
            for epoch in stats[k][k2]:
                stats[k][k2][epoch] = np.mean(stats[k][k2][epoch])
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


def read_features_dir(features_dir,feature_list,queries):
    features= {}
    for feature in feature_list:
        features[feature]={}
        for query in queries:
            filename = features_dir+"/"+feature+"_"+query
            with open(filename) as file:
                for line in file:
                    doc = line.split()[0]
                    val = float(line.split()[1].rstrip())
                    features[feature][doc]=val
    return features

def analyze_raw_fetures(ranked_lists,features):
    stats = {f:{"reference":{},"next":{},"top":{}} for f in features.keys()}
    for epoch in ranked_lists:
        for f in stats:
            for k in stats[f]:
                stats[f][k][epoch]=[]
        for query in ranked_lists[epoch]:
            top_docs = ranked_lists[epoch][query][:3]
            ref_doc = ranked_lists[epoch][query][-1]
            next_doc = ranked_lists[epoch][query][-2]
            for feature in features:
                stats[feature]["reference"][epoch].append(features[feature][ref_doc])
                stats[feature]["next"][epoch].append(features[feature][next_doc])
                stats[feature]["top"][epoch].append(np.mean([features[feature][doc] for doc in top_docs]))
    for f in stats:
        for k in stats[f]:
            for epoch in stats[f][k]:
                stats[f][k][epoch] = np.mean(stats[f][k][epoch])
    return stats


if __name__=="__main__":
    feature_list = ["docCoverQueryNum","docCoverQueryRatio","docLen","docBM25","docLMIR.DIR","docLMIR.JM","docEnt","docStopCover","docFracStops"]
    features_dir = "Features"
    post_features_dir = "Features_post"

    trec="../trecs/trec_file_original_sorted.txt"
    post_trec="../trecs/trec_file_post_sorted.txt"
    queries_file="../data/queries.xml"
    modified_queries = read_queries_file("../data/queries_seo_exp.xml")
    features_file ="../data/features_original"
    featues=read_features_file(features_file)
    queries = read_queries_file(queries_file)

    ranked_lists = read_trec_file(trec)

    original_features = read_features_dir(features_dir, feature_list,modified_queries)
    post_features = read_features_dir(post_features_dir, feature_list,modified_queries)

    original_features_stats = analyze_raw_fetures(ranked_lists,original_features)
    post_features_stats = analyze_raw_fetures(ranked_lists,post_features)
    for f in feature_list:
        original_features_stats[f]["post"]=post_features_stats[f]["reference"]
    legends = ["reference","next", "top","post"]
    colors = ['b', 'r',"k","y"]
    for f in feature_list:
        ys = [[original_features_stats[f][k][e] for e in sorted(list(original_features_stats[f][k].keys()))] for k in legends]
        x = sorted(list(original_features_stats[f]["next"].keys()))
        plot_metric(ys, x, f.lower().replace('.',''), f.replace('.',''), "Epochs", legends, colors)



    scores = read_trec_scores(trec)
    post_scores = read_trec_scores(post_trec)
    doc_texts = load_file("../data/documents.trectext")
    updated_doc_texts = load_file("../data/updated_documents.trectext")


    stats = doc_frequency_eval(ranked_lists,queries,doc_texts)
    updated_stats = doc_frequency_eval(ranked_lists,queries,updated_doc_texts)
    stats["post"]=updated_stats["reference"]
    legends = ["reference", "next", "top", "post"]
    colors = ['b', 'r', "k", "y"]
    ys = [[stats[k][e] for e in sorted(list(stats[k].keys()))] for k in legends]
    x = sorted(list(stats["reference"].keys()))
    plot_metric(ys, x, "plt/qtf_comp_docs", "Avg QTF", "Epochs", legends, colors)

    stats = compare_scores(scores,ranked_lists)
    post_stats = compare_scores(post_scores,ranked_lists)
    stats["post next"] = post_stats["next"]
    stats["post reference"] = post_stats["reference"]
    legends = ["reference","next","post reference","post next"]
    colors = ['b', 'r','k','y']
    ys = [[stats[k][e] for e in sorted(list(stats[k].keys()))] for k in legends]
    x = sorted(list(stats["reference"].keys()))
    plot_metric(ys, x, "plt/scores_comp_docs", "AvgScore", "Epochs", legends, colors)

    # feature_stats = analyze_fetures_diff(featues, ranked_lists)
    # fname_dict = {"sum":"plt/sum_diff_features","avg":"plt/avg_diff_features","std":"plt/std_diff_features"}
    # axis_dict = {"sum":"Sum","avg":"Avg","std":"Std"}
    # legends = ["next", "top"]
    # colors = ['b', 'r']
    # for key in fname_dict:
    #     ys = [[feature_stats[k][key][e] for e in sorted(list(feature_stats[k][key].keys()))] for k in legends]
    #     x = sorted(list(feature_stats["next"][key].keys()))
    #     plot_metric(ys, x, fname_dict[key], axis_dict[key], "Epochs", legends, colors)



    # summary_analysis_file = "../data/summary_analysis.txt"
    # stats,occ_stats = compare_frequecies(summary_analysis_file)
    # legends = ["sentence","summary"]
    # colors = ['b','r']
    # ys =[[stats[k][e] for e in sorted(list(stats[k].keys()))] for k in legends]
    # x = sorted(list(stats["sentence"].keys()))
    # plot_metric(ys,x,"plt/qtf_comp_summaries","Avg","Epochs",legends,colors)
    #
    # ys = [[occ_stats[k][e] for e in sorted(list(occ_stats[k].keys()))] for k in legends]
    # x = sorted(list(occ_stats["sentence"].keys()))
    # plot_metric(ys, x, "plt/qto_comp_summaries", "Avg", "Epochs", legends, colors)