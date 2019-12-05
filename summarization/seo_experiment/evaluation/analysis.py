import numpy as np
import matplotlib.pyplot as plt
from summarization.seo_experiment.utils import read_trec_file
import csv

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


def read_trec_scores(trec_file):
    stats = {}
    with open(trec_file) as file:
        for line in file:
            doc = line.split()[2]
            epoch = doc.split("-")[1]
            if int(epoch)<7:
                continue
            query = doc.split("-")[2]
            score = float(line.split()[4])
            if epoch not in stats:
                stats[epoch]={}
            if query not in stats[epoch]:
                stats[epoch][doc]=score
    return stats

def compare_lists(original_lists,updated_lists,reference_index):
    stats ={}
    for epoch in original_lists:
        stats[epoch]={}
        for query in original_lists[epoch]:
            original_list = original_lists[epoch][query]
            updated_list = updated_lists[epoch][query]
            reference_doc = original_list[reference_index]
            original_index = original_list.index(reference_doc)
            new_index = updated_list.index(reference_doc)
            increase = abs(new_index-original_index) if new_index<original_index else 0
            stats[epoch][query]=increase
    return stats


def normalize_histograms_for_plot(histograms,max_increase):
    for epoch in histograms:
        for i in range(max_increase+1):
            if i not in histograms[epoch]:
                histograms[epoch][i]=0
    return histograms

def histogram(rank_increase_stats):
    hisograms={}
    for epoch in rank_increase_stats:
        hisograms[epoch]={}
        for query in rank_increase_stats[epoch]:
            increase = rank_increase_stats[epoch][query]
            if increase not in hisograms[epoch]:
                hisograms[epoch][increase]=0
            hisograms[epoch][increase]+=1
    return normalize_histograms_for_plot(hisograms,4)

def get_average_increase(rank_increase_stats):
    average_stats = {}
    for epoch in rank_increase_stats:
        average_stats[epoch] = np.mean([rank_increase_stats[epoch][q] for q in rank_increase_stats[epoch]])
    return [average_stats[e] for e in sorted(list(average_stats.keys()))]

# def plot_metric(y,x,fname,y_label,x_label,plot=True):
#
#     params = {'legend.fontsize': 'x-large',
#               'figure.figsize': (13, 8),
#               'axes.labelsize': 'x-large',
#               'axes.titlesize': 'x-large',
#               'xtick.labelsize': 'small',
#               'ytick.labelsize': 'x-large',
#               'font.family': 'serif'}
#
#
#     plt.rcParams.update(params)
#     plt.figure()
#     if plot:
#         plt.plot(x, y, color='r', linewidth=5,markersize=5, mew=1)
#         plt.xticks(x, fontsize=25)
#     else:
#         plt.bar(x=[i for i in range(len(y))],height=[int(i) for i in y],color='b')
#         plt.xticks([i for i in range(len(y))], fontsize=25)
#     # plt.xticks(x,fontsize=15)
#     plt.yticks(fontsize=25)
#     plt.ylabel(y_label, fontsize=30)
#     plt.xlabel(x_label, fontsize=30)
#     plt.legend(loc="best")
#     plt.savefig(fname+".png")
#     plt.clf()
#
def read_trec_file(trec_file):
    stats = {}
    with open(trec_file) as file:
        for line in file:
            doc = line.split()[2]
            epoch = doc.split("-")[1]
            if int(epoch)<7:
                continue
            query = doc.split("-")[2]
            if epoch not in stats:
                stats[epoch]={}
            if query not in stats[epoch]:
                stats[epoch][query]=[]
            stats[epoch][query].append(doc)
    return stats


def read_annotations(fname):
    stats = {}
    with open(fname) as file:
        reader =csv.DictReader(file)
        for row in reader:
            query = row["Input.query_id"]
            initial_rank = row["Input.initial_rank"]
            if query not in stats:
                stats[query]={}
            if initial_rank not in stats[query]:
                stats[query][initial_rank]=0
            annotation = row["Answer.this_document_is"]
            if stats[query][initial_rank]>=5:
                continue
            if annotation.lower()=="valid":
                stats[query][initial_rank]+=1
    return stats

def analyze_annotations(stats):
    final_stats={}
    for qid in stats:
        r=qid[-1]
        if r not in final_stats:
            final_stats[r]={}
        for initial_rank in stats[qid]:
            if initial_rank not in final_stats[r]:
                final_stats[r][initial_rank]=[]
            tag = 0
            if stats[qid][initial_rank]>=3:
                tag = 1
            final_stats[r][initial_rank].append(tag)
    for r in final_stats:
        for initial_rank in final_stats[r]:
            final_stats[r][initial_rank]=np.mean(final_stats[r][initial_rank])
    return final_stats

def read_waterloo(fname):
    stats = {}
    with open(fname) as f:
        for line in f:
            doc = line.split()[0]

            waterloo = int(line.split()[1].rstrip())
            stats[doc]=waterloo
    return stats

def analyze_waterloo(ranked_list,index,waterloo):
    stats={}
    for epoch in ranked_list:
        if int(epoch)<7:
            continue
        stats[epoch]=[]
        for qid in ranked_list[epoch]:
            ref_index = ranked_list[epoch][qid][index]
            waterloo_score = waterloo[ref_index]
            if waterloo_score>=60:
                stats[epoch].append(1)
            else:
                stats[epoch].append(0)
    for epoch in stats:
        stats[epoch] = np.mean(stats[epoch])
    return stats

if __name__=="__main__":
    stats = read_annotations("../data/summarization_quality_annotations.csv")
    final_annotation_stats = analyze_annotations(stats)
    waterloo_scores = read_waterloo("waterloo_scores_file.txt")
    for i in [1,2,3,4]:

        original_trec="trecs_comp/trec_file_original_sorted.txt"
        updated_trec="trecs_comp/trec_file_post_"+str(i)+"_sorted.txt"
        bot_summary_trec_ext="trecs_comp/trec_file_bot_summary_1_post_"+str(i)+"_sorted.txt"
        bot_trec="trecs_comp/trec_file_bot_regular_post_"+str(i)+"_sorted.txt"
        original_lists = read_trec_file(original_trec)
        updated_lists = read_trec_file(updated_trec)
        bot_lists = read_trec_file(bot_trec)
        bot_summary_ext_lists = read_trec_file(bot_summary_trec_ext)
        rank_increase_stats = compare_lists(original_lists,updated_lists,i)
        bot_summary_ext_increase_stats = compare_lists(original_lists,bot_summary_ext_lists,i)
        bot_increase_stats = compare_lists(original_lists,bot_lists,i)
        averages = get_average_increase(rank_increase_stats)
        bot_summary_ext_averages = get_average_increase(bot_summary_ext_increase_stats)
        bot_averages=get_average_increase(bot_increase_stats)
        ys=[averages,bot_summary_ext_averages,bot_averages]
        legends=["Summarization","Bot+Summary","Bot"]
        colors=["b","r","k"]
        plot_metric(ys,[7,8],"plt/average_increase_"+str(i),"Rank Increase","Epochs",legends,colors)

    for i in ["1","4"]:
        all_original_quality = analyze_waterloo(original_lists,int(i),waterloo_scores)
        original_quality = [all_original_quality[e] for e in ["07","08"]]
        quality = [final_annotation_stats[r][i] for r in ["7","8"]]
        ys = [original_quality,quality]
        legends = ["Original","Summarization"]
        colors = ["b", "r"]
        plot_metric(ys,[7,8],"plt/average_quality_"+str(i),"Quality Ratio","Epochs",legends,colors)

