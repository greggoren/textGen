import numpy as np
import matplotlib.pyplot as plt
from summarization.seo_experiment.utils import read_trec_file


def read_trec_scores(trec_file):
    stats = {}
    with open(trec_file) as file:
        for line in file:
            doc = line.split()[2]
            epoch = doc.split("-")[1]
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

def plot_metric(y,x,fname,y_label,x_label,plot=True):

    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (13, 8),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'small',
              'ytick.labelsize': 'x-large',
              'font.family': 'serif'}


    plt.rcParams.update(params)
    plt.figure()
    if plot:
        plt.plot(x, y, color='r', linewidth=5,markersize=5, mew=1)
        plt.xticks(x, fontsize=25)
    else:
        plt.bar(x=[i for i in range(len(y))],height=[int(i) for i in y],color='b')
        plt.xticks([i for i in range(len(y))], fontsize=25)
    # plt.xticks(x,fontsize=15)
    plt.yticks(fontsize=25)
    plt.ylabel(y_label, fontsize=30)
    plt.xlabel(x_label, fontsize=30)
    plt.legend(loc="best")
    plt.savefig(fname+".png")
    plt.clf()



if __name__=="__main__":
    original_trec="../trecs/trec_file_original_sorted.txt"
    updated_trec="../trecs/trec_file_post_sorted.txt"
    original_lists = read_trec_file(original_trec)
    updated_lists = read_trec_file(updated_trec)
    rank_increase_stats = compare_lists(original_lists,updated_lists,-1)
    histograms = histogram(rank_increase_stats)
    averages = get_average_increase(rank_increase_stats)
    plot_metric(averages,[i+1 for i in range(len(averages))],"plt/average_increase","Rank Increase","Epochs")
    for epoch in histograms:
        h = histograms[epoch]
        plot_metric([h[i] for i in range(5)],[],"plt/histogram_"+str(epoch),"#","Rank Increase",False)