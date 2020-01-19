import numpy as np
import matplotlib.pyplot as plt
from summarization.seo_experiment.utils import read_trec_file
import csv
import os
from scipy.stats import ttest_rel


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

def update_doc(doc):
    return "-".join([doc.split("-")[0], str(int(doc.split("-")[1]) + 1).zfill(2), doc.split("-")[2], doc.split("-")[3]])

def compare_lists_dynamic(original_lists,updated_lists,reference_index):
    stats ={}
    sig_stats={"human":{},"bot":{}}
    success_bot = {}
    success_human = {}
    success_bot_over_human = {}
    average_rank_bot={}
    ties_human={}
    ties_bot={}
    ties_bot_over_human={}
    average_rank_human={}
    for epoch in original_lists:
        if int(epoch)==8:
            continue
        success_bot[str(int(epoch)+1)]=[]
        for key in sig_stats:
            sig_stats[key][str(int(epoch)+1)]={}
        success_human[str(int(epoch)+1)]=[]
        ties_human[str(int(epoch)+1)]=[]
        ties_bot[str(int(epoch)+1)]=[]
        ties_bot_over_human[str(int(epoch)+1)]=[]
        success_bot_over_human[str(int(epoch)+1)]=[]
        average_rank_bot[str(int(epoch)+1)]=[]
        average_rank_human[str(int(epoch)+1)]=[]
        stats[str(int(epoch)+1)]={}
        for query in original_lists[epoch]:
            fixed_query = query[:-1]+str(int(query[-1])+1)
            original_list = original_lists[str(int(epoch)+1)][fixed_query]
            updated_list = updated_lists[str(int(epoch)+1)][fixed_query]
            reference_doc = original_lists[epoch][query][reference_index]
            original_index = original_list.index(update_doc(reference_doc))
            new_index = updated_list.index(reference_doc)
            if new_index<reference_index:
                success_bot[str(int(epoch)+1)].append(1)
            else:
                success_bot[str(int(epoch) + 1)].append(0)
            if original_index<reference_index:
                success_human[str(int(epoch) + 1)].append(1)
            else:
                success_human[str(int(epoch) + 1)].append(0)
            if original_index==reference_index:
                ties_human[str(int(epoch) + 1)].append(1)
            else:
                ties_human[str(int(epoch) + 1)].append(0)
            if new_index<original_index:
                success_bot_over_human[str(int(epoch) + 1)].append(1)
            else:
                success_bot_over_human[str(int(epoch) + 1)].append(0)
            if new_index==original_index:
                ties_bot_over_human[str(int(epoch) + 1)].append(1)
            else:
                ties_bot_over_human[str(int(epoch) + 1)].append(0)
            average_rank_bot[str(int(epoch) + 1)].append(new_index+1)
            sig_stats["human"][str(int(epoch)+1)][query]=original_index+1
            sig_stats["bot"][str(int(epoch)+1)][query]=new_index+1
            average_rank_human[str(int(epoch) + 1)].append(original_index+1)
            increase = abs(new_index-original_index) if new_index<original_index else 0
            stats[str(int(epoch)+1)][fixed_query]=increase
    for e in success_bot:
        success_bot[e]=np.mean(success_bot[e])
        success_human[e]=np.mean(success_human[e])
        success_bot_over_human[e]=np.mean(success_bot_over_human[e])
        ties_bot[e]=np.mean(ties_bot[e])
        ties_human[e]=np.mean(ties_human[e])
        ties_bot_over_human[e]=np.mean(ties_bot_over_human[e])
        if e in average_rank_bot:
            average_rank_bot[e]=np.mean(average_rank_bot[e])
            average_rank_human[e] = np.mean(average_rank_human[e])
    return stats,average_rank_bot,average_rank_human,success_bot,success_human,success_bot_over_human,ties_bot,ties_human,ties_bot_over_human,sig_stats




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
    return average_stats
    # return [average_stats[e] for e in sorted(list(average_stats.keys()))]

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

def create_results_table(out_fname,results_dict,sig_stats=None):
    if not os.path.exists(os.path.dirname(out_fname)):
        os.makedirs(os.path.dirname(out_fname))
    with open(out_fname,'w') as out:
        cols = "{|"+"c|"*(len(list(results_dict[list(results_dict.keys())[0]].keys()))+1)+"}"
        out.write("\\begin{tabular}"+cols+"\n")
        out.write("\\hline\n")
        header_suffix = " & ".join(['Round'+str(int(r)) for r in sorted(list(results_dict[list(results_dict.keys())[0]].keys()))])+" \\\\ \n"
        out.write("Method & "+header_suffix)
        out.write("\\hline\n")

        for key in sorted(results_dict.keys()):
            line = key
            flag =True
            if key.lower().__contains__("human") or key.lower().__contains__("untouched"):
                flag=False
            for r in sorted(list(results_dict[key].keys())):
                sig_sign = ""
                if sig_stats is not None and flag:
                    if sig_stats[r][0]<=0.05:
                        sig_sign="^*"
                line+=" & $"+str(round(results_dict[key][r],3))+sig_sign+"$"
            line+=" \\\\ \n"
            out.write(line)
            out.write("\\hline\n")
        out.write("\\end{tabular}\n")




def create_results_table_aggregated(out_fname,results_dict,header_suffix,level=None):
    if not os.path.exists(os.path.dirname(out_fname)):
        os.makedirs(os.path.dirname(out_fname))
    with open(out_fname,'w') as out:
        out.write("\\begin{tabular}{|c|c|}\n")
        out.write("\\hline\n")
        out.write("Method & "+header_suffix+"\\\\ \n")
        out.write("\\hline\n")

        for key in sorted(results_dict.keys()):
            line = key
            flag =True
            if key.lower().__contains__("human") or key.lower().__contains__("untouched"):
                flag=False

            sig_sign = ""
            if level is not None and flag:
                if level<=0.05:
                    sig_sign="^*"
            line+=" & $"+str(round(results_dict[key],3))+sig_sign+"$"
            line+=" \\\\ \n"
            out.write(line)
            out.write("\\hline\n")
        out.write("\\end{tabular}\n")


def read_trec_file(trec_file):
    stats = {}
    with open(trec_file) as file:
        for line in file:
            doc = line.split()[2]
            epoch = doc.split("-")[1]
            query = doc.split("-")[2]
            if epoch not in stats:
                stats[epoch]={}
            if query not in stats[epoch]:
                stats[epoch][query]=[]
            stats[epoch][query].append(doc)
    return stats


def read_dynamic_trec_file(trec_file):
    stats = {}
    with open(trec_file) as file:
        for line in file:
            doc = line.split()[2]
            query = line.split()[0]
            epoch = query[-1]
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
    sig_stats={}
    final_stats={}
    for qid in stats:
        r=qid[-1].zfill(2)
        if r not in final_stats:
            final_stats[r]={}
            sig_stats[r]={}
        for initial_rank in stats[qid]:
            if initial_rank not in final_stats[r]:
                final_stats[r][initial_rank]=[]
                sig_stats[r][initial_rank]={}
            tag = 0
            if stats[qid][initial_rank]>=3:
                tag = 1
            final_stats[r][initial_rank].append(tag)
            sig_stats[r][initial_rank][qid]=tag
    for r in final_stats:
        for initial_rank in final_stats[r]:
            final_stats[r][initial_rank]=np.mean(final_stats[r][initial_rank])
    return final_stats,sig_stats

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
    sig_stats={}
    for epoch in ranked_list:
        # if int(epoch)<7:
        #     continue
        stats[epoch]=[]
        sig_stats[epoch]={}
        for qid in ranked_list[epoch]:
            ref_index = ranked_list[epoch][qid][index]
            waterloo_score = waterloo[ref_index]
            tag=0
            if waterloo_score>=60:
                tag=1
            stats[epoch].append(tag)
            sig_stats[epoch][qid]=tag
    for epoch in stats:
        stats[epoch] = np.mean(stats[epoch])
    return stats,sig_stats

def analyze_waterloo_dynamic(ranked_list,index,waterloo):
    stats={}
    sig_stats={}
    for epoch in ranked_list:
        if epoch=="08":
            continue
        sig_stats[str(int(epoch)+1)]={}
        stats[str(int(epoch)+1)]=[]
        for qid in ranked_list[epoch]:
            ref_index = ranked_list[epoch][qid][index]
            waterloo_score = waterloo[update_doc(ref_index)]
            tag = 0
            if waterloo_score>=60:
                tag=1
            stats[str(int(epoch) + 1)].append(tag)
            sig_stats[str(int(epoch) + 1)][qid]=tag
    for epoch in stats:
        stats[epoch] = np.mean(stats[epoch])
    return stats,sig_stats

def parse_file(fname):
    stats = {}
    with open(fname) as file:
        for line in file:
            epoch = line.split()[0]
            query = line.split()[1]
            score = float(line.split()[2].rstrip())
            if epoch not in stats:
                stats[epoch]={}
            stats[epoch][query]=score
        return stats

def get_vectors_for_local_test(stats):
    converted_stats={}
    epochs = list(stats.keys())
    epochs = sorted(epochs)
    for epoch in epochs:
        sorted_queries = sorted(list(stats[epoch].keys()), key=lambda x: int(x.split("_")[0]))
        converted_stats[epoch]=[stats[epoch][q] for q in sorted_queries]
    return converted_stats,epochs


def run_local_significance(test_stats,control_stats,control_epochs,test_epochs):
    stats = {}
    for index,test_epoch in enumerate(test_epochs):
        control_epoch = control_epochs[index]
        test_vector = test_stats[test_epoch]
        control_vector = control_stats[control_epoch]
        diff,level = ttest_rel(test_vector,control_vector)
        if np.isnan(level):
            diff=0
            level=1
        stats[test_epoch]=(diff,level)
    return stats

def preprocess_static_sig_stats(stats,initial_rank):
    result={}
    for epoch in stats:
        result[epoch]={}
        for qid in stats[epoch][initial_rank]:
            result[epoch][qid]=stats[epoch][initial_rank][qid]
    return result


def transform_stats(stats,epochs_avoid):
    result = {}
    filled_epochs_avoid = [e.zfill(2) for e in epochs_avoid]
    for epoch in stats:
        if epoch in epochs_avoid or epoch in filled_epochs_avoid:
            continue
        for query in stats[epoch]:
            fixed_query = query[:-2].zfill(3)
            if fixed_query not in result:
                result[fixed_query]=[]
            result[fixed_query].append(stats[epoch][query])
    for qid in result:
        result[qid]=np.mean(result[qid])
    return result

def average_results(stats):
    results = []
    for qid in stats:
        results.append(stats[qid])
    return np.mean(results)
def analyze_sig(stats,avoid_epochs):
    sig = []
    for epoch in stats:
        if epoch in avoid_epochs or epoch in [e.zfill(2) for e in avoid_epochs]:
            continue
        for query in stats[epoch]:
            sig.append(stats[epoch][query])
    return np.mean(sig)

def run_sig_test(stats1,stats2):
    vector1 = [stats1[q] for q in sorted(list(stats1.keys()))]
    vector2 = [stats2[q] for q in sorted(list(stats2.keys()))]
    diff,level = ttest_rel(vector1,vector2)
    return level


if __name__=="__main__":
    # stats = read_annotations("quality_annotations/quality_2.csv")
    stats = read_annotations("old_bot_quality/quality_old.csv")
    # stats = read_annotations("quality_0_bot/quality_0_bot.csv")
    epochs_avoid = ["7"]
    final_annotation_stats,static_sig_stats = analyze_annotations(stats)
    waterloo_scores = read_waterloo("waterloo_scores_file.txt")
    for i in [1,2,3,4]:
        original_trec="trecs_comp/trec_file_original_sorted.txt"
        original_lists = read_trec_file(original_trec)
        # """STATIC ANALYSIS"""
        # updated_trec="trecs_comp/trec_file_post_"+str(i)+"_sorted.txt"
        # bot_summary_trec_ext="trecs_comp/trec_file_bot_summary_1_post_"+str(i)+"_sorted.txt"
        # bot_trec="trecs_comp/trec_file_bot_regular_post_"+str(i)+"_sorted.txt"
        # updated_lists = read_trec_file(updated_trec)
        # bot_lists = read_trec_file(bot_trec)
        # bot_summary_ext_lists = read_trec_file(bot_summary_trec_ext)
        # rank_increase_stats = compare_lists(original_lists,updated_lists,i)
        # bot_summary_ext_increase_stats = compare_lists(original_lists,bot_summary_ext_lists,i)
        # bot_increase_stats = compare_lists(original_lists,bot_lists,i)
        # averages = get_average_increase(rank_increase_stats)
        # bot_summary_ext_averages = get_average_increase(bot_summary_ext_increase_stats)
        # bot_averages=get_average_increase(bot_increase_stats)
        # ys={"Summarization":averages,"Summary+Bot":bot_summary_ext_averages,"Bot":bot_averages}
        # create_results_table("tables/Static_analysis_experiment_"+str(i)+".tex",ys)

        """DYNAMIC ANALYSIS"""
        # summarization_trec = "dynamic_trecs/trec_file_summarization_post_"+str(i)+"_sorted.txt"
        # bot_trec = "dynamic_trecs/trec_file_bot_regular_post_"+str(i)+"_sorted.txt"
        bot_trec = "old_model_dynamic_trecs/trec_file_bot_regular_post_"+str(i)+"_sorted.txt"
        # bot_trec = "bot_0_trecs/trec_file_bot_regular_0_post_"+str(i)+"_sorted.txt"
        # bot_summary_trec = "dynamic_trecs/trec_file_bot_summary_post_"+str(i)+"_sorted.txt"
        original_ranks = read_dynamic_trec_file(original_trec)
        # summarization_ranks = read_dynamic_trec_file(summarization_trec)
        bot_ranks = read_dynamic_trec_file(bot_trec)
        # bot_summary_ranks = read_dynamic_trec_file(bot_summary_trec)
        # summarization_increase,average_rank_summarization,average_rank_human,success_summarization,success_human,success_summarization_over_human = compare_lists_dynamic(original_ranks,summarization_ranks,i)
        bot_increase,average_rank_bot,average_rank_human,success_bot,success_human,success_bot_over_human,ties_bot,ties_human,ties_bot_over_human,sig_stats = compare_lists_dynamic(original_ranks,bot_ranks,i)
        # bot_summary_increase,average_rank_bot_summary,average_rank_human,success_bot_summary,success_human,success_bot_summary_over_human = compare_lists_dynamic(original_ranks,bot_summary_ranks,i)
        # average_summary_increase = get_average_increase(summarization_increase)
        average_bot_increase = get_average_increase(bot_increase)
        # average_bot_summary_increase = get_average_increase(bot_summary_increase)
        ys = {"Bot": average_bot_increase}
        create_results_table("tables/Dynamic_analysis_experiment_old" + str(i) + ".tex", ys)
        ys = {"Bot Success":success_bot,"Human success":success_human}
        create_results_table("tables/dynamic_success_rate_old"+str(i)+".tex",ys)
        ys = {"Bot over human":success_bot_over_human,"Ties":ties_bot_over_human}
        create_results_table("tables/dynamic_success_over_human_rate_old"+str(i)+".tex",ys)
        ys = {"Average Bot rank":average_rank_bot,"Average rank human":average_rank_human}

        human_vectors,control_epochs = get_vectors_for_local_test(sig_stats["human"])
        bot_vectors,test_epochs = get_vectors_for_local_test(sig_stats["bot"])

        sig_results = run_local_significance(bot_vectors,human_vectors,control_epochs,test_epochs)
        create_results_table("tables/dynamic_average_rank_old"+str(i)+".tex",ys,sig_results)
        aggregated_results_human = transform_stats(sig_stats["human"],epochs_avoid)
        aggregated_results_bot = transform_stats(sig_stats["bot"],epochs_avoid)
        ys = {"Average rank bot":average_results(aggregated_results_bot),"Average rank human bot":average_results(aggregated_results_human)}
        create_results_table_aggregated("tables/aggregated_average_rank_old"+str(i)+".tex",ys,"Average rank",run_sig_test(aggregated_results_bot,aggregated_results_human))

    epochs_avoid = ["6"]
    for i in ["1","2","3","4"]:
        # all_original_quality,bot_sig_stats = analyze_waterloo(original_lists,int(i),waterloo_scores)
        dynamic_original_quality,dynamic_sig_stats = analyze_waterloo_dynamic(original_lists,int(i),waterloo_scores)
        # original_quality = {e:all_original_quality[e] for e in  [str(i).zfill(2) for i in range(1,8)]}
        quality = {r:final_annotation_stats[r][i] for r in [str(i).zfill(2) for i in range(1,8)]}
        dynamic_vectors,dynamic_epochs = get_vectors_for_local_test(dynamic_sig_stats)
        static_sig_stats.pop("08")
        bot_vectors,bot_epochs = get_vectors_for_local_test(static_sig_stats)
        static_vectors,static_epochs = get_vectors_for_local_test(preprocess_static_sig_stats(static_sig_stats,i))
        sig_results = run_local_significance(bot_vectors,dynamic_vectors,dynamic_epochs,bot_epochs)
        # ys = {"Human Bot":dynamic_original_quality,"Bot":quality,"Untouched document":original_quality}
        # create_results_table("tables/dynamic_quality_old" + i + ".tex", ys,sig_results)

        aggregated_bot_quality = transform_stats(static_sig_stats,epochs_avoid)
        aggregated_human_quality = transform_stats(dynamic_sig_stats,epochs_avoid)
        # ys = {"Human bot":average_results(aggregated_human_quality),"Bot":average_results(aggregated_bot_quality)}
        ys = {"Human bot":analyze_sig(dynamic_sig_stats,epochs_avoid),"Bot":analyze_sig(static_sig_stats,epochs_avoid)}
        create_results_table_aggregated("tables/aggregated_quality_"+i+".tex",ys,"Quality ratio",run_sig_test(aggregated_bot_quality,aggregated_human_quality))