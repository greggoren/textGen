import matplotlib.pyplot as plt
import os

def read_file(fname):
    results = {}
    steps = []
    with open(fname) as file:
        for line in file:
            step = int(line.split("/")[-1].split("\t")[0].split("_")[1].split(".")[0])
            steps.append(step)
            results[step] = float(line.split("\t")[1].rstrip())
    y_axis = [i[1] for i in sorted(results.items(),key=lambda item:item[0])]
    return y_axis,sorted(steps)

def get_ylabel(fname):
    if fname.__contains__("BLEU"):
        return "BLEU"
    elif fname.__contains__("COVERAGE"):
        return "Query Coverage"
    elif fname.__contains__("ACCURACY"):
        return "Accuracy"
    elif fname.__contains__("SIMILARITY"):
        return "Similarity"

def generate_graphs_results_dir(results_dir):
    for file in os.listdir(results_dir):
        fname = results_dir+file
        if not os.path.isfile(fname):
            continue
        y,x = read_file(fname)
        plot_metric(y,x,"_".join(file.split("_")[:-1]),get_ylabel(file),"Steps",None,None,13)


def plot_metric(y,x,fname,y_label,x_label,legends=None,colors=None,jumps=15):
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

    # plt.axis([1, len(x), 1.1 * np.min(x), 2 * np.max(x)])
    plt.xticks(x[::jumps],fontsize=15)

    plt.yticks(fontsize=25)
    plt.ylabel(y_label, fontsize=30)
    plt.xlabel(x_label, fontsize=30)
    plt.legend(loc="best")
    plt.savefig(fname+".png")
    plt.clf()

generate_graphs_results_dir("results/")