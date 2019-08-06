import matplotlib.pyplot as plt


def parse_loss_file(filename):
    losses = []
    with open(filename) as file:
        for line in file:
            cleaned = line.replace(",","")
            loss = float(cleaned.split("loss = ")[1].split()[0])
            losses.append(loss)
        x = [i*100 for i,a in enumerate(losses)]
        return losses,x

def parse_metrics(filename):
    bleus,losses,accs,steps = [],[],[],[]
    with open(filename) as file:
        for line in file:
            loss = float(line.split("loss = ")[1].split()[0].replace(",",""))
            losses.append(loss)
            acc_per_seq = float(line.split("accuracy_per_sequence = ")[1].split()[0].replace(",",""))
            accs.append(acc_per_seq)
            bleu = float(line.split("approx_bleu_score = ")[1].split()[0].replace(",",""))
            bleus.append(bleu)
            step = float(line.split("global_step = ")[1].split()[0].replace(",",""))
            steps.append(step)
        return bleus,losses,accs,steps


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

eval_filename = "results/eval.out"
random_eval_filename = "results/evaluation_random.txt"
loss_filename = "results/loss.out"
random_loss_filename = "results/loss_random.txt"

losses,steps = parse_loss_file(loss_filename)
rlosses,rsteps = parse_loss_file(random_loss_filename)
bleus,losses_,accs,steps_ = parse_metrics(eval_filename)
rbleus,rlosses_,raccs,rsteps_ = parse_metrics(random_eval_filename)



legends = ["Borda","Random"]
colors = ["b","r"]
items = min(len(steps),len(rsteps))
obj = [losses[:items],rlosses[:items]]

plot_metric(obj,steps[:items],"loss_train","CELoss","Steps",legends,colors,500)

items = min(len(steps_),len(rsteps_))
obj = [losses_[:items],rlosses_[:items]]
plot_metric(obj,steps_[:items],"loss_test","CELoss","Steps",legends,colors,50)

# plot_metric(bleus,steps_,"bleu_test","BLEU","Steps",None,None,50)
obj = [accs[:items],raccs[:items]]
plot_metric(obj,steps_[:items],"acc_per_seq_test","Accuracy","Steps",legends,colors,50)