import matplotlib.pyplot as plt

def plot_metric(y,fname,y_label):
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (13, 8),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large',
              'font.family': 'serif'}
    plt.rcParams.update(params)
    plt.figure()
    x = [i +1 for i in range(y)]
    plt.plot(x, y, color='b', linewidth=5,markersize=10, mew=1)
    plt.xticks(x, fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel(y_label, fontsize=30)
    plt.xlabel("Epoch", fontsize=30)
    plt.savefig(fname)
    plt.clf()