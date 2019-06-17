import matplotlib.pyplot as plt
import os
import torch
import pickle

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


def save_loss_history(obj,epoch,lr,batch_size):
    dir_name = "loss_history/"+str(lr) + "_" + str(batch_size)+"/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fname = dir_name+"loss_progress_"+str(epoch)+".pkl"
    with open(fname,'wb') as f:
        pickle.dump(obj,f)

def save_model(net,epoch,lr,batch_size,logger=None):
    models_dir = "models/"+str(lr) + "_" + str(batch_size)+"/"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_name = "model_" + str(epoch)
    torch.save(net, models_dir + model_name)
    if logger is not None:
        logger.info("Model Saved")