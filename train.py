import torch.optim as optim
import torch
import torch.cuda as cuda
from model.SequenceToSequence import Seq2seq
from dataLoader.DataLoader import Loader
from torch.utils.data import DataLoader
import os
from dataLoader.Collator import PadCollator
import logging
import matplotlib.pyplot as plt
import sys

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


def train_model(lr,batch_size,epochs,hidden_size,n_layers,w2v_model,SOS_idx,EOS_idx,PAD_idx,df,logger=None):
    prnt = False
    if logger is not None:
        prnt = True
    rows,cols = w2v_model.wv.vectors.shape
    # net = Seq2seq(cols, rows+1, hidden_size,SOS_idx,EOS_idx ,n_layers,w2v_model.wv.vectors)
    net = Seq2seq(cols,rows+2,hidden_size,SOS_idx,EOS_idx,n_layers,w2v_model.wv.vectors)
    net = net.double()
    if cuda.is_available():
        if prnt:
            logger.info("cuda is on!!")
        net.cuda()
        net.share_memory()

    collator = PadCollator(PAD_idx)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_idx)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    data = Loader(df,w2v_model,PAD_idx,EOS_idx,SOS_idx)
    loss_history = []

    if prnt:
        logger.info("Training Initialization")
    for epoch in range(epochs):
        data_loading = DataLoader(data, num_workers=0, shuffle=True, batch_size=batch_size, collate_fn=collator)
        running_loss = 0.0
        running_loss_for_plot = 0.0
        for i, batch in enumerate(data_loading):
            sequences, labels, lengths = batch

            # forward + backward + optimize
            y_hat = net.forward_train(sequences,sequences,lengths)
            optimizer.zero_grad()
            loss = criterion(y_hat,sequences)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_for_plot += loss.item()
            # if i % 1000 == 999:  # print every 1000 mini-batches
            if prnt:
                logger.info('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / (i+1)))
                running_loss = 0.0
        loss_history.append(running_loss_for_plot/i)
    if prnt:
        logger.info("Training Is Done")
    models_dir = "models/"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_name = "model_"+str(lr)+"_"+str(batch_size)+"_"+str(epochs)
    torch.save(net,models_dir+model_name)
    if prnt:
        logger.info("Model Saved")
    # plot_metric(loss_history,"CELoss.png","CE Loss")
    # if prnt:
    #     logger.info("Plot Finished")
    return net,models_dir+model_name