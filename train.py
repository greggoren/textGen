import torch.optim as optim
import torch
from model.SequenceToSequence import Seq2seq
from dataLoader.DataLoader import Loader
from torch.utils.data import DataLoader
from dataLoader.Collator import PadCollator
from dataLoader.DefCollate import DefCollator
import pandas as pd
from torch import nn
from loss.Utils import save_loss_history
from loss.Utils import save_model

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)

class CustomDataParallel(nn.DataParallel):
    def __init__(self, model):
        super(CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def train_model(lr,batch_size,epochs,hidden_size,n_layers,w2v_model,SOS_idx,EOS_idx,PAD_idx,data_set_file_path,logger=None):
    prnt = False
    if logger is not None:
        prnt = True
    if prnt:
        logger.info("RUNNING WITH PARAMS: lr=" +str(lr)+" batch_size="+str(batch_size)+" epochs="+str(epochs))
    rows,cols = w2v_model.wv.vectors.shape
    chunks = pd.read_csv(data_set_file_path,delimiter=",",header=0,chunksize=100000)
    net = Seq2seq(cols,rows+3,hidden_size,SOS_idx,EOS_idx,PAD_idx,n_layers,w2v_model.wv.vectors)
    net = net.double()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net = CustomDataParallel(net)
    collator = PadCollator(PAD_idx,device)
    def_collator = DefCollator()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_idx)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    loss_history = []

    if prnt:
        logger.info("Training Initialization")
    for epoch in range(epochs):
        running_batch_num = 0
        for df in chunks:
            data = Loader(df, w2v_model, PAD_idx, EOS_idx, SOS_idx)
            data_loading = DataLoader(data, num_workers=10, shuffle=True, batch_size=batch_size,collate_fn=def_collator)
            running_loss = 0.0
            running_loss_for_plot = 0.0
            for i, batch in enumerate(data_loading):
                running_batch_num+=1
                batch = collator(batch)
                sequences,labels, lengths = batch

                # forward + backward + optimize
                y_hat = net.forward_train(sequences,sequences,lengths)
                optimizer.zero_grad()
                loss = criterion(y_hat,sequences)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                running_loss_for_plot += loss.item()
                if running_batch_num % 1000 == 999:  # print every 1000 mini-batches
                    if prnt:
                        logger.info('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / (running_batch_num)))
                        running_loss = 0.0

        loss_history.append(running_loss_for_plot/running_batch_num)
        save_loss_history(loss_history,epoch,lr,batch_size)
        if epoch%10==0:
            save_model(net,epoch,lr,batch_size,logger)
    if prnt:
        logger.info("Training Is Done")


