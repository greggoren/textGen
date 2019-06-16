import torch.optim as optim
import torch
from model.SequenceToSequence import Seq2seq
from dataLoader.DataLoader import Loader
from torch.utils.data import DataLoader
import os
from dataLoader.Collator import PadCollator
from dataLoader.DefCollate import DefCollator
import pickle

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

def train_model(lr,batch_size,epochs,hidden_size,n_layers,w2v_model,SOS_idx,EOS_idx,PAD_idx,df,logger=None):
    prnt = False
    if logger is not None:
        prnt = True
    rows,cols = w2v_model.wv.vectors.shape
    net = Seq2seq(cols,rows+2,hidden_size,SOS_idx,EOS_idx,n_layers,w2v_model.wv.vectors)
    net = net.double()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if prnt:
    #     logger.info("cuda is on!!")
    net.to(device)
    # net.share_memory()

    collator = PadCollator(PAD_idx,device)
    def_collator = DefCollator()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_idx)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    data = Loader(df,w2v_model,PAD_idx,EOS_idx,SOS_idx)
    loss_history = []

    if prnt:
        logger.info("Training Initialization")
    for epoch in range(epochs):
        data_loading = DataLoader(data, num_workers=10, shuffle=True, batch_size=batch_size,collate_fn=def_collator)
        running_loss = 0.0
        running_loss_for_plot = 0.0
        for i, batch in enumerate(data_loading):
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
            if i % 1000 == 999:  # print every 1000 mini-batches
                if prnt:
                    logger.info('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (i+1)))
                    running_loss = 0.0

        loss_history.append(running_loss_for_plot/(i+1))
        save_loss_history(loss_history,epoch,lr,batch_size)
        if epoch%10==0:
            save_model(net,epoch,lr,batch_size,logger)
    if prnt:
        logger.info("Training Is Done")


