import torch.optim as optim
import torch
from model.Seq2SeqAttnNet import Seq2seqAttn
from dataLoader.DataLoader import Loader
from torch.utils.data import DataLoader
from dataLoader.Collator import PadCollator
from dataLoader.DefCollate import DefCollator
import pandas as pd
from torch import nn
from loss.Utils import save_loss_history
from loss.Utils import save_model
from parallel.Parallel import DataParallelModel, DataParallelCriterion
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)

class CustomDataParallel(DataParallelModel):
    def __init__(self, model):
        super(CustomDataParallel, self).__init__(model,device_ids=[1,0])

    def __getattr__(self, name):
        try:
            return super(CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)





def train_model(lr,batch_size,epochs,hidden_size,n_layers,w2v_model,SOS_idx,EOS_idx,PAD_idx,data_set_file_path,random_seed,p,bidrectional,logger=None):
    #t
    prnt = False
    if logger is not None:
        prnt = True
    if prnt:
        logger.info("RUNNING WITH PARAMS: lr=" +str(lr)+" batch_size="+str(batch_size)+" dropout="+str(p)+" epochs="+str(epochs))
    rows,cols = w2v_model.wv.vectors.shape
    df = pd.read_csv(data_set_file_path,delimiter=",",header=0,nrows=500000)
    max_len = int(df["proc_len"].max())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_idx,reduction='none')
    # criterion = DataParallelCriterion(criterion, device_ids=[1, 0])
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    net = Seq2seqAttn(cols, hidden_size, SOS_idx, EOS_idx, PAD_idx, n_layers, w2v_model.wv.vectors, criterion, random_seed, p,device, max_len,bidrectional)
    # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    net = net.double()
    net.to(device)
    net.apply(init_weights)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net = CustomDataParallel(net)
    collator = PadCollator(PAD_idx,device)
    def_collator = DefCollator()



    loss_history = []

    if prnt:
        logger.info("Training Initialization")
    for epoch in range(epochs):
        running_batch_num = 0
        running_loss_for_plot = 0.0
        data = Loader(df, w2v_model, PAD_idx, EOS_idx, SOS_idx)
        data_loading = DataLoader(data, num_workers=4, shuffle=True, batch_size=batch_size,collate_fn=def_collator)
        running_loss = 0.0

        for i, batch in enumerate(data_loading):
            running_batch_num+=1
            batch = collator(batch)
            sequences,labels, lengths = batch

            loss = net(sequences,sequences,lengths)
            optimizer.zero_grad()
            if isinstance(loss,list):
                tmp_loss=0.0
                for item in loss:
                    tmp_loss+=item.to(device)
                tmp_loss=tmp_loss/len(loss)

                loss = tmp_loss
                del tmp_loss
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_for_plot += loss.item()

            logger.info("IN EPOCH: "+str(epoch)+" RUNNING BATCH: "+str(running_batch_num))
            if running_batch_num % 1000 == 999:  # print every 1000 mini-batches
                if prnt:
                    logger.info('[%d, %5d] loss: %.3f' %
                          (epoch + 1, running_batch_num, running_loss / 1000))

                    running_loss = 0.0
            # del loss,y_hat
            del loss

        loss_history.append(running_loss_for_plot/running_batch_num)
        save_loss_history(loss_history,epoch,lr,batch_size)
        save_model(net,epoch,lr,batch_size,logger)
    if prnt:
        logger.info("Training Is Done")


