import torch.optim as optim
import torch
import torch.cuda as cuda
from model.SequenceToSequence import Seq2seq
from dataLoader.DataLoader import Loader
from torch.utils.data import DataLoader
import os
from dataLoader.Collator import PadCollator
from loss.FlatennedLoss import CELoss

def train_model(lr,batch_size,epochs,hidden_size,n_layers,w2v_model,SOS_idx,EOS_idx,PAD_idx,df):
    rows,cols = w2v_model.wv.vectors.shape
    net = Seq2seq(cols, rows+1, hidden_size,SOS_idx,EOS_idx ,n_layers)
    net = net.double()
    if cuda.is_available():
        print("cuda is on!!")
        net.cuda()
    collator = PadCollator(PAD_idx)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_idx)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    data = Loader(df,w2v_model)
    data_loading = DataLoader(data, num_workers=10, shuffle=True, batch_size=batch_size,collate_fn=collator)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(data_loading):
            sequences, labels, lengths = batch

            # forward + backward + optimize
            y_hat = net.forward_train(sequences,sequences)
            optimizer.zero_grad(y_hat,sequences)
            loss = criterion()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
    models_dir = "models/"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_name = "model_"+str(lr)+"_"+str(batch_size)+"_"+str(epochs)
    torch.save(net,models_dir+model_name)
    return net,models_dir+model_name