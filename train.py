import torch.optim as optim
import torch
import torch.cuda as cuda
from model.SimpleLSTMAE import LSTMAE
from dataLoader.DataLoader import Loader
from torch.utils.data import DataLoader
import os
from dataLoader.utilis import pad_and_sort_batch


def train_model(lr,momentum,input_dir,batch_size,epochs,lang,input_size,hidden_size,stacked_layers):
    net = LSTMAE(input_size,hidden_size,stacked_layers,True,len(lang.word2index))
    net = net.double()
    if cuda.is_available():
        print("cuda is on!!")
        net.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    data = Loader(input_dir)
    data_loading = DataLoader(data, num_workers=5, shuffle=True, batch_size=batch_size,collate_fn=pad_and_sort_batch)
    epochs = epochs
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(data_loading):
            inputs, labels = batch

            # forward + backward + optimize
            out1, out2 = net(inputs)
            optimizer.zero_grad()
            loss = criterion(out1, out1, labels)
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
    model_name = "model_"+str(lr)+"_"+str(momentum)+"_"+str(batch_size)+"_"+str(epochs)
    torch.save(net,models_dir+model_name)
    return net,models_dir+model_name