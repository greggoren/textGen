from torch.utils.data import Dataset
import torch
import numpy as np
import string

"""
To work with utils functions the Loader must return:
def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.targets[idx]), self.sequence_lengths[idx]


Then needed to do this:
train_gen = Data.DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=pad_and_sort_batch)
"""
class Loader(Dataset):
    def __init__(self,df,model,PAD_idx,EOS_idx,SOS_idx):
        self.df = df
        self.model = model
        self.PAD_idx = PAD_idx
        self.EOS_idx = EOS_idx
        self.SOS_idx = SOS_idx

    def sequence2index(self,text):
        text = str(text)
        print("text",text)
        seq = [self.model.wv.vocab.get(token).index for token in text.split()]
        seq.append(self.EOS_idx)
        # seq.insert(0,self.SOS_idx)
        return seq


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.ix[idx]
        sequence= self.sequence2index(row['proc_sentence'])
        length = int(row['proc_len'])+1
        return sequence,sequence,length