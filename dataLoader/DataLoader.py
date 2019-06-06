from torch.utils.data import Dataset
import pickle

class Loader(Dataset):
    def __init__(self,input_dir):
        self.input_dir = input_dir


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_file = self.input_dir+str(idx)
        input_file_access = open(input_file,"rb")
        vectors = pickle.load(input_file_access)
        input_file_access.close()
        label = self.labels[idx]
        return vectors,label