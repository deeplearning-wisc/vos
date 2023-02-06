from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data=data
        self.label = label

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

class MyDataset_outlier(Dataset):
    def __init__(self, data, label, outlier_label):
        self.data=data
        self.label = label
        self.outlier_label = outlier_label

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx],  self.outlier_label[idx], 
