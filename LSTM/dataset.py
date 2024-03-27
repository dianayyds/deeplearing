import numpy as np
from pandas import read_csv
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from parser_my import args


def getData(corpusFile, sequence_length, batchSize):
    temperature_data = read_csv(corpusFile, encoding='gbk')
    temperature_max = temperature_data['温度(单位:℃)'].max()
    temperature_min = temperature_data['温度(单位:℃)'].min()
    df = temperature_data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    sequence = sequence_length
    X = []
    Y = []
    for i in range(df.shape[0] - sequence):
        X.append(np.array(df.iloc[i:(i + sequence), ].values, dtype=np.float32))
        Y.append(np.array(df.iloc[(i + sequence), 0], dtype=np.float32))
    total_len = len(Y)
    train_X, train_Y = X[:int(args.split_rate * total_len)], Y[:int(args.split_rate * total_len)]
    test_X, test_Y = X[int(args.split_rate * total_len):], Y[int(args.split_rate * total_len):]
    train_loader = DataLoader(dataset=Mydataset(train_X, train_Y, transform=transforms.ToTensor()),
                              batch_size=batchSize)
    test_loader = DataLoader(dataset=Mydataset(test_X, test_Y),
                             batch_size=batchSize)
    return temperature_max, temperature_min, train_loader, test_loader



class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)
