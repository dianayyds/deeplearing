import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from LSTMModel import lstm
from dataset import getData
from parser_my import args
import torch


def eval():
    model = lstm(input_size=args.input_size,
                 hidden_size=args.hidden_size,
                 num_layers=args.layers,
                 output_size=args.output_size,
                 dropout=args.dropout,
                 batch_first=args.batch_first)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    temperature_max, temperature_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length,
                                                                          args.batch_size)
    for idx, (x, label) in enumerate(test_loader):
        if args.useGPU:
            x = x.squeeze(1).cuda()
        else:
            x = x.squeeze(1)
        pred = model(x)
        list = pred.data.squeeze(1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())

    preds = (np.array(preds)) * (temperature_max - temperature_min) + temperature_min
    labels = (np.array(labels)) * (temperature_max - temperature_min) + temperature_min

    rmse = np.sqrt(mean_squared_error(labels, preds))
    r2 = r2_score(labels, preds)

    print("均方根误差（RMSE）: %.5f" % rmse)
    print("决定系数（R2）: %.5f" % r2)

    for i in range(len(preds)):
        print('预测值是%.2f,真实值是%.2f' % (preds[i][0], labels[i]))

eval()
