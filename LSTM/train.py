from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from parser_my import args
from dataset import getData


def train():
    model = lstm(input_size=args.input_size,
                 hidden_size=args.hidden_size,
                 num_layers=args.layers,
                 output_size=args.output_size,
                 dropout=args.dropout,
                 batch_first=args.batch_first)
    print(model)
    model.to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    temperature_max, temperature_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length,
                                                                          args.batch_size)
    for i in range(args.epochs):
        total_loss = 0
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                data = data.squeeze(1).cuda()
                pred = model(Variable(data).cuda())
                pred = pred[1, :, :]
                label = label.unsqueeze(1).cuda()
            else:
                data = data.squeeze(1)
                pred = model(Variable(data))
                pred = pred[1, :, :]
                label = label.unsqueeze(1)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if i % 10 == 0:
            print(f"{'*' * 50}_epoch:{i}_{'*' * 50}")
            print(total_loss)
            torch.save({'state_dict': model.state_dict()}, args.save_file)
    torch.save({'state_dict': model.state_dict()}, args.save_file)


train()
