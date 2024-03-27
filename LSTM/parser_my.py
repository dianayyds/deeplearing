import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--corpusFile', default='./data/13.csv')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--layers', default=2, type=int)
parser.add_argument('--input_size', default=1, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--output_size', default=1, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--batch_first', default=True, type=bool)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--split_rate', default=0.85, type=float)
parser.add_argument('--sequence_length', default=4, type=int)
parser.add_argument('--useGPU', default=True, type=bool)
parser.add_argument('--save_file', default='./model/13.pkl')

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device
