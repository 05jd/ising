import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset


def parse_option_args():
    # Training settings
    parser = argparse.ArgumentParser(description='2D-Ising Example')
    parser.add_argument('--train-ratio', type=float, default=0.7, metavar='M',
                        help='ratio of training dataset (default: 0.7)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging '
                             'training status')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                        metavar='DIR',
                        help='Checkpoint directory to store a trained model')
    args = parser.parse_args()
    return args


class ConvNet(nn.Module):
    """ A simple convolution net.

    # Args.
        config: list of int or 'p',
            integer: #channels of a convolution layer
            'p': a pooling layer
            e.g. cfg = [6, 6, 'p', 12, 12]
                conv(6) -> conv(6) -> 2D-pooling -> conv(12) -> conv(12)
                -> global-pooling -> fc(2) (w/ softmax)
    """

    def __init__(self, config=(6, 6, 'p', 12, 12)):
        super(ConvNet, self).__init__()

        self.layers = []
        in_channels = 1  # single channel
        for cfg in config:
            if isinstance(cfg, int):
                layer = nn.Sequential(
                    nn.Conv2d(in_channels, cfg, 3, padding=1),
                    nn.ReLU())
                in_channels = cfg
            else:
                assert cfg == 'p'
                layer = nn.MaxPool2d(2, 2)
            self.layers.append(layer)
        self.fc = nn.Linear(in_channels, 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # (N, C, W, H)
        x = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze()
        return self.fc(x)


def load_dataset(train_ratio=0.7):
    print('Loading dataset')
    inputs_path = './data/Ising2DFM_reSample_L40_T=All.pkl'
    inputs = np.load(inputs_path, allow_pickle=True)
    inputs = np.unpackbits(inputs).reshape((-1, 1, 40, 40))
    inputs = inputs / 255. - 0.5  # in [-0.5, 0.5]
    print('| input data has normalized into [-0.5, 0.5]')

    labels_path = './data/Ising2DFM_reSample_L40_T=All_labels.pkl'
    labels = np.load(labels_path, allow_pickle=True)

    # split into train / val
    num_samples = len(labels)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    num_train_samples = int(num_samples * train_ratio)
    x_train = torch.tensor(inputs[indices[:num_train_samples]]).float()
    y_train = torch.tensor(labels[indices[:num_train_samples]])
    x_val = torch.tensor(inputs[indices[num_train_samples:]]).float()
    y_val = torch.tensor(labels[indices[num_train_samples:]])
    print('| #train samples: %d' % len(y_train))
    print('| #val samples  : %d' % len(y_val))
    return (x_train, y_train), (x_val, y_val)


def load_dataloader(args):
    (x_train, y_train), (x_val, y_val) = load_dataset(args.train_ratio)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)

    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)
    return train_loader, val_loader


def train(model, device, train_loader, criterion, optimizer, epoch,
          log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()), end='\r')


def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    args = parse_option_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_dataloader(args)

    model = ConvNet()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        print('Epoch %d' % epoch)
        train(model, device, train_loader, criterion, optimizer, epoch)
        evaluate(model, device, val_loader, criterion)
        scheduler.step()
    print('Finished Training')


if __name__ == '__main__':
    main()
