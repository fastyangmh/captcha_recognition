# import
import torch
import torch.nn as nn

# def


def calculate_dim(Hin, Win, kernel_size, stride=1, padding=0, dilation=1):
    Hin = ((Hin+2*padding-dilation*(kernel_size-1)-1)/stride)+1
    Win = ((Win+2*padding-dilation*(kernel_size-1)-1)/stride)+1
    return int(Hin), int(Win)

# class


class DEEPCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, paddings, dilations, Hin, Win, hidden_dim, n_hidden, out_dim, dropout=0.5):
        super(DEEPCNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, stride=s, padding=p, dilation=d)
                                    for i, o, k, s, p, d in zip(in_channels, out_channels, kernel_sizes, strides, paddings, dilations)])
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=s, padding=p, dilation=d)
                                    for k, s, p, d in zip(kernel_sizes, strides, paddings, dilations)])
        for k, s, p, d in zip(kernel_sizes, strides, paddings, dilations):
            Hin, Win = calculate_dim(Hin, Win, k, s, p, d)
            Hin, Win = calculate_dim(Hin, Win, k, s, p, d)
        in_sizes = [Win]+[hidden_dim]*(n_hidden-1)
        out_sizes = [hidden_dim]*n_hidden
        self.layers = nn.ModuleList([nn.Linear(in_size, out_size) for (
            in_size, out_size) in zip(in_sizes, out_sizes)])
        self.last_layer = nn.Linear(hidden_dim, out_dim)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=3)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for conv, pool in zip(self.convs, self.pools):
            x = self.leakyrelu(self.dropout(pool(conv(x))))
        for layer in self.layers:
            x = self.leakyrelu(self.dropout(layer(x)))
        x = self.softmax(self.dropout(self.last_layer(x)))
        return x.squeeze(1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_hidden, dropout=0.5):
        super(MLP, self).__init__()
        in_sizes = [in_dim]+[hidden_dim]*(n_hidden-1)
        out_sizes = [hidden_dim]*n_hidden
        self.layers = nn.ModuleList([nn.Linear(in_size, out_size) for (
            in_size, out_size) in zip(in_sizes, out_sizes)])
        self.last_layer = nn.Linear(hidden_dim, out_dim)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.softmax = nn.Softmax(dim=3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = self.leakyrelu(self.dropout(layer(x)))
        x = self.softmax(self.dropout(self.last_layer(x)))
        return x.squeeze(1)
