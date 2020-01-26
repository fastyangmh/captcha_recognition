# import
import torch
import torch.nn as nn

# def


def calculate_dim(Hin, Win, kernel_size, padding=0, dilation=1):
    for k in kernel_size:
        for _ in range(2):
            Hin = Hin+2*padding-dilation*(k-1)-1+1
            Win = Win+2*padding-dilation*(k-1)-1+1
    for _ in range(2):
        Hin = Hin+2*padding-dilation*(kernel_size[-1]-1)-1+1
        Win = Win+2*padding-dilation*(kernel_size[-1]-1)-1+1
    return Hin, Win

# class


class DEEPCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, Hin, Win, hidden_dim, n_hidden, out_dim, dropout=0.5):
        super(DEEPCNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(
            in_channels=i, out_channels=j, kernel_size=k) for i, j, k in zip(in_channels, out_channels, kernel_size)])
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=i, stride=1) for i in kernel_size])
        self.last_conv = nn.Conv2d(
            in_channels=out_channels[-1], out_channels=1, kernel_size=kernel_size[-1])
        self.last_pool = nn.MaxPool2d(kernel_size=kernel_size[-1], stride=1)
        Hout, Wout = calculate_dim(Hin, Win, kernel_size)
        in_sizes = [Wout]+[hidden_dim]*(n_hidden-1)
        out_sizes = [hidden_dim]*n_hidden
        self.layers = nn.ModuleList([nn.Linear(in_size, out_size) for (
            in_size, out_size) in zip(in_sizes, out_sizes)])
        self.last_layer = nn.Linear(
            in_features=hidden_dim, out_features=out_dim)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for conv, pool in zip(self.convs, self.pools):
            x = pool(conv(x))
        x = self.last_pool(self.last_conv(x))
        for layer in self.layers:
            x = self.dropout(self.leakyrelu(layer(x)))
        x = self.softmax(self.last_layer(x))
        return x
