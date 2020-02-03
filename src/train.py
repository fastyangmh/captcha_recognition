# import
from imageio import imread
import numpy as np
import string
from os.path import basename, join
from glob import glob
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import *
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

#global parameters
USE_CUDA = torch.cuda.is_available()


# def


def train_loop(dataloader, model, optimizer, criterion, epochs):
    train_loader, test_loader = dataloader
    criterion_bce, criterion_mse = criterion
    history = []
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = []
        for x, y in train_loader:
            if USE_CUDA:
                x, y = x.cuda(), y.cuda()
            yhat = model(x)
            loss = criterion_bce(yhat, y)+criterion_mse(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        history.append(np.mean(total_loss))
        if not epoch % 100:
            ohe = onehotencoder()
            _, acc = evaluation(train_loader, model, ohe)
            print('Train accuracy:{}, loss:{}'.format(
                round(np.mean(acc), 8), round(history[-1], 8)))
    return history


def evaluation(dataloader, model, decoder):
    predict = []
    accuracy = []
    model.eval()
    for x, y in dataloader:
        if USE_CUDA:
            x, y = x.cuda(), y.cuda()
        pred = model(x).cpu().data.numpy()
        pred = np.argmax(pred, 2)
        y = torch.argmax(y, 2).cpu().data.numpy()
        acc = accuracy_score(decoder.decoding(y), decoder.decoding(pred))
        predict.append(pred)
        accuracy.append(acc)
    return predict, accuracy

# class


class onehotencoder:
    def __init__(self,):
        self.alphabet_dict = {}
        keys = list(string.digits)+list(string.ascii_lowercase)
        values = np.identity(len(keys))
        for k, v in zip(keys, values):
            self.alphabet_dict[k] = v
            self.alphabet_dict[np.argmax(v)] = k

    def encoding(self, label):
        encode = []
        for l in label:
            code = []
            for c in l:
                code.append(self.alphabet_dict[c])
            encode.append(np.array(code))
        '''
        in order to easy training, i expand dimensions and swap axes from initial dimension.
        initial dimention is (1040,5,36)
        '''
        return np.array(encode)  # np.expand_dims(encode, 1).swapaxes(2, 3)

    def decoding(self, label):
        decode = []
        for l in label:
            code = ''
            #index = np.argmax(l, 1)
            index = l
            for i in index:  # [0]:
                code += self.alphabet_dict[i]
            decode.append(code)
        return decode


if __name__ == "__main__":
    # parameters
    data_path = './data'
    batch_size = 128
    lr = 0.01
    epochs = 20

    # load data
    print('load data')
    data = []
    label = []
    files = glob(join(data_path, '*.png'))
    for f in files:
        # 'L' (8-bit pixels, black and white)
        data.append(imread(f, pilmode='L'))
        label.append(basename(f)[:-4])
    data = np.array(data)
    label = np.array(label)

    # preprocessing
    print('preprocessing')
    data = np.expand_dims(data/255., 1)
    ohe = onehotencoder()
    X_train, X_test, Y_train, Y_test = train_test_split(
        data, ohe.encoding(label), test_size=0.3)
    train_set = TensorDataset(torch.from_numpy(
        X_train).float(), torch.from_numpy(Y_train).float())
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_set = TensorDataset(torch.from_numpy(
        X_test).float(), torch.from_numpy(Y_test).float())
    test_loader = DataLoader(dataset=test_set, num_workers=4, pin_memory=True)

    # create model
    print('create model')
    deepcnn = DEEPCNN(in_channels=[1], out_channels=[1], kernel_sizes=[3], strides=[
                      3], paddings=[0], dilations=[1], Hin=50, Win=200, hidden_dim=40, n_hidden=2, out_dim=36)
    optimizer = optim.Adam(deepcnn.parameters(), lr=lr)
    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()
    if USE_CUDA:
        deepcnn = deepcnn.cuda()
'''
    # train
    print('train')
    history = train_loop((train_loader, test_loader), deepcnn,
                         optimizer, (criterion_bce, criterion_mse), epochs)

    # evaluation
    print('evaluation')
    train_pred, train_acc = evaluation(train_loader, deepcnn, ohe)
    print('Training dataset accuracy: {}'.format(round(np.mean(train_acc), 4)))
    test_pred, test_acc = evaluation(test_loader, deepcnn, ohe)
    print('Test dataset accuracy: {}'.format(round(np.mean(test_acc), 4)))
'''