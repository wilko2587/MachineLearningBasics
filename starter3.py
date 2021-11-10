import sys
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def read_mnist(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show_mnist(file_name,mode):
    
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
                   
def read_insurability(file_name):
    
    count = 0
    data = []
    with open(file_name,'rt') as f:
        for line in f:
            if count > 0:
                line = line.replace('\n','')
                tokens = line.split(',')
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == 'Good':
                        cls = 0
                    elif tokens[3] == 'Neutral':
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls],[x1,x2,x3]])
            count = count + 1
    return(data)
               
def classify_insurability():
    
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    # insert code to train simple FFNN and produce evaluation metrics
    toscale = MinMaxScaler()
    train_numpy = toscale.fit_transform(train)
    test_numpy = toscale.transform(test)
    train_tensor = torch.tensor(train_numpy)

    class CustomInsurabilityDataset(torch.utils.data.Dataset):
        def __init__(self, file, scaling):
            self.df = pd.read_csv(file)
            self.toscale = scaling

        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, idx):
            rawv = self.df.iloc[idx].values
            if type(idx) == int:
                rawv = rawv.reshape(1, -1)
            rawv = self.toscale.transform(rawv)
            data = torch.tensor(rawv[:, :-1], dtype=torch.float32)
            label = torch.tensor(rawv[:, -1], dtype=torch.float32)
            return data, label

    train_data = CustomInsurabilityDataset('three_train.csv', toscale)
    test_data = CustomInsurabilityDataset('three_test.csv', toscale)

    train_load = DataLoader(train_data, batch_size=64, shuffle=True)
    test_load = DataLoader(test_data, batch_size=64, shuffle=True)

    class FeedForward(nn.Module):
        def __init__(self):
            super(FeedForward, self).__init__()
            self.lin1 = nn.Linear(8, 32)
            self.relu1 = nn.LeakyReLU()
            self.lin2 = nn.Linear(32, 16)
            self.relu2 = nn.LeakyReLU()
            self.lin_out = nn.Linear(16, 1)

        def forward(self, x):
            x = self.lin1(x)
            x = self.relu1(x)
            x = self.lin2(x)
            x = self.relu2(x)
            x = self.lin_out(x)
            return x

    #ff = FeedForward()
    #print(ff)

    def stable_softmax(X):
        exps = np.exp(ff - np.max(ff))
        return exps/np.sum(exps)

    loss_func = stable_softmax(ff)

    optimize = torch.optim.SGD(ff.parameters(), lr=1e-2, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    ff.eval()
    a, b = train_data[0]
    with torch.no_grad():
        pred1 = ff(a)
    print('prediction:', pred1)
    print('target:', b)
    print('error:', stable_softmax(pred1))

    def train(dataloader, mod1, optimize):
        mod1.train()
        train_loss_val = []

        now = datetime.datetime.now()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # make predictions and retrieve error
            predicted = mod1(X)
            loss = stable_softmax(predicted)

            # backpropogation
            optimize.zero_grad()
            loss.backward()
            optimize.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                intervs = 10 * len(X)
                past = datetime.datetime.now()
                intervs /= (past - now).total_seconds()
                print(f"loss: {loss:>6f} [{current:>5d}/{17000}] ({intervs:.1f} its/sec)")
                now = past
                train_loss_value.append(loss)
        return train_loss_value

    def test(dataloader, mod1):
        size = len(dataloader)
        num_batches = 170
        mod1.eval()
        test_loss_value = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                predict_v = mod1(X)
                test_loss_value += stable_softmax(pred).item()
        test_loss_value /= num_batches
        print(f"Avg Loss: {test_loss_value:>8f}\n")
        return test_loss_value

    ff = FeedForward().to(device)
    loss_func = stable_softmax(ff)
    optimize = torch.optim.SGD(ff.parameters(), lr=1e-2, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    epochs = 10
    train_loss_values = []
    test_loss_values = []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n------------------------------- \n")
        losses_values = train(train_loader, ff, loss_func, optimize)
        train_loss_values.append(losses_values)
        test_loss_values.append(test(test_loader, ff, loss_func))

    plt.plot([i for i in range(len(train_loss))], torch.tensor(train_loss).mean(axis=1))

    plt.plot([i for i in range(len(test_loss))], test_loss)

    ff.eval()
    with torch.no_grad():
        x, y = train_data[4]
        pred = ff(x)
        print(pred)
        print(y)
    print(stable_softmax(ff))

def classify_mnist():
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    show_mnist('mnist_test.csv','pixels')
    
    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics
    
def classify_mnist_reg():
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    show_mnist('mnist_test.csv','pixels')
    
    # add a regularizer of your choice to classify_mnist()
    
def classify_insurability_manual():
    
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN
    
    
def main():
    classify_insurability()
    classify_mnist()
    classify_mnist_reg()
    classify_insurability_manual()
    
if __name__ == "__main__":
    main()
