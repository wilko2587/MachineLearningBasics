import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import txt_preprocess_james as tpj
import torch


class NeuralNet(nn.Module):
    '''
    High level class for a nn.Module neural network
    '''

    def __init__(self):
        super(NeuralNet, self).__init__()

    def forward(self, X):
        # NB: good pytorch practice: never apply softmax function at the end of .forward()!!!
        # This breaks nn.CrossEntropyLoss() which already has a softmax function built into it
        logits = self._stack(X)
        return logits

    def predict_proba(self, X):
        # Forward pass but applies a softmax at the output. Same functionality as in sklearn.Logistic_Loss
        # predicts probabilities of the class labels
        logits = self._stack(X)
        return nn.Softmax(dim=1)(logits)

    def predict(self, X):
        # predicts the class labels of a dataset X. Same functionality as in sklearn.Logistic_Loss
        probs = self.predict_proba(X)
        return torch.argmax(probs, dim=1)

    def score(self, X, y):
        # returns "score" of the model comparing self.predict(X) to y, using MSE
        preds = self.predict(X)
        MSE = sum(preds != y) / len(y)
        return float(MSE)

    def fit(self, traindata, validdata,
            lr=1e-4, momentum=0.9, batchsize=100, l2=0., max_epoch=20, tol=1e-4,
            verbose='vv'):
        '''

        Trains the NN using cross entropy loss. Use just like sklearn modules :)

        :param lr: learning rate (step size). Smaller = foolproof better but slower
        :param momentum: momentum parameter to help learning
        :param batchsize: we'll train using minibatch gradient descent. This parameter determines the batch size
        :param l2: l2 regularization parameter (0 = no regularisation)
        :param max_iter: maximum number of iterations before stopping training
        :param tol: training criterion tolerance for stopping training
        :param verbose: bool
        :param loadedfiles: integer, number of cache'd files to be read into memory at once for minibatch
        '''

        # CUDA for PyTorch - uses your GPU if you have one
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True

        trainingloader = DataLoader(dataset=traindata, batch_size=batchsize, shuffle=True)
        validloader = DataLoader(dataset=validdata, batch_size=batchsize, shuffle=True)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=l2)
        reason = "Max iterations reached" # ititialise a reason for exitting training

        learning_curve= []
        validation_curve = []
        for epoch in range(max_epoch):  # loop over the dataset multiple times

            # Training
            train_losses = []  # we'll store the losses over each minibatch, and average at the end to get a "stable" loss
            for X_batch, y_batch in trainingloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Send to GPU
                optimizer.zero_grad()
                logits = self(X_batch)
                tloss = criterion(logits, y_batch)
                if verbose == 'vv':
                    print('[%d] mid-epoch train loss: %.5f' %
                          (epoch + 1, tloss.item()), end='\r', flush=True)

                tloss.backward()
                optimizer.step()
                train_losses.append(tloss.item())

            ave_train_loss = sum(train_losses)/len(train_losses)

            # Validation
            valid_losses = []
            with torch.set_grad_enabled(False): # turn off gradients when calculating validation data
                for X_batch, y_batch in validloader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Send to GPU
                    logits = self(X_batch)
                    vloss = criterion(logits, y_batch)
                    valid_losses.append(vloss.item())

            ave_valid_loss = sum(valid_losses)/len(valid_losses)

            # stopping criteria
            if ave_train_loss <= tol:
                reason = "Learning criteria achieved"
                break

            # print statistics
            if epoch % 1 == 0:  # print out loss as model is training
                if verbose == 'v' or verbose == 'vv':
                    print('[%d] train loss: %.5f valid loss: %.5f' %
                          (epoch + 1, ave_train_loss, ave_valid_loss))#, end='\r', flush=True)

            learning_curve.append(ave_train_loss)
            validation_curve.append(ave_valid_loss)

        if verbose: print("Training complete: %s \n final loss (train/valid) %.5f/%.5f" % (reason, ave_train_loss, ave_valid_loss))

        plt.plot(learning_curve, label='training loss')
        plt.plot(validation_curve, label='validation loss')
        plt.show()
        return None


class FeedForward(NeuralNet):
    def __init__(self, inputs, layers, outputs, activationfunc=nn.Tanh):
        '''

        :param inputs: integer, number of input features
        :param layers: list of integers representing sizes of hidden layers. ie: [5,4] will give a NN with 2 hidden
                        layers, first hidden layer of 5 nodes, and second hidden layer of 4 nodes
        :param outputs: number of output nodes
        :param activationfunc: default tanh. activation function to use throughout the network
        '''

        super(FeedForward, self).__init__()
        layer_sizes = [inputs] + layers + [outputs]
        sequentials = []

        for i in range(len(layer_sizes) - 1):
            if i != 0: sequentials.append(activationfunc())

            sequentials.append(nn.Linear(layer_sizes[i],
                                         layer_sizes[i + 1]))

        self._stack = nn.Sequential(*sequentials)

class LSTM(NeuralNet):
    def __init__(self, inputs, hiddendim, outputs):
        '''

        :param inputs: integer, number of input features
        :param hiddendim: dimensions of hidden layer
        :param outputs: number of output nodes
        '''

        super(LSTM, self).__init__()
        sequentials = [nn.LSTM(inputs, hiddendim),
                       nn.Linear(hiddendim, outputs)]

        self._stack = nn.Sequential(*sequentials)


if __name__ == '__main__':
    traindata = tpj.my_corpus('wiki.train.txt', windowlength=5)
    validdata= tpj.my_corpus('wiki.valid.txt', windowlength=5)
    traindata.generate_embeddings()
    validdata.inherit_embeddings(traindata)

    #testdata = txt_preprocess.my_corpus('wiki.test.txt')
    embedding_size = traindata.embedding_size()
    vocab_size = traindata.vocab_size()
    print('datalength: ', len(traindata))

    ff = FeedForward(embedding_size*5, [1000], vocab_size, activationfunc=nn.ReLU)
    ff.fit(traindata, validdata, lr=1e-3, batchsize=20, max_epoch=20)
