import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import txt_preprocess_josh
import torch


class FeedForward(nn.Module):

    def __init__(self, window_length, embed_dim, vocab_size):
        super(FeedForward, self).__init__()
        self.embeds = nn.Embedding(vocab_size,100)
        self.lin1 = nn.Linear(500,1000)
        self.lin2 = nn.Linear(1000,vocab_size)

    def forward(self, X):
        X = self.embeds(X)
        X = torch.flatten(X,start_dim=1)
        X = F.relu(self.lin1(X))
        X = self.lin2(X)

        return X

def training_loop(traindata, validdata, model):

    lr=1e-4
    momentum=0.9
    batchsize=100
    l2=0.
    max_epoch=20
    tol=1e-4,
    verbose='vv'

    # CUDA for PyTorch - uses your GPU if you have one
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    trainingloader = DataLoader(dataset=traindata, batch_size=batchsize, shuffle=True)
    validloader = DataLoader(dataset=validdata, batch_size=batchsize, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2)
    reason = "Max iterations reached" # ititialise a reason for exitting training

    learning_curve= []
    validation_curve = []
    for epoch in range(max_epoch):  # loop over the dataset multiple times
        print(f'epoch {epoch+1}')
        # Training
        train_losses = []  # we'll store the losses over each minibatch, and average at the end to get a "stable" loss
        for X_batch, y_batch in trainingloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Send to GPU
            optimizer.zero_grad()
            logits = model(X_batch)
            tloss = criterion(logits, y_batch)
            tloss.backward()
            optimizer.step()
            train_losses.append(tloss.item())
            if verbose == 'vv':
                print('[%d] mid-epoch train loss: %.5f' %
                      (epoch + 1, tloss.item()), end='\r', flush=True)

        ave_train_loss = sum(train_losses)/len(train_losses)

        # Validation
        valid_losses = []
        with torch.set_grad_enabled(False): # turn off gradients when calculating validation data
            for X_batch, y_batch in validloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Send to GPU
                logits = model(X_batch)
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

if __name__ == '__main__':
    traindata = txt_preprocess_josh.my_corpus('wiki.train.txt')
    validdata = txt_preprocess_josh.my_corpus('wiki.valid.txt')

    vocabsize = len(traindata._tokenmap)
    window_size = 5

    model = FeedForward(vocabsize*5, [100], vocabsize)

    training_loop(traindata,validdata,model)

    # this puts all the embeddings in a list one after another
    # need to stack or flatten these, and this should be input layer to model

    # embeddings = list()
    # for each in traindata[0][0]:
    #     embeddings.append(model.embeds(torch.tensor(traindata._tokenmap[each],dtype=torch.long)))
    # embeddings = torch.stack(embeddings).flatten(0)
    #
    # print(embeddings)

    # model.fit(traindata, validdata, lr=1e-3, batchsize=20, max_epoch=20)

    # lookup = torch.tensor(traindata._tokenmap['with'],dtype=torch.long)
    #
    # hello_embed = model.embeds(lookup)
    # print(hello_embed)


