import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import wiki_dataset
from dataloader import wiki_dataloader
import torchmetrics
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import time
import matplotlib.pyplot as plt
import numpy as np
import nltk

nltk.download('punkt')


class FeedForward(pl.LightningModule):

    def __init__(self, context,
                 embed_dim,
                 vocab_size,
                 dropout=0.2,
                 lr=1e-3,
                 trainweights=None):
        super(FeedForward, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lr = lr
        nn.init.uniform_(self.embed.weight, a=-0.1, b=0.1)  # initialise weights in range -0.1->0.1 with uniform distro
        self.lin1 = nn.Linear(context * embed_dim, embed_dim)
        nn.init.uniform_(self.lin1.weight, a=-0.1, b=0.1)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.drop1 = nn.Dropout(p=dropout)
        self.lin2 = nn.Linear(embed_dim, vocab_size)
        self.lin2.weight = self.embed.weight  # tied embeddings

        self.loss = nn.CrossEntropyLoss(weight=trainweights)
        self.viewloss = nn.CrossEntropyLoss() # weights change the results of the loss, so we initialise an unweighted
                                                # loss to keep track and use for perplexity calcs

    def forward(self, X):
        X = self.embed(X)
        X = torch.flatten(X, start_dim=1)
        X = torch.tanh(self.bn1(self.lin1(X)))
        X = self.drop1(X)
        X = self.lin2(X)
        return X

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def _step(self, batch, batch_idx, logstring):
        data, label = batch
        logits = self(data)
        loss = self.loss(logits, label)
        viewloss = self.viewloss(logits, label)
        tensorboard_logs = {'loss': {logstring: viewloss.detach()}}
        self.log("{} loss".format(logstring), viewloss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")


def test_hparam(hparam, values=[], logpath="./FeedForward_logs/", tpu_cores=None, gpus=1):
    '''

    pretty neat function to test a hparam (as titled in FeedForward.__init__()) between a set of values.
    Will automatically log for you

    hparam = string of hyperparam to vary
    values = values of hparam to try
    tpu_cores: either None, or 1 or 8
    gpus = None, or integer
    '''

    # Load datasets
    train = wiki_dataset('./wiki.train.txt', training=True, token_map='create', window=5)
    valid = wiki_dataset('./wiki.valid.txt', training=False, token_map=train.token_map, window=5)
    test = wiki_dataset('./wiki.test.txt', training=False, token_map=train.token_map, window=5)
    datasets = [train, valid, test]

    # default feedforward params
    params = {'context': train.window,
              'embed_dim': 100,
              'vocab_size': len(train.unique_tokens),
              'dropout': 0,
              'lr': 1e-3}

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=64, num_workers=8)

    for hparam_val in values:

        params[hparam] = hparam_val
        # Make model and train
        model = FeedForward(**params,
                            trainweights=torch.log(1./train.token_count()))

        tb_logger = pl_loggers.TensorBoardLogger(logpath, name="{}_{}".format(hparam, hparam_val))
        trainer = pl.Trainer(gradient_clip_val=0.5, logger=tb_logger, max_epochs=10, tpu_cores=tpu_cores, gpus=gpus)

        trainer.fit(model, dataloader)
        model.eval()  # freeze the model
        result = trainer.test(model, dataloader)
        print('printing some example sentences from test set')
        print('--> format: sentence (true) [predicted]')
        for idx in np.random.randint(0, 1000, size=10):
            features, groundTruth = test[idx]
            fpass = model.forward(features.unsqueeze(dim=0))
            pred = np.argmax(torch.softmax(fpass.detach().squeeze(dim=0), 0))
            sentence = ' '.join([test.decode_int(i) for i in features])
            nextword = test.decode_int(groundTruth)
            nextpred = test.decode_int(pred)
            print('{} ({}) [{}]'.format(sentence, nextword, nextpred))
    return


if __name__ == '__main__':
    test_hparam('lr', values=[1e-3], tpu_cores=None, gpus=None)
