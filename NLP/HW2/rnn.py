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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import time
import matplotlib.pyplot as plt

class rnn(pl.LightningModule):
    def __init__(self, n_vocab, embedding_size, hidden_size, num_layers, dropout, lr):
        super(rnn, self).__init__()
        self.lr = lr
        self.embed = nn.Embedding(n_vocab, embedding_size)
        nn.init.uniform_(self.embed.weight, a=-0.1, b=0.1)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False, dropout=dropout,lr=lr)
        nn.init.uniform_(self.rnn.weight_ih_l0, a=-0.1, b=0.1)
        nn.init.uniform_(self.rnn.weight_hh_l0, a=-0.1, b=0.1)
        nn.init.uniform_(self.rnn.weight_ih_l1, a=-0.1, b=0.1)
        nn.init.uniform_(self.rnn.weight_hh_l1, a=-0.1, b=0.1)
        self.fc = nn.Linear(hidden_size, n_vocab)
        # self.fc.weight = self.embed.weight
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embed(x)
        x, hidden = self.rnn(x)
        x = x[:, -1, :]
        logits = self.fc(x)
        return logits

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        data, label = batch
        logits = self.forward(data)
        loss = self.loss(logits, label)
        tensorboard_logs = {'loss': {'train': loss.detach()}}
        self.log("train loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        data, label = batch
        logits = self.forward(data)
        loss = self.loss(logits, label)
        tensorboard_logs = {'loss': {'val': loss.detach()}}
        self.log("val loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        data, label = batch
        logits = self.forward(data)
        loss = self.loss(logits, label)
        tensorboard_logs = {'loss': {'test': loss.detach()}}
        self.log("test loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

def test_hparam(hparam, values = [], logpath="./RNN_logs/", tpu_cores=None, gpus=1):
    '''

    pretty neat function to test a hparam (as titled in RNN.__init__()) between a set of values.
    Will automatically log for you

    hparam = string of hyperparam to vary
    values = values of hparam to try
    tpu_cores: either None, or 1 or 8
    gpus = None, or integer
    '''

    # Load datasets
    train = wiki_dataset('./wiki.train.txt', training=True, token_map='create', window=30)
    valid = wiki_dataset('./wiki.valid.txt', training=False, token_map=train.token_map, window=30)
    test = wiki_dataset('./wiki.test.txt', training=False, token_map=train.token_map, window=30)
    datasets = [train, valid, test]

    # default RNN params
    params = {'n_vocab':len(train.unique_tokens),
              'embedding_size':100,
              'hidden_size':100,
              'num_layers':2,
              'dropout':0,
              'lr':1e-3}

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=64, num_workers=2)

    for hparam_val in values:

        if hparam != 'gradient_clip_val':
            params[hparam] = hparam_val

        # Make model and train
        model = RNN(**params,
                        trainweights = torch.log(1. / train.token_count()))

        tb_logger = pl_loggers.TensorBoardLogger(logpath, name="{}_{}".format(hparam, hparam_val))
        if hparam == 'gradient_clip_val':
            trainer = pl.Trainer(gradient_clip_val=hparam_val, logger=tb_logger, max_epochs=20, tpu_cores=tpu_cores, gpus=gpus)
        else:
            trainer = pl.Trainer(gradient_clip_val=0.5, logger=tb_logger, max_epochs=20, tpu_cores=tpu_cores, gpus=gpus)

        trainer.fit(model, dataloader)
        model.eval()
        result = trainer.test(model, dataloader)
        print('printing some example sentences from test set')
        print('--> format: sentence (true) [predicted]')
        for idx in np.random.randint(0, 1000, size=10):
            features, groundTruth = test[idx]
            fpass = model.forward(features.unsqueeze(dim=0))
            pred = np.argmax(torch.softmax(fpass.detach().squeeze(dim=0), 0))
            sentence = ''.join([test.decode_int(i) for i in features])
            nextword = test.decode_int(groundTruth)
            nextpred = test.decode_int(pred)
            print('{} ({}) [{}]'.format(sentence, nextword, nextpred))
    return
