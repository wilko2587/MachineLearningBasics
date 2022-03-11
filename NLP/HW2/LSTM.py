import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from torch import optim
from dataset import wiki_dataset
from dataloader import wiki_dataloader
import pytorch_lightning.loggers as pl_loggers
torch.manual_seed(1)


class LSTM1(pl.LightningModule):
    def __init__(self,
                 n_vocab,
                 embedding_size,
                 num_layers,
                 dropout=0,
                 lr=1e-3,
                 trainweights=None):
        super(LSTM1, self).__init__()

        self.embed = nn.Embedding(n_vocab, embedding_size)

        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=embedding_size,
                            num_layers=num_layers,
                            batch_first=False, dropout=dropout)

        self.fc = nn.Linear(embedding_size, n_vocab)  # transpose of embedding layer; need same weights but transposed
        self.fc.weight = self.embed.weight  # tie embeddings

        self.loss = nn.CrossEntropyLoss(weight=trainweights)
        self.viewloss = nn.CrossEntropyLoss()  # weights change the results of the loss, so we initialise an unweighted
        # loss to keep track and use for perplexity calcs
        self.lr = lr

        self.state = None # initialise

    def forward(self, x):
        x = self.embed(x)
        x, state = self.lstm(x, self.state)
        self.state = (state[0].detach(), state[1].detach())
        # x = x[:, -1, :]
        logits = self.fc(x)  # logit from running x through linear layer
        return logits

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def _step(self, batch, batch_idx, logstring):
        data, label = batch
        label = torch.cat([data[:,1:], label.unsqueeze(dim=1)], dim=1)
        logits = self(data)
        loss = self.loss(logits.flatten(start_dim=0, end_dim=1), label.flatten(start_dim=0, end_dim=1))
        viewloss = self.viewloss(logits.flatten(start_dim=0, end_dim=1), label.flatten(start_dim=0, end_dim=1))
        tensorboard_logs = {'loss': {logstring: viewloss.detach()}}
        self.log("{} loss".format(logstring), viewloss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def predict(self, x):
        '''
        return a single predicton from a feature vector
        '''
        fpass = self.forward(x.unsqueeze(dim=0))[:, -1, :]
        return np.argmax(torch.softmax(fpass.detach().squeeze(dim=0), 0))


def test_hparam(hparam, values=[], logpath="./LSTM_logs/", tpu_cores=None, gpus=1):
    '''

    pretty neat function to test a hparam (as titled in LSTM1.__init__()) between a set of values.
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

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=20, num_workers=2, unk_threshold=0.4)

    # default LSTM params
    params = {'n_vocab': len(train.unique_tokens),
              'embedding_size': 100,
              'num_layers': 2,
              'dropout': 0,
              'lr': 1e-3}

    for hparam_val in values:

        if hparam != 'gradient_clip_val':
            params[hparam] = hparam_val

        # Make model and train
        model = LSTM1(**params,
                      trainweights=torch.log(1. / train.token_count()))

        tb_logger = pl_loggers.TensorBoardLogger(logpath, name="{}_{}".format(hparam, hparam_val))
        if hparam == 'gradient_clip_val':
            trainer = pl.Trainer(gradient_clip_val=hparam_val, logger=tb_logger, max_epochs=20, tpu_cores=tpu_cores,
                                 gpus=gpus)
        else:
            trainer = pl.Trainer(gradient_clip_val=0, logger=tb_logger, max_epochs=20, tpu_cores=tpu_cores, gpus=gpus)

        trainer.fit(model, dataloader)
        model.eval()
        result = trainer.test(model, dataloader)
        print('printing some example sentences from test set')
        print('--> format: sentence (true) [predicted]')
        for idx in np.random.randint(0, 1000, size=10):
            features, groundTruth = test[idx]
            pred = model.predict(features)
            sentence = ''.join([test.decode_int(i) for i in features])
            nextword = test.decode_int(groundTruth)
            nextpred = test.decode_int(pred)
            print('{} ({}) [{}]'.format(sentence, nextword, nextpred))
    return

def run_LSTM(lr=1e-3, dropout=0, grad_clipping=0, punctuation_allowed=True, unk_threshold=1, logdir='FinalModel',
             tpu_cores=8, gpus=None, max_epochs=20):
    # Load datasets
    train = wiki_dataset('./wiki.train.txt', training=True, token_map='create', window=30)
    valid = wiki_dataset('./wiki.valid.txt', training=False, token_map=train.token_map, window=30)
    test = wiki_dataset('./wiki.test.txt', training=False, token_map=train.token_map, window=30)
    datasets = [train, valid, test]

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=20, num_workers=2, unk_threshold=unk_threshold)

    # default LSTM params
    params = {'n_vocab': len(train.unique_tokens),
              'embedding_size': 100,
              'num_layers': 2,
              'dropout': dropout,
              'lr': lr}

    # Make model and train
    model = LSTM1(**params,
                  trainweights=torch.log(1. / train.token_count()))

    tb_logger = pl_loggers.TensorBoardLogger(logdir, name="{}_{}_{}".format(lr, dropout, grad_clipping))
    trainer = pl.Trainer(gradient_clip_val=grad_clipping, logger=tb_logger, max_epochs=20, tpu_cores=tpu_cores,
                         gpus=gpus)

    trainer.fit(model, dataloader)
    model.eval()
    result = trainer.test(model, dataloader)
    print('printing some example sentences from test set')
    print('--> format: sentence (true) [predicted]')
    for idx in np.random.randint(0, 1000, size=10):
        features, groundTruth = test[idx]
        pred = model.predict(features)
        sentence = ''.join([test.decode_int(i) for i in features])
        nextword = test.decode_int(groundTruth)
        nextpred = test.decode_int(pred)
        print('{} ({}) [{}]'.format(sentence, nextword, nextpred))


if __name__ == '__main__':
    # Load datasets
    train = wiki_dataset('./wiki.train.txt', training=True, token_map='create', window=30)
    valid = wiki_dataset('./wiki.valid.txt', training=False, token_map=train.token_map, window=30)
    test = wiki_dataset('./wiki.test.txt', training=False, token_map=train.token_map, window=30)
    datasets = [train, valid, test]

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=64, num_workers=2)

    params = {'n_vocab': len(train.unique_tokens),
              'embedding_size': 100,
              'num_layers': 2,
              'dropout': 0,
              'lr': 1e-3}

    model = LSTM1(**params,
                  trainweights=torch.log(1. / train.token_count()))

    #tb_logger = pl_loggers.TensorBoardLogger('./Logs/', name="logname")
    #trainer = pl.Trainer(gradient_clip_val=0, logger=tb_logger, max_epochs=20, tpu_cores=None, gpus=None)

    #trainer.fit(model, dataloader)
    print('--> format: sentence (true) [predicted]')
    for idx in np.random.randint(0, 1000, size=10):
        features, groundTruth = test[idx]
        pred = model.predict(features)
        sentence = ' '.join([test.decode_int(i) for i in features])
        nextword = test.decode_int(groundTruth)
        nextpred = test.decode_int(pred)
        print('{} ({}) [{}]'.format(sentence, nextword, nextpred))
