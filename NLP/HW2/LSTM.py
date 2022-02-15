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


class LSTM1(pl.LightningModule):
    def __init__(self,
                 n_vocab,
                 embedding_size,
                 num_layers,
                 dropout=0,
                 lr = 1e-3):

        super(LSTM1, self).__init__()

        self.embed = nn.Embedding(n_vocab, embedding_size)

        self.lstm = nn.LSTM(input_size = embedding_size,
                            hidden_size = embedding_size,
                            num_layers = num_layers,
                            batch_first=False, dropout=dropout)

        self.fc = nn.Linear(embedding_size, n_vocab) #transpose of embedding layer; need same weights but transposed
        self.fc.weight = self.embed.weight # tie embeddings

        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        logits = self.fc(x) #logit from running x through linear layer
        return logits

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr)

    def _step(self, batch, batch_idx, logstring):
        data, label = batch
        logits = self(data)
        loss = self.loss(logits, label)
        tensorboard_logs = {'loss': {logstring: loss.detach()}}
        self.log("{} loss".format(logstring), loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")


def test_hparam(hparam, values = [], logpath="./LSTM_logs/", tpu_cores=None, gpus=1):
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

    # default LSTM params
    params = {'n_vocab':len(train.unique_tokens),
              'embedding_size':100,
              'num_layers':2,
              'dropout':0,
              'lr':1e-3}

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=64, num_workers=2)

    for hparam_val in values:

        if hparam != 'gradient_clip_val':
            params[hparam] = hparam_val

        # Make model and train
        model = LSTM1(**params)

        tb_logger = pl_loggers.TensorBoardLogger(logpath, name="{}_{}".format(hparam, hparam_val))
        if hparam == 'gradient_clip_val':
            trainer = pl.Trainer(gradient_clip_val=values, logger=tb_logger, max_epochs=20, tpu_cores=tpu_cores, gpus=gpus)
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

if __name__ == '__main__':
    test_hparam("lr", values=[1e-4, 1e-3])
