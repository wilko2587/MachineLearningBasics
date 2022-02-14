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
import nltk
nltk.download('punkt')


class FeedForward(pl.LightningModule):

    def __init__(self, context,
                 embed_dim,
                 vocab_size,
                 dropout=0.8,
                 lr = 1e-3):

        super(FeedForward, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lr = lr
        nn.init.uniform_(self.embed.weight, a=-0.1, b=0.1) # initialise weights in range -0.1->0.1 with uniform distro
        self.lin1 = nn.Linear(context * embed_dim, embed_dim)
        nn.init.uniform_(self.lin1.weight, a=-0.1, b=0.1) # initialise weights in range -0.1->0.1 with uniform distro
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.drop1 = nn.Dropout(p=dropout)
        self.loss = nn.CrossEntropyLoss()
        self.lin2 = nn.Linear(embed_dim, vocab_size)
        self.lin2.weight = self.embed.weight # tied embeddings

    def forward(self, X):
        X = self.embed(X)
        X = torch.flatten(X, start_dim=1)
        X = torch.tanh(self.bn1(self.lin1(X)))
        X = self.drop1(X)
        X = self.lin2(X)
        return X

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def _step(self, batch, batch_idx, label):
        data, label = batch
        logits = self(data)
        loss = self.loss(logits, label)
        tensorboard_logs = {'loss': {label: loss.detach()}}
        self.log("{} loss".format(label), loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")


def test_hparam(hparam, values = [], logpath="./FeedForward_logs/", tpu_cores=None, gpus=1):
    '''

    pretty neat function to test a hparam (as titled in FeedForward.__init__()) between a set of values.
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

    # default feedforward params
    params = {'context': train.window,
              'embed_dim':100,
              'vocab_size':len(train.unique_tokens),
              'dropout':0,
              'lr':1e-3}

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=64, num_workers=8)

    for hparam_val in values:

        params[hparam] = hparam_val
        # Make model and train
        model = FeedForward(**params)

        tb_logger = pl_loggers.TensorBoardLogger(logpath, name="{}_{}".format(hparam, hparam_val))
        trainer = pl.Trainer(gradient_clip_val=0.5, logger=tb_logger, max_epochs=20, tpu_cores=tpu_cores, gpus=gpus)

        trainer.fit(model, dataloader)
        result = trainer.test(model, dataloader)
    return
