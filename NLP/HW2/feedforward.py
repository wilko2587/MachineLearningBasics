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

    def __init__(self, context, embed_dim, vocab_size,
                 dropout=0.8):
        super(FeedForward, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.embed.weight, a=-0.1, b=0.1) # initialise weights in range -0.1->0.1 with uniform distro
        self.lin1 = nn.Linear(context * embed_dim, embed_dim)
        nn.init.uniform_(self.lin1.weight, a=-0.1, b=0.1) # initialise weights in range -0.1->0.1 with uniform distro
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.drop1 = nn.Dropout(p=dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X):
        X = self.embed(X)
        X = torch.flatten(X, start_dim=1)
        X = torch.tanh(self.bn1(self.lin1(X)))
        X = self.drop1(X)
        X = torch.mm(X, self.embed.weight.transpose(1,0))
        return X

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        data, label = batch
        logits = self.forward(data)
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters()).item()
        l1_norm = sum(p.abs().sum() for p in self.parameters()).item()
        loss = self.loss(logits, label)# + 0.001*l2_norm + 0.001*l1_norm
        tensorboard_logs = {'loss': {'train': loss.detach()}}
        self.log("training loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        data, label = batch
        logits = self.forward(data)
        loss = self.loss(logits, label)
        tensorboard_logs = {'loss': {'val': loss.detach()}}
        self.log("validation loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        data, label = batch
        logits = self.forward(data)
        loss = self.loss(logits, label)
        tensorboard_logs = {'loss': {'test': loss.detach()}}
        self.log("test loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}


def test_dropout(dropouts = [0.2, 0.4, 0.6, 0.8], logpath="./FeedForward_logs/", tpu_cores=None, gpus=1): # Todo: put dropout as a feature in LSTM1

    # Load datasets
    train = wiki_dataset('./wiki.train.txt', training=True, token_map='create', window=30)
    valid = wiki_dataset('./wiki.valid.txt', training=False, token_map=train.token_map, window=30)
    test = wiki_dataset('./wiki.test.txt', training=False, token_map=train.token_map, window=30)
    datasets = [train, valid, test]

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=64, num_workers=8)

    for dropout in dropouts:

        # Make model and train
        model = FeedForward(context=train.window, embed_dim=100, vocab_size=len(train.unique_tokens),
                            hidden_size=1000,
                            dropout=dropout)

        tb_logger = pl_loggers.TensorBoardLogger("./LSTM_logs/", name="dropout_{}".format(dropout))
        trainer = pl.Trainer(gradient_clip_val=0.5, logger=tb_logger, max_epochs=20, tpu_cores=tpu_cores, gpus=gpus)

        trainer.fit(model, dataloader)
        result = trainer.test(model, dataloader)
    return


if __name__ == '__main__':
    # Load datasets
    train = wiki_dataset('./wiki.train.txt', training=True, token_map='create')
    valid = wiki_dataset('./wiki.valid.txt', training=False, token_map=train.token_map)
    test = wiki_dataset('./wiki.test.txt', training=False, token_map=train.token_map)
    datasets = [train, valid, test]

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=20)

    # Make model and train
    model = FeedForward(context=train.window, embed_dim=100, vocab_size=len(train.unique_tokens))
    tb_logger = pl_loggers.TensorBoardLogger("./lightning_logs/", name="ff")
    trainer = pl.Trainer(logger=tb_logger, max_epochs=10)
    trainer.fit(model, dataloader)
    result = trainer.test(model, dataloader)
    print(result)
