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


class FeedForward(pl.LightningModule):

    def __init__(self, context, embed_dim, vocab_size):
        super(FeedForward, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lin1 = nn.Linear(context * embed_dim, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.drop1 = nn.Dropout(p=0.5)
        self.lin2 = nn.Linear(1000, vocab_size)

        #l2_norm = sum(p.pow(2.0).sum() for p in self.parameters()).item()

        self.loss = nn.CrossEntropyLoss() #+ l2_norm

    def forward(self, X):
        X = self.embed(X)
        X = torch.flatten(X, start_dim=1)
        X = torch.tanh(self.bn1(self.lin1(X)))
        X = self.drop1(X)
        X = self.lin2(X)
        return X

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        data, label = batch
        logits = self.forward(data)
        loss = self.loss(logits, label)
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

if __name__ == '__main__':
    # Load datasets
    train = wiki_dataset('../wiki.train.txt', training=True, token_map='create')
    valid = wiki_dataset('../wiki.valid.txt', training=False, token_map=train.token_map)
    test = wiki_dataset('../wiki.test.txt', training=False, token_map=train.token_map)
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
