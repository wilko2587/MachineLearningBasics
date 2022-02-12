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


class RecurrentNet(pl.LightningModule):

    def __init__(self, context, embed_dim, vocab_size, hidden_size, num_layers):
        super(RecurrentNet, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=context*embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X):
        batch_size = X.size(0)
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        X = self.embed(X)
        X = torch.flatten(X, start_dim=1)

        output, hidden = self.rnn(X,hidden)
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)

        return output, hidden

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        data, label = batch
        logits, hidden = self.forward(data)
        # l2_norm = sum(p.pow(2.0).sum() for p in self.parameters()).item()
        # l1_norm = sum(p.abs().sum() for p in self.parameters()).item()
        loss = self.loss(logits, label)# + 0.001*l2_norm + 0.001*l1_norm
        tensorboard_logs = {'loss': {'train': loss.detach()}}
        self.log("training loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        data, label = batch
        logits, hidden  = self.forward(data)
        loss = self.loss(logits, label)
        tensorboard_logs = {'loss': {'val': loss.detach()}}
        self.log("validation loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        data, label = batch
        logits, hidden = self.forward(data)
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
    model = RecurrentNet(context=train.window, embed_dim=100, vocab_size=len(train.unique_tokens), hidden_size=100, num_layers=2)
    tb_logger = pl_loggers.TensorBoardLogger("./lightning_logs/", name="ff")
    trainer = pl.Trainer(logger=tb_logger, max_epochs=1)
    trainer.fit(model, dataloader)
    result = trainer.test(model, dataloader)
    print(result)
