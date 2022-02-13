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

class rnn(pl.LightningModule):
    def __init__(self, n_vocab, embedding_size, hidden_size, num_layers, seq_size):
        super(rnn, self).__init__()

        self.n_vocab = n_vocab
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_size = seq_size

        self.prev_state = None

        self.embed = nn.Embedding(n_vocab, embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_vocab)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, prev_state):
        x = self.embed(x)
        x, hidden = self.rnn(x)
        x = x[:, -1, :]
        logits = self.fc(x)

        return logits, hidden

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        data, label = batch
        logits, self.prev_state = self.forward(data, self.prev_state)
        self.prev_state = [state.detach() for state in self.prev_state]  # holding onto the numbers, not the gradient
        # l2_norm = sum(p.pow(2.0).sum() for p in self.parameters()).item()
        # l1_norm = sum(p.abs().sum() for p in self.parameters()).item()
        loss = self.loss(logits, label)  # + 0.001*l2_norm + 0.001*l1_norm
        tensorboard_logs = {'loss': {'train': loss.detach()}}
        self.log("training loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        data, label = batch
        logits, self.prev_state = self.forward(data, self.prev_state)
        loss = self.loss(logits, label)
        tensorboard_logs = {'loss': {'val': loss.detach()}}
        self.log("validation loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        data, label = batch
        logits, self.prev_state = self.forward(data, self.prev_state)
        loss = self.loss(logits, label)
        tensorboard_logs = {'loss': {'test': loss.detach()}}
        self.log("test loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

if __name__ == '__main__':
    # Load datasets
    train = wiki_dataset('wiki.train.txt', training=True, token_map='create', window=30)
    valid = wiki_dataset('wiki.valid.txt', training=False, token_map=train.token_map, window=30)
    test = wiki_dataset('wiki.test.txt', training=False, token_map=train.token_map, window=30)
    datasets = [train, valid, test]

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=20)

    # Make model and train
    model = rnn(n_vocab=len(train.unique_tokens), embedding_size=100, hidden_size=100, num_layers=2, seq_size=30)
    tb_logger = pl_loggers.TensorBoardLogger("./lightning_logs/", name="ff")
    trainer = pl.Trainer(logger=tb_logger, max_epochs=1, gpus=1)
    trainer.fit(model, dataloader)
    result = trainer.test(model, dataloader)
    print(result)
