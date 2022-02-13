import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from torch import optim
from dataset import wiki_dataset
from dataloader import wiki_dataloader
import pytorch_lightning.loggers as pl_loggers


class LSTM1(pl.LightningModule):
    def __init__(self, n_vocab,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 seq_size):
        super(LSTM1, self).__init__()
        self.seq_size = seq_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.lstm = nn.LSTM(input_size = embedding_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first=True, dropout=0.5)
        # nn.utils.clip_grad_norm_(self.lstm.parameters(), clip) - between backwards and optimizer step
        self.prev_state = None
        self.embed = nn.Embedding(n_vocab, embedding_size)
        self.loss = nn.CrossEntropyLoss()
        self.fc = nn.Linear(hidden_size, n_vocab) #transpose of embedding layer; need same weights but transposed

    def forward(self, x, prev_state):
        x = self.embed(x)
        #x = torch.flatten(x, start_dim=1)
        if self.prev_state is None:
            x , state = self.lstm(x)
        else:
            x, state = self.lstm(x, prev_state)
        x = x[:, -1, :]
        logits = self.fc(x) #logit from running x through linear layer

        return logits, state

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        data, label = batch
        logits, self.prev_state = self.forward(data, self.prev_state)
        self.prev_state = [state.detach() for state in self.prev_state] #holding onto the numbers, not the gradient
        #need to move tensors back to gpu/cuda
        #print(f'{logits.shape}, {label.shape}')
        # l2_norm = sum(p.pow(2.0).sum() for p in self.parameters()).item()
        # l1_norm = sum(p.abs().sum() for p in model.parameters()).item()
        loss = self.loss(logits, label)  # + l2_norm + l1_norm
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


def test_hiddensize(sizes=[50,100,200,500], logpath="./LSTM_logs/"):
    # Load datasets
    train = wiki_dataset('./wiki.train.txt', training=True, token_map='create', window=30)
    valid = wiki_dataset('./wiki.valid.txt', training=False, token_map=train.token_map, window=30)
    test = wiki_dataset('./wiki.test.txt', training=False, token_map=train.token_map, window=30)
    datasets = [train, valid, test]

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=20)

    for hidden_size in sizes:

        # Make model and train
        model = LSTM1(n_vocab=len(train.unique_tokens),
                  num_layers=2,
                  seq_size=30,
                  embedding_size=100,
                  hidden_size=hidden_size)

        tb_logger = pl_loggers.TensorBoardLogger(logpath, name="hiddensize_{}".format(hidden_size))
        trainer = pl.Trainer(gradient_clip_val=0.5, logger=tb_logger, max_epochs=20)

        trainer.fit(model, dataloader)
        result = trainer.test(model, dataloader)
    return


def test_dropout(): # Todo: put dropout as a feature in LSTM1

    # Load datasets
    train = wiki_dataset('./wiki.train.txt', training=True, token_map='create', window=30)
    valid = wiki_dataset('./wiki.valid.txt', training=False, token_map=train.token_map, window=30)
    test = wiki_dataset('./wiki.test.txt', training=False, token_map=train.token_map, window=30)
    datasets = [train, valid, test]

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=20)

    for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:

        # Make model and train
        model = LSTM1(n_vocab=len(train.unique_tokens),
                  num_layers=2,
                  seq_size=30,
                  embedding_size=100,
                  hidden_size=100)

        tb_logger = pl_loggers.TensorBoardLogger("./LSTM_logs/", name="dropout_{}".format(dropout))
        trainer = pl.Trainer(gradient_clip_val=0.5, logger=tb_logger, max_epochs=20)

        trainer.fit(model, dataloader)
        result = trainer.test(model, dataloader)
    return


if __name__ == '__main__':
    # Load datasets
    train = wiki_dataset('./wiki.train.txt', training=True, token_map='create', window=30)
    valid = wiki_dataset('./wiki.valid.txt', training=False, token_map=train.token_map, window=30)
    test = wiki_dataset('./wiki.test.txt', training=False, token_map=train.token_map, window=30)
    datasets = [train, valid, test]

    # Load dataloader
    dataloader = wiki_dataloader(datasets=datasets, batch_size=20)

    # Make model and train
    model = LSTM1(n_vocab=len(train.unique_tokens),
              num_layers=2,
              seq_size=30,
              embedding_size=100,
              hidden_size=100)

    tb_logger = pl_loggers.TensorBoardLogger("./LSTM_logs/", name="LSTM")
    trainer = pl.Trainer(gradient_clip_val=0.5, logger=tb_logger, max_epochs=10)
    trainer.fit(model, dataloader)
    result = trainer.test(model, dataloader)
