from dataset import wiki_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class wiki_dataloader(pl.LightningDataModule):

    def __init__(self, datasets, batch_size):
        super().__init__()
        self.train_dataset = datasets[0]
        self.valid_dataset = datasets[1]
        self.test_dataset = datasets[2]

        self.token_map = datasets[0].token_map
        self.batch_size = batch_size
        self.setup() # remove before running this

    def setup(self, stage=None):
        self.train = [[each[0],each[1]] for each in self.train_dataset]
        self.val = [[each[0],each[1]] for each in self.valid_dataset]
        self.test = [[each[0],each[1]] for each in self.test_dataset]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=8)

if __name__ == "__main__":
    train = wiki_dataset('../wiki.train.txt', training=True, token_map='create')
    valid = wiki_dataset('../wiki.valid.txt', training=False, token_map=train.token_map)
    test = wiki_dataset('../wiki.test.txt', training=False, token_map=train.token_map)

    datasets = [train,valid,test]

    dataloader = wiki_dataloader(datasets=datasets, batch_size=100)

