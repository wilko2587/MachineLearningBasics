from dataset import wiki_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch


class wiki_dataloader(pl.LightningDataModule):

    def __init__(self, datasets, batch_size, num_workers=2, unk_threshold=1.):
        '''
        datasets: expects [train,valid,test]

        unk_threshold = threshold for the % of unks allowed in a sequence. eg: 0.4 = drops all samples
        where more than 40% of the sentence is made of "unk". Default "1" = no effect
        '''
        super().__init__()
        self.num_workers = num_workers

        self.train_dataset = datasets[0]
        self.valid_dataset = datasets[1]
        self.test_dataset = datasets[2]

        self.token_map = datasets[0].token_map
        self.batch_size = batch_size

        self.train = [[each[0], each[1]] for each in self.train_dataset if
                      torch.sum(each[0] == datasets[0].token_map['<unk>']) <= int(unk_threshold * datasets[0].window)]
        self.val = [[each[0], each[1]] for each in self.valid_dataset if
                    torch.sum(each[0] == datasets[0].token_map['<unk>']) <= int(unk_threshold * datasets[0].window)]
        self.test = [[each[0], each[1]] for each in self.test_dataset if
                     torch.sum(each[0] == datasets[0].token_map['<unk>']) <= int(unk_threshold * datasets[0].window)]

#        self.train = [[each[0], each[1]] for each in self.train_dataset]
#        self.val = [[each[0], each[1]] for each in self.valid_dataset]
#        self.test = [[each[0], each[1]] for each in self.test_dataset]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    train = wiki_dataset('./wiki.train.txt', training=True, token_map='create', window=5)
    valid = wiki_dataset('./wiki.valid.txt', training=False, token_map=train.token_map, window=5)
    test = wiki_dataset('./wiki.test.txt', training=False, token_map=train.token_map, window=5)

    datasets = [train, valid, test]

    dataloader = wiki_dataloader(datasets=datasets, batch_size=64, unk_threshold=1)
    # dataloader2 = wiki_dataloader(datasets=datasets, batch_size=64, unk_threshold=0.1)
