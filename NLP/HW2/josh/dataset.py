import nltk
import torch
from torch.utils.data import Dataset

class wiki_dataset(Dataset):
    '''
    Dataset module.
    '''

    def __init__(self, file, training, token_map, window=5):
        '''
        File: data location
        training: True if training set, False otherwise. makes sure that no tokens get saved for val/test that are not
        in training.
        token_map: 'create' if training data, else provide it token map from training data
        window: sliding window length
        '''

        super().__init__()
        self.window = window
        unk = '<unk>'
        token_list = list()

        with open(file, 'r') as f:
            for line in f:
                toks = nltk.word_tokenize(line.strip().lower())
                for tok in toks:
                    if training == False:
                        if tok not in token_map:
                            token_list.append(unk) # for valid/test set, add <unk> if token not in training set
                        else:
                            token_list.append(tok)
                    else:
                        token_list.append(tok) # this is for training set

        self.tokens = token_list
        self.unique_tokens = list(set(self.tokens))

        if token_map == 'create': # create a token_map from training data
            self.token_map = {self.unique_tokens[i]: i for i in range(len(self.unique_tokens))}
        else:
            self.token_map = token_map

    def __len__(self):
        return len(self.tokens) # returns number of tokens

    def __getitem__(self, idx):
        # returns data, label where data is the idx of the tokens and label is idx of label
        tokens = self.tokens[idx:idx+self.window]
        token_idx_list = list()

        for each in tokens:
            token_idx_list.append(self.token_map[each])

        token_tensor = torch.tensor(token_idx_list, dtype=torch.long)

        target_token = self.tokens[idx+self.window]
        target = torch.tensor(self.token_map[target_token],dtype=torch.long)

        return [token_tensor,target]

if __name__ == "__main__":
    train = wiki_dataset('../wiki.train.txt', training=True, token_map='create')
    valid = wiki_dataset('../wiki.valid.txt', training=False, token_map=train.token_map)
    test = wiki_dataset('../wiki.test.txt', training=False, token_map=train.token_map)



