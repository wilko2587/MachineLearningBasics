import nltk
import torch
from torch.utils.data import Dataset
import re
from collections import Counter
import pandas as pd
from nltk.corpus import stopwords
nltk.download('stopwords')

class wiki_dataset(Dataset):

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

        tokenizer = nltk.RegexpTokenizer(r"\w+|[\.|\,|\(|\)|\?|\!]")#|(\.|\,|\(|\)|\?|\!)") # new tokenizer ignores any punctuation
        # stop_words = set(stopwords.words('english'))

        with open(file, 'r') as f:
            for line in f:
                #toks = nltk.word_tokenize(line.strip().lower())
                toks = tokenizer.tokenize(line.lower())
                # toks = self._tag_sequence(toks)
                for tok in toks:
                    # if tok not in stop_words:
                    if training == False:
                        if tok not in token_map:
                            token_list.append(unk) # for valid/test set, add <unk> if token not in training set
                        else:
                            token_list.append(tok)
                    else:
                        token_list.append(tok) # this is for training set

        token_list.append(unk)

        self.tokens = token_list
        self.unique_tokens = list(set(self.tokens))

        if token_map == 'create': # create a token_map from training data
            self.token_map = {self.unique_tokens[i]: i for i in range(len(self.unique_tokens))}
        else:
            self.token_map = token_map

    def __len__(self):
        return len(self.tokens) - self.window # returns number of tokens

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

    def token_count(self):
        '''
        returns histogram of tokens found in self.token_list
        '''

        histogram = pd.Series(Counter(self.tokens))
        histogram = histogram.reindex(self.token_map.keys(), fill_value=0)
        return torch.tensor(histogram.values)

    def _tag_sequence(self, sequence):
        '''
        method to take a list of tokens (sequence) and to replace years with <year>,
        days with <days>, decimals with <decimal> etc
        '''

        # Now replace years, ints, decimals, days, numbers with tags
        sequence = [re.sub('^[12][0-9]{3}$', '<year>', tok) for tok in sequence]  # tag years
        sequence = [re.sub('^[0-9]+', '<integer>', tok) for tok in sequence]  # tag integers
        sequence = [re.sub('^[0-9]+\.+[0-9]*$', '<decimal>', tok) for tok in sequence]  # tag decimals
        sequence = [re.sub('(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
                           '<dayofweek>', tok) for tok in sequence]  # tag days of week
        sequence = [re.sub('(january|february|march|april|may|june|july|august|september|october|november|december)',
                           '<month>', tok) for tok in sequence]  # tag month
        sequence = [re.sub('^[0-9]+(st|nd|rd|th)',
                           '<days>', tok) for tok in sequence]  # tag days (in date) - can have errors in this
        sequence = [re.sub('^[0-9]', '<other>', tok) for tok in sequence]  # tag all remaining numbers
        sequence = [re.sub('(unk)', '<unk>', tok) for tok in sequence] # reformat the unks to <unk> (previously "unk")
        return sequence

    def decode_int(self, num):
        inv_token_map = {v: k for k, v in self.token_map.items()}
        return inv_token_map[int(num)]

if __name__ == "__main__":
    train = wiki_dataset('./wiki.train.txt', training=True, token_map='create')
    valid = wiki_dataset('./wiki.valid.txt', training=False, token_map=train.token_map)
    test = wiki_dataset('./wiki.test.txt', training=False, token_map=train.token_map)


