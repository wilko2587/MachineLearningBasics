import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class my_corpus(Dataset):
    __doc__ = '''
    corpus class from HW1, repurposed to handle HW2 data.

    Inherits from torch.utils.data.Dataset, with the addition of __len__() and __getitem__() methods
    allowing efficient integration with pytorch models and a widely understandable usability
    '''

    def __init__(self, filename, windowlength=5):
        super().__init__()

        self._windowlength = windowlength

        print('importing data to memory...')
        token_list = list()

        with open(filename, 'r') as f:
            for line in f:
                toks = nltk.word_tokenize(line.strip().lower())
                for tok in toks:
                    token_list.append(tok)

        self._token_list = token_list

    def generate_embeddings(self):
        '''
        method to generate:
        1) a vocabulary dictionary. Consists of unique tokens, with a unique integer mapping for each token
        2) an embeddings space from the vocabulary using nn.Embedding
        '''
        unique_tokens = list(set(self._token_list))  # only keep uniques

        print('building corpus...')
        self._tokenmap = {unique_tokens[i]: i for i in range(len(unique_tokens))}

        print('generating embeddings...')
        self._embeds = nn.Embedding(len(self._tokenmap), 100) # 100 dimensional embedding space

    def embedding_size(self):
        return self._embeds.embedding_dim

    def vocab_size(self):
        return len(self._tokenmap)

    def inherit_embeddings(self, corpus):
        '''
        method to inherit the embeddings from another corpus.
        corpus must be an instance of my_corpus.
        This method replaces this class' instance of self._tokenmap with corpus._tokenmap
            and self._embeds with corpus._embeds
        NB: this class could contain vocabulary that's undefined in corpus._tokenmap. We will need
        to convert these tokens to <unk>
        '''
        assert isinstance(corpus, my_corpus)
        self._tokenmap = corpus._tokenmap
        self._embeds = corpus._embeds
        # convert tokens in self._token_list that don't exist in self._tokenmap into <unk>
        new_list = []
        for token in self._token_list:
            if token not in self._tokenmap.keys():
                new_list.append('<unk>')
            else:
                new_list.append(token)

    def __len__(self):
        return len(self._token_list) - self._windowlength # length is entire list of tokens leaving one
                                                            # windowlength on end

    def __getitem__(self, idx):
        tokens = self._token_list[idx:idx+self._windowlength]
        token_list = [self._tokenmap[each] for each in tokens]
        token_tensor = torch.tensor(token_list)
        target_token = self._token_list[idx+self._windowlength]
        target = self._tokenmap[target_token]
        return self._embeds(token_tensor).flatten(), \
               torch.tensor(target)

if __name__ == "__main__":
    corpus = my_corpus('wiki.train.txt')
