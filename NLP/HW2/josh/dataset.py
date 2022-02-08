import nltk
import torch
from torch.utils.data import Dataset


class my_corpus(Dataset):
    '''
    Differences from existing code:
    Only kept self._tokens, self._tokenmap.
    Len returns len(self._tokenmap.
    Get item returns ([tokens of sliding window length], integer index from self._tokenmap of target).
    Plan to use embedding layer in PyTorch. This will allow us to update the weights in this embedding layer
    after each minibatch.
    Commented out other stuff.
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

        self._tokens = token_list
        unique_tokens = list(set(self._tokens))  # only keep uniques

        print('building corpus...')
        self._tokenmap = {unique_tokens[i]: i for i in range(len(unique_tokens))}

        print('generating embeddings...')
        # self._generate_word_embeddings()

        print('{} corpus initialisation complete.'.format(filename))

    def __len__(self):
        return len(self._tokenmap)

    def __getitem__(self, idx):
        tokens = self._tokens[idx:idx+self._windowlength]
        token_list = list()

        for each in tokens:
            token_list.append(self._tokenmap[each])
        token_tensor = torch.tensor(token_list)
        target_token = self._tokens[idx+self._windowlength]
        target = self._tokenmap[target_token]
        return (token_tensor,target)

if __name__ == "__main__":
    main()
