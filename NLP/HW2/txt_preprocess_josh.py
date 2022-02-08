import nltk
import torch
from nltk.corpus import stopwords
import regex as re
import time
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
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

    # def _generate_word_embeddings(self):
    #     embeddings = {}
    #     nullvector = [0.] * len(self._tokenmap) # initialising an empty word embedding
    #     for token in self._tokenmap:
    #         i = self._tokenmap[token]
    #         one_hot = nullvector.copy()
    #         one_hot[i] = 1.
    #         embeddings[token] = one_hot
    #     self._embeddings = embeddings

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
    #
    # def get_stopwords(self):
    #     '''
    #     uses nltk's built in functions to find the stopwords from the corpus.
    #     returns a list of the stopwords
    #     '''
    #
    #     stop_words = set(stopwords.words('english'))
    #     stopwords_list = [t for t in self._tokens if t.lower() in stop_words]
    #     return stopwords_list
    #
    # def tag_corpus(self):
    #     '''
    #     calls _tag_sequence to update the tokens stored within the corpus
    #     '''
    #
    #     self._token_list = self._tag_sequence(self._token_list)  # tag the tokens as required
    #     self._validdata = self._tag_sequence(self._validdata)
    #     self._testdata = self._tag_sequence(self._testdata)
    #
    #     unique_tokens = list(set(self._token_list))  # update the unique tokens in the corpus
    #     self._tokenmap = {unique_tokens[i]: i for i in range(len(unique_tokens))}
    #     return
    #
    # def _tag_sequence(self, sequence):
    #     '''
    #     method to take a list of tokens (sequence) and to replace years with <year>,
    #     days with <days>, decimals with <decimal> etc
    #     '''
    #
    #     # Now replace years, ints, decimals, days, numbers with tags
    #     sequence = [re.sub('^[12][0-9]{3}$*', '<year>', tok) for tok in sequence]  # tag years
    #     sequence = [re.sub('^[0-9]*$', '<integer>', tok) for tok in sequence]  # tag integers
    #     sequence = [re.sub('^[0-9]+\.+[0-9]*$', '<decimal>', tok) for tok in sequence]  # tag decimals
    #     sequence = [re.sub('(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
    #                        '<dayofweek>', tok) for tok in sequence]  # tag days of week
    #     sequence = [re.sub('(january|february|march|april|may|june|july|august|september|october|november|december)',
    #                        '<month>', tok) for tok in sequence]  # tag month
    #     sequence = [re.sub('^[0-9]+(st|nd|rd|th)',
    #                        '<days>', tok) for tok in sequence]  # tag days (in date) - can have errors in this
    #     sequence = [re.sub('^[0-9]', '<other>', tok) for tok in sequence]  # tag all remaining numbers
    #
    #     return sequence
    #
    # def get_counts(self):
    #     token_count = dict()
    #
    #     for each in self._tokens:
    #         token_count[each] = token_count.get(each, 0) + 1
    #
    #     self._tokencount = token_count
    #
    # def get_avg(self, l):
    #     '''
    #
    #     calculates average word length for a given list of words
    #     '''
    #     length = [len(tok) for tok in l]
    #     avg_length = sum(length) / len(length)
    #
    #     return avg_length

    # def wordvec_length(self):
    #     return len(self._embeddings)
#
#     def encode_as_ints(self, sequence):
#
#         int_represent = []
#         print('encode this sequence: %s' % sequence)
#         print('as a list of integers.')
#         sequence = sequence.lower()
#         tokens = nltk.word_tokenize(sequence)
#         tokens = self._tag_sequence(tokens)
#
#         for t in tokens:
#             try:
#                 int_represent.append(self._tokenmap[t])
#             except KeyError:
#                 int_represent.append(self._tokenmap['<unk>'])
#
#         return (int_represent)
#
#     def encode_as_text(self, int_represent):
#
#         text = ''
#         print('encode this list', int_represent)
#         print('as a text sequence.')
#         inv_map = {v: k for k, v in self._tokenmap.items()}  # inverted self._tokenmap (switch values/keys)
#
#         for i in int_represent:
#             token = inv_map[i]
#             text = text + ' ' + token
#
#         return (text)
#
#     def huggingface(self):
#         """
#         implementation of tokenization using Huggingface Wordpiece tokenization
#         """
#         print('Huggingface tokenization:')
#         unk_tokens = "<UNK>"
#         spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]
#         tokenizer = Tokenizer(WordPiece(unk_tokens=unk_tokens))
#         trainer = WordPieceTrainer(vocab_size=5000, special_tokens=spl_tokens)
#         tokenizer.pre_tokenizer = Whitespace()
#
#         tokenizer.train([f"wiki.train.txt"], trainer)  # training the tokenzier
#         tokenizer.save("./tokenizer-trained.json")
#         tokenizer = Tokenizer.from_file("./tokenizer-trained.json")
#         input_string = input("PLease enter a sentence to tokenize: ")
#         output = tokenizer.encode(input_string)
#
#         print("Tokenized text: ", output.tokens)
#
#
# def main():
#     corpus = my_corpus('wiki.train.txt')
#
#     corpus.huggingface()
#
#     text = input('Please enter a test sequence to encode and recover: ')
#     print(' ')
#     ints = corpus.encode_as_ints(text)
#     print(' ')
#     print('integer encoding: ', ints)
#
#     print(' ')
#     text = corpus.encode_as_text(ints)
#     print(' ')
#     print('this is the encoded text: %s' % text)
#

if __name__ == "__main__":
    main()
