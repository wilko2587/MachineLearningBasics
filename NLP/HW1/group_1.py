import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import regex as re
import time
import pandas as pd
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

nltk.download('punkt')
nltk.download('stopwords')


class my_corpus():

    def __init__(self, params):
        super().__init__()

        print('setting parameters')
        self.params = params

        print('building corpus')
        token_list = list()
        sentence_list = list()

        with open(('source_text.txt')) as f:
            for line in f:
                toks = nltk.word_tokenize(line.strip().lower())
                sent_toks = nltk.sent_tokenize(line.strip().lower())

                for tok in toks:
                    token_list.append(tok)
                for tok in sent_toks:
                    sentence_list.append(tok)

        unique_tokens = list(set(token_list))  # only keep uniques

        print('building token map')
        self._tokens = token_list
        self._tokenmap = {unique_tokens[i]: i for i in range(len(unique_tokens))}

        # self._tokenmap always needs an <unk>
        self._tokenmap['<unk>'] = len(unique_tokens)
        self._unktokens = []  # initialise container. List holding tokens that get mapped to <unk>

    def get_stopwords(self):
        '''
        uses nltk's built in functions to find the stopwords from the corpus.
        returns a list of the stopwords
        '''

        stop_words = set(stopwords.words('english'))
        stopwords_list = [t for t in self._tokens if t.lower() in stop_words]
        return stopwords_list

    def tag_corpus(self):
        '''
        calls _tag_sequence to update the tokens stored within the corpus
        '''

        token_list = self._tag_sequence(self._tokens)  # tag the tokens as required

        unique_tokens = list(set(token_list))  # only keep uniques

        self._tokens = token_list
        self._tokenmap = {unique_tokens[i]: i for i in range(len(unique_tokens))}

        # self._tokenmap always needs an <unk>
        self._tokenmap['<unk>'] = len(unique_tokens)
        return

    def _tag_sequence(self, sequence):
        '''
        method to take a list of tokens (sequence) and to replace years with <year>,
        days with <days>, decimals with <decimal> etc
        '''

        # Now replace years, ints, decimals, days, numbers with tags
        sequence = [re.sub('^[12][0-9]{3}$*', '<year>', tok) for tok in sequence]  # tag years
        sequence = [re.sub('^[0-9]*$', '<integer>', tok) for tok in sequence]  # tag integers
        sequence = [re.sub('^[0-9]+\.+[0-9]*$', '<decimal>', tok) for tok in sequence]  # tag decimals
        sequence = [re.sub('(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
                           '<dayofweek>', tok) for tok in sequence]  # tag days of week
        sequence = [re.sub('(january|february|march|april|may|june|july|august|september|october|november|december)',
                           '<month>', tok) for tok in sequence]  # tag month
        sequence = [re.sub('^[0-9]+(st|nd|rd|th)',
                           '<days>', tok) for tok in sequence]  # tag days (in date) - can have errors in this
        sequence = [re.sub('^[0-9]', '<other>', tok) for tok in sequence]  # tag all remaining numbers

        return sequence

    def get_counts(self):
        token_count = dict()

        for each in self._tokens:
            token_count[each] = token_count.get(each, 0) + 1

        self._tokencount = token_count

    def generate_datasets(self, split=None, reset_vocab_to_training=True):
        '''
        returns three corpi, for training, validation and testing.
        :param split: list of three, determining the split. ie: [80,10,10] for an 80%, 10%, 10% split
        '''
        if split is None:
            split = [80, 10, 10]

        # lets use the first 80% of tokens as training, the next 10% as valid, and the final 10% as test
        # Ideally, we don't want to split a single sentence between different data sets
        # because we want each each dataset to be as self-contained as possible, and to make sense as
        # a stand-alone piece. We know sentences occur often, so lets find the indices of a perfect 80/10/10 split,
        # and "round-up" our indices to the end-of-sentence.

        Nwords = len(self._tokens)
        trainend_index = int(
            Nwords * float(split[0]) / sum(split))  # index of where the training data ends (and valid starts)
        validend_index = int(
            Nwords * float(sum(split[0:2])) / sum(split))  # index of where the valid data ends (and test starts)

        d1 = self._tokens[trainend_index:].index('.')  # number of extra tokens we need to move trainend_index to get
        # to the end of the sentence

        d2 = self._tokens[validend_index:].index('.')  # number of extra tokens we need to move validend_index to get
        # to the end of the sentence

        # round up trainend_index and validend_index accordingly
        trainend_index += d1
        validend_index += d2

        traindata = self._tokens[:trainend_index]
        validdata = self._tokens[trainend_index:validend_index]
        testdata = self._tokens[validend_index:]

        self._traindata = traindata
        self._validdata = validdata
        self._testdata = testdata

        if reset_vocab_to_training:
            # this will set the corpus to the training set. validation/test data still held in self._validdata etc
            print("NB: setting corpus to the training set")
            unique_tokens = list(set(traindata))
            self._tokens = traindata
            self._tokenmap = {unique_tokens[i]: i for i in range(len(unique_tokens))}

        return traindata, validdata, testdata

    def get_avg(self, l):
        '''
        calculates average word length for a given list of words
        '''
        length = [len(tok) for tok in l]
        avg_length = sum(length) / len(length)

        return avg_length

    def print_summary_stats(self):
        '''
        print out some basic summary stats of the corpus
        '''
        train, valid, test = self.generate_datasets()
        print('======')
        print("Printing summary statistics:")
        stats = pd.DataFrame(
            {"Metric":
                 ["Number of tokens in training data",
                  "Number of tokens in validation data",
                  "Number of tokens in test data",
                  "Size of vocabulary",
                  "Number of <unk> tokens",
                  "Number of stopwords",
                  "Size of <unk> vocab",
                  "Number of validation data tokens not in training data",
                  "Number of test data tokens not in training data",
                  "Average word length in training data",
                  "Average word length in validation data",
                  "Average word length in test data",
                  "Number of training data tokens not in validation and test data"],
             "Result":
                 [round(len(train), 2),
                  round(len(valid), 2),
                  round(len(test), 2),
                  round(len(self._tokenmap), 2),
                  round(self._tokens.count("<unk>"), 2),
                  round(len(self.get_stopwords()), 2),
                  round(len(self._unktokens), 2),
                  round(len([t for t in self._validdata if t not in self._traindata]), 2),
                  round(len([t for t in self._testdata if t not in self._traindata]), 2),
                  round(self.get_avg(train), 2),
                  round(self.get_avg(valid), 2),
                  round(self.get_avg(test), 2),
                  round(len([t for t in self._traindata if (t not in self._validdata) and (t not in self._testdata)]),
                        2)
                  ]
             }
        )

        print(stats)

    def encode_as_ints(self, sequence):

        int_represent = []
        print('encode this sequence: %s' % sequence)
        print('as a list of integers.')
        sequence = sequence.lower()
        tokens = nltk.word_tokenize(sequence)
        tokens = self._tag_sequence(tokens)

        for t in tokens:
            try:
                int_represent.append(self._tokenmap[t])
            except KeyError:
                int_represent.append(self._tokenmap['<unk>'])

        return (int_represent)

    def encode_as_text(self, int_represent):

        text = ''
        print('encode this list', int_represent)
        print('as a text sequence.')
        inv_map = {v: k for k, v in self._tokenmap.items()}  # inverted self._tokenmap (switch values/keys)

        for i in int_represent:
            token = inv_map[i]
            text = text + ' ' + token

        return (text)

    def threshold(self, threshold):
        '''
        Takes a threshold value, parses token list, replaces those below threshold with <UNK>
        Creates new list at self._tokens_threshold.
        Create new list with any removed tokens at self._tokens_removed
        '''

        unk = '<unk>'

        tokens_threshold = pd.Series(self._tokens.copy())  # put through pandas to get on C level
        saved_old_tokens = self._tokens.copy()
        token_counts = Counter(self._tokens)  # histogram of token counts (done on C level super fast!)
        sorted_counts = dict(sorted(token_counts.items(), key=lambda item: item[1]))  # histogram, but sorted ascending
        thresh_index = next(i for i, v in enumerate(sorted_counts.values()) if
                            v >= threshold)  # index where counts becomes geq than threshold
        unk_tokens = list(sorted_counts.keys())[0:thresh_index]  # all the tokens whose counts were less than threshold
        tokens_threshold[tokens_threshold.isin(unk_tokens)] = unk
        tokens_threshold = tokens_threshold.to_list()

        unique_tokens = list(set(tokens_threshold))
        # reset self._tokens and self._tokenmap accordingly
        self._tokenmap = {unique_tokens[i]: i for i in range(len(unique_tokens))}
        self._tokens = tokens_threshold
        self._unktokens = unk_tokens
        return

    def huggingface(self):
        """
        implementation of tokenization using Huggingface Wordpiece tokenization
        """
        print('Huggingface tokenization:')
        unk_tokens = "<UNK>"
        spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]
        tokenizer = Tokenizer(WordPiece(unk_tokens=unk_tokens))
        trainer = WordPieceTrainer(vocab_size=5000, special_tokens=spl_tokens)
        tokenizer.pre_tokenizer = Whitespace()

        tokenizer.train([f"source_text.txt"], trainer)  # training the tokenzier
        tokenizer.save("./tokenizer-trained.json")
        tokenizer = Tokenizer.from_file("./tokenizer-trained.json")
        input_string = input("PLease enter a sentence to tokenize: ")
        output = tokenizer.encode(input_string)

        print("Tokenized text: ", output.tokens)


def main():
    corpus = my_corpus(None)

    t0 = time.time()
    corpus.tag_corpus()
    t1 = time.time()
    print('tag time taken: ', t1 - t0)

    t0 = time.time()
    corpus.threshold(3)
    t1 = time.time()
    print('threshold time taken: ', t1 - t0)

    # split corpus into training/validation/test.
    # Redefine corpus to just be the training portion
    train, valid, test = corpus.generate_datasets(split=[80, 10, 10],
                                                  reset_vocab_to_training=True)

    # write to txt
    with open('train.txt', 'w') as f:
        f.write(' '.join(train))
    with open('valid.txt', 'w') as f:
        f.write(' '.join(valid))
    with open('test.txt', 'w') as f:
        f.write(' '.join(test))

    # print some summary stats
    corpus.print_summary_stats()

    corpus.huggingface()

    text = input('Please enter a test sequence to encode and recover: ')
    print(' ')
    ints = corpus.encode_as_ints(text)
    print(' ')
    print('integer encodeing: ', ints)

    print(' ')
    text = corpus.encode_as_text(ints)
    print(' ')
    print('this is the encoded text: %s' % text)


if __name__ == "__main__":
    main()
