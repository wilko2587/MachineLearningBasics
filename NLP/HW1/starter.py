import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import regex as re

nltk.download('punkt')

class my_corpus():

    def __init__(self, params):
        super().__init__()

        print('setting parameters')
        self.params = params

        print('building corpus')
        token_list = list()

        with open(('source_text.txt')) as f:
            for line in f:
                toks = nltk.word_tokenize(line.strip().lower())
                for tok in toks:
                    token_list.append(tok)

        unique_tokens = list(set(token_list)) # only keep uniques

        print('building token map')
        self._tokens = token_list
        self._tokenmap = {unique_tokens[i]:i for i in range(len(unique_tokens))}

        # self._tokenmap always needs an <unk>
        self._tokenmap['<unk>'] = len(unique_tokens)

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

        token_list = self._tag_sequence(self._tokens) # tag the tokens as required

        unique_tokens = list(set(token_list)) # only keep uniques

        self._tokens = token_list
        self._tokenmap = {unique_tokens[i]:i for i in range(len(unique_tokens))}

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
        sequence = [re.sub('(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
                              '<dayofweek>', tok) for tok in sequence]  # tag days of week
        sequence = [re.sub('(Janurary|February|March|April|May|June|July|August|September|October|November|December)',
                              '<month>', tok) for tok in sequence]  # tag month
        sequence = [re.sub('^[0-9]+(st|nd|rd|th)',
                              '<days>', tok) for tok in sequence]  # tag days (in date) - can have errors in this
        sequence = [re.sub('^[0-9]', '<other>', tok) for tok in sequence]  # tag all remaining numbers

        return sequence


    def get_counts(self):
        token_count = dict()

        for each in self._tokens:
            token_count[each] = token_count.get(each,0) + 1

        self._tokencount = token_count


    def generate_datasets(self, split=None):
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
        trainend_index = int(Nwords*float(split[0]/sum(split))) # index of where the training data ends (and valid starts)
        validend_index = int(Nwords*float(sum(split[0:1])/sum(split))) # index of where the valid data ends (and test starts)

        d1 = self._tokens[trainend_index:].index('.') # number of extra tokens we need to move trainend_index to get
                                                        # to the end of the sentence

        d2 = self._tokens[validend_index:].index('.')  # number of extra tokens we need to move validend_index to get
                                                        # to the end of the sentence

        # round up trainend_index and validend_index accordingly
        trainend_index += d1
        validend_index += d2

        traindata = self._tokens[:trainend_index]
        validdata = self._tokens[trainend_index:validend_index]
        testdata  = self._tokens[validend_index:]

        self._traindata = traindata
        self._validdata = validdata
        self._testdata = testdata

        return traindata, validdata, testdata


    def print_summary_stats(self):
        '''
        print out some basic summary stats of the corpus
        '''
        train, valid, test = self.generate_datasets()
        print('======')
        print("Number of tokens in:"
              "--> training data: {} tokens"
              "--> validation data: {} tokens"
              "--> testing data: {} tokens\n".format(len(train), len(valid), len(test)))

        print("Size of vocabulary: {} tokens\n".format(len(self._tokenmap)))

        print("Number of <unk> tokens: {}\n".format(self._tokens.count("<unk>")))

        print("Number of stopwords: {}\n".format(len(self.get_stopwords())))

    def encode_as_ints(self, sequence):

        int_represent = []
        print('encode this sequence: %s' % sequence)
        print('as a list of integers.')
        tokens = nltk.word_tokenize(sequence)
        tokens = self._tag_sequence(tokens)

        for t in tokens:
            try:
                int_represent.append(self._tokenmap[t])
            except KeyError:
                int_represent.append(self._tokenmap['<unk>'])

        return(int_represent)


    def encode_as_text(self, int_represent):

        text = ''
        print('encode this list', int_represent)
        print('as a text sequence.')
        inv_map = {v: k for k, v in self._tokenmap.items()} #inverted self._tokenmap (switch values/keys)

        for i in int_represent:
            token = inv_map[i]
            text = text + ' ' + token

        return(text)

    def threshold(self,threshold):
        '''
        Takes a threshold value, parses token list, replaces those below threshold with <unk>
        Creates new list at self._tokens_threshold.
        Create new list with any removed tokens at self._tokens_removed
        '''

        unk = '<unk>'
        tokens_removed = list() # keep track of which tokens were omitted
        tokens_threshold = list() # new list with thresholded tokens

        for each in self._tokenmap.keys(): # loop through vocab
            if self._tokens.count(each) <= threshold: # used lessthanorequal because with threshold = 3 it has more "effect"
                tokens_removed.append(each)
                tokens_threshold.append(unk)
            else:
                tokens_threshold.append(each)

        unique_tokens = list(set(tokens_threshold))
        # reset self._tokens and self._tokenmap accordingly
        self._tokenmap = {unique_tokens[i]: i for i in range(len(unique_tokens))}
        self._tokens = tokens_threshold
        return

def main():
    corpus = my_corpus(None)

    train, valid, test = corpus.generate_datasets(split=[80, 10, 10])

    # write to txt
    with open('train.txt','w') as f:
        f.write(' '.join(train))
    with open('valid.txt','w') as f:
        f.write(' '.join(valid))
    with open('test.txt', 'w') as f:
        f.write(' '.join(test))

    text = input('Please enter a test sequence to encode and recover: ')
    print(' ')
    ints = corpus.encode_as_ints(text)
    print(' ')
    print('integer encodeing: ',ints)

    print(' ')
    text = corpus.encode_as_text(ints)
    print(' ')
    print('this is the encoded text: %s' % text)


if __name__ == "__main__":
    main()
