import nltk
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
                tokens = nltk.word_tokenize(line.strip().lower())
                for tok in tokens:
                    token_list.append(tok)

        unique_tokens = list(set(token_list))
        self._tokens = token_list

        print('building token map')
        self._tokenmap = {unique_tokens[i]:i for i in range(len(unique_tokens))}

    def replace_numbers(self):

        # this mostly works for years
        self._tokens_trimmed = [re.sub('^[12][0-9]{3}$*', '<year>', tok) for tok in corpus._tokens]

        # this matches integers
        self._tokens_trimmed = [re.sub('^[0-9]*$', '<integer>', tok) for tok in corpus._tokens_trimmed]


    def get_counts(self):
        token_count = dict()

        for each in self._tokens:
            token_count[each] = token_count.get(each,0) + 1

        self._tokencount = token_count

    def encode_as_ints(self, sequence):
        
        int_represent = []
        print('encode this sequence: %s' % sequence)
        print('as a list of integers.')
        tokens = nltk.word_tokenize(sequence)

        for t in tokens:
            int_represent.append(self._tokenmap[t])

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
    
    #traverses tokens in tokenized list corpus_tok, identifies numbers and classifies as either year or int
    def conv_to_year_int(self, corpus_tok):
        for tok in corpus_tok:
            if int(tok) != null:
                if len(tok) == 4:
                    corpus_tok.replace(tok, '<year>')
                else:
                    corpus_tok.replace(tok, '<int>')
        return corpus_tok

    # def token_counts(self, corpus_tok):
    #     tok_cnt = {}
    #
    #     for tok in corpus_tok:
    #         tok_cnt[tok] = corpus_tok.count(tok)
    #         if tok_cnt[tok] < 3:
    #             # corpus_tok = re.sub(r"\bword\b", "unk", corpus_tok)
    #             corpus_tok.replace(tok, 'unk')
    #     return corpus_tok

    def threshold(self,threshold):
        '''
        Takes a threshold value, parses token list, replaces those below threshold with <UNK>
        Creates new list at self._tokens_threshold.
        Create new list with any removed tokens at self._tokens_removed
        '''

        unk = '<unk>'
        self._tokens_threshold = list()
        self._tokens_removed = list()

        for each in self._tokens:
            if self._tokencount[each] < threshold:
                self._tokens_removed.append(each)
                self._tokens_threshold.append(unk)
            else:
                self._tokens_threshold.append(each)

def tokens(sequence):
    return sent_tokenize(sequence)
    
def main():
    corpus = my_corpus(None)
    
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
    # main()

    corpus = my_corpus(None)
    #
    # for each in corpus._tokens:
    #     if re.search('^[12][0-9]{3}$', each):
    #         print(each)

    # token = list()
    #
    # with open(('source_text.txt')) as f:
    #     for line in f:
    #         tokens = nltk.word_tokenize(line.strip().lower())
    #         for tok in tokens:
    #             token.append(tok)
