import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')


class my_corpus():
    def __init__(self, params):
        super().__init__() 
        
        self.params = params
        print('setting parameters')

        print('building corpus')
        with open('source_text.txt') as f:
            lines = f.readlines()

        fulltext = ' '.join(lines).lower()

        tokens = nltk.word_tokenize(fulltext)
        unique_tokens = list(set(tokens))
        self._tokens = tokens
        self._tokenmap = {unique_tokens[i]:i for i in range(len(unique_tokens))}

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

    def token_counts(self, corpus_tok)
        tok_cnt = {}
        for tok in corpus_tok:
            tok_cnt[tok] = corpus_tok.count(tok)
            if tok_cnt[tok] < 3:
                # corpus_tok = re.sub(r"\bword\b", "unk", corpus_tok)
                corpus_tok.replace(tok, 'unk')

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

    lower = tokens(text)
    print(' ')
    print('this is the tokenized text: %s' % text)
    
if __name__ == "__main__":
    main()