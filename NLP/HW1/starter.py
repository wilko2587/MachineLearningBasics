import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')


class my_corpus():
    def __init__(self, params):
        super().__init__() 
        
        self.params = params
        print('setting parameters')
    
    def encode_as_ints(self, sequence):
        
        int_represent = []
        
        print('encode this sequence: %s' % sequence)
        print('as a list of integers.')
        
        return(int_represent)
    
    def encode_as_text(self,int_represent):

        text = ''
        
        print('encode this list', int_represent)
        print('as a text sequence.')
        
        return(text)

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
        
    
    
              