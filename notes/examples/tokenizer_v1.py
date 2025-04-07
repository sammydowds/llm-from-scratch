import urllib.request
import re

UNKOWN_WORD_TOKEN = "<|unk|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"
SPECIAL_TOKENS = [END_OF_TEXT_TOKEN, UNKOWN_WORD_TOKEN]

class TokenizerV1:
    def __init__(self, vocab):
        self.str_to_id = vocab 
        self.id_to_str= { i:s for s,i in vocab.items() }

    def encode(self, text):
        tokens = self.tokenize(text)
        tokens = [
            item if item in self.str_to_id else UNKOWN_WORD_TOKEN for item in tokens
        ]
        ids = [self.str_to_id[s] for s in tokens]
        return ids

    def decode(self, ids):
        text = " ".join([self.id_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) 
        return text
    
    @classmethod
    def create_vocab(self, text):
        all_words = sorted(list(set(self.tokenize(text))))
        all_words.extend(SPECIAL_TOKENS)
        return { token:integer for integer,token in enumerate(all_words) }
    
    def tokenize(self, text):
        result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        return [item.strip() for item in result if item.strip()] 