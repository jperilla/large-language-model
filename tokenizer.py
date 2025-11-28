import re

# Tokenizer Class
class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {index: token for token, index in vocab.items()}

    def encode(self, text):
        preproccessed = re.split(r'([,.:;!?_"()\']|--|\s+)', text)
        preproccessed = [item.strip() for item in preproccessed if item.strip()]
        preproccessed = [item if item in self.str_to_int else "<|unk|>" for item in preproccessed]
        ids = [self.str_to_int[token] for token in preproccessed]
        return ids

    def decode(self, ids):
        text = ' '.join([self.int_to_str[id_] for id_ in ids if id_ in self.int_to_str])
        return re.sub (r'\s+([,.:;!?_"()\'])', r'\1', text)

    