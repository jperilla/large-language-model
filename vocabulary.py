
import re

class Vocabulary:
    def __init__(self, raw_text):
        preproccessed = re.split(r'([,.:;!?_"()\']|--|\s+)', raw_text)
        preproccessed = [token for token in preproccessed if token.strip()]
        self.all_words = sorted(set(preproccessed))
        self.all_words.extend(["<|endoftext|>", "<|unk|>"]) # Add special tokens
        self.token_to_id = {token: integer for integer, token in enumerate(self.all_words)}
        self.id_to_token = {integer: token for integer, token in enumerate(self.all_words)}

    def size(self):
        return len(self.token_to_id)
    
    def token_to_index(self, token):
        return self.token_to_id.get(token, None)

    def index_to_token(self, index):
        return self.id_to_token.get(index, None)