
import file_utils
from tokenizer import SimpleTokenizer
from vocabulary import Vocabulary

# Read the text file
raw_text = file_utils.read_file_local("./input/the-verdict.txt")
#print("Total characters in text:", len(raw_text))
#print(raw_text[:99])  # Print the first 100 characters to verify content

# Create vocabulary
vocab = Vocabulary(raw_text)

# Tokenize some text
tokenizer = SimpleTokenizer(vocab.token_to_id)
some_text = "Hello, do you like tea?"
new_text = "Hi, I was tired -- all the time."
extended_text = " <|endoftext|> ".join([some_text, new_text])

# Try a BPE tokenizer from tiktoken
from importlib.metadata import version
import tiktoken

tiktokenizer = tiktoken.get_encoding("gpt2")
assert tiktokenizer.decode(tiktokenizer.encode("hello world")) == "hello world"

enc_text = tiktokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
#print(len(tiktokenizer.encode(raw_text)))


# Create input and target pairs, where context is the input sequence
# and the desired output is the next token, this is used for worf prediction
enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:-context_size]
y = enc_sample[1:context_size + 1]

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    #print(tiktokenizer.decode(context), " --> ", tiktokenizer.decode([desired]))  



# Use GPT Dataset and DataLoader
from datasets import GPTDataset, create_dataloader

dataloader = create_dataloader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print("First batch input IDs:", first_batch)
print("\n")
second_batch = next(data_iter)
print("Second batch input IDs:", second_batch)