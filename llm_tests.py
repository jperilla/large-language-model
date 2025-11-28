
import file_utils
from gpt_models import DummyGPTModel
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
#print("First batch input IDs:", first_batch)
#print("\n")
second_batch = next(data_iter)
#print("Second batch input IDs:", second_batch)

# Convert tokens to embeddings - small example
import torch
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#rint(embedding_layer.weight)
#print(embedding_layer(torch.tensor([3])))

# Convert tokens to embeddings - larger example
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
#print("Input IDs shape:", inputs.shape)
token_embeddings = token_embedding_layer(inputs)
#print("Token embeddings shape:", token_embeddings.shape)
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
#print(pos_embeddings.shape)
input_embeddings = token_embeddings + pos_embeddings
#print(input_embeddings.shape)


# Testing Self-Attention mechanism on small input
inputs = torch.tensor([[0.43, 0.15, 0.89],
                       [0.55, 0.87, 0.66],
                       [0.57, 0.85, 0.64],
                       [0.22, 0.58, 0.33],
                       [0.77, 0.25, 0.10],
                       [0.05, 0.80, 0.55]])

d_in = inputs.shape[1]
d_out = 2
batch = torch.stack([inputs, inputs], dim=0)  # Create a batch of size 2
context_length = batch.shape[1]

#print(d_in, d_out)
from attention import SelfAttentionV1
torch.manual_seed(123)
sa_v1 = SelfAttentionV1(d_in, d_out)
#print(sa_v1(inputs))

from attention import SelfAttentionV2
torch.manual_seed(789)
sa_v2 = SelfAttentionV2(d_in, d_out)
#print(sa_v2(inputs))

from attention import CausalAttention
torch.manual_seed(123)
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vec = ca(batch)
#print("context_vec shape:", context_vec.shape)

# Testing Dummy GPT Model
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

# encode texts and add to batch
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)  # Shape: (batch_size, seq_length)
print(batch)

dummy_gpt_model = DummyGPTModel(GPT_CONFIG_124M)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
#print("Logits shape:", logits.shape)  # Expected shape: (batch_size, seq
#print(logits)

# Testing Full GPT Model
from gpt_models import GPTModel

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch", batch)
print("Output shape:", out.shape) 
print(out) # Expected shape: (batch_size, seq