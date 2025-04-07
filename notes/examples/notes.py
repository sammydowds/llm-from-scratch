import torch
from data_loader_v1 import create_dataloader_v1 
import tiktoken 
from attention import SelfAttentionV1

# https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt
with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

# ------------- SECTION 1: Tokenize and Embed ----------------

# init tokenizer
tokenizer = tiktoken.get_encoding('gpt2')
vocab_size = tokenizer.n_vocab 

# Create embeddings layer - vector 256 dimension per input token
output_dim = 256 
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Tokenize and get input, targets 
max_length = 4
dataloader = create_dataloader_v1(raw_text, tokenizer=tokenizer, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# Embed a batch of inputs 
token_embeddings = token_embedding_layer(inputs)

# Create embedding layer for word position - GPT absolute position 
context_length = max_length 
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
abs_pos_vector = torch.arange(context_length) 
pos_embeddings = pos_embedding_layer(abs_pos_vector)

# Encode word position into batch of embeddings
input_embeddings = token_embeddings + pos_embeddings

# input_embeddings ready LLM training 
print("\n-------- Sample of Data processed (Input) 1/8 Batch --------")
print("\nSample Context (Words):\n", tokenizer.decode(inputs[0].tolist()))
print("\nSample Context (Tokens):\n", inputs[0])
print("\nSample Context (Embeddings):\n", token_embeddings[0])
print("\nSample Context (Embeddings + Word Position Applied):\n", input_embeddings[0])

# ------------- SECTION 2: Attention ----------------
sample_batch = input_embeddings[0] 

# EXAMPLE - Simplified Self Attention

# 1.0: Context Vector: calculate context vector for second token in first batch

# Scores and weights
query = sample_batch[1] # embedding for second token 
attn_scores_2 = torch.empty(sample_batch.shape[0])
for i, x_i in enumerate(sample_batch):
    attn_scores_2[i] = torch.dot(x_i, query)
attn_weights_2_tmp = torch.softmax(attn_scores_2, dim=0)
print("Attention weights, 2nd token, first batch\n", attn_weights_2_tmp)

# Context vector 
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(sample_batch):
    context_vec_2 += attn_weights_2_tmp[i] * x_i
print("Context Vector, 2nd token, first batch\n", context_vec_2)

# 1.1: Context Vectors for all input tokens in the batch

# Scores and weights
attn_scores = sample_batch @ sample_batch.T 
print("Batch attention scores\n", attn_scores)
attn_weights = torch.softmax(attn_scores, dim=-1)
print("Batch attention weights\n", attn_weights)

# Context vectors
all_context_vecs = attn_weights @ sample_batch 
print("Batch context vectors\n", all_context_vecs)

# EXAMPLE - scaled dot-product attention

# Introducing 3 trainable weight matrices - query, key, value
x_2 = sample_batch[1]
d_in = sample_batch.shape[1]
d_out = sample_batch.shape[1] 
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 2-D vectors
query_2 = x_2 @ W_query
keys = sample_batch @ W_key 
values = sample_batch @ W_value
keys_2 = keys[1]

# score
attn_score_2 = query_2 @ keys.T
print("Attention Score (Omega) for query 2 (2nd token):", attn_score_2)

# Context vector for token 2
weights_2 = torch.softmax(attn_score_2 / keys.shape[1]**0.5, dim=-1) 
print("Weights for query 2 (Alpha)\n", weights_2)
context_vector_2 = weights_2 @ values 
print("Context vector token 2\n", context_vector_2)

# Use v1 class for simple SelfAttention
torch.manual_seed(123)
sa = SelfAttentionV1(d_in=sample_batch.shape[-1], d_out=sample_batch.shape[-1])
print(sa(sample_batch))