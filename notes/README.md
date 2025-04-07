# Misc notes on data loading 

## Embeddings

Vectors are used to represent text. These are referred to as embeddings which can represent a word, series of words, sentences, or paragraphs. 

### Dimensions

Smaller models such as GPT-2 (117M-125M parameters) use an embedding size of 768 dimensions, while larger ones such as GPT-3 (175B parameters) uses an embedding size of 12, 288 dimensions.

### Preparation

Splitting text into words, converting words to tokens, and turning tokens into embedding vectors. 

### Steps

1. Create tokens
2. Create Vocab: Unique set of tokens, labeled with IDs via BPE tokenizer
3. Handle special context tokens (`<|endoftext|>`)
4. Extract data samples for training (input, target pairs - x, y via sliding window approach)
5. Create initial input embeddings
6. Encode word positions into input embeddings

Note: Run `examples/sample.py` to see the above process in action.
