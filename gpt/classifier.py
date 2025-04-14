from trained_model import get_small_gpt_2_model
import torch
from configs import GPT_SMALL
import tiktoken

gpt = get_small_gpt_2_model()

# default layers
print(gpt)

# tweak output to 2 nodes
torch.manual_seed(123)
num_classes = 2
gpt.out_head = torch.nn.Linear(
    in_features=GPT_SMALL["emb_dim"],
    out_features=2,
)

# tweak layer params
for param in gpt.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in gpt.final_norm.parameters():
    param.requires_grad = True

tokenizer = tiktoken.get_encoding('gpt2')
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)

print(inputs)
with torch.no_grad():
    outputs = gpt(inputs)
print(outputs)
