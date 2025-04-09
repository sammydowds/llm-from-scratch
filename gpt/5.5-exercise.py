import torch
from generate import text_to_token_ids, generate, token_ids_to_text
from trained_model import get_small_gpt_2_model
from loaders import get_verdict_data_loaders
import tiktoken
from train import calc_loss_loader
from configs import GPT_SMALL

# get GPT-2 small 
model = get_small_gpt_2_model()
tokenizer = tiktoken.get_encoding('gpt2')

train_loader, val_loader = get_verdict_data_loaders()
print("GPT-2-SMALL: Train loss:\n", calc_loss_loader(train_loader, model, torch.device("cpu")))
print("GPT-2-SMALL: Val loss:\n", calc_loss_loader(val_loader, model, torch.device("cpu")))

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves", tokenizer).to(torch.device("cpu")),
    max_new_tokens=30,
    context_size=GPT_SMALL["context_length"],
    top_k=1,
    temperature=1.0
)
print(token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer))
