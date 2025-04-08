import torch
from trained_model import get_small_gpt_2_model
from loaders import get_verdict_data_loaders
import tiktoken
from train import calc_loss_loader

# get GPT-2 small 
model = get_small_gpt_2_model()
tokenizer = tiktoken.get_encoding('gpt2')

train_loader, val_loader = get_verdict_data_loaders()
print("GPT-2-SMALL: Train loss:\n", calc_loss_loader(train_loader, model, torch.device("cpu")))
print("GPT-2-SMALL: Val loss:\n", calc_loss_loader(val_loader, model, torch.device("cpu")))
