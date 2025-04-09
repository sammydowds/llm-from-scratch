from configs import GPT_SMALL, DEFAULT, model_configs 
from loaders import get_verdict_data_loaders 
from gpt import GPTModel
from train import train_model_simple
import tiktoken
import torch
import os
import urllib.request

SIMPLY_TRAINED_MODEL_CACHE_PATH = "simply-trained-model.pth"
SMALL_GPT_2_CACHE_PATH = "gpt2-small-124M.pth"
SMALL_GPT_2_REMOTE_PATH = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{SMALL_GPT_2_CACHE_PATH}"

def get_simply_trained_model(skip_cache = False):
    torch.manual_seed(123)
    model = GPTModel(GPT_SMALL)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1
    )
    tokenizer = tiktoken.get_encoding('gpt2')
    device = torch.device("cpu")
    num_epochs = 10

    # check cache 
    if not skip_cache:
        cached = None
        try: 
            cached = torch.load(SIMPLY_TRAINED_MODEL_CACHE_PATH)
        except:
            print("Unabled to find cached model.")
        if cached:
            print("Found cached model, skipping training.")
            model.load_state_dict(cached['model_state_dict'])
            optimizer.load_state_dict(cached['optimizer_state_dict'])
            model.eval()
            return model, { "meta": {}, "tokenizer": tokenizer, "optimizer": optimizer }
    else:
        print("Skipping cache read. Initializing training...")

    train_loader, val_loader = get_verdict_data_loaders()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    # cache model, optimizer
    if not skip_cache:
        print("Training complete. Caching model.")
        torch.save({ "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, SIMPLY_TRAINED_MODEL_CACHE_PATH)

    return model, { "tokenizer": tokenizer, "optimizer": optimizer, "meta": { "train_losses": train_losses, "val_losses": val_losses, "tokens_seen": tokens_seen }}  

def get_small_gpt_2_model():
    if not os.path.exists(SMALL_GPT_2_CACHE_PATH):
        urllib.request.urlretrieve(SMALL_GPT_2_REMOTE_PATH, SMALL_GPT_2_CACHE_PATH)
        print(f"Downloaded to {SMALL_GPT_2_CACHE_PATH}")
    
    model = GPTModel(GPT_SMALL)
    model.load_state_dict(torch.load(SMALL_GPT_2_CACHE_PATH, weights_only=True))
    model.eval()
    model.to(torch.device("cpu"))

    return model
