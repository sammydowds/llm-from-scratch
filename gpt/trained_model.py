from configs import GPT_CONFIG_124M 
from loaders import create_dataloader_v1
from gpt import GPTModel
from train import train_model_simple
import tiktoken
import torch

SIMPLY_TRAINED_MODEL_CACHE_PATH = "simply-trained-model.pth"

def get_simply_trained_model(file_path = "the-verdict.txt", skip_cache = False):
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
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
            return model, { "meta": {}, "tokenizer": tokenizer, "optimizer": optimizer }
    else:
        print("Skipping cache read. Initializing training...")

    with open(file_path, "r", encoding='utf-8') as file:
        text_data = file.read()
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    
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
